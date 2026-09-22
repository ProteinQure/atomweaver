"""AtomWeaver inference sampler: reverse-diffusion side-chain design.

Loads a trained AtomWeaver model, samples all-atom side-chain point clouds for each
input backbone (+ optional target) PDB under the production shell prior + sampling
recipe, and exports the clouds. Identity is read off separately by
``apply_hybrid_readout.py`` (the b2_balanced NDM + logistic-regression read-out);
``design.py`` orchestrates sample -> read-out -> designs.

Most users should run ``design.py`` (the one-command front door). This sampler is the
implementation step it invokes; the additional flags below are advanced/diagnostic and are
not needed for standard design.

Modes:
  * JOINT (default): design ALL peptide side chains together in one coupled sample.
  * SUBSET (--design-positions 3,7,10): design the named positions (1-based, matching
    PDB residue order), holding the rest fixed to the input side chains.
"""

import warnings
from string import ascii_lowercase, ascii_uppercase, digits

# Quiet the noisy load-time warnings from the ML stack (torch / sklearn version notes);
# real problems still surface as DeprecationWarning / RuntimeWarning, which stay visible.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from typing import Optional

import torch
import typer

from atomweaver.joint_diffusion.datasets import peptide_chain_ccds
from atomweaver.joint_diffusion.sampling import SamplingConfig, apply_sampling_config

app = typer.Typer(add_completion=False)

CANON = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

THREE2ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

_ELEM_NAME = {1: "C", 2: "N", 3: "O", 4: "S"}  # model element encoding (PAD=0 skipped)
_BB_NAMES = ("N", "CA", "C", "O")


# Read-time discretiser: normalized distance-matrix (NDM) correlation.


def _atom_line(n, aname, rname, ch, resnum, xyz, el):
    x, y, z = xyz
    return (
        f"ATOM  {n:5d}  {aname:<3s} {rname:>3s} {ch}{resnum:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
    )


def _target_pdb_records(pdb_path, peptide_chain):
    """Keep original target atom records, including atoms outside the conditioning crop."""
    return [
        line
        for line in Path(pdb_path).read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM", "ANISOU")) and (line[21].strip() or "A") != peptide_chain
    ]


def _make_cloud_pdb(batch, out, p_idx, ccds, gt_seq, *, target_records):
    """Export the original target alongside input-peptide and generated-cloud copies.

    Preserve target atom records verbatim. Choose unused chain IDs for the peptide
    copies (prefer B/C) and identify their roles for the read-out in a REMARK.
    p_idx contains exported peptide positions in sequence order. Generated residues
    remain UNK until the hybrid read-out assigns their identities.
    """
    target_chains = {line[21] for line in target_records}
    available = [chain for chain in ascii_uppercase + ascii_lowercase + digits if chain not in target_chains]
    if len(available) < 2:
        raise ValueError("Cloud export needs two unused PDB chain IDs for the peptide copies")
    generated_chain = "C" if "C" in available else available[0]
    available.remove(generated_chain)
    input_chain = "B" if "B" in available else available[0]
    bb = batch["backbone_coords"][0].cpu()  # (L,4,3)
    bbm = batch["backbone_mask"][0].cpu()  # (L,4)
    gsc = batch["sidechain_coords"][0].cpu()  # GT sidechain (L,maxsc,3)
    gscm = batch["sidechain_mask"][0].cpu()
    psc = out["sidechain_coords"][0].cpu()  # generated sidechain (L,maxsc,3)
    pscm = out["predicted_mask"][0].cpu()
    pel = out["element_types"][0].cpu()  # (L,maxsc) model encoding
    lines = [
        f"REMARK  True sequence:        {gt_seq}",
        "REMARK  Target atom records preserved from input",
        f"REMARK  ATOMWEAVER_CHAINS {input_chain} {generated_chain}",
        f"REMARK  Chain {input_chain} = GT peptide, {generated_chain} = generated cloud",
    ]
    previous_chain = None
    for record in target_records:
        if previous_chain is not None and record[21] != previous_chain:
            lines.append("TER")
        lines.append(record)
        previous_chain = record[21]
    if target_records:
        lines.append("TER")
    n = max((int(line[6:11]) for line in target_records), default=0) + 1
    # Input peptide and generated cloud use separate, unoccupied chain IDs.
    for chain, resnames, sc, scm, el_src in (
        (input_chain, [ccds[p] for p in p_idx], gsc, gscm, None),
        (generated_chain, ["UNK"] * len(p_idx), psc, pscm, pel),
    ):
        for k, p in enumerate(p_idx):
            rn = (resnames[k] or "UNK")[:3].upper()
            for ai, an in enumerate(_BB_NAMES):
                if bool(bbm[p, ai]):
                    lines.append(_atom_line(n, an, rn, chain, k + 1, bb[p, ai].tolist(), an[0]))
                    n += 1
            for si in range(scm.shape[1]):
                if bool(scm[p, si]):
                    el = _ELEM_NAME.get(int(el_src[p, si]), "C") if el_src is not None else "C"
                    lines.append(_atom_line(n, f"{el}{si + 1}", rn, chain, k + 1, sc[p, si].tolist(), el))
                    n += 1
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _load_model(checkpoint, residue_db, device, coord_process_type=None):
    """Rebuild the eval model holder from a checkpoint (public-inference loader).

    Loads the checkpoint through a direct call into the shipped ``model_loader`` module,
    so inference no longer needs the training script on disk.
    """
    from atomweaver.joint_diffusion.model_loader import load_model

    return load_model(Path(checkpoint), residue_db, device)


def _device_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _target_kwargs(batch):
    out = {}
    for k in [
        "target_backbone_coords",
        "target_backbone_mask",
        "target_residue_types",
        "target_seq_mask",
        "target_coords",
        "target_mask",
        "target_atom_type",
        "target_atom_element_type",
        "target_atom_residue_type",
        "target_atom_is_backbone",
    ]:
        if k in batch:
            out[k] = batch[k]
    return out


@app.command()
def main(
    pdb_dir: str = typer.Option(..., help="Directory of input PDBs (peptide backbone + optional target)."),
    checkpoint: str = typer.Option(..., help="Model checkpoint (.ckpt/.pt)."),
    sampling_db: str = typer.Option(..., help="Residue DB for model load (builds name_to_idx)."),
    repo: str = typer.Option(".", help="atomweaver checkout root."),
    design_positions: str = typer.Option(
        "",
        "--design-positions",
        help="Comma-separated 1-based residue positions to design, holding the rest fixed (e.g. '3,7,10'). "
        "Default empty = JOINT: design ALL peptide side chains together in one coupled sample.",
    ),
    num_samples: int = typer.Option(1, help="samples per input structure"),
    num_steps: int = typer.Option(250),
    export_clouds: str = typer.Option("", help="write per-design atom-cloud PDBs to this dir"),
    device: str = typer.Option("cuda"),
    max_pdbs: int = typer.Option(0, help="0 = all"),
    reserved_slot0: Optional[bool] = typer.Option(
        None,
        "--reserved-slot0/--no-reserved-slot0",
        help="reserved-slot0 slotization for the eval dataset (slot0=N-connecting, slot1=Cbeta, radial, "
        "cap-13). DEFAULT: auto-detected from the checkpoint's stored reserved_slot0 hparam (+ cross-checked "
        "against the model's per-slot fill-rate buffer). An explicit value must agree with the checkpoint.",
    ),
    reserved_slot0_prefix_exempt: bool = typer.Option(
        True,
        "--reserved-slot0-prefix-exempt/--no-reserved-slot0-prefix-exempt",
        help=(
            "exempt reserved-slot0 from the hard prefix constraint at sampling. Enabled by default to prevent "
            "the slot-0/prefix collapse; pass --no-reserved-slot0-prefix-exempt only for an intentional ablation."
        ),
    ),
):
    from functools import partial

    from torch.utils.data import DataLoader

    from atomweaver.joint_diffusion.datasets import (
        PeptideDataset,
        collate_peptides,
        peptide_chain_from_filename,
        resolve_reserved_slot0,
    )

    if num_steps < 2:
        raise typer.BadParameter("--num-steps must be >= 2 (single-step sampling is not supported).")

    coord_process_type = ""  # production: use the checkpoint's coord process
    graph_num_edge_types = 3  # production: peptide graph regime
    inpaint_mode = "clean"  # production: clean pinning for --design-positions subset

    # Resolve the sampling recipe first, so a bad value is caught in milliseconds rather than after
    # the model has loaded.
    cfg = SamplingConfig.from_env()

    lm = _load_model(Path(checkpoint), sampling_db, device, coord_process_type=(coord_process_type or None))
    model = lm.model
    model.eval()
    apply_sampling_config(model, cfg)

    # AUTO-DETECT the reserved-slot0 layout from the checkpoint (stored hparam) so the eval dataset
    # is slotized to match how the model was trained WITHOUT relying on a manual --reserved-slot0 flag.
    # A stored value wins by default; an explicit flag that DISAGREES errors; older ckpts (no stored
    # value) fall back to the flag with a warning. Cross-checked against the model's _slot_fill_rate
    # buffer (authoritative): slot0<<slot1 => reserved, slot0-high => not-reserved.
    reserved_slot0 = resolve_reserved_slot0(
        stored=getattr(lm, "_ckpt_reserved_slot0", None),
        cli_value=reserved_slot0,
        slot_fill_rate=getattr(model, "_slot_fill_rate", None),
    )
    typer.echo(f"[reserved-slot0] resolved slot layout: reserved_slot0={reserved_slot0}")
    # REGIME MATCH: force the sampling-time graph edge-type count. The eval loader does not
    # thread graph_num_edge_types out of the ckpt (it is absent from saved hyper_parameters),
    # so without this a graph-2-trained monomer would sample at graph-3 (activating the frozen
    # binder-target row 2) -> off-regime. 0 = no override (peptide graph-3 default preserved).
    # Lives on the denoiser (SidechainDenoiser owns _graph_num_edge_types + the graph build).
    if graph_num_edge_types > 0:
        model.denoiser._graph_num_edge_types = int(graph_num_edge_types)
        typer.echo(
            f"[regime] forced sampling graph_num_edge_types={graph_num_edge_types} "
            f"(embedding rows / num_edge_types={model.denoiser.num_edge_types})"
        )
    # Match the dataset's atom-slot width to the loaded model (v60-lineage = 16, legacy = 14)
    # so GT sidechains aren't truncated; falls back to 14 if the model lacks the flow buffer.
    eval_max_sc = int(getattr(model.coord_flow, "_shell_radii", torch.zeros(14)).shape[0])

    dataset = PeptideDataset(
        pdb_dir, max_binder_length=32, max_sidechain_atoms=eval_max_sc, reserved_slot0=reserved_slot0
    )
    collate = partial(collate_peptides, name_to_idx=lm.name_to_idx)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    typer.echo(f"[sampler] {len(test_loader.dataset)} input PDB(s); steps={num_steps} samples/input={num_samples}")

    _design_pos = None
    if design_positions.strip():
        _design_pos = sorted({int(x) - 1 for x in design_positions.split(",") if x.strip() != ""})
        typer.echo(f"[sampler] SUBSET design: positions {[p + 1 for p in _design_pos]} (1-based); rest held fixed")
    else:
        typer.echo("[sampler] JOINT design: all peptide side chains sampled together (default)")

    n_done = 0
    for bi, batch in enumerate(test_loader):
        if max_pdbs and n_done >= max_pdbs:
            break
        batch = _device_batch(batch, device)
        pdb_name = (batch.get("pdb_names") or ["?"])[0]
        pdb_path = str(Path(pdb_dir) / f"{pdb_name}.pdb")
        if not Path(pdb_path).exists():
            continue
        peptide_chain, ccds = peptide_chain_ccds(pdb_path, pep_chain=peptide_chain_from_filename(pdb_path))
        target_records = _target_pdb_records(pdb_path, peptide_chain) if export_clouds else []
        B, L, maxsc = batch["sidechain_mask"].shape
        if len(ccds) != L:
            typer.echo(f"  [skip {pdb_name}] CCD/len mismatch ({len(ccds)} vs {L})")
            continue
        seq_mask = batch.get("seq_mask")
        gt_coords, gt_el, gt_m = batch["sidechain_coords"], batch["sidechain_element_types"], batch["sidechain_mask"]
        # BUG-B fix (2026-06-25): relocate ghost (mask=0) sidechain slots to the residue CA before pinning,
        # mirroring TRAINING's coords_for_diffusion = torch.where(sidechain_mask, coords, ca_expanded) (models.py
        # ~5033). Without this, inpaint pins context ghost slots at the origin (0,0,0) -> off-manifold SE(3) graph
        # nodes for context residues -> deflated subset design. Separate var so gt_coords stays untouched.
        _ca_pin = batch["backbone_coords"][..., 1, :].unsqueeze(2).expand_as(gt_coords)  # (B,L,maxsc,3) CA broadcast
        gt_coords_pin = torch.where(gt_m.unsqueeze(-1).bool(), gt_coords, _ca_pin)

        # Chirality sign per position. DEFAULT = blind L-init EVERYWHERE (oracle-chirality OFF).
        # Standing decision (2026-08-31): we ALWAYS L-init nowadays -- the pseudo-Cβ source
        # cone points L for ALL positions, so any correct D is GENUINE discovery, not oracle-fed
        # placement. The cone is only consumed inside sample() when the checkpoint has
        # pseudo_cb_direction set.

        # Blind L-init for all positions: identity handedness is never fed to the cone.
        chirality = None

        common = dict(
            backbone_coords=batch["backbone_coords"],
            backbone_mask=batch["backbone_mask"],
            sidechain_mask=torch.ones_like(gt_m),
            seq_mask=seq_mask,
            num_steps=num_steps,
            neighbor_x0_packing_recycles=cfg.recycles,
            element_sampling_temp_max=1.0,
            reserved_slot0_prefix_exempt=reserved_slot0_prefix_exempt,
            chirality=chirality,
            **_target_kwargs(batch),
        )

        design_position_groups = [None] if _design_pos is None else [_design_pos]
        for dp in design_position_groups:
            design = torch.zeros(B, L, dtype=torch.bool, device=device)
            if dp is None:
                design = (
                    seq_mask.bool().clone()
                    if seq_mask is not None
                    else torch.ones(B, L, dtype=torch.bool, device=device)
                )
                inpaint_kw = {}
            else:
                for _p in dp:
                    if 0 <= _p < L:
                        design[:, _p] = True
                inpaint_kw = {
                    "design_mask": design,
                    "inpaint_gt_coords": gt_coords_pin,
                    "inpaint_gt_elements": gt_el,
                    "inpaint_gt_mask": gt_m,
                    "inpaint_mode": inpaint_mode,
                }
            for _s in range(num_samples):
                out = model.sample(**common, **({} if dp is None else inpaint_kw))
                pred_coords, pred_mask, pred_el = out["sidechain_coords"], out["predicted_mask"], out["element_types"]
                if export_clouds:
                    pep_pos = [p for p in range(L) if bool(design[0, p])]
                    gt_seq = "".join(THREE2ONE.get(ccds[p], "X") for p in pep_pos)
                    Path(export_clouds).mkdir(parents=True, exist_ok=True)
                    pdb_str = _make_cloud_pdb(batch, out, pep_pos, ccds, gt_seq, target_records=target_records)
                    (Path(export_clouds) / f"{pdb_name}_s{_s}.pdb").write_text(pdb_str)
        n_done += 1
    typer.echo(f"[sampler] done: {n_done} input(s) sampled.")


if __name__ == "__main__":
    app()
