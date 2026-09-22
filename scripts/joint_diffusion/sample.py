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

import os
import sys
import warnings

# Quiet the noisy load-time warnings from the ML stack (torch / sklearn version notes / biotite);
# real problems still surface as DeprecationWarning / RuntimeWarning, which stay visible.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from typing import Optional

import torch
import typer

# ``sampling_knobs`` sits beside this script. Run as a script, this directory is already on sys.path;
# imported as a library it is not, so state the path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sampling_knobs  # noqa: E402
from sampling_knobs import SamplingConfig  # noqa: E402

from atomweaver.joint_diffusion.datasets import peptide_chain_ccds  # noqa: E402

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


def _make_cloud_pdb(batch, out, p_idx, ccds, idx_to_ccd, r_top1_row, des_seq, gt_seq):
    """3-chain cloud PDB: A=target bb, B=GT peptide (bb+GT sidechain), C=generated cloud.

    Mirrors the training-time writer: coords are absolute; generated sidechain atoms are
    named {element}{slot}; GT/designed residue names give chains B/C identities (NCAA CCDs
    preserved on chain B). p_idx = peptide positions (list of L-indices, in order).
    """
    bb = batch["backbone_coords"][0].cpu()  # (L,4,3)
    bbm = batch["backbone_mask"][0].cpu()  # (L,4)
    gsc = batch["sidechain_coords"][0].cpu()  # GT sidechain (L,maxsc,3)
    gscm = batch["sidechain_mask"][0].cpu()
    psc = out["sidechain_coords"][0].cpu()  # generated sidechain (L,maxsc,3)
    pscm = out["predicted_mask"][0].cpu()
    pel = out["element_types"][0].cpu()  # (L,maxsc) model encoding
    lines = [
        f"REMARK  Designed (de-novo): {des_seq}",
        f"REMARK  True sequence:        {gt_seq}",
        "REMARK  Chain A = target bb, B = GT peptide, C = generated cloud",
    ]
    n = 1
    # Chain A: target backbone (placeholder GLY resname -- context only)
    tbb = batch.get("target_backbone_coords")
    if tbb is not None:
        tbb = tbb[0].cpu()
        tbbm = batch.get("target_backbone_mask")
        tbbm = tbbm[0].cpu() if tbbm is not None else torch.ones(tbb.shape[:2], dtype=torch.bool)
        for ri in range(tbb.shape[0]):
            for ai, an in enumerate(_BB_NAMES):
                if bool(tbbm[ri, ai]):
                    lines.append(_atom_line(n, an, "GLY", "A", ri + 1, tbb[ri, ai].tolist(), an[0]))
                    n += 1
        lines.append("TER")
    # Chain B (GT) + Chain C (generated)
    for chain, resnames, sc, scm, el_src in (
        ("B", [ccds[p] for p in p_idx], gsc, gscm, None),
        ("C", [idx_to_ccd.get(int(r_top1_row[p]), "GLY") for p in p_idx], psc, pscm, pel),
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


def _checkpoint_provenance_name(checkpoint: str = "") -> str:
    """
    Compact checkpoint identity for per-row provenance.

    ``last.ckpt`` alone is not an identity in this workflow: both checkpoint axes in the cyclosporin
    grid are usually loaded from a run directory ending in ``checkpoints/last.ckpt``. Keep named
    checkpoint files unchanged, but qualify generic ``last.ckpt`` with the run directory so the grid
    can verify an epoch label from the CSV itself.
    """
    if not checkpoint:
        return ""
    path = Path(checkpoint)
    if path.name != "last.ckpt":
        return path.name
    run_dir = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    return f"{run_dir.name}/{path.name}" if run_dir.name else path.name


def apply_sampling_config(model, cfg: SamplingConfig) -> None:
    """
    Apply a resolved sampling recipe to an already-built model.

    Factored out of ``main`` because it was only ever reachable through the Typer CLI. Studies that
    import this eval as a library call ``load_model`` directly and so silently
    ran with every knob here unapplied, while knobs read deeper in the model (geometric reconcile,
    the absorbing override) did fire. The NCAA panel therefore reported a shell-variance recipe for
    runs that never used one, and the only tell was a missing log line.

    Byte-identical to the untouched model when ``cfg`` is at its defaults; every branch prints what
    it did.

    APPLIED AT MOST ONCE PER MODEL. The shell knobs mutate registered buffers in place, so a second
    call would compound them -- thickness by ``vs`` instead of ``sqrt(vs)``, radial variance by
    ``vs**2`` -- while printing the same line both times. No caller does that today, but this
    function exists precisely so library callers invoke it themselves, and the CLI's own ``main``
    still calls it, so a driver that did both would silently sample at a variance nobody configured.
    Re-entry is a no-op with a log line rather than an error, because the safe reading of a second
    call is "make sure the knobs are applied", not "apply them twice".

    Parameters
    ----------
    model
        The built model, mutated in place.
    cfg
        The resolved recipe. Build it with :meth:`SamplingConfig.resolve_all` at a CLI entry point,
        or :meth:`SamplingConfig.from_env` to reproduce the historical environment-only behaviour.
    """
    if getattr(model, "_sampling_env_overrides_applied", False):
        typer.echo("[sampling-env] already applied to this model; skipping (these knobs do not compound)")
        return
    model._sampling_env_overrides_applied = True
    if os.environ.get("ATOMWEAVER_VERBOSE"):
        for line in cfg.describe():
            typer.echo(line)
    _var_scale = cfg.shell_var_scale
    if _var_scale is not None:
        # Negative values are rejected in SamplingConfig.validate, before the model is even built:
        # (-0.25) ** 0.5 is complex in Python 3, so both the mul_ and the f-string below would fail
        # with something unrecognisable after an hour of load time.
        _vs = float(_var_scale)
        _cf = model.coord_flow
        _before = _cf._shell_source_var.clone() if hasattr(_cf, "_shell_source_var") else None
        with torch.no_grad():
            if hasattr(_cf, "_shell_thickness"):
                _cf._shell_thickness.mul_(_vs**0.5)
            if hasattr(_cf, "_shell_radial_var"):
                _cf._shell_radial_var.mul_(_vs)
            if all(hasattr(_cf, _b) for _b in ("_shell_source_var", "_shell_radii", "_shell_thickness")):
                _cf._shell_source_var.copy_(_cf._shell_radii**2 / 3.0 + _cf._shell_thickness**2)
        # This knob scales the JITTER, and the difference from scaling the source variance is not
        # small. The source draw is Ca + r*d + tau*eta*r, whose per-axis variance is
        # r^2/3 + tau^2*r^2, and only the tau^2*r^2 term moves -- at the production tau=0.3 the
        # untouched r^2/3 directional term is the dominant one. So var_scale=0.25 changes the source
        # variance by x0.84, not x0.25, and var_scale=0.1 by x0.81, not x0.10. The old line read
        # "variance x0.25", and that string is where the var025 run labels came from. Report the
        # REALISED ratio, measured off the buffer rather than re-derived, so the label cannot drift.
        _realised = float((_cf._shell_source_var.sum() / _before.sum()).item()) if _before is not None else float("nan")
        typer.echo(
            f"[shell-var-scale] jitter variance x{_vs} (thickness x{_vs**0.5:.3f}); "
            f"SOURCE variance x{_realised:.3f} -- the shell-radius term r^2/3 is not scaled"
        )


def apply_sampling_env_overrides(model) -> None:
    """
    Apply the sampling knobs from the environment alone.

    .. deprecated::
        Resolve a :class:`SamplingConfig` and call :func:`apply_sampling_config` instead, so the
        recipe is visible and can be recorded. Kept because in-flight drivers call this by name;
        it resolves from the environment, which warns per variable.
    """
    apply_sampling_config(model, SamplingConfig.from_env())


@app.command()
def main(
    pdb_dir: str = typer.Option(..., help="Directory of input PDBs (peptide backbone + optional target)."),
    checkpoint: str = typer.Option(..., help="Model checkpoint (.ckpt/.pt)."),
    sampling_db: str = typer.Option(..., help="Residue DB for model load (builds name_to_idx)."),
    ref_db: str = typer.Option(..., "--ref-db", help="Reference residue library (.pt) for cloud residue-name labels."),
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
        "against the model's per-slot fill-rate buffer). Pass --reserved-slot0/--no-reserved-slot0 only to "
        "override for an older ckpt that did not store it; if it DISAGREES with the stored value the eval errors.",
    ),
    reserved_slot0_prefix_exempt: bool = typer.Option(
        False,
        help="exempt reserved-slot0 from the hard prefix constraint at sampling (the slot-0/prefix collapse fix)",
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
    cfg = SamplingConfig.resolve_all()
    _undeclared = sampling_knobs.undeclared_atomweaver_vars()
    if _undeclared and os.environ.get("ATOMWEAVER_VERBOSE"):
        # Not an error -- a scratch variable is not a defect -- but a variable the package does not
        # recognise is worth surfacing in case it was meant to take effect.
        typer.echo(
            f"[sampling-config] WARNING undeclared ATOMWEAVER_* in the environment: {', '.join(_undeclared)}. "
            "If any of these is meant to affect sampling, add it to sampling_knobs.KNOWN_KNOBS."
        )

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

    # Reference library -> type-index -> CCD map (cosmetic cloud residue-name labels; identity is read
    # off later by apply_hybrid_readout from the cloud GEOMETRY, not these names).
    _db = torch.load(ref_db, weights_only=False)
    ccd_to_idx: dict = {}
    for _i, _m in enumerate(_db["metadata"]):
        for _pid in _m.get("pdb_ids") or []:
            if _pid:
                ccd_to_idx.setdefault(_pid, _i)
    idx_to_ccd = {v: k for k, v in ccd_to_idx.items()}
    # Per-CCD L/D chirality lookup, consumed only by the opt-in --oracle-chirality cone steering.

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
        _, ccds = peptide_chain_ccds(pdb_path, pep_chain=peptide_chain_from_filename(pdb_path))
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
                    _placeholder = torch.zeros(L, dtype=torch.long)
                    gt_seq = "".join(THREE2ONE.get(ccds[p], "X") for p in pep_pos)
                    Path(export_clouds).mkdir(parents=True, exist_ok=True)
                    pdb_str = _make_cloud_pdb(batch, out, pep_pos, ccds, idx_to_ccd, _placeholder, "", gt_seq)
                    (Path(export_clouds) / f"{pdb_name}_s{_s}.pdb").write_text(pdb_str)
        n_done += 1
    typer.echo(f"[sampler] done: {n_done} input(s) sampled.")


if __name__ == "__main__":
    app()
