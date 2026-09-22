#!/usr/bin/env python3
"""Apply a HYBRID (Learned + NDM-geometric) read-out to exported cloud PDBs.

This is the hybrid sibling of ``apply_learned_readout.py``. It produces a Learned-logits-shaped
per-position distribution over the candidate vocabulary (= the Learned head's ``classes``), but
blends in the NDM geometric matcher so that NDM drives NON-canonical identity while the Learned
head owns canonical identity. The combination is HAND-CRAFTED (fixed, interpretable formulas from
``hybrid_lib.py``): a parameterized NDM "cliff" transform turns the raw geometric score vector into
a distribution, and a per-candidate-class blend merges it with the Learned distribution. Nothing is
fitted -- every knob (eps, m, beta, gamma) is a named constant in the chosen preset.

Two inputs:
  * Learned head (--head) : the SAME frozen logreg bundle apply_learned_readout.py consumes
                             (bundle = {clf, classes, prior}). Defines the candidate vocabulary.
  * NDM ref-DB (--ref-db) : the reference residue library .pt (reference_library.pt),
                             scored with build_eval_discretizer at penalties atom0.5/elem0.3/chir50.

Output ``preds.json`` mirrors apply_learned_readout.py's schema (meta / classes / designs), where
each design carries per-site ``argmax`` (top-1) AND ``probs`` (the [L, C] normalized distribution,
aligned to ``classes``) -- the logits-shaped output for the DMS correlation pipeline. Optionally
also dumps the distributions to a .pt/.npz via --dump-distributions.

CPU only. Element vocab pinned to 5 (production). Ships to the DMS pipeline.

Example
-------
    python apply_hybrid_readout.py \
        --head   data/readout_head_full300.joblib \
        --ref-db data/reference_library.pt \
        --clouds out/clouds \
        --preset b2_balanced \
        --out    out/preds.json
"""

import glob
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("ATOMWEAVER_ELEMENT_VOCAB", "5")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import joblib
import numpy as np
import torch
import typer

torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))

# hybrid_lib.py sits next to this script.
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_lib as H  # noqa: E402

app = typer.Typer(add_completion=False, help=__doc__)

MAXSC = 16
# canon-20 ship head (20-class); --canon20 swaps to it when --head is not given, restricting the
# candidate vocabulary to the 20 canonicals (NCAA columns vanish -> pure canon-20 read-out).
CANON20_HEAD = "data/readout_head_canon20.joblib"
ELMAP = {"C": 1, "N": 2, "O": 3, "S": 4}  # sidechain char -> DB element index (+1); default 4=X
BB_EL = [1, 0, 0, 2]
IU20 = torch.triu_indices(20, 20, 1)
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


# ---------------------------------------------------------------------------- cloud parse + tensors
def parse_cloud(path):
    chains = {"B": {}, "C": {}}
    order = {"B": [], "C": []}
    for ln in open(path):
        if not ln.startswith("ATOM"):
            continue
        ch = ln[21]
        if ch not in ("B", "C"):
            continue
        an = ln[12:16].strip()
        rn = ln[17:20].strip()
        rnum = int(ln[22:26])
        xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
        d = chains[ch]
        if rnum not in d:
            d[rnum] = {"rn": rn, "bb": {}, "sc": []}
            order[ch].append(rnum)
        r = d[rnum]
        if an in ("N", "CA", "C", "O"):
            r["bb"][an] = xyz
        else:
            try:
                slot = int(an[1:]) - 1
            except ValueError:
                slot = len(r["sc"])
            r["sc"].append((slot, an[0], xyz))
    return [chains["C"][r] for r in order["C"]], [chains["B"][r] for r in order["B"]]


def build_tensors(Cres):
    L = len(Cres)
    bb = torch.zeros(L, 4, 3)
    bm = torch.zeros(L, 4, dtype=torch.bool)
    sc = torch.zeros(L, MAXSC, 3)
    sm = torch.zeros(L, MAXSC, dtype=torch.bool)
    sel = torch.zeros(L, MAXSC, dtype=torch.long)
    for i, rr in enumerate(Cres):
        for bi, bn in enumerate(["N", "CA", "C", "O"]):
            if bn in rr["bb"]:
                bb[i, bi] = torch.tensor(rr["bb"][bn])
                bm[i, bi] = True
        for slot, elc, xyz in rr["sc"]:
            if 0 <= slot < MAXSC:
                sc[i, slot] = torch.tensor(xyz)
                sm[i, slot] = True
                sel[i, slot] = ELMAP.get(elc, 4)
    return bb, bm, sc, sm, sel


def learned_features(Cres, bb, bm, sc, sm, sel):
    """294-dim feature identical to apply_learned_readout.build_features / score_shipcfg.cpp_features."""
    L = len(Cres)
    xyz = torch.cat([bb, sc], 1)
    m = torch.cat([bm, sm], 1)
    fe = torch.full((L, 4 + MAXSC), 4, dtype=torch.long)
    for j in range(4):
        fe[:, j] = torch.where(bm[:, j], torch.tensor(BB_EL[j]), torch.tensor(4))
    for j in range(MAXSC):
        e = sel[:, j]
        fmap = torch.tensor([4, 0, 1, 2, 3])[e.clamp(0, 4)]
        fe[:, 4 + j] = torch.where(sm[:, j], fmap, torch.tensor(4))
    xyz = xyz[:, :20]
    m = m[:, :20]
    fe = fe[:, :20]
    d = torch.cdist(xyz, xyz)[:, IU20[0], IU20[1]]
    rp = m[:, IU20[0]] & m[:, IU20[1]]
    d = torch.where(rp, d, torch.full_like(d, -1.0))
    ei = fe.clone()
    ei[~m] = 4
    ei = ei.clamp(0, 4)
    eo = torch.zeros(L, 20, 5)
    eo.scatter_(2, ei.unsqueeze(2), 1.0)
    feat = torch.cat([d, eo.reshape(L, -1)], 1).numpy()

    def dih(p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1 = b1 / (np.linalg.norm(b1) + 1e-9)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))

    pp = np.zeros((L, 4), np.float32)

    def g(i, a):
        b = Cres[i]["bb"]
        return np.array(b[a]) if a in b else None

    for i in range(L):
        N, CA, C = g(i, "N"), g(i, "CA"), g(i, "C")
        if i > 0 and all(x is not None for x in (g(i - 1, "C"), N, CA, C)):
            a = dih(g(i - 1, "C"), N, CA, C)
            pp[i, 0] = np.sin(np.radians(a))
            pp[i, 1] = np.cos(np.radians(a))
        if i < L - 1 and all(x is not None for x in (N, CA, C, g(i + 1, "N"))):
            bt = dih(N, CA, C, g(i + 1, "N"))
            pp[i, 2] = np.sin(np.radians(bt))
            pp[i, 3] = np.cos(np.radians(bt))
    return np.concatenate([np.nan_to_num(feat, nan=-1.0), pp], 1).astype(np.float32)


@app.command()
def main(
    head: Optional[str] = typer.Option(
        None, "--head", help="Learned head bundle (.joblib); defines candidate vocab. Required unless --canon20."
    ),
    ref_db: Optional[str] = typer.Option(
        None,
        "--ref-db",
        help="NDM reference residue library (.pt). "
        "Required only when NDM is actually used; skipped for --canon20 / pure-canonical vocab.",
    ),
    clouds: str = typer.Option(..., "--clouds", help="Directory of exported cloud PDBs."),
    out: str = typer.Option(..., "--out", help="Output preds.json."),
    preset: str = typer.Option(
        "b2_balanced",
        "--preset",
        help=f"Named hybrid preset (default b2_balanced, DMS-validated): {list(H.SHIP_PRESETS)}",
    ),
    natfreq: bool = typer.Option(
        False,
        "--natfreq/--no-natfreq",
        help="Multiply the blended distribution by the head's SwissProt natfreq prior (bundle['prior']) "
        "before argmax, biasing composition toward natural (canon-heavy) frequencies. Default off "
        "(uniform); composable with --canon20.",
    ),
    canon20: bool = typer.Option(
        False,
        "--canon20/--no-canon20",
        help=f"Convenience: when --head is not given, swap to the canon-20 ship head ({CANON20_HEAD}), "
        "restricting the candidate vocabulary to the 20 canonicals (no NCAA candidates). "
        "Composable with --natfreq.",
    ),
    atom_penalty: float = typer.Option(0.5, "--atom-penalty"),
    elem_penalty: float = typer.Option(0.3, "--elem-penalty"),
    chir_penalty: float = typer.Option(50.0, "--chir-penalty"),
    dump_distributions: Optional[str] = typer.Option(
        None, "--dump-distributions", help="Also dump [L,C] probs to .pt/.npz."
    ),
):
    """Read residue identity off cloud PDBs with a hand-crafted Learned+NDM hybrid."""
    if preset not in H.SHIP_PRESETS:
        raise typer.BadParameter(f"--preset must be one of {list(H.SHIP_PRESETS)}")
    pr = H.SHIP_PRESETS[preset]

    # --- resolve the head: --canon20 supplies the canon-20 ship head only when --head is not given ---
    if head is None:
        if canon20:
            head = CANON20_HEAD
        else:
            raise typer.BadParameter("--head is required (or pass --canon20 to use the canon-20 ship head).")
    elif canon20:
        typer.echo(f"[apply_hybrid_readout] NOTE: explicit --head given; --canon20 head-swap ignored (using {head}).")

    # --- Learned head defines the candidate vocabulary ---
    bundle = joblib.load(head)
    clf = bundle["clf"]
    classes = [str(c) for c in bundle["classes"]]
    cls_idx = {c: i for i, c in enumerate(classes)}
    is_canon = np.array([c in CANON for c in classes])
    # SwissProt natfreq prior aligned to `classes` (only consumed when --natfreq); required if requested.
    prior_vec = bundle.get("prior")
    if natfreq:
        if prior_vec is None:
            raise typer.BadParameter("--natfreq requested but head bundle has no 'prior' vector.")
        prior_vec = np.asarray(prior_vec, dtype=np.float64).ravel()
        if prior_vec.shape[0] != len(classes):
            raise typer.BadParameter(f"head 'prior' length {prior_vec.shape[0]} != n_classes {len(classes)}.")

    # --- decide whether the NDM geometric matcher is actually consumed ([2]) ---
    # The blend only reads qN (the NDM cliff) when the family is not pure-Learned AND there is at
    # least one non-canonical candidate. A canon-only vocabulary (--canon20, or any all-canon head)
    # collapses every blend to pure Learned (w_canon=1 => P=pL), so NDM/ref-DB are never needed.
    need_ndm = (pr["family"] != "pure_learned") and (not bool(is_canon.all()))
    if need_ndm and ref_db is None:
        raise typer.BadParameter(
            "--ref-db is required for this preset/head (NDM is used). "
            "Pass --canon20 (or a pure-canonical head / pure_learned preset) to skip it."
        )

    NDM = ndm_col = num_types = representable = rot2type = None
    if need_ndm:
        # --- NDM discretizer over the ref DB at the task penalties ---
        # Locate sample.py: env override, else the sibling file next to this script
        # (portable across checkouts -- this script ships in scripts/joint_diffusion/ alongside it).
        eval_mod = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.py")
        spec = importlib.util.spec_from_file_location("evalmod", eval_mod)
        EV = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(EV)
        discs, ccd_to_idx, _sa, _ss, representable, db, rot2type = EV.build_eval_discretizer(
            ref_db,
            device="cpu",
            element_mismatch_penalty=elem_penalty,
            atom_mismatch_penalty=atom_penalty,
            chirality_mismatch_penalty=chir_penalty,
        )
        NDM = discs["NDM"]
        num_types = len(db["metadata"])
        # candidate col in the DB type space per Learned class (-1 => not in DB; NDM score = -inf there)
        ndm_col = np.array([ccd_to_idx.get(c, -1) for c in classes], dtype=np.int64)
    else:
        typer.echo("[apply_hybrid_readout] canon-only vocab -> NDM/ref-DB skipped (blend collapses to pure Learned).")

    files = sorted(glob.glob(os.path.join(clouds, "*.pdb")))
    if not files:
        raise typer.BadParameter(f"no *.pdb under {clouds}")
    typer.echo(
        f"[apply_hybrid_readout] preset={preset} classes={len(classes)} files={len(files)} "
        f"penalties=atom{atom_penalty}/elem{elem_penalty}/chir{chir_penalty}"
    )

    designs = []
    dist_store = {}
    for path in files:
        name = Path(path).stem
        Cres, Bres = parse_cloud(path)
        if not Cres:
            continue
        L = len(Cres)
        bb, bm, sc, sm, sel = build_tensors(Cres)
        if need_ndm:
            # NDM logits -> [L, num_types], masked to representable
            outp = NDM(
                predicted_coords=sc.unsqueeze(0),
                predicted_mask=sm.unsqueeze(0),
                backbone_coords=bb.unsqueeze(0),
                backbone_mask=bm.unsqueeze(0),
                predicted_element_types=(sel.unsqueeze(0) - 1),
            )
            lg = outp
            lg = lg.masked_fill(~representable.view(1, 1, -1), float("-inf"))[0].numpy()  # [L, num_types]
            # gather NDM score onto the Learned class order (-inf for classes absent from DB)
            sN = np.full((L, len(classes)), -np.inf, dtype=np.float64)
            has = ndm_col >= 0
            sN[:, has] = lg[:, ndm_col[has]]
        else:
            # canon-only vocab: NDM is unused (w_canon=1 zeroes the qN term), so a dummy score matrix
            # keeps the exact apply_preset code path (output byte-identical) with no NDM forward pass.
            sN = np.zeros((L, len(classes)), dtype=np.float64)

        F = learned_features(Cres, bb, bm, sc, sm, sel)
        pL = np.asarray(clf.predict_proba(F), dtype=np.float64)  # [L, len(classes)]

        P = H.apply_preset(pL, sN, is_canon, pr)  # [L, C] normalized distribution
        if natfreq:  # post-hoc natfreq prior multiply (composable; default off -> byte-identical)
            P = H.apply_natfreq(P, prior_vec)
        rs = P.sum(1)
        assert np.allclose(rs, 1.0, atol=1e-5), f"{name} rows not normalized: {rs.min()},{rs.max()}"

        # strictly-positive export: floor every prob at 1e-12 and renormalize so NO class is ever
        # exactly 0 in the output and rows still sum to 1. round(6) is dropped so tiny-but-nonzero
        # blend probs survive (float32 preserves ~1e-12). This lets any consumer log() safely with no
        # floor of their own (fixes dropped-canonicals -inf in the DMS log-averaging AND the prior
        # round(6) tie artifact). The floor is far below the max, so argmax is unchanged.
        P = np.maximum(P, 1e-12)
        P = P / P.sum(axis=1, keepdims=True)

        argmax = [classes[i] for i in P.argmax(1)]
        rec = {
            "name": name,
            "length": L,
            "sites": list(range(1, L + 1)),
            "argmax": argmax,
            "probs": P.astype(np.float32).tolist(),
        }
        gt = [Bres[i]["rn"] for i in range(min(len(Bres), L))]
        if gt:
            rec["gt"] = gt
        designs.append(rec)
        if dump_distributions:
            dist_store[name] = P.astype(np.float32)

    payload = {
        "meta": {
            "head": os.path.abspath(head),
            "ref_db": (os.path.abspath(ref_db) if ref_db else None),
            "clouds": os.path.abspath(clouds),
            "preset": preset,
            "preset_params": pr,
            "penalties": {"atom": atom_penalty, "elem": elem_penalty, "chir": chir_penalty},
            "n_designs": len(designs),
            "n_classes": len(classes),
            "natfreq": bool(natfreq),
            "canon20": bool(canon20),
            "element_vocab": os.environ.get("ATOMWEAVER_ELEMENT_VOCAB"),
            "mode": "hybrid",
        },
        "classes": classes,
        "designs": designs,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(out, "w"), indent=1, default=float)
    typer.echo(f"[apply_hybrid_readout] wrote {out} ({len(designs)} designs)")
    if dump_distributions:
        ext = Path(dump_distributions).suffix.lower()
        if ext == ".pt":
            torch.save({"classes": classes, "distributions": dist_store}, dump_distributions)
        elif ext == ".npz":
            np.savez(
                dump_distributions,
                classes=np.array(classes),
                names=np.array(list(dist_store.keys())),
                **{f"P__{k}": v for k, v in dist_store.items()},
            )
        else:
            raise typer.BadParameter("--dump-distributions must end in .pt or .npz")
        typer.echo(f"[apply_hybrid_readout] dumped distributions -> {dump_distributions}")


if __name__ == "__main__":
    app()
