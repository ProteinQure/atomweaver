#!/usr/bin/env python3
r"""Fit a *Learned* readout head (self-distilled multinomial logistic regression) for a CUSTOM
residue subset, using the EXACT recipe of the shipped clean-300 A6 heads.

Why this exists
---------------
The Learned discretizer is the frozen sklearn head consumed by ``apply_learned_readout.py``: it maps
a per-site 294-dim geometric+chemical feature to a distribution over a residue vocabulary. The shipped
heads cover the settled clean-300 vocabulary. This CLI re-runs the *same* fit recipe over an arbitrary
user-chosen subset of residue types (e.g. "just my 20 canonicals + these 5 NCAAs"), producing a bundle
that ``apply_learned_readout.py`` consumes UNCHANGED (identical ``dict(clf, classes, prior, meta)``
shape and identical 294-dim feature contract).

This is the "coords-first, discretize-after" design in action: a new residue type is introduced by
adding ONE reference structure to the ref-DB -- there is NO model
retraining. This CLI only re-fits the cheap CPU read-off head over whatever subset you name.

The recipe (all knobs baked in to match the ship heads byte-for-byte)
--------------------------------------------------------------------
FEATURES (294-dim, must mirror ``apply_learned_readout.py`` / ``fit_ship_heads.py``):
  * 190 backbone-inclusive pairwise distances = upper-triangle of the 20x20 cdist over
         [N, CA, C, O, <16 sidechain slots>]; unresolved pairs set to -1.
  * 100 element one-hots = 20 slots x 5 classes (PAD/GHOST=4, C=0, N=1, O=2, X/other=3).
  * 4 phi/psi sin/cos = (sin phi, cos phi, sin psi, cos psi).
  reserved_slot0 convention => Cbeta-first; MAXSC=16 sidechain slots; 20 atoms = 4 backbone + 16 sc.

FIT SET (self-distillation UNION corrupted reference rotamers):
  * model clouds : per-site 294-dim features exported from a checkpoint's own samples
                       (``--ckpt-fit-cache`` npz with CPP[N,294] + RES[N] arrays), self-labelled
                       by their GT residue. These are the "self-distillation" targets.
  * reference rotamers: canonical + NCAA rotamers pulled from ``--ref-db`` and CORRUPTED (atom-drop
                       0.20, coordinate jitter sigma 0.60) to look like noisy predicted clouds; phi/psi
                       are sampled from the parent canonical's Ramachandran grid. Reference rotamers are
                       LOAD-BEARING: they are the ONLY signal for unseen / zero-shot types that have no
                       model clouds, so a subset can include types the checkpoint never emitted.

SAMPLE WEIGHTS = balanced-class-weight x popweight:
  * balanced : N / (n_classes * count[class]) -- equalizes class mass.
  * popweight: reference rotamers additionally weighted by their phi/psi population at the sampled
               (phi,psi). ``--popweight a6`` uses the A6 POOLED-Ramachandran fallback for parentless
               rotamers (recommended, matches the ship A6 heads); ``a5`` uses uniform weight 1.0 for
               parentless rotamers; ``none`` disables popweight entirely (balanced only). Model-cloud
               rows always get popweight 1.0; popweight is normalized WITHIN each class to mean 1 so it
               only reweights a type's rotamers, never the class balance.

MODEL: StandardScaler -> multinomial LogisticRegression (max_iter=300, C=1.0). lbfgs is slow to
  converge on the full feature set; pass ``--warmstart-from <existing head.joblib>`` to initialize
  coef_/intercept_ from an existing head (class-aligned: shared classes copied, new classes zero-init).
  This is 7-9x faster. Without it lbfgs typically caps at max_iter; a warning is emitted if it does.

PRIOR: the natfreq (SwissProt) vector is computed and carried into the bundle as ``prior`` (aligned to
  ``classes``). It is an OPTIONAL runtime toggle -- the PRIMARY readout is UNIFORM (plain predict_proba
  argmax). ``apply_learned_readout.py --prior natfreq`` multiplies it back in.

--residues (the CUSTOM SUBSET spec)
-----------------------------------
Accepts either:
  (a) a plain-text file with one CCD code per line (blank lines / ``#`` comments ignored), or
  (b) a ``final_sets.json``-style JSON with ``keep`` / ``holdout`` (and optionally ``canonical``) lists.
      The subset = canonical U keep U holdout; entries under ``holdout`` are treated as REFERENCE-ONLY
      (their model clouds are dropped -> zero-shot behaviour). If no ``canonical`` key is present
      the 20 standard canonicals are included.

The fitted classes = (requested subset) INTERSECT (types present in ``--ref-db``), further restricted to
the 20 canonicals when ``--scope canon20``. Any requested CCD MISSING from the ref-DB is warned about and
listed: to fit a head over it you must first add its reference structure to the ``--ref-db``
library (coords-first: one reference structure, no model retrain), then re-run this CLI.

Output bundle
-------------
``dict(clf, classes, prior, meta)`` -- identical shape to the ship heads, so ``apply_learned_readout.py``
consumes it unchanged. ``meta`` records feature_spec, scope, popweight mode, corruption, ref_db,
phipsi_table, ckpt, and subset size.

CPU only by design (no GPU is touched).

Example
-------
    python scripts/joint_diffusion/fit_learned_readout.py \
        --ckpt-fit-cache data/readout_train_clouds.npz \
        --ref-db  data/reference_library.pt \
        --phipsi-table data/phipsi_populations.pt \
        --residues my_subset.txt \
        --scope full --popweight a6 \
        --warmstart-from data/readout_head_full300.joblib \
        --out my_subset_head.joblib
"""

import os

# CPU-only + production element vocab must be pinned BEFORE importing torch-dependent code.
os.environ.setdefault("ATOMWEAVER_ELEMENT_VOCAB", "5")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # keep off the GPUs; this is a CPU readout fit

import collections
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import typer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))

app = typer.Typer(add_completion=False, help=__doc__)

# ----------------------------------------------------------------------------------------------
# Baked-in recipe constants (mirror fit_ship_heads.py / fit_eval_clean300.py exactly)
# ----------------------------------------------------------------------------------------------
CANON = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]  # fmt: skip
CANON_S = set(CANON)
# natfreq (SwissProt) prior -- carried into the bundle; PRIMARY readout stays uniform.
NAT = {
    "ALA": .0825, "ARG": .0553, "ASN": .0406, "ASP": .0546, "CYS": .0138, "GLN": .0393, "GLU": .0672,
    "GLY": .0707, "HIS": .0227, "ILE": .0591, "LEU": .0966, "LYS": .0580, "MET": .0241, "PHE": .0386,
    "PRO": .0474, "SER": .0656, "THR": .0534, "TRP": .0110, "TYR": .0292, "VAL": .0687,
}  # fmt: skip
FLOOR = 1e-4
CAP_M = 1000  # max model-cloud rows per class
CAP_R = 400  # max (corrupted) reference rows per class
DROP, SIGMA = 0.20, 0.60  # reference-rotamer corruption: atom-drop prob, coordinate jitter sigma
COLD_MAX_ITER = 300
WARM_MAX_ITER_DEFAULT = 40

FEATURE_SPEC = (
    "294-dim: 190 backbone-inclusive pairwise distances (triu of 20x20 atoms, includes the "
    "4 backbone atoms) + 100 element one-hots (20 slots x 5 = PAD/GHOST,C,N,O,X) + 4 phi/psi "
    "sin/cos (prevC-N-CA-C, N-CA-C-nextN)"
)
SLOT_CONV = "reserved_slot0 => Cbeta-first; MAXSC=16 sidechain slots; 20 atoms = 4 backbone (N,CA,C,O) + 16 sidechain"

S20 = 20
iu20 = torch.triu_indices(S20, S20, 1)


# ----------------------------------------------------------------------------------------------
# Feature build (VERBATIM from fit_ship_heads.py / fit_eval_clean300.py)
# ----------------------------------------------------------------------------------------------
def _feats_from_coords(xyz, m, el):
    """290-dim geometry+element block (190 pairwise dists + 100 element one-hots)."""
    d = torch.cdist(xyz, xyz)[:, iu20[0], iu20[1]]
    rp = m[:, iu20[0]] & m[:, iu20[1]]
    d = torch.where(rp, d, torch.full_like(d, -1.0))
    ei = el.clone()
    ei[~m] = 4
    ei = ei.clamp(0, 4).long()
    eo = torch.zeros(xyz.shape[0], S20, 5)
    eo.scatter_(2, ei.unsqueeze(2), 1.0)
    return torch.cat([d, eo.reshape(xyz.shape[0], -1)], 1).numpy()


# curated NCAA -> canonical parent map (VERBATIM); used to source a rotamer's phi/psi grid.
_CUR = {
    "SEP": "SER", "TPO": "THR", "PTR": "TYR", "TYS": "TYR", "OMY": "TYR", "FTY": "TYR", "IYR": "TYR", "DTY": "TYR",
    "NAL": "PHE", "DPN": "PHE", "MEA": "PHE", "0BN": "PHE", "HT7": "PHE", "MK8": "LEU", "MP8": "LEU", "NLE": "LEU",
    "LE1": "LEU", "MLE": "LEU", "NVA": "VAL", "ABU": "ABA", "HYP": "PRO", "HZP": "PRO", "FP9": "PRO", "DPR": "PRO",
    "PCA": "GLU", "CGU": "GLU", "B3E": "GLU", "ALY": "LYS", "M3L": "LYS", "MLY": "LYS", "MLZ": "LYS", "KCR": "LYS",
    "B8R": "LYS", "2MR": "ARG", "NMM": "ARG", "HMR": "ARG", "CIR": "ARG", "ORN": "ARG", "ORQ": "ARG", "DAL": "ALA",
    "AIB": "ALA", "MAA": "ALA", "SAR": "GLY", "CHG": "VAL", "CSO": "CYS", "CY3": "CYS", "SEC": "CYS", "DAB": "LYS",
    "HLX": "LEU", "B3L": "LEU", "ASA": "ASP",
}  # fmt: skip


class RefLib:
    """Holds the reference-rotamer DB + phi/psi side-table and builds the corrupted reference set.

    This bundles the exact ``build_reference`` / popweight / phi/psi-fallback logic from
    ``fit_ship_heads.py`` so a head fit here is byte-compatible with ``apply_learned_readout.py``.
    """

    def __init__(self, ref_db_path: str, phipsi_path: str):
        db = torch.load(ref_db_path, map_location="cpu", weights_only=False)
        self.coords = db["coords"]
        self.masks = db["masks"]
        self.elem = db["element_types"]
        self.rid = db["residue_ids"]
        md = db["metadata"]
        self.code_of = {
            i: (str(md[i].get("ccd_code") or md[i].get("ccd_code") or md[i]["name"]).strip()) for i in range(len(md))
        }
        self.rows = collections.defaultdict(list)
        for ei in range(self.coords.shape[0]):
            self.rows[self.code_of[int(self.rid[ei])]].append(ei)
        self.tags = {}
        for i, m in enumerate(md):
            self.tags[self.code_of[i]] = m.get("rotamer_tags", [])
        self.ref_codes = set(self.code_of.values())

        pp = torch.load(phipsi_path, map_location="cpu", weights_only=False)
        self.ppc = pp["canonical"]
        self.well_idx = {"g+": 0, "t": 1, "g-": 2}
        self.nbins = pp["meta"]["nbins"]
        self.binw = pp["meta"]["binw"]

        # chirality lookup (D-AA mirror for sampled phi/psi)
        try:
            from atomweaver.joint_diffusion.data_utils import build_chirality_lookup

            cl = build_chirality_lookup(db)
            self.chir = {c: int(cl.get(c, 1)) for c in self.code_of.values()}
        except Exception as e:  # pragma: no cover - defensive; L default is the common case
            typer.echo(f"[fit_learned_readout] chirality lookup FAILED, defaulting L: {e!r}")
            self.chir = dict.fromkeys(self.code_of.values(), 1)

        # pooled Ramachandran grid (A6 fallback for parentless rotamers)
        pooled = np.zeros((self.nbins, self.nbins), np.float64)
        for aa in CANON:
            g = self.ppc.get(aa, {}).get("grid_P")
            if g is None:
                continue
            ga = np.asarray(g, np.float64)
            pooled += ga.sum(axis=2) if ga.ndim == 3 else ga
        occ = pooled[pooled > 0]
        med = float(np.median(occ)) if occ.size else 1.0
        self.pooled_grid = pooled / (med if med > 0 else 1.0)
        self.ref_branch = collections.Counter()

    # -- phi/psi parent sourcing --------------------------------------------------------------
    def parent_of(self, code):
        if code in _CUR:
            return _CUR[code]
        if code in CANON_S:
            return code
        for t in self.tags.get(code, []):
            gr = (t.get("phipsi_population") or {}).get("grid_ref")
            if gr:
                return gr.split("/")[0]
        return "LEU"

    def _bin(self, a):
        return int(min(self.nbins - 1, max(0, (a + 180.0) // self.binw)))

    def sample_pp_rows(self, code, parent_pp, n, rng):
        p = self.parent_of(code)
        src = parent_pp.get(p, parent_pp["_POOLED"])
        if src is None or len(src) == 0:
            src = parent_pp["_POOLED"]
        rows = src[rng.integers(0, len(src), n)].copy()
        if self.chir.get(code, 1) < 0:
            rows[:, 0] *= -1
            rows[:, 2] *= -1
        return rows.astype(np.float32)

    def rotamer_popweight_ex(self, code, tag, sincos, pooled_fallback):
        """phi/psi population weight of a reference rotamer (A5 uniform-/A6 pooled-fallback)."""
        ppop = tag.get("phipsi_population") if isinstance(tag, dict) else None
        sphi, cphi, spsi, cpsi = sincos

        def _pooled():
            s_phi, s_psi = sphi, spsi
            if ppop and ppop.get("mirror"):
                s_phi, s_psi = -s_phi, -s_psi
            phi = np.degrees(np.arctan2(s_phi, cphi))
            psi = np.degrees(np.arctan2(s_psi, cpsi))
            return float(max(self.pooled_grid[self._bin(phi)][self._bin(psi)], 0.01))

        def _fb():
            rp = tag.get("rotamer_population") if isinstance(tag, dict) else None
            if rp is not None:
                return float(rp), "rp"
            return (_pooled() if pooled_fallback else 1.0), "fallback"

        if not ppop or ppop.get("grid_ref") is None:
            return _fb()
        aa, well = ppop["grid_ref"].split("/")
        grid = self.ppc.get(aa, {}).get("grid_P")
        if grid is None or well not in self.well_idx:
            return _fb()
        s_phi, s_psi = sphi, spsi
        if ppop.get("mirror"):
            s_phi, s_psi = -s_phi, -s_psi
        phi = np.degrees(np.arctan2(s_phi, cphi))
        psi = np.degrees(np.arctan2(s_psi, cpsi))
        return float(max(grid[self._bin(phi)][self._bin(psi)][self.well_idx[well]], 0.01)), "specific"

    @staticmethod
    def corrupt(xyz, m, el, drop, sigma, reps):
        X = xyz.repeat(reps, 1, 1).clone()
        M = m.repeat(reps, 1).clone()
        E = el.repeat(reps, 1).clone()
        sc = slice(4, 20)
        if sigma > 0:
            jit = torch.zeros_like(X)
            jit[:, sc, :] = torch.randn((X.shape[0], 16, 3)) * sigma
            X = X + jit
        if drop > 0:
            dm = (torch.rand(X.shape[0], 16) < drop) & M[:, sc]
            M[:, sc] = M[:, sc] & ~dm
        return X, M, E

    def build_reference(self, parent_pp, class_list):
        """Return Xr[n,294], yr[n], W(A5)[n], W6(A6)[n] for the corrupted reference rotamers."""
        rng = np.random.default_rng(42)
        X, y, W, W6 = [], [], [], []
        for c in class_list:
            rows = self.rows.get(c, [])
            if not rows:
                continue
            idx = torch.tensor(rows)
            xyz, m, el = self.coords[idx].clone(), self.masks[idx].clone().bool(), self.elem[idx].clone().long()
            nrot = xyz.shape[0]
            reps = max(1, int(np.ceil(CAP_R / nrot)))
            Xc, Mc, Ec = self.corrupt(xyz, m, el, DROP, SIGMA, reps)
            Fc = _feats_from_coords(Xc, Mc, Ec)
            pp = self.sample_pp_rows(c, parent_pp, Fc.shape[0], rng)
            Fc = np.concatenate([Fc, pp], 1)
            tags = self.tags.get(c, [])
            rot_of_row = np.tile(np.arange(nrot), reps)
            if len(Fc) > CAP_R:
                sel = rng.choice(len(Fc), CAP_R, replace=False)
                Fc, pp, rot_of_row = Fc[sel], pp[sel], rot_of_row[sel]
            w = np.ones(len(Fc), np.float32)
            w6 = np.ones(len(Fc), np.float32)
            for k in range(len(Fc)):
                ri = int(rot_of_row[k])
                tag = tags[ri] if ri < len(tags) else {}
                wv, br = self.rotamer_popweight_ex(c, tag, pp[k], False)
                w6v, _ = self.rotamer_popweight_ex(c, tag, pp[k], True)
                w[k], w6[k] = wv, w6v
                self.ref_branch[br] += 1
            X.append(Fc)
            y += [c] * len(Fc)
            W.append(w)
            W6.append(w6)
        if not X:
            return np.zeros((0, 294), np.float32), np.array([], "U4"), np.zeros(0, np.float32), np.zeros(0, np.float32)
        return np.concatenate(X), np.array(y, "U4"), np.concatenate(W), np.concatenate(W6)


def build_parent_pp(fit_cpp, fit_res):
    """Per-canonical phi/psi rows (cols 290:294) from model clouds, + a pooled fallback."""
    pp = {}
    for c in CANON:
        m = fit_res == c
        if m.sum() >= 20:
            pp[c] = fit_cpp[m][:, 290:294].copy()
    if pp:
        pp["_POOLED"] = np.concatenate(list(pp.values()), 0)
    else:
        pp["_POOLED"] = np.zeros((1, 4), np.float32)
    return pp


# ----------------------------------------------------------------------------------------------
# --residues parsing
# ----------------------------------------------------------------------------------------------
def parse_residue_spec(path: str):
    """Parse the CUSTOM SUBSET spec. Returns (requested_set, holdout_set, canonical_set).

    Plain-text: one CCD per line (``#`` comments / blanks ignored). No holdouts; canonical = the 20
    standard canonicals present in the list.
    JSON (final_sets.json-style): subset = canonical U keep U holdout; ``holdout`` entries are
    reference-only. ``canonical`` defaults to the 20 standard canonicals if absent.
    """
    raw = Path(path).read_text()
    text = raw.strip()
    if text.startswith("{"):
        obj = json.loads(raw)
        keep = [str(c).strip() for c in obj.get("keep", [])]
        holdout = [str(c).strip() for c in obj.get("holdout", [])]
        canonical = [str(c).strip() for c in obj.get("canonical", CANON)]
        requested = set(keep) | set(holdout) | set(canonical)
        return requested, set(holdout), set(canonical)
    # plain text
    codes = []
    for ln in raw.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            codes.append(ln)
    requested = set(codes)
    return requested, set(), requested & CANON_S


# ----------------------------------------------------------------------------------------------
# warm-start (class-aligned)
# ----------------------------------------------------------------------------------------------
def _warm_start_fit(X, y, sw, warmstart_path, max_iter):
    """Fit StandardScaler->LogisticRegression, initializing coef_/intercept_ from an existing head.

    Class-aligned: for each fit class present in the warm-start head, copy its coef_ row / intercept;
    classes absent from the head are zero-initialized. The warm-start head's scaler is NOT reused (the
    subset's feature statistics differ), so the transferred coef_ is an approximate but strong
    initialization -- lbfgs refines it. Returns (pipeline, n_iter, base_classes).
    """
    base = joblib.load(warmstart_path)
    base_lr = base["clf"].named_steps["logisticregression"]
    base_cls = [str(c) for c in base["classes"]]
    base_coef = np.asarray(base_lr.coef_, np.float64)  # (n_base_classes, n_feat) for multinomial
    base_int = np.asarray(base_lr.intercept_, np.float64).ravel()

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    new_classes = np.array(sorted(set(y.tolist())))  # LogisticRegression sorts classes_
    n_new, n_feat = len(new_classes), Xs.shape[1]
    coef = np.zeros((n_new, n_feat), np.float64)
    intercept = np.zeros(n_new, np.float64)
    n_copied = 0
    if base_coef.ndim == 2 and base_coef.shape[0] == len(base_cls) and base_coef.shape[1] == n_feat:
        bmap = {c: i for i, c in enumerate(base_cls)}
        for i, c in enumerate(new_classes):
            if c in bmap:
                coef[i] = base_coef[bmap[c]]
                intercept[i] = base_int[bmap[c]]
                n_copied += 1
    else:
        typer.echo(
            f"[fit_learned_readout] WARN warm-start head coef_ shape {base_coef.shape} incompatible "
            f"(base_classes={len(base_cls)}, n_feat={n_feat}); cold-initializing coef_."
        )

    lr = LogisticRegression(max_iter=max_iter, C=1.0, warm_start=True)
    lr.coef_ = coef
    lr.intercept_ = intercept
    lr.classes_ = new_classes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr.fit(Xs, y, sample_weight=sw)
    clf = Pipeline([("standardscaler", scaler), ("logisticregression", lr)])
    typer.echo(
        f"[fit_learned_readout] warm-start: copied {n_copied}/{n_new} class rows from "
        f"{Path(warmstart_path).name} ({len(base_cls)} classes)"
    )
    return clf, int(np.max(lr.n_iter_)), base_cls


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
@app.command()
def main(
    ckpt_fit_cache: str = typer.Option(
        "weights/readout_train_clouds.npz",
        "--ckpt-fit-cache",
        help="Training-clouds .npz (Release asset, download into weights/): keys CPP[N,294] features + RES[N] GT CCD per row.",
    ),
    ref_db: str = typer.Option(..., "--ref-db", help="Reference rotamer DB .pt (reference_library.pt)."),
    phipsi_table: str = typer.Option(..., "--phipsi-table", help="phi/psi populations .pt (phipsi_populations.pt)."),
    residues: str = typer.Option(
        ...,
        "--residues",
        help="CUSTOM SUBSET spec: plain-text (one CCD/line) OR final_sets.json-style {keep,holdout,canonical}. "
        "Requested CCDs missing from --ref-db are warned + skipped.",
    ),
    out: str = typer.Option(..., "--out", help="Output head bundle path (.joblib)."),
    scope: str = typer.Option(
        "full", "--scope", help="'full' = the whole requested subset; 'canon20' = restrict to the 20 canonicals."
    ),
    popweight: str = typer.Option(
        "a6",
        "--popweight",
        help="Reference-rotamer phi/psi popweight: 'a6' (pooled-Ramachandran fallback, matches ship A6), "
        "'a5' (uniform-1.0 fallback for parentless), 'none' (balanced class weight only).",
    ),
    warmstart_from: Optional[str] = typer.Option(
        None,
        "--warmstart-from",
        help="Existing head .joblib to warm-start coef_/intercept_ from (class-aligned; 7-9x faster). "
        "Without it lbfgs usually caps at max_iter=300.",
    ),
    ckpt_name: str = typer.Option("custom", "--ckpt-name", help="Label recorded in meta['ckpt'] (e.g. atomweaver)."),
):
    """Fit a Learned readout head for a custom residue subset with the shipped A6 recipe."""
    t_start = time.time()
    scope = scope.lower()
    popweight = popweight.lower()
    if scope not in ("full", "canon20"):
        raise typer.BadParameter("--scope must be 'full' or 'canon20'")
    if popweight not in ("a6", "a5", "none"):
        raise typer.BadParameter("--popweight must be 'a6', 'a5', or 'none'")

    # ---- reference DB + phi/psi ----
    reflib = RefLib(ref_db, phipsi_table)

    # ---- custom subset ----
    requested, holdout_set, _canonical_set = parse_residue_spec(residues)
    present = requested & reflib.ref_codes
    missing = sorted(requested - reflib.ref_codes)
    if missing:
        typer.echo(
            f"[fit_learned_readout] WARN {len(missing)} requested CCD(s) NOT in --ref-db (skipped): "
            f"{missing}\n  -> add each residue's reference structure to --ref-db (coords-first: "
            f"one reference structure, no model retrain), then re-run."
        )
    if scope == "canon20":
        classes_set = present & CANON_S
    else:
        classes_set = present
    CLASSES = sorted(classes_set)
    if len(CLASSES) < 2:
        raise typer.BadParameter(
            f"need >=2 fittable classes (subset INTERSECT ref-db{' INTERSECT canon20' if scope == 'canon20' else ''}), "
            f"got {len(CLASSES)}: {CLASSES}"
        )
    holdout_in = sorted(holdout_set & classes_set)  # reference-only within the fit
    typer.echo(
        f"[fit_learned_readout] subset: requested={len(requested)} present_in_refdb={len(present)} "
        f"fit_classes={len(CLASSES)} scope={scope} holdout(ref-only)={holdout_in}"
    )

    # ---- model clouds (self-distillation targets) ----
    z = np.load(ckpt_fit_cache)
    fit_cpp = z["CPP"].astype(np.float32)
    fit_res = z["RES"].astype("U4")
    # drop model-cloud rows for reference-only (holdout) classes -> zero-shot behaviour
    if holdout_in:
        keep = ~np.isin(fit_res, holdout_in)
        fit_cpp, fit_res = fit_cpp[keep], fit_res[keep]
    parent_pp = build_parent_pp(fit_cpp, fit_res)
    rng = np.random.default_rng(0)
    Xm, ym = [], []
    for c in CLASSES:
        ci = np.where(fit_res == c)[0]
        if len(ci) == 0:
            continue
        if len(ci) > CAP_M:
            ci = rng.choice(ci, CAP_M, replace=False)
        Xm.append(fit_cpp[ci])
        ym += [c] * len(ci)
    Xm = np.concatenate(Xm).astype(np.float32) if Xm else np.zeros((0, 294), np.float32)
    ym = np.array(ym, "U4")
    n_model_types = len(set(ym.tolist()))

    # ---- corrupted reference rotamers ----
    Xr, yr, wr_a5, wr_a6 = reflib.build_reference(parent_pp, CLASSES)
    Xr = Xr.astype(np.float32)
    if len(Xm) == 0 and len(Xr) == 0:
        raise typer.BadParameter("empty fit set: no model clouds and no reference rotamers for the subset")

    X = np.concatenate([Xm, Xr]) if len(Xr) else Xm
    y = np.concatenate([ym, yr]) if len(yr) else ym
    is_ref = np.concatenate([np.zeros(len(Xm), bool), np.ones(len(Xr), bool)])

    # ---- sample weights: balanced class weight x popweight ----
    cnt = collections.Counter(y.tolist())
    nC = len(cnt)
    N = len(y)
    balanced = np.array([N / (nC * cnt[v]) for v in y], np.float64)

    def norm_within_class(w, subset):
        out = w.astype(np.float64).copy()
        for c in set(y.tolist()):
            m = (y == c) & subset
            if m.sum() and out[m].sum() > 0:
                out[m] *= m.sum() / out[m].sum()
        return out

    def popw_ref_of(wref_raw):
        # model-cloud rows -> 1.0; ref rows -> per-row popweight, normalized within class to mean 1.
        p = np.concatenate([np.ones(len(Xm), np.float32), wref_raw]) if len(wref_raw) else np.ones(len(X), np.float32)
        p[~is_ref] = 1.0
        p = norm_within_class(p, is_ref)
        p[~is_ref] = 1.0
        return p

    if popweight == "a6":
        popw = popw_ref_of(wr_a6)
    elif popweight == "a5":
        popw = popw_ref_of(wr_a5)
    else:  # none
        popw = np.ones(len(X), np.float64)
    sw = balanced * popw

    _n = int(is_ref.sum())
    _b = reflib.ref_branch
    FB = {
        "ref_rows": _n,
        "specific": _b.get("specific", 0),
        "rotamer_population": _b.get("rp", 0),
        "uniform_fallback": _b.get("fallback", 0),
        "frac_fallback": round(_b.get("fallback", 0) / max(_n, 1), 4),
    }
    typer.echo(
        f"[fit_learned_readout] fit set: model={len(Xm)} ({n_model_types} types) ref={len(Xr)} classes={nC} "
        f"| popweight={popweight} REF branch: specific={FB['specific']} rp={FB['rotamer_population']} "
        f"fallback={FB['uniform_fallback']}"
    )

    # ---- fit ----
    t0 = time.time()
    if warmstart_from:
        max_iter = int(os.environ.get("WARM_MAXITER", WARM_MAX_ITER_DEFAULT))
        clf, niter, base_cls = _warm_start_fit(X, y, sw, warmstart_from, max_iter)
        fit_kind = "warmstart"
        cap = max_iter
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=COLD_MAX_ITER, C=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X, y, logisticregression__sample_weight=sw)
        niter = int(np.max(clf.named_steps["logisticregression"].n_iter_))
        fit_kind = "cold"
        cap = COLD_MAX_ITER
    dt = time.time() - t0
    if niter >= cap:
        typer.echo(
            f"[fit_learned_readout] WARN lbfgs hit max_iter={cap} (n_iter={niter}) -- not converged. "
            f"{'Increase WARM_MAXITER or use a closer --warmstart-from head.' if warmstart_from else 'Pass --warmstart-from for a 7-9x speedup / convergence.'}"
        )

    cls = np.array([str(c) for c in clf.named_steps["logisticregression"].classes_])
    prior = np.array([NAT.get(c, FLOOR) for c in cls])

    meta = {
        "readout": "logreg_bbphipsi",
        "ckpt": ckpt_name,
        "scope": scope,
        "popweight_mode": popweight,
        "popweight": "ref_rotamers_only" if popweight != "none" else "disabled",
        "fallback": ("pooled_phipsi_grid" if popweight == "a6" else ("uniform_1.0" if popweight == "a5" else "n/a")),
        "variant": ("A6" if popweight == "a6" else ("A5" if popweight == "a5" else "balanced_only")),
        "prior_note": "bundle prior=natfreq SwissProt (OPTIONAL); PRIMARY readout=uniform predict_proba argmax",
        "feature_spec": FEATURE_SPEC,
        "slot_convention": SLOT_CONV,
        "corruption": {"atom_drop": DROP, "jitter_sigma": SIGMA},
        "phi_psi": "retained(sampled parent-grid)",
        "ref_db": os.path.basename(ref_db),
        "phipsi_table": os.path.basename(phipsi_table),
        "ckpt_fit_cache": os.path.basename(ckpt_fit_cache),
        "residues_spec": os.path.basename(residues),
        "subset_requested": len(requested),
        "subset_missing_from_refdb": missing,
        "holdout_ref_only": holdout_in,
        "n_classes": len(cls),
        "model_n": len(Xm),
        "ref_n": len(Xr),
        "fallback_stats": FB,
        "fit_kind": fit_kind,
        "n_iter": niter,
        "warmstart_from": (os.path.basename(warmstart_from) if warmstart_from else None),
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "classes": cls, "prior": prior, "meta": meta}, out)
    typer.echo(
        f"[fit_learned_readout] [{fit_kind} n_iter={niter}] saved {out}  "
        f"classes={len(cls)} model_n={len(Xm)} ref_n={len(Xr)}  ({dt:.1f}s fit / {time.time() - t_start:.1f}s total)"
    )


if __name__ == "__main__":
    app()
