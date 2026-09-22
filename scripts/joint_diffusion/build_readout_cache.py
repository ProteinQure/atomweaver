"""Build the learned-readout per-alphabet cache (self-contained).

Produces `readout_cache.pkl` with BOTH modulators (canon 20-class + unconstrained all-representable), trained
ONLY on GT reference rotamers + Gaussian coordinate noising (zero-leak; reproducible at inference for any
alphabet from the reference library alone), class-balanced (min samples/class) with a Laplace output floor, PLUS
the calibration temperature persisted into the cache.

Provenance note (see also learned_readout.py): the MODULATORS are reference-only. The calibration temperature
(`--global-t`) is a single scalar fit SEPARATELY on held-out MODEL clouds (standard temperature-scaling); pass
the fitted value here (default 2.12 from the ep274 fit) so the cache is self-contained. A new alphabet keeps the
same modulator pipeline; only the temperature may want re-fitting on a small model-cloud validation set.

Usage:
  python build_readout_cache.py --eval-db <DB.pt> --out readout_cache.pkl [--global-t 2.12] \
      [--sigmas 0.2,0.4,0.6,0.8] [--min-per-class 192]
"""

import argparse
import time

import learned_readout
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def build(eval_db, out, global_t=2.12, sigmas=(0.2, 0.4, 0.6, 0.8), min_per=192, seed=0, device="cpu"):
    import joblib

    from scripts.joint_diffusion.sample import build_eval_discretizer

    discs, ccd_to_idx, sc_a, sc_s, rep, db, r2t = build_eval_discretizer(
        eval_db, device, chirality_mismatch_penalty=0.0
    )
    idx_to_ccd = {v: k for k, v in ccd_to_idx.items()}
    CANON = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
    coords, masks, elem, bbi = db["coords"], db["masks"], db["element_types"], db["backbone_indices"]
    S = 16
    iu = torch.triu_indices(S, S, 1)

    def feats(xyz, m, el):
        d = torch.cdist(xyz, xyz)[:, iu[0], iu[1]]
        rp = m[:, iu[0]] & m[:, iu[1]]
        d = torch.where(rp, d, torch.full_like(d, -1.0))
        eo = torch.zeros(xyz.shape[0], S, 5)
        ei = el.clone()
        ei[~m] = 4
        ei = ei.clamp(0, 4)
        eo.scatter_(2, ei.long().unsqueeze(2), 1.0)
        return torch.cat([d, eo.reshape(xyz.shape[0], -1)], 1).numpy()

    reptypes = [t for t in range(rep.numel()) if bool(rep[t]) and (r2t == t).any()]
    canon_t = [t for t in reptypes if idx_to_ccd.get(t) in CANON]
    g = torch.Generator().manual_seed(seed)
    rx, ry, t0 = [], [], time.time()
    for t in reptypes:
        rots = (r2t == t).nonzero().flatten().tolist()
        npat = max(8, int(np.ceil(min_per / (len(rots) * len(sigmas)))))  # class-balance rare types
        for ri in rots:
            m = masks[ri]
            bset = set(bbi[ri].tolist())
            sc = [j for j in range(coords.shape[1]) if bool(m[j]) and j not in bset and int(elem[ri, j]) >= 0]
            x0 = torch.zeros(S, 3)
            mm = torch.zeros(S, dtype=torch.bool)
            ee = torch.full((S,), -1)
            for j in sc:
                ss = j - 4  # full DB slot -> sidechain-slot (reserved slot4 -> ss0), matches model atom-name convention
                if 0 <= ss < S:
                    x0[ss] = coords[ri, j]
                    mm[ss] = True
                    ee[ss] = int(elem[ri, j])
            for s in sigmas:
                nz = torch.randn(npat, S, 3, generator=g) * s
                nz[:, ~mm, :] = 0
                rx.append(
                    feats(x0.unsqueeze(0) + nz, mm.unsqueeze(0).expand(npat, -1), ee.unsqueeze(0).expand(npat, -1))
                )
                ry.append(np.full(npat, t))
    X = np.nan_to_num(np.concatenate(rx), nan=-1.0)
    Y = np.concatenate(ry)
    print(f"[build] X={X.shape} classes={len(reptypes)} canon={len(canon_t)} gen={time.time() - t0:.0f}s", flush=True)

    cmask = np.isin(Y, canon_t)
    t0 = time.time()
    clf_c = LogisticRegression(max_iter=300, C=1.0, n_jobs=8).fit(X[cmask], Y[cmask])
    tc = time.time() - t0
    t0 = time.time()
    clf_u = LogisticRegression(max_iter=200, C=1.0, n_jobs=8).fit(X, Y)
    tu = time.time() - t0
    print(f"[fit] canon({len(canon_t)})={tc:.1f}s unconstrained({len(reptypes)})={tu:.0f}s", flush=True)

    # natural-frequency prior (SwissProt %); canonicals get their frequency, NCAAs a small floor so the
    # unconstrained readout stays NCAA-capable but strongly down-weighted. Stored in the cache; apply_readout
    # normalizes, so only relative scale matters.
    NAT = {
        "ALA": 8.25,
        "ARG": 5.53,
        "ASN": 4.06,
        "ASP": 5.46,
        "CYS": 1.38,
        "GLN": 3.93,
        "GLU": 6.72,
        "GLY": 7.07,
        "HIS": 2.27,
        "ILE": 5.91,
        "LEU": 9.66,
        "LYS": 5.84,
        "MET": 2.41,
        "PHE": 3.86,
        "PRO": 4.74,
        "SER": 6.56,
        "THR": 5.34,
        "TRP": 1.08,
        "TYR": 2.92,
        "VAL": 6.87,
    }
    # Exact offline (0.705) scale: canonical freqs as FRACTIONS (f/100), NCAA floor 1e-6 -> canon:NCAA ratio ~0.0825/1e-6.
    _ncaa_floor = 1e-6
    prior_canon = {int(t): NAT.get(idx_to_ccd.get(t), 100.0) / 100.0 for t in canon_t}
    prior_unconstrained = {
        int(t): (NAT[idx_to_ccd[t]] / 100.0 if idx_to_ccd.get(t) in NAT else _ncaa_floor) for t in reptypes
    }

    cache = {
        "clf_c": clf_c,
        "clf_u": clf_u,
        "canon_t": canon_t,
        "reptypes": reptypes,
        "floor": 1e-6,
        "calib": {"global_T": float(global_t)},  # persisted (fit separately on held-out model clouds)
        "prior_canon": prior_canon,  # natural-freq prior (stored in the cache)
        "prior_unconstrained": prior_unconstrained,
        "slot_convention": {"S": S, "reserved_slot0": True},  # readout is slot-order sensitive; guarded in eval
        "library_fingerprint": learned_readout.library_fingerprint(db),  # classifier labels are DB type indices
        # ...and the geometry those labels were actually fit on: identity alone lets a library
        # rebuilt with the same residue list and rotamer count pass with different conformers.
        "geometry_digest": learned_readout.geometry_digest(db),
    }
    joblib.dump(cache, out)
    print(f"[cache] saved {out}  (canon+unconstrained modulators + calib global_T={global_t})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--global-t", type=float, default=2.12)
    ap.add_argument("--sigmas", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--min-per-class", type=int, default=192)
    a = ap.parse_args()
    build(a.eval_db, a.out, a.global_t, tuple(float(x) for x in a.sigmas.split(",")), a.min_per_class)
