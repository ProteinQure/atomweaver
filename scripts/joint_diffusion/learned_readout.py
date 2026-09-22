"""Learned discretization readout (per-alphabet cache, zero-leak).

Pipeline:  NDM geometry -> learned modulator (predict_proba over types) x pluggable prior -> calibrated logits.
The MODULATORS (canon 20-class + unconstrained all-representable) are built OFFLINE from GT-rotamer noising ONLY
(no model/test clouds) -> zero-leak, reproducible at inference from the reference library alone for any alphabet.
PROVENANCE CAVEAT: the calibration scalar (global temperature) is fit SEPARATELY on held-out MODEL clouds
(standard temperature-scaling; see calibration_finish.py), so that one scalar is NOT reproducible from the
reference library alone -- a new alphabet needs a small model-cloud validation set or a default T. User
temperature layers on top of the calibrated logits.

Feature convention (must match the cache's training):
  - MAXSC=16 sidechain slots, contiguous by the fixed sidechain-slot index (reserved slot0 -> Proline's ring atom).
  - features = upper-tri pairwise distance matrix (ghost-involving pairs -> -1) ++ per-slot element one-hot
    {C,N,O,X,PAD} (5).  Elements are DB ids {C:0,N:1,O:2,X:3}; ghost via mask.
"""

import joblib
import numpy as np
import torch

MAXSC = 16


def cloud_features(pred_coords, pred_mask, pred_elem_db, S=MAXSC):
    """pred_coords [B,L,s,3], pred_mask [B,L,s] bool, pred_elem_db [B,L,s] long (DB ids; ghost anything+masked).
    Returns [B*L, F] float32 (padded/truncated to S slots).
    """
    B, L, s, _ = pred_coords.shape
    x = torch.zeros(B, L, S, 3)
    m = torch.zeros(B, L, S, dtype=torch.bool)
    e = torch.full((B, L, S), -1, dtype=torch.long)
    k = min(s, S)
    x[..., :k, :] = pred_coords[..., :k, :]
    m[..., :k] = pred_mask[..., :k].bool()
    e[..., :k] = pred_elem_db[..., :k].long()
    x = x.reshape(B * L, S, 3)
    m = m.reshape(B * L, S)
    e = e.reshape(B * L, S)
    iu = torch.triu_indices(S, S, 1)
    d = torch.cdist(x, x)[:, iu[0], iu[1]]
    rp = m[:, iu[0]] & m[:, iu[1]]
    d = torch.where(rp, d, torch.full_like(d, -1.0))
    eo = torch.zeros(B * L, S, 5)
    ei = e.clone()
    ei[~m] = 4  # ghost slots take the PAD column; this is the only legitimate use of index 4
    # The one-hot is five wide because the cache was fit under the production 5-element vocabulary
    # {C, N, O, X, PAD}. Under ATOMWEAVER_ELEMENT_VOCAB=12 -- which is what diffusion.py DEFAULTS to --
    # DB ids run past 3, and the old `clamp(0, 4)` folded every one of them (P, F, Cl, Br, I, Se, B)
    # into index 4, the GHOST column: every heteroatom past sulfur silently encoded as "no atom",
    # with a feature vector of the right shape and no warning anywhere. Refuse instead: this readout
    # cannot represent those elements, and a wrong answer is worse than no answer.
    real_out_of_range = m & (e > 3)
    if bool(real_out_of_range.any()):
        offending = sorted({int(v) for v in e[real_out_of_range].unique()})
        raise ValueError(
            f"cloud_features received DB element id(s) {offending} on real slots, but the learned "
            f"readout's one-hot only spans {{C:0, N:1, O:2, X:3}} plus PAD. This is the signature of "
            "running under ATOMWEAVER_ELEMENT_VOCAB=12 with a cache built for the production vocab 5; "
            "set ATOMWEAVER_ELEMENT_VOCAB=5 to match the cache, or rebuild the cache."
        )
    ei = ei.clamp(0, 4)
    eo.scatter_(2, ei.unsqueeze(2), 1.0)
    F = torch.cat([d, eo.reshape(B * L, -1)], 1).numpy()
    return np.nan_to_num(F, nan=-1.0).astype(np.float32)


def load_cache(path):
    return joblib.load(path)


def geometry_digest(eval_db: dict) -> str:
    """
    A digest of the actual reference GEOMETRY and elements the modulators were fit on.

    The identity fingerprint below records residue codes and counts, which is what makes a class id
    mean something -- but it is not what the classifier learned from. ``build_readout_cache`` fits
    on ``db["coords"]`` and ``db["element_types"]``, and the residue-database builder stores
    ``num_rotamers`` as the REQUESTED count, so rebuilding the library with the same residue list
    and the same ``--num-rotamers`` produces conformers that differ while every identity field
    matches exactly. Without this the cache would validate against geometry it never saw.

    Elements are digested too: they feed the per-slot one-hot in :func:`cloud_features`, and a
    reassignment leaves atom counts untouched.

    Empty string when the database carries no coordinates, so a metadata-only stub (as tests build)
    still fingerprints rather than raising.
    """
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for key in ("coords", "element_types"):
        value = eval_db.get(key)
        if value is None:
            continue
        tensor = value.detach().cpu().contiguous() if hasattr(value, "detach") else torch.as_tensor(value)
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.float().numpy().tobytes())
    return digest.hexdigest()


def library_fingerprint(eval_db: dict) -> list[dict]:
    """Ordered residue identities whose type indices the learned classifiers were trained to emit."""
    return [
        {
            "type_index": i,
            "ccd_code": str(meta.get("ccd_code", "")),
            "num_atoms": int(meta.get("num_atoms", 0)),
            "num_rotamers": int(meta.get("num_rotamers", 1)),
        }
        for i, meta in enumerate(eval_db["metadata"])
    ]


def validate_cache_library(cache: dict, eval_db: dict, context: str = "learned-readout") -> None:
    """
    Fail if the cache was not built from THIS residue library.

    Two checks, because a class id is only meaningful if both hold. The identity fingerprint says
    the class ids still name the residues they were trained to name. The geometry digest says the
    conformers behind them are the ones the modulator actually saw -- see :func:`geometry_digest`
    for why identity alone lets a rebuilt library pass.
    """
    expected = library_fingerprint(eval_db)
    found = cache.get("library_fingerprint")
    if found is None:
        raise RuntimeError(
            f"{context} cache lacks library_fingerprint metadata; rebuild it with build_readout_cache.py "
            "before trusting learned class ids."
        )
    if found != expected:
        mismatch = next((i for i, (a, b) in enumerate(zip(found, expected, strict=False)) if a != b), None)
        if mismatch is None:
            mismatch = min(len(found), len(expected))
        raise RuntimeError(
            f"{context} cache was built for a different residue library/order at type index {mismatch}; "
            "rebuild the cache for this eval DB before scoring."
        )
    # Absent on a cache built before the digest existed. Warn rather than refuse: those caches are
    # still identity-correct, and the identity check above is the one that keeps class ids honest.
    stored_digest = cache.get("geometry_digest")
    current_digest = geometry_digest(eval_db)
    if stored_digest is None:
        print(
            f"{context} WARNING: cache predates the geometry digest, so only residue IDENTITY could be "
            "checked. A library rebuilt with the same residue list would pass this. Rebuild the cache "
            "to get the stronger guard.",
            flush=True,
        )
    elif current_digest and stored_digest != current_digest:
        raise RuntimeError(
            f"{context} cache names the same residues in the same order, but was fit on DIFFERENT "
            f"reference geometry (digest {stored_digest[:12]} vs {current_digest[:12]}). The library "
            "has been rebuilt since; rebuild the cache against it before scoring."
        )


def apply_readout(features, cache, mode="canon", prior=None, temperature=1.0, floor=1e-6):
    """Features [N,F] -> (logits [N, ncls], classes). Calibrated (global T), prior-modulated, user-temperature.
    mode: 'canon'|'unconstrained'. prior: {type_idx: weight} or None (uniform).

    ORDER: temperature FIRST, then the prior. Dividing log(P * w) by T gives
    ``P**(1/T) * w**(1/T)`` -- the prior gets raised to 1/T along with the model's own
    probabilities, so at the cached global_T = 2.12 a prior asking for 100:1 delivers 8.8:1 and the
    weights a caller passes are not the weights that apply. Calibration is a statement about the
    classifier's confidence and has no business rescaling a prior supplied from outside it, so the
    temperature is applied to the model's probabilities and the prior multiplied onto the result.

    Inert at this panel's defaults (``prior=None``), and identical to the old ordering whenever
    ``T == 1``. It changes the ``--readout-prior cache`` path, which is the one that was wrong.
    """
    clf = cache["clf_c"] if mode == "canon" else cache["clf_u"]
    P = clf.predict_proba(features)
    P = P + floor
    P = P / P.sum(1, keepdims=True)
    T = cache.get("calib", {}).get("global_T", 1.0) * float(temperature)
    if T <= 0:
        # np.log(P) is negative everywhere, so dividing by 0 sends every class to -inf with only a
        # numpy warning: topk then sees nothing finite, every frequency comes out zero, and the CSV
        # reads as a legitimately unconfident model.
        raise ValueError(f"readout temperature must be > 0, got {T} (calibration x user temperature)")
    logits = np.log(P + 1e-12) / T
    if prior is not None:
        pw = np.array([prior.get(int(c), floor) for c in clf.classes_])
        logits = logits + np.log(pw + 1e-12)[None, :]
    return logits, clf.classes_
