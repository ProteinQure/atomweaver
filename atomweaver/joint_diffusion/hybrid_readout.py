"""Hand-crafted hybrid Learned+NDM read-out: NDM cliff transforms + per-class blend families.

NO fitting anywhere. Every knob (eps, m, beta, gamma, T) is set by reasoning; test1k is a
guardrail, not an objective. All functions operate on per-position vectors over a FIXED candidate
vocabulary (canonical + NCAA), with a boolean `is_canon` mask aligned to the candidate columns.

Two inputs per position:
  * pL : Learned head predict_proba over the candidate columns (already a proper distribution
         once restricted+renormalized). Owns canonical identity.
  * sN : NDM geometric similarity SCORE over the candidate columns (higher = better match;
         -inf / NaN for non-representable candidates). Drives NCAA identity.

The pipeline is:  sN --cliff(eps,m,...)-->  qN (a distribution) ;  then blend(pL, qN, is_canon).
Everything returns a row-normalized distribution over the candidate columns.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-12


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _row_norm(p: np.ndarray) -> np.ndarray:
    """Clip negatives, renormalize each row to sum 1; uniform fallback on all-zero rows."""
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    s = p.sum(1, keepdims=True)
    z = (s <= 0).ravel()
    if z.any():
        p[z] = 1.0
        s = p.sum(1, keepdims=True)
    return p / s


def learned_dist(pL_full: np.ndarray) -> np.ndarray:
    """Restrict+renormalize the Learned proba (already over candidate cols) to a clean distribution."""
    return _row_norm(np.asarray(pL_full, dtype=np.float64))


def _eps_effective(sorted_desc: np.ndarray, eps: float, eps_mode: str) -> float:
    """Resolve the epsilon knob to absolute score units for one position.

    `sorted_desc` = finite scores sorted high->low.  eps_mode:
      * 'abs' -- eps is already in raw NDM-score units.
      * 'range' -- eps is a FRACTION of the per-position finite range (max-min). NOTE: the NDM
                        score floor is dominated by the chirality/count PENALTY (down to ~-90), so
                        this range is huge (~58) and eps=range is COARSE near the top competition.
      * 'topk_range' -- eps is a FRACTION of the range of the TOP-10 finite scores (max - 10th),
                        a clean near-tie scale that ignores the penalty floor. Preferred relative mode.
    """
    if eps_mode == "abs":
        return float(eps)
    if eps_mode == "range":
        rng = max(float(sorted_desc[0] - sorted_desc[-1]), _TINY)
        return float(eps) * rng
    if eps_mode == "topk_range":
        k = min(10, sorted_desc.size)
        rng = max(float(sorted_desc[0] - sorted_desc[k - 1]), _TINY)
        return float(eps) * rng
    raise ValueError(f"eps_mode must be 'abs' | 'range' | 'topk_range', got {eps_mode!r}")


# ----------------------------------------------------------------------------------------------
# NDM cliff transforms: score vector -> distribution (per position)
# ----------------------------------------------------------------------------------------------
def cliff_row(
    s: np.ndarray,
    *,
    kind: str = "largest_gap",
    eps: float = 0.02,
    eps_mode: str = "range",
    m: float = 0.8,
    below: str = "even",
    above_split: str = "equal",
    T: float = 0.1,
    plateau_guard: bool = True,
) -> np.ndarray:
    """Turn ONE NDM score vector `s` (candidate cols; -inf/NaN => excluded) into a distribution.

    Parameters
    ----------
    kind : 'top_plateau' | 'walk_plateau' | 'largest_gap' | 'softmax'
        * top_plateau : winners = {c : s_max - s_c <= eps_eff} ("within eps of the max"; this is
                         also the 'exact-tie' family since NDM ties are only ever near-ties, and the
                         'relative-threshold' family when eps is set to a deliberate delta).
        * walk_plateau : sort desc; walk down while each CONSECUTIVE gap <= eps_eff; STOP at the first
                         gap > eps_eff (the coordinator's 2nd definition). Top-focused; robust to the
                         penalty floor (the huge drop to the chirality/count-penalized tail).
        * largest_gap : sort desc; cut at the LARGEST consecutive gap that EXCEEDS eps_eff; winners
                         are the block above that cut. Given the penalty floor this usually cuts at
                         the big drop to the penalized tail => a BROAD "geometrically-plausible" set.
        * softmax : smooth baseline, softmax(s / T) over live candidates (eps/m/below ignored).
    eps, eps_mode : cliff tolerance; see `_eps_effective`. Governs tie-detection & gap significance.
    m : mass assigned to the winner block (rest gets 1-m). Ignored when below='zero' (winners get 1).
    below : 'even' (losers share 1-m equally) | 'zero' (losers get 0; winners share full mass 1).
    above_split : 'equal' | 'proportional' (winners split their mass proportional to their score,
                  shifted to be positive within the block).
    T : softmax temperature (kind='softmax' only).
    plateau_guard : when True (default), clamp the winner mass up to ``m_eff = max(m, nwin/n)`` so the
                  winner per-member score (m_eff/nwin) is always >= the tail per-member score
                  ((1-m_eff)/(n-nwin)). Without it, a huge winner plateau (nwin/n > m, i.e. the
                  k>4n/5 case) makes the tail per-member score EXCEED the winner per-member score,
                  inverting the ranking so a tail candidate can win argmax. With the shipped m=0.8 the
                  clamp NEVER triggers unless nwin > 4n/5, so on normal data m_eff == m and the output
                  is byte-identical. No-op for below='zero' (losers get 0 mass anyway).
    """
    s = np.asarray(s, dtype=np.float64).ravel()
    live = np.isfinite(s)
    n = int(live.sum())
    q = np.zeros_like(s)
    if n == 0:
        return q  # caller renormalizes; all-zero -> uniform fallback upstream
    if n == 1:
        q[live] = 1.0
        return q

    sv = s[live]
    smax, smin = float(sv.max()), float(sv.min())

    if kind == "softmax":
        z = (sv - smax) / max(T, 1e-6)
        w = np.exp(z)
        q[live] = w / w.sum()
        return q

    order = np.argsort(-sv)  # indices into sv, best first
    ss = sv[order]
    eps_eff = _eps_effective(ss, eps, eps_mode)

    if kind == "top_plateau":
        win_local = (smax - sv) <= eps_eff  # boolean over live candidates (original order)
    elif kind in ("walk_plateau", "largest_gap"):
        gaps = ss[:-1] - ss[1:]  # consecutive descending gaps, length n-1
        sig = gaps > eps_eff
        if not sig.any():
            win_local = np.ones(n, dtype=bool)  # flat plateau, no cliff -> all co-winners
        else:
            if kind == "walk_plateau":
                cut = int(np.argmax(sig))  # FIRST significant gap from the top
            else:
                cut = int(np.argmax(np.where(sig, gaps, -np.inf)))  # LARGEST significant gap
            keep_sorted = np.zeros(n, dtype=bool)
            keep_sorted[: cut + 1] = True  # block above the cut
            win_local = np.zeros(n, dtype=bool)
            win_local[order] = keep_sorted
    else:
        raise ValueError(f"cliff kind must be top_plateau|walk_plateau|largest_gap|softmax, got {kind!r}")

    nwin = int(win_local.sum())
    qv = np.zeros(n)
    # plateau guard ([1]): keep every winner per-member score >= every tail per-member score.
    # m_eff/nwin >= (1-m_eff)/(n-nwin) <=> m_eff >= nwin/n, so clamp m up to nwin/n when the winner
    # plateau is huge (nwin/n > m). Never triggers on normal data (nwin <= 4n/5 with m=0.8).
    m_eff = float(m)
    if plateau_guard and below != "zero" and nwin > 0:
        m_eff = max(m_eff, nwin / n)
    win_mass = 1.0 if below == "zero" else m_eff
    lose_mass = 0.0 if below == "zero" else (1.0 - m_eff)

    if above_split == "proportional" and nwin > 0:
        wsc = sv[win_local]
        wsc = wsc - wsc.min() + _TINY  # shift positive within the winner block
        qv[win_local] = win_mass * wsc / wsc.sum()
    else:  # equal
        qv[win_local] = win_mass / max(nwin, 1)

    nlose = n - nwin
    if nlose > 0 and lose_mass > 0:
        if below == "even":
            qv[~win_local] = lose_mass / nlose
    q[live] = qv
    return q


def cliff_dist(S: np.ndarray, **kw) -> np.ndarray:
    """Vectorized wrapper: apply `cliff_row` to each row of an [L, C] score matrix."""
    S = np.asarray(S, dtype=np.float64)
    out = np.zeros_like(S)
    for i in range(S.shape[0]):
        out[i] = cliff_row(S[i], **kw)
    return _row_norm(out)


# ----------------------------------------------------------------------------------------------
# blend families: (pL, qN, is_canon) -> distribution over candidate cols
# ----------------------------------------------------------------------------------------------
def blend_b1_loglinear(pL: np.ndarray, qN: np.ndarray, is_canon: np.ndarray, beta: float) -> np.ndarray:
    """B1 log-linear per-class: logP(c) ~ w_c*log pL(c) + (1-w_c)*log qN(c).

    w_canon = 1 (canon identity from Learned only); w_ncaa = beta. Both inputs are distributions;
    floored at _TINY before the log. Result is a valid distribution (softmax of the weighted logs).
    """
    pL = learned_dist(pL)
    qN = _row_norm(qN)
    w = np.where(is_canon[None, :], 1.0, float(beta))
    lp = w * np.log(np.clip(pL, _TINY, None)) + (1.0 - w) * np.log(np.clip(qN, _TINY, None))
    lp = lp - lp.max(1, keepdims=True)
    return _row_norm(np.exp(lp))


def blend_b2_convex(pL: np.ndarray, qN: np.ndarray, is_canon: np.ndarray, beta: float) -> np.ndarray:
    """B2 convex per-class: P(c) ~ w_c*pL(c) + (1-w_c)*qN(c), renormalized.

    w_canon = 1; w_ncaa = beta. beta=0 => NCAA mass comes entirely from NDM, canon mass from Learned.
    """
    pL = learned_dist(pL)
    qN = _row_norm(qN)
    w = np.where(is_canon[None, :], 1.0, float(beta))
    return _row_norm(w * pL + (1.0 - w) * qN)


def blend_b3_gated(pL: np.ndarray, qN: np.ndarray, is_canon: np.ndarray, gamma: float) -> np.ndarray:
    """B3 gated sub-order override (lowest expected canon regression).

    * canon-vs-NCAA TOTAL mass split AND within-canon ordering come from Learned.
    * within-NCAA ordering comes from the NDM cliff.
    * gamma scales the NCAA total mass: M_ncaa' = clip(gamma*M_ncaa, 0, 1) ; M_canon' = 1 - M_ncaa'.
      gamma=1 leaves the Learned canon/NCAA mass split (and thus canon behaviour) untouched.
    """
    pL = learned_dist(pL)
    qN = _row_norm(qN)
    canon = is_canon[None, :]
    Mc = (pL * canon).sum(1, keepdims=True)  # Learned canon mass
    Mn = (pL * (~canon)).sum(1, keepdims=True)  # Learned NCAA mass
    Mn2 = np.clip(float(gamma) * Mn, 0.0, 1.0)
    Mc2 = 1.0 - Mn2
    out = np.zeros_like(pL)
    # canon block: preserve Learned within-canon shape, rescale to Mc2
    cshape = pL * canon
    cs = cshape.sum(1, keepdims=True)
    out += np.where(cs > 0, Mc2 * cshape / np.clip(cs, _TINY, None), 0.0)
    # NCAA block: take NDM cliff within-NCAA shape, rescale to Mn2
    nshape = qN * (~canon)
    ns = nshape.sum(1, keepdims=True)
    # if NDM gives no NCAA mass at a position, fall back to Learned NCAA shape so mass is conserved
    fallback = pL * (~canon)
    fs = fallback.sum(1, keepdims=True)
    nshape = np.where(ns > 0, nshape, fallback)
    ns = np.where(ns > 0, ns, fs)
    out += np.where(ns > 0, Mn2 * nshape / np.clip(ns, _TINY, None), 0.0)
    return _row_norm(out)


# ----------------------------------------------------------------------------------------------
# preset dispatch
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# SHIPPABLE PRESETS (hand-crafted; NO fitting). eps in absolute NDM-score units (the regime that
# behaves well; full-range-relative eps is polluted by the chirality/count penalty floor). The
# near-tie cliff is walk_plateau (walk down, break at first gap > eps), m=0.8, below='even' -- which
# keeps NDM's strong top-3/top-10 NCAA ranking. Canon identity is owned by the Learned head in every
# blended preset (per-class weight w_canon=1), so the 20-way canonical top-1 is preserved (~0.235).
# ----------------------------------------------------------------------------------------------
_WALK = {
    "kind": "walk_plateau",
    "eps": 0.02,
    "eps_mode": "abs",
    "m": 0.8,
    "below": "even",
    "above_split": "equal",
    "T": 0.1,
    "plateau_guard": True,
}  # guard: winner per-member >= tail per-member (no k>4n/5 inversion)

SHIP_PRESETS = {
    # pure-NDM cliff, logits-shaped -- reference anchor for the DMS pipeline (canon NOT protected here).
    "ndm_cliff_anchor": {"family": "pure_ndm", "cliff": dict(_WALK)},
    # lowest canon regression: Learned sets canon/NCAA mass split + canon order; NDM orders WITHIN NCAA.
    "b3_gated_g1": {"family": "B3", "gamma": 1.0, "cliff": dict(_WALK)},
    # literal task spec: canon = Learned, NCAA identity entirely from NDM (convex, beta=0).
    "b2_ncaa_ndm": {"family": "B2", "beta": 0.0, "cliff": dict(_WALK)},
    # balanced convex: NCAA = 1/2 Learned + 1/2 NDM (retains Learned's NCAA top-1 strength).
    "b2_balanced": {"family": "B2", "beta": 0.5, "cliff": dict(_WALK)},
    # log-linear consensus: NCAA ~ geometric mean of Learned & NDM (distinct product-form family).
    "b1_loglinear": {"family": "B1", "beta": 0.5, "cliff": dict(_WALK)},
    # NDM-heavy dial: gamma=2 doubles the NCAA mass budget (most NDM-leaning blend; modest canon cost).
    "b3_gated_g2": {"family": "B3", "gamma": 2.0, "cliff": dict(_WALK)},
}


def apply_preset(pL: np.ndarray, sN: np.ndarray, is_canon: np.ndarray, preset: dict) -> np.ndarray:
    """Apply a full hybrid preset dict to [L,C] Learned proba + [L,C] NDM scores.

    preset keys:
        family      : 'pure_learned' | 'pure_ndm' | 'B1' | 'B2' | 'B3'
        cliff       : sub-dict of kwargs for cliff_dist (kind, eps, eps_mode, m, below, above_split, T)
        beta / gamma: blend knob for the chosen family
    Returns an [L,C] row-normalized distribution over the candidate columns.
    """
    fam = preset["family"]
    if fam == "pure_learned":
        return learned_dist(pL)
    cliff_kw = preset.get("cliff", {})
    qN = cliff_dist(sN, **cliff_kw)
    if fam == "pure_ndm":
        return qN
    if fam == "B1":
        return blend_b1_loglinear(pL, qN, is_canon, preset["beta"])
    if fam == "B2":
        return blend_b2_convex(pL, qN, is_canon, preset["beta"])
    if fam == "B3":
        return blend_b3_gated(pL, qN, is_canon, preset["gamma"])
    raise ValueError(f"unknown family {fam!r}")
