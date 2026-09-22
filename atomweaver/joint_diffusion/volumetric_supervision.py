"""Faithful supervision for the volumetric self-occupancy head (standalone pretraining).

This module provides the loss + region-classification machinery to pretrain
:class:`~atomweaver.joint_diffusion.volumetric_head.VolumetricOccupancyHead` to *self-occupancy-level*
BEFORE it is deep-injected into the full flow-matching trunk. The motivation is signal budget:
grafted into the live model the head only ever sees leftover gradient; trained standalone it gets
100% of the training signal on the one task it exists to do (predict a residue's own side-chain
occupancy density from a target-aware backbone context).

Why a separate module (vs. the head's own ``volumetric_occupancy_loss``)
-----------------------------------------------------------------------
The head ships a *simplified* anti-leak loss: a single empty-weighted MSE that derives the "empty"
bucket at run time from where the GT splat is ~0. Self-occupancy's real recipe is richer and is what we
reproduce here:

* **Per-region bucketing.** Query points are labelled center / around / context / near-empty /
  medium-empty / far-empty and the MSE is taken *per bucket*, then weighted and summed.
* **Empty up-weighting (load-bearing).** The three EMPTY buckets are weighted ``1.75×`` (self-occupancy
  default). This is the anti-leak term that sharpens the field: occupancy must fall to ~0 off the
  side chain, and mispredicting empty space costs more than a same-size error on the side chain.
* **Context as label, not target.** The regressed density is the masked residue's OWN side chain
  ONLY (self-occupancy ``query_sampling.py``: ``occupancy = _build_occupancy_targets(query_pos,
  target_coords)`` with ``target_coords`` = the masked residue's own atoms). Neighbour + target
  atoms enter as the head's INPUT and as the CONTEXT *region label* (weight 1.0) that weights the
  loss on query points sitting near context -- those points regress to ~0 own-density. Context is
  NEVER added to the target density; doing so would make the head re-predict its own input and blur
  the own-sidechain identity signal.
* **Charge OFF.** Self-occupancy's charge head is weight 0 in this recipe; not modelled here.

Region logic adapted to OUR fixed lattice
------------------------------------------
Self-occupancy *samples* fresh query points into each bucket every step (masked-center at exact atom
centers, masked-around within a radius of an atom, distance-banded empties, ...). We instead have a
FIXED Fibonacci-shell lattice (:func:`~atomweaver.joint_diffusion.volumetric_head.build_query_points`)
evaluated once per residue, so we *classify* each fixed lattice point into the reference's buckets by
distance:

* distance to the nearest OWN (masked) side-chain atom -> ``center`` / ``around``;
* distance to the nearest CONTEXT atom (neighbour backbone + target) -> ``context``;
* otherwise distance to the nearest atom of any kind -> ``near`` / ``medium`` / ``far`` empty tiers.

Thresholds mirror the reference's nm bands converted to Å (self-occupancy works in nm; ``×10``): masked-around
``0.12 nm -> 1.2 Å``; near-empty ``0.25 nm -> 2.5 Å``; medium-empty ``0.35 nm -> 3.5 Å``. The extra
``center`` sub-band (default ``0.5 Å``) has no self-occupancy analogue -- the reference's masked-center is *exactly*
at atom centers, which our fixed lattice never lands on, so we carve the tightest own-atom shell as
``center`` and the rest of the ``≤1.2 Å`` own-atom shell as ``around``.

The GT regression target here is the masked residue's OWN side-chain occupancy field ONLY
(faithful; :func:`build_full_occupancy_target` with its default ``include_context=False``),
built with the EXACT splat transform used by the head (:func:`local_frame_splat` mirrors
``VolumetricOccupancyHead._splat_density``), so predictions and targets live on the identical
lattice + sigma and are directly comparable. Context atoms condition the prediction and label the
CONTEXT bucket, but are not part of the target (``include_context=True`` restores the legacy
own+context field for A/B only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

from .volumetric_head import per_atom_sigma_from_ids, splat_gaussian_kernel

# --- Region ids (mirror the reference's QUERY_TYPE_* ordering exactly) ---------------------------------
REGION_CENTER = 0  # self-occupancy QUERY_TYPE_MASKED_CENTER
REGION_AROUND = 1  # self-occupancy QUERY_TYPE_MASKED_AROUND
REGION_CONTEXT = 2  # self-occupancy QUERY_TYPE_CONTEXT
REGION_NEAR_EMPTY = 3  # self-occupancy QUERY_TYPE_NEAR_EMPTY
REGION_MEDIUM_EMPTY = 4  # self-occupancy QUERY_TYPE_MEDIUM_EMPTY
REGION_FAR_EMPTY = 5  # self-occupancy QUERY_TYPE_FAR_EMPTY

REGION_NAMES = ("center", "around", "context", "near_empty", "medium_empty", "far_empty")
_EMPTY_REGIONS = (REGION_NEAR_EMPTY, REGION_MEDIUM_EMPTY, REGION_FAR_EMPTY)


@dataclass(frozen=True)
class RegionThresholds:
    """Distance thresholds (Å) that bin fixed lattice points into self-occupancy buckets.

    Defaults mirror the reference's nm bands ``×10`` (self-occupancy works in nm). ``center_radius`` has no self-occupancy
    analogue -- see the module docstring -- and carves the tightest own-atom shell.

    Attributes
    ----------
    center_radius : float
        Own-atom distance ``≤`` this -> ``center`` (default 0.5 Å).
    around_radius : float
        Own-atom distance ``≤`` this (and ``>`` ``center_radius``) -> ``around`` (default 1.2 Å;
        self-occupancy ``masked_around_radius_nm`` 0.12).
    context_radius : float
        Context-atom distance ``≤`` this -> ``context`` (default 1.2 Å).
    near_empty_max : float
        Nearest-any-atom distance ``<`` this -> ``near_empty`` (default 2.5 Å; self-occupancy 0.25 nm).
    medium_empty_max : float
        Nearest-any-atom distance ``<`` this -> ``medium_empty`` (default 3.5 Å; self-occupancy 0.35 nm);
        ``≥`` it -> ``far_empty``.
    """

    center_radius: float = 0.5
    around_radius: float = 1.2
    context_radius: float = 1.2
    near_empty_max: float = 2.5
    medium_empty_max: float = 3.5


@dataclass(frozen=True)
class OccupancyLossWeights:
    """Per-bucket occupancy MSE weights (self-occupancy recipe; charge omitted).

    Empty buckets are up-weighted (self-occupancy default 1.75) -- the load-bearing anti-leak term. Build
    with :meth:`from_empty_weight` to set all three empties from one scalar.
    """

    center: float = 1.0
    around: float = 1.0
    context: float = 1.0
    near_empty: float = 1.75
    medium_empty: float = 1.75
    far_empty: float = 1.75

    @classmethod
    def from_empty_weight(cls, empty_weight: float = 1.75) -> OccupancyLossWeights:
        """All three empty buckets set to ``empty_weight``; center/around/context stay 1.0."""
        return cls(
            center=1.0,
            around=1.0,
            context=1.0,
            near_empty=float(empty_weight),
            medium_empty=float(empty_weight),
            far_empty=float(empty_weight),
        )

    def as_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """(6,) weight vector indexed by ``REGION_*``."""
        return torch.tensor(
            [self.center, self.around, self.context, self.near_empty, self.medium_empty, self.far_empty],
            device=device,
            dtype=dtype,
        )


def local_frame_splat(
    query_local: torch.Tensor,
    coords_global: torch.Tensor,
    weights: torch.Tensor,
    R: torch.Tensor,  # noqa: N803
    ca: torch.Tensor,
    sigma: float,
    per_atom_sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian-splat a per-residue atom cloud onto the fixed local-frame query lattice.

    Byte-for-byte the same transform as
    :meth:`~atomweaver.joint_diffusion.volumetric_head.VolumetricOccupancyHead._splat_density`
    (``local = Rᵀ·(x - CA)``, unit-height Gaussian at each query point, ``sum`` composition), but
    written as a free function so the standalone trainer can splat BOTH the own side chain and the
    shared context atoms onto the identical lattice + sigma the head predicts on. Works for any
    per-residue atom count ``M`` (own ``K`` or broadcast context ``N``).

    Parameters
    ----------
    query_local : torch.Tensor
        (Q, 3) fixed local-frame query lattice (origin = CA).
    coords_global : torch.Tensor
        (B, L, M, 3) atom coordinates in the GLOBAL frame.
    weights : torch.Tensor
        (B, L, M) per-atom weight (0/1 validity mask, or soft weights).
    R : torch.Tensor
        (B, L, 3, 3) per-residue local frame (basis vectors as columns; ``global = R·local``).
    ca : torch.Tensor
        (B, L, 3) per-residue frame origin (CA).
    sigma : float
        Gaussian width (Å), matching the head's ``sigma``.
    per_atom_sigma : torch.Tensor, optional
        (B, L, M) per-atom Gaussian width. ``None`` (default) => the scalar-``sigma`` path (byte-identical
        to the pre-existing splat); when given, atom ``m`` splats with its own width (broadcast over ``Q``).

    Returns
    -------
    torch.Tensor
        (B, L, Q) summed occupancy density at the query points.
    """
    q = query_local.shape[0]
    query_local = query_local.to(coords_global.dtype)  # (Q, 3)
    rel = coords_global - ca.unsqueeze(2)  # (B, L, M, 3)
    local = torch.einsum("blij,blkj->blki", R.transpose(-1, -2), rel)  # (B, L, M, 3)
    d2 = ((query_local.view(1, 1, q, 1, 3) - local.unsqueeze(2)) ** 2).sum(dim=-1)  # (B, L, Q, M)
    kernel = splat_gaussian_kernel(d2, sigma, per_atom_sigma)  # (B, L, Q, M)
    w = weights.to(kernel.dtype).unsqueeze(2)  # (B, L, 1, M)
    return (kernel * w).sum(dim=-1)  # (B, L, Q) -- sum composition


def build_full_occupancy_target(
    query_local: torch.Tensor,
    R: torch.Tensor,  # noqa: N803
    ca: torch.Tensor,
    sigma: float,
    own_coords_global: torch.Tensor,
    own_mask: torch.Tensor,
    context_coords_global: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    include_context: bool = False,
    sigma_table: torch.Tensor | None = None,
    own_element_ids: torch.Tensor | None = None,
    context_element_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Occupancy regression target = the masked residue's OWN side chain, splatted (faithful).

    Self-occupancy regresses the density of the masked residue's OWN side-chain atoms ONLY (its
    ``query_sampling.py`` builds ``occupancy = _build_occupancy_targets(query_pos, target_coords)``
    where ``target_coords`` are the masked residue's own atoms). Context atoms (neighbour backbone +
    target) are used as the head's *input* (they condition the prediction) and as the CONTEXT-region
    *label* that weights the loss -- points near a context atom regress to ~0 own-density, weighted
    like any other bucket -- but they are **never** added to the regressed density itself. Adding
    context to the target makes the head spend capacity re-predicting the context it already receives
    as input, blurring the small own-sidechain identity signal to chance at readout.

    Therefore the DEFAULT (``include_context=False``) is own-only:
    ``density = local_frame_splat(own_coords, own_mask, ...)``. The splat uses the head's exact
    transform (:func:`local_frame_splat`), so the target is directly comparable to
    ``vol_density_pred``. ``include_context=True`` reproduces the legacy own+context field and exists
    only for A/B comparison -- it is NOT the production objective.

    Parameters
    ----------
    query_local : torch.Tensor
        (Q, 3) fixed local-frame query lattice.
    R, ca : torch.Tensor
        (B, L, 3, 3), (B, L, 3) per-residue local frame + CA origin.
    sigma : float
        Gaussian width (Å).
    own_coords_global : torch.Tensor
        (B, L, K, 3) the residue's own GT side-chain atoms (global frame).
    own_mask : torch.Tensor
        (B, L, K) own real-atom mask.
    context_coords_global : torch.Tensor, optional
        (B, N, 3) shared context atoms (binder backbone + target), broadcast to all residues. Only
        splatted into the target when ``include_context=True`` (A/B); ignored by default.
    context_mask : torch.Tensor, optional
        (B, N) context validity mask. Only used when ``include_context=True``.
    include_context : bool, default False
        A/B toggle. False (default, faithful) => target is own side chain ONLY. True => the
        legacy own+context field (adds the context splat), kept for comparison only.
    sigma_table : torch.Tensor, optional
        ``(num_element_types,)`` per-element-id sigma table (:func:`build_element_sigma_table`). When given
        (with element ids) the own/context atoms splat with per-element sigma; ``None`` (default) => the
        scalar-``sigma`` path (byte-identical to the uniform-sigma target).
    own_element_ids : torch.Tensor, optional
        (B, L, K) own-side-chain per-atom MODEL element ids. Used only when ``sigma_table`` is given.
    context_element_ids : torch.Tensor, optional
        (B, N) shared-context per-atom MODEL element ids (broadcast to all residues). Used only when
        ``sigma_table`` is given AND ``include_context=True``.

    Returns
    -------
    torch.Tensor
        (B, L, Q) occupancy target (own-only by default; own+context when ``include_context=True``).
    """
    own_sigma = (
        per_atom_sigma_from_ids(own_element_ids, sigma_table)
        if sigma_table is not None and own_element_ids is not None
        else None
    )
    density = local_frame_splat(
        query_local, own_coords_global, own_mask.to(R.dtype), R, ca, sigma, per_atom_sigma=own_sigma
    )
    if include_context and context_coords_global is not None and context_mask is not None:
        length = R.shape[1]
        ctx = context_coords_global.unsqueeze(1).expand(-1, length, -1, -1)  # (B, L, N, 3)
        ctx_w = context_mask.to(R.dtype).unsqueeze(1).expand(-1, length, -1)  # (B, L, N)
        ctx_sigma = (
            per_atom_sigma_from_ids(context_element_ids, sigma_table).unsqueeze(1).expand(-1, length, -1)
            if sigma_table is not None and context_element_ids is not None
            else None
        )
        density = density + local_frame_splat(query_local, ctx, ctx_w, R, ca, sigma, per_atom_sigma=ctx_sigma)
    return density


def _nearest_distance(query_local: torch.Tensor, atoms_local: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Distance from each query point to the nearest valid atom (both in the same local frame).

    Parameters
    ----------
    query_local : torch.Tensor
        (Q, 3) query lattice.
    atoms_local : torch.Tensor
        (B, L, M, 3) atom coordinates in the residue's local frame.
    atom_mask : torch.Tensor
        (B, L, M) atom validity; invalid atoms are pushed to +inf so they never win the min.

    Returns
    -------
    torch.Tensor
        (B, L, Q) distance to the nearest valid atom (``+inf`` where a residue has no valid atom).
    """
    b, length, m = atoms_local.shape[:3]
    q = query_local.shape[0]
    if m == 0:
        return torch.full((b, length, q), float("inf"), device=atoms_local.device, dtype=atoms_local.dtype)
    ql = query_local.to(atoms_local.dtype).view(1, 1, q, 1, 3)
    d2 = ((ql - atoms_local.unsqueeze(2)) ** 2).sum(dim=-1)  # (B, L, Q, M)
    invalid = ~atom_mask.bool().unsqueeze(2)  # (B, L, 1, M)
    d2 = d2.masked_fill(invalid.expand_as(d2), float("inf"))
    return d2.min(dim=-1).values.sqrt()  # (B, L, Q)


def classify_query_regions(
    query_local: torch.Tensor,
    gt_sidechain_local: torch.Tensor,
    context_local: torch.Tensor,
    own_mask: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    thresholds: RegionThresholds | None = None,
) -> torch.Tensor:
    """Label each fixed lattice point with a self-occupancy region id (``REGION_*``).

    Precedence (highest first): ``center`` -> ``around`` -> ``context`` -> empty tiers. A point close
    to an own side-chain atom is ``center`` / ``around`` regardless of nearby context; a point close
    to a context atom (but not to an own atom) is ``context``; everything else is an empty tier keyed
    by its distance to the nearest atom of *any* kind (own or context), mirroring the reference's
    window-atom distance bands.

    Parameters
    ----------
    query_local : torch.Tensor
        (Q, 3) fixed local-frame query lattice.
    gt_sidechain_local : torch.Tensor
        (B, L, K, 3) the residue's own side-chain atoms in ITS local frame.
    context_local : torch.Tensor
        (B, L, N, 3) context atoms (neighbour backbone + target) in the residue's local frame.
    own_mask : torch.Tensor, optional
        (B, L, K) own real-atom mask. None => all valid.
    context_mask : torch.Tensor, optional
        (B, L, N) context validity. None => all valid.
    thresholds : RegionThresholds, optional
        Distance bands (Å). None => defaults.

    Returns
    -------
    torch.Tensor
        (B, L, Q) ``long`` region ids in ``[0, 5]``.
    """
    th = thresholds or RegionThresholds()
    b, length = gt_sidechain_local.shape[:2]
    q = query_local.shape[0]
    device = gt_sidechain_local.device

    if own_mask is None:
        own_mask = torch.ones(gt_sidechain_local.shape[:3], device=device, dtype=torch.bool)
    if context_mask is None:
        context_mask = torch.ones(context_local.shape[:3], device=device, dtype=torch.bool)

    d_own = _nearest_distance(query_local, gt_sidechain_local, own_mask)  # (B, L, Q)
    d_ctx = _nearest_distance(query_local, context_local, context_mask)  # (B, L, Q)
    d_any = torch.minimum(d_own, d_ctx)  # (B, L, Q)

    # Start everyone in far_empty, then refine downward by distance, then overlay the atom-close
    # buckets in ascending precedence so the tightest label wins.
    labels = torch.full((b, length, q), REGION_FAR_EMPTY, dtype=torch.long, device=device)
    labels = torch.where(d_any < th.medium_empty_max, torch.full_like(labels, REGION_MEDIUM_EMPTY), labels)
    labels = torch.where(d_any < th.near_empty_max, torch.full_like(labels, REGION_NEAR_EMPTY), labels)
    labels = torch.where(d_ctx <= th.context_radius, torch.full_like(labels, REGION_CONTEXT), labels)
    labels = torch.where(d_own <= th.around_radius, torch.full_like(labels, REGION_AROUND), labels)
    labels = torch.where(d_own <= th.center_radius, torch.full_like(labels, REGION_CENTER), labels)
    return labels


def region_occupancy_loss(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    region_labels: torch.Tensor,
    weights: OccupancyLossWeights | None = None,
    seq_mask: torch.Tensor | None = None,
    query_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-bucket, empty-up-weighted occupancy MSE (self-occupancy ``MultiTaskLoss`` density term).

    ``total = Σ_bucket weight[bucket] · mean_MSE_in_bucket``. Empty buckets carry ``weight > 1``
    (default 1.75), so a leak into empty space costs ``1.75×`` a same-size error on the side chain.
    Mirrors the reference's ``MultiTaskLoss``: MSE is meaned *within* a bucket, then weighted and summed
    across buckets (NOT a single global weighted mean).

    Parameters
    ----------
    pred_density, target_density : torch.Tensor
        (B, L, Q) predicted / target occupancy at the query points.
    region_labels : torch.Tensor
        (B, L, Q) region ids from :func:`classify_query_regions`.
    weights : OccupancyLossWeights, optional
        Per-bucket weights. None => self-occupancy defaults (empties 1.75).
    seq_mask : torch.Tensor, optional
        (B, L) valid-residue mask. Points on invalid residues are dropped. None => all valid.
    query_valid_mask : torch.Tensor, optional
        (B, L, Q) per-QUERY validity. For the atom-anchored sampler, masked (padded) own/context slots are
        excluded from the loss. None (default) => every query is valid (byte-identical to the fixed-lattice
        path, which has no per-query mask).

    Returns
    -------
    total : torch.Tensor
        Scalar weighted occupancy loss.
    components : dict[str, torch.Tensor]
        Per-bucket mean MSE (scalar tensors), keyed by ``REGION_NAMES`` (``0`` where a bucket is
        empty in this batch).
    """
    w = weights or OccupancyLossWeights()
    wvec = w.as_tensor(device=pred_density.device, dtype=pred_density.dtype)  # (6,)
    sq = (pred_density - target_density) ** 2  # (B, L, Q)

    if seq_mask is not None:
        valid = seq_mask.bool().unsqueeze(-1).expand_as(region_labels)  # (B, L, Q)
    else:
        valid = torch.ones_like(region_labels, dtype=torch.bool)
    if query_valid_mask is not None:
        valid = valid & query_valid_mask.bool()  # drop padded own/context slots (atom-anchored path)

    total = pred_density.new_zeros(())
    components: dict[str, torch.Tensor] = {}
    for region_id, name in enumerate(REGION_NAMES):
        bucket = (region_labels == region_id) & valid
        if bucket.any():
            bucket_mse = sq[bucket].mean()
        else:
            bucket_mse = pred_density.new_zeros(())
        components[name] = bucket_mse
        total = total + wvec[region_id] * bucket_mse
    return total, components


def _density_score(pred: torch.Tensor, ref: torch.Tensor, method: str) -> torch.Tensor:
    """Similarity between a predicted field and a reference field over the query axis.

    ``neg_mse`` -> ``-mean((pred - ref)²)`` (maximised, ``=0``, when they match); ``cosine`` ->
    cosine similarity over the query axis. Broadcasting is left to the caller.
    """
    if method == "neg_mse":
        return -((pred - ref) ** 2).mean(dim=-1)
    if method == "cosine":
        return F.cosine_similarity(pred, ref, dim=-1, eps=1e-8)
    raise ValueError(f"score method must be 'neg_mse' or 'cosine', got {method!r}")


def volumetric_decoy_classification_loss(
    pred_density: torch.Tensor,
    gt_ref_density: torch.Tensor,
    decoy_ref_densities: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    score: str = "neg_mse",
    gt_ref_rotamer_mask: torch.Tensor | None = None,
    decoy_ref_rotamer_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discriminative "on-top" loss: pred must score its GT reference density above all decoys.

    Cross-entropy over ``{GT + decoys}`` where the logit of each candidate is its temperature-scaled
    similarity (:func:`_density_score`) to the predicted field. The GT reference always sits at
    column 0, so this is minimised exactly when the predicted field is closest (in the chosen score)
    to the GT reference -- i.e. when ``pred == gt_ref_density`` the GT logit is maximal and the model
    picks GT. This complements the regression loss with a contrastive pressure to *distinguish* the
    correct occupancy shape from confusable alternatives.

    Per-rotamer references (optional). A discretization TYPE is generally MULTI-ROTAMER, and both the
    atom-disc path and the recovery gate reduce per rotamer then take the BEST rotamer of each type.
    When ``gt_ref_rotamer_mask`` / ``decoy_ref_rotamer_mask`` are supplied, each candidate carries ALL
    its rotamer densities -- ``gt_ref_density`` is ``(B, L, R, Q)`` and ``decoy_ref_densities`` is
    ``(B, L, D, R, Q)`` -- and each candidate's logit is its BEST (max-similarity) rotamer, mirroring
    that reduction rather than collapsing a type to an arbitrary first rotamer. Padded rotamer slots
    (mask ``False``) are excluded from the max. Without the masks the historical single-reference-per-type
    API applies (``(B, L, Q)`` GT / ``(B, L, D, Q)`` decoys).

    Parameters
    ----------
    pred_density : torch.Tensor
        (B, L, Q) predicted occupancy field.
    gt_ref_density : torch.Tensor
        (B, L, Q) -- or (B, L, R, Q) with ``gt_ref_rotamer_mask`` -- reference density of the GT type.
    decoy_ref_densities : torch.Tensor
        (B, L, D, Q) -- or (B, L, D, R, Q) with ``decoy_ref_rotamer_mask`` -- decoy reference densities.
    seq_mask : torch.Tensor, optional
        (B, L) valid-residue mask. None => all valid.
    temperature : float
        Softmax temperature applied to the scores (smaller => sharper).
    score : str
        ``"neg_mse"`` (default) or ``"cosine"``.
    gt_ref_rotamer_mask : torch.Tensor, optional
        (B, L, R) True where a GT rotamer slot is real. Enables the max-over-rotamer GT reduction.
    decoy_ref_rotamer_mask : torch.Tensor, optional
        (B, L, D, R) True where a decoy rotamer slot is real. Enables the max-over-rotamer decoy reduction.

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy (0 when the batch has no valid positions).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0 (got {temperature})")
    if gt_ref_rotamer_mask is not None:
        # gt_ref_density: (B, L, R, Q); pred (B, L, Q) -> (B, L, 1, Q) broadcasts over the rotamer axis.
        gt_rot_score = _density_score(pred_density.unsqueeze(2), gt_ref_density, score)  # (B, L, R)
        gt_score = gt_rot_score.masked_fill(~gt_ref_rotamer_mask.bool(), float("-inf")).amax(dim=-1)  # (B, L)
    else:
        gt_score = _density_score(pred_density, gt_ref_density, score)  # (B, L)
    if decoy_ref_rotamer_mask is not None:
        # decoy_ref_densities: (B, L, D, R, Q); pred -> (B, L, 1, 1, Q) broadcasts over decoys + rotamers.
        decoy_rot_score = _density_score(
            pred_density.unsqueeze(2).unsqueeze(3), decoy_ref_densities, score
        )  # (B, L, D, R)
        decoy_score = decoy_rot_score.masked_fill(~decoy_ref_rotamer_mask.bool(), float("-inf")).amax(dim=-1)
    else:
        decoy_score = _density_score(pred_density.unsqueeze(2), decoy_ref_densities, score)  # (B, L, D)
    scores = torch.cat([gt_score.unsqueeze(-1), decoy_score], dim=-1)  # (B, L, 1+D)
    # Scale-invariance (root-cause fix for the dead-flat vol_decoy_ce). Raw neg_mse (or cosine) gaps
    # between the GT reference and the decoys are tiny -- per-position candidate std is ~1e-2 for the
    # weak early density field -- so a temperature=1.0 softmax is ~uniform and the CE pins at ln(1+D)
    # (= ln 20 for n_decoys=19) with a vanishing gradient to ``pred_density``; the head never learns
    # the contrast. Z-scoring each position's candidate scores over the candidate axis (subtract mean,
    # divide by std) makes the logit spread O(1) REGARDLESS of the absolute neg_mse magnitude, so the
    # softmax is non-uniform and a real gradient flows. This is gradient-preserving: the mean/std reduce
    # over the CANDIDATE axis only, and ``pred_density`` (which enters every candidate score) keeps its
    # grad. ``unbiased=False`` + eps keeps it finite for the degenerate n_decoys=0 / equal-score cases.
    # Applied on the merged candidate scores => covers BOTH the rotamer-mask and simple branches, and
    # both the FT and pretrain-only call sites (they share this fn). ``temperature`` is now a pure
    # post-standardization sharpness knob (smaller => sharper) rather than an absolute-scale knob.
    scores = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True, unbiased=False) + 1e-6)
    logits = scores / float(temperature)  # (B, L, 1+D)

    b, length, n_cls = logits.shape
    logits_flat = logits.reshape(b * length, n_cls)
    target = torch.zeros(b * length, dtype=torch.long, device=logits.device)  # GT at column 0
    if seq_mask is not None:
        target = target.masked_fill(~seq_mask.reshape(-1).bool(), -100)
    if (target != -100).sum() == 0:
        return pred_density.new_zeros(())
    return F.cross_entropy(logits_flat, target, ignore_index=-100)


class StratifiedDecoySampler:
    """Cluster-stratified decoy type sampler (training-time InfoNCE supervision for the volumetric head).

    Ports the atom-discretization decoy scheme used by the InfoNCE stratified path: for each masked
    position it returns ``1 GT + n_decoys`` candidate residue *types*, drawn so the contrast is

    * **canonical-fallback** -- when the GT is canonical and a coin flip (``canonical_fallback_prob``)
      lands, the decoys are OTHER canonical types (a pure canonical-vs-canonical contrast);
    * **cluster-stratified otherwise** -- ``n_in`` in-cluster decoys (same Qupid/ECFP4 cluster as GT)
      + ``n_out`` out-of-cluster decoys spread across the other clusters, where ``n_in`` **ramps**
      ``in_cluster_start -> in_cluster_end`` over the first ``ramp_end_frac`` of training (call
      :meth:`set_epoch_progress` each epoch);
    * **padding** -- any shortfall is padded from the valid clustered universe.

    Returns TYPE indices (into whatever per-type density bank the trainer maintains), with the GT at
    column 0. Simplifications: representability / holdout guards collapse to "GT must
    be clustered", and there is no secondary (alt) clustering coin -- one clustering view.

    Parameters
    ----------
    cluster_id_per_type : torch.Tensor
        (n_types,) cluster id per residue type; ``< 0`` => unclustered (never sampled as a decoy).
    is_canonical_per_type : torch.Tensor, optional
        (n_types,) bool; True for the 20 canonical amino acids.
    n_decoys : int
        Number of decoys per position (subset width = ``n_decoys + 1``).
    canonical_fallback_prob : float
        P(take the all-canonical contrast) when GT is canonical.
    in_cluster_start, in_cluster_end : int
        In-cluster decoy count at the start / end of the ramp.
    ramp_end_frac : float
        Fraction of training over which ``n_in`` ramps to ``in_cluster_end``.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        cluster_id_per_type: torch.Tensor,
        is_canonical_per_type: torch.Tensor | None = None,
        n_decoys: int = 19,
        canonical_fallback_prob: float = 0.0,
        in_cluster_start: int = 2,
        in_cluster_end: int = 8,
        ramp_end_frac: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.cluster_id = cluster_id_per_type.long().cpu()
        self.n_types = int(self.cluster_id.shape[0])
        self.is_canonical = (
            is_canonical_per_type.bool().cpu()
            if is_canonical_per_type is not None
            else torch.zeros(self.n_types, dtype=torch.bool)
        )
        self.n_decoys = int(n_decoys)
        self.canonical_fallback_prob = float(canonical_fallback_prob)
        self.in_cluster_start = int(in_cluster_start)
        self.in_cluster_end = int(in_cluster_end)
        self.ramp_end_frac = float(ramp_end_frac)
        self._progress = 0.0
        self._g = torch.Generator()
        if seed is not None:
            self._g.manual_seed(int(seed))

        # Per-cluster type pools (clustered types only) + canonical index list.
        self.cluster_to_indices: dict[int, torch.Tensor] = {}
        for cid in sorted(set(self.cluster_id.tolist())):
            if cid < 0:
                continue
            self.cluster_to_indices[int(cid)] = torch.where(self.cluster_id == cid)[0].long()
        self.n_clusters = len(self.cluster_to_indices)
        self.canonical_indices = torch.where(self.is_canonical)[0].long()
        self.valid_universe = torch.where(self.cluster_id >= 0)[0].long()

    def set_epoch_progress(self, progress: float) -> None:
        """Set training progress in ``[0, 1]`` (drives the in-cluster ramp)."""
        self._progress = float(max(0.0, min(1.0, progress)))

    def _current_in_cluster_count(self) -> int:
        """Ramp ``in_cluster_start -> in_cluster_end`` over ``[0, ramp_end_frac]``."""
        if self.ramp_end_frac <= 0.0:
            frac = 1.0
        else:
            frac = min(1.0, self._progress / self.ramp_end_frac)
        n_in = self.in_cluster_start + frac * (self.in_cluster_end - self.in_cluster_start)
        return int(max(0, min(self.n_decoys, round(n_in))))

    def _randperm(self, n: int) -> torch.Tensor:
        return torch.randperm(n, generator=self._g)

    def _pad_from_universe(self, chosen: set[int], n_needed: int) -> list[int]:
        """Draw ``n_needed`` extra type indices from the valid universe, excluding ``chosen``."""
        if n_needed <= 0:
            return []
        avail = [int(x) for x in self.valid_universe.tolist() if int(x) not in chosen]
        if not avail:
            return []
        avail_t = torch.tensor(avail, dtype=torch.long)
        pick = avail_t[self._randperm(avail_t.shape[0])[:n_needed]]
        return pick.tolist()

    def sample(
        self,
        targets_flat: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``(n_pos, n_decoys+1)`` candidate TYPE indices; GT at column 0.

        Parameters
        ----------
        targets_flat : torch.Tensor
            (n_pos,) GT residue-type index per position.
        valid_mask : torch.Tensor, optional
            (n_pos,) bool; invalid positions get an arbitrary filled row + ``gt_pos = -100``.

        Returns
        -------
        subset_idx : torch.Tensor
            (n_pos, n_decoys+1) type indices; column 0 is the GT.
        gt_subset_pos : torch.Tensor
            (n_pos,) column holding the GT (always 0 here; ``-100`` for invalid / unclustered GT).
        """
        targets = targets_flat.long().cpu()
        n_pos = targets.shape[0]
        width = self.n_decoys + 1
        subset = torch.zeros((n_pos, width), dtype=torch.long)
        gt_pos = torch.zeros((n_pos,), dtype=torch.long)

        n_in_target = self._current_in_cluster_count()
        n_out_target = self.n_decoys - n_in_target
        vmask = valid_mask.bool().cpu() if valid_mask is not None else None

        for i in range(n_pos):
            if vmask is not None and not bool(vmask[i]):
                subset[i] = torch.arange(width)
                gt_pos[i] = -100
                continue
            gt_idx = int(targets[i])
            gt_cluster = int(self.cluster_id[gt_idx]) if 0 <= gt_idx < self.n_types else -1
            gt_is_canon = bool(self.is_canonical[gt_idx]) if 0 <= gt_idx < self.n_types else False

            # Canonical-fallback contrast.
            if (
                gt_is_canon
                and self.canonical_fallback_prob > 0.0
                and self.canonical_indices.numel() > 0
                and float(torch.rand((), generator=self._g)) < self.canonical_fallback_prob
            ):
                others = self.canonical_indices[self.canonical_indices != gt_idx]
                k = min(others.shape[0], self.n_decoys)
                chosen_types = [gt_idx]
                if k > 0:
                    chosen_types += others[self._randperm(others.shape[0])[:k]].tolist()
                chosen_types += self._pad_from_universe(set(chosen_types), width - len(chosen_types))
                chosen_types = (chosen_types + [gt_idx] * width)[:width]
                subset[i] = torch.tensor(chosen_types, dtype=torch.long)
                gt_pos[i] = 0
                continue

            # Unclustered GT: no valid in/out contrast -> ignore this position.
            if gt_cluster < 0:
                subset[i] = torch.arange(width)
                gt_pos[i] = -100
                continue

            # Stratified: GT + in-cluster + out-of-cluster.
            chosen_types = [gt_idx]
            in_pool = self.cluster_to_indices.get(gt_cluster, torch.empty(0, dtype=torch.long))
            in_pool = in_pool[in_pool != gt_idx]
            if in_pool.numel() > 0 and n_in_target > 0:
                take = min(int(in_pool.shape[0]), n_in_target)
                chosen_types += in_pool[self._randperm(in_pool.shape[0])[:take]].tolist()

            out_ids = [c for c in self.cluster_to_indices if c != gt_cluster]
            if out_ids and n_out_target > 0:
                per = max(1, n_out_target // len(out_ids))
                for cid in out_ids:
                    if len(chosen_types) >= width:
                        break
                    pool = self.cluster_to_indices[cid]
                    take = min(int(pool.shape[0]), per, width - len(chosen_types))
                    if take > 0:
                        chosen_types += pool[self._randperm(pool.shape[0])[:take]].tolist()

            chosen_types += self._pad_from_universe(set(chosen_types), width - len(chosen_types))
            chosen_types = (chosen_types + [gt_idx] * width)[:width]
            subset[i] = torch.tensor(chosen_types, dtype=torch.long)
            gt_pos[i] = 0

        return subset, gt_pos
