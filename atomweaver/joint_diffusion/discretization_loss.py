"""Geometric read-out (discretization) loss for AtomWeaver residue identity.

Extracted verbatim from the training-only loss code for the public inference
package: this is the strict distance-matrix / US-align residue-cloud matcher used at
read time to score a predicted side-chain cloud against the reference residue library
(``build_eval_discretizer`` in ``sample.py``). Only ``DiscretizationLoss``
is needed on the inference path; the training LightningModule and its helpers are not
shipped. Behaviour is byte-identical to the original class.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812


class DiscretizationLoss(nn.Module):
    """
    Unified discretization loss supporting multiple methods and backbone inclusion.

    This combines the best features of the distance matrix approaches with optional
    backbone atom inclusion for better structure matching.

    Parameters
    ----------
    residue_database : torch.Tensor
        Reference residue coordinates of shape (num_residues, max_atoms, 3).
    residue_masks : torch.Tensor
        Masks for reference residues of shape (num_residues, max_atoms).
    backbone_indices : torch.Tensor, optional
        Indices of backbone atoms (N, CA, C, O) in each reference residue.
        Shape: (num_residues, 4). Value of -1 means atom not found.
    method : Literal["normalized"]
        Comparison method: normalized distance-matrix correlation (NDM).
    temperature : float
        Temperature for softmax over similarity scores (lower = sharper).
    include_backbone : bool
        Whether to include backbone atoms when comparing structures.
        Only works if backbone_indices is provided.
    use_backbone_for_alignment_only : bool
        If True, backbone atoms are used for frame alignment but the distance
        matrix loss is computed on sidechain atoms only, making the loss
        rotamer-invariant (doesn't penalize chi1 rotation around CA-CB bond).
    n_iters : int
        Number of alignment iterations (retained for signature compatibility).
    combined_weight : float
        Retained for signature compatibility (unused by the NDM path).
    """

    def __init__(
        self,
        residue_database: torch.Tensor,
        residue_masks: torch.Tensor,
        backbone_indices: torch.Tensor | None = None,
        element_types: torch.Tensor | None = None,
        rotamer_to_type: torch.Tensor | None = None,
        method: Literal["normalized"] = "normalized",
        temperature: float = 1.0,
        include_backbone: bool = True,
        use_backbone_for_alignment_only: bool = True,
        n_iters: int = 2,
        combined_weight: float = 0.5,
        atom_mismatch_penalty: float = 0.0,
        element_mismatch_penalty: float = 0.0,
        # Chirality-mismatch penalty (2026-08 spec). The NDM correlation is computed on a
        # distance matrix, which is MIRROR-INVARIANT: an L-handed predicted cloud correlates
        # identically with an L reference and its D-isomer, so a D decoy can out-rank the correct
        # L for free. When > 0 AND backbone_coords are supplied at forward, candidates whose
        # handedness disagrees with the predicted cloud's handedness are penalized (finite,
        # confidence-weighted). Default 0.0 => byte-identical to the pre-chirality behavior.
        chirality_mismatch_penalty: float = 0.0,
        shortlist_k: int = 0,
        # InfoNCE-style stratified decoy sampling (2026-05-27 spec).
        # When decoy_sampling="stratified", per-position CE is taken over a subset
        # of (1 GT + n_decoys) classes instead of the full N-way softmax. Decoys
        # are sampled stratified by cluster (Qupid/ECFP4) with a curriculum that
        # ramps in-cluster count from start->end over the first ramp_end_frac of
        # training. When GT is canonical and rand()<canonical_fallback_prob, we
        # instead restrict the softmax to the 20 canonicals (matches the canonical setup).
        # Optional secondary clustering (cluster_id_per_type_alt) -- when supplied
        # together with alt_sampling_prob>0, each stratified call independently
        # picks alt vs primary at the per-batch coin flip (hedges a single
        # clustering being a poor proxy for chemistry/geometry).
        decoy_sampling: Literal["full", "stratified"] = "full",
        cluster_id_per_type: torch.Tensor | None = None,
        is_canonical_per_type: torch.Tensor | None = None,
        n_decoys: int = 19,
        canonical_fallback_prob: float = 0.0,
        in_cluster_start: int = 2,
        in_cluster_end: int = 8,
        in_cluster_std: float = 0.0,
        ramp_end_frac: float = 0.5,
        cluster_id_per_type_alt: torch.Tensor | None = None,
        alt_sampling_prob: float = 0.0,
        model_max_sidechain_atoms: int | None = None,
        # Slot off-by-one fix (2026-09). The reference is built dense-packed (backbone 0-3,
        # sidechain contiguous from slot 4 -- CB at sidechain-index 0), but predictions arrive
        # in reserved_slot0 layout (slot 0 = ghost/masked, CB at slot 1). Because the NDM /
        # combined distance-matrix correlation is POSITIONAL (cell [i,j] vs reference [i,j]),
        # every sidechain atom was compared one slot off. When True (default), the predicted
        # sidechain is dense-packed (masked-TRUE atoms compacted to contiguous indices from 0,
        # order preserved) BEFORE the backbone prepend / distance-matrix comparison, matching
        # the reference layout. It is a no-op on an already-dense prediction (stable sort).
        # Env override ATOMWEAVER_DISC_REPACK={0,1} wins over the constructor value for ablation.
        repack_prediction: bool = True,
    ):
        super().__init__()

        if decoy_sampling not in {"full", "stratified"}:
            raise ValueError(
                f"decoy_sampling must be 'full' or 'stratified', got {decoy_sampling!r}. "
                "Guards against a silent full-N-softmax fallback from a typo'd --disc-decoy-sampling "
                "(e.g. 'stratfied'), which would otherwise construct fine and ignore the cluster pools."
            )

        self.method = method
        self.temperature = temperature
        self.include_backbone = include_backbone and backbone_indices is not None
        self.use_backbone_for_alignment_only = use_backbone_for_alignment_only
        self.n_iters = n_iters
        self.combined_weight = combined_weight
        self.atom_mismatch_penalty = atom_mismatch_penalty
        self.element_mismatch_penalty = element_mismatch_penalty
        self.chirality_mismatch_penalty = float(chirality_mismatch_penalty)
        self.shortlist_k = shortlist_k

        # Dense-pack the predicted sidechain before the (positional) distance-matrix comparison so
        # its layout matches the dense-packed reference. Default-ON bug fix; env override for ablation.
        _repack_env = os.environ.get("ATOMWEAVER_DISC_REPACK")
        if _repack_env is not None:
            self.repack_prediction = _repack_env.strip() not in {"0", "false", "False", ""}
        else:
            self.repack_prediction = bool(repack_prediction)

        # InfoNCE state ─────────────────────────────────────────────────────
        self.decoy_sampling = decoy_sampling
        self.n_decoys = int(n_decoys)
        self.canonical_fallback_prob = float(canonical_fallback_prob)
        self.in_cluster_start = int(in_cluster_start)
        self.in_cluster_end = int(in_cluster_end)
        self.in_cluster_std = float(in_cluster_std)
        self.ramp_end_frac = float(ramp_end_frac)
        # Shared-memory float buffer so DataLoader-fork workers see updates from
        # main process's set_epoch_progress (mirrors PeptideDataset's K-ramp pattern).
        self._epoch_progress = torch.zeros((), dtype=torch.float32)
        if decoy_sampling == "stratified":
            self._epoch_progress.share_memory_()
            if cluster_id_per_type is None or is_canonical_per_type is None:
                raise ValueError(
                    "decoy_sampling='stratified' requires cluster_id_per_type and "
                    "is_canonical_per_type tensors (built from DB metadata + cluster CSV "
                    "in the LightningModule)."
                )
        # ── max_sc-aware representability filter ────────────────────────────
        # A K-slot model structurally cannot emit a sidechain with >K heavy atoms, so any
        # reference whose sidechain exceeds the model's slot budget is meaningless both as a
        # DECOY (unreachable) and as a snap/GT TARGET. Build a per-TYPE boolean mask; a budget
        # of None (or >= the DB's widest sidechain) makes this all-True -- an exact no-op for the
        # full-width 16-slot model, and active only for the trimmed 14-slot model (ONE knob,
        # correct for both). Filtering the candidate POOLS here (upstream of _sample_decoy_subset)
        # means a pool shrunk below the needed decoy count degrades through the SAME existing
        # "not enough in-cluster -> draw more out-of-cluster -> pad from valid_universe" fallback as
        # any naturally-small cluster; no new special-case path is introduced.
        self.model_max_sidechain_atoms = model_max_sidechain_atoms
        _num_types_early = (
            int(rotamer_to_type.max().item()) + 1 if rotamer_to_type is not None else residue_database.shape[0]
        )
        if model_max_sidechain_atoms is not None:
            _per_rot_total = residue_masks.sum(dim=-1).long()  # backbone + sidechain atoms present
            if backbone_indices is not None:
                _n_bb = (backbone_indices >= 0).sum(dim=-1).long()
            else:
                _n_bb = torch.zeros_like(_per_rot_total)
            _per_rot_sc = (_per_rot_total - _n_bb).clamp(min=0)  # sidechain-only atom count per rotamer
            if rotamer_to_type is not None:
                # A type is representable only if EVERY rotamer fits -> reduce by max sc-count over rotamers.
                _type_sc = torch.zeros(_num_types_early, dtype=torch.long)
                _type_sc.scatter_reduce_(0, rotamer_to_type.long(), _per_rot_sc, reduce="amax", include_self=True)
            else:
                _type_sc = _per_rot_sc.long()
            _representable = _type_sc <= int(model_max_sidechain_atoms)
        else:
            _representable = torch.ones(_num_types_early, dtype=torch.bool)
        self.register_buffer("_type_representable", _representable)
        # ────────────────────────────────────────────────────────────────────

        if cluster_id_per_type is not None:
            self.register_buffer("_cluster_id_per_type", cluster_id_per_type.long())
            # Per-cluster pools: dict[int, LongTensor] of type indices (CPU; sampling
            # happens once per batch, then indices are moved to GPU).
            self._cluster_to_indices: dict[int, torch.Tensor] = {}
            # Size-safe: on a length mismatch fall back to all-True so the explicit per-type shape
            # guard below fires the clean ValueError instead of a cryptic broadcast error here.
            _rep_prim = (
                _representable.to(cluster_id_per_type.device)
                if _representable.shape[0] == cluster_id_per_type.shape[0]
                else torch.ones_like(cluster_id_per_type, dtype=torch.bool)
            )
            for cid in sorted(set(cluster_id_per_type.cpu().tolist())):
                if cid < 0:
                    continue  # unclustered (e.g. holdout NCAAs): skipped in sampling pools
                idx = torch.where((cluster_id_per_type == cid) & _rep_prim)[0]
                if idx.numel() == 0:
                    continue  # cluster emptied by the max_sc filter: drop it (as any absent cluster)
                self._cluster_to_indices[int(cid)] = idx.long()
            self._n_clusters = len(self._cluster_to_indices)
        else:
            self._cluster_id_per_type = None
            self._cluster_to_indices = {}
            self._n_clusters = 0
        # Secondary clustering (e.g. ECFP4 alongside Qupid): mirrors the primary
        # buffer + pool dict + n_clusters triple so the per-batch dispatcher can
        # swap them as a unit. Validated against alt_sampling_prob>0.
        self.alt_sampling_prob = float(alt_sampling_prob)
        if not 0.0 <= self.alt_sampling_prob <= 1.0:
            raise ValueError(f"alt_sampling_prob must be in [0, 1] (got {alt_sampling_prob}).")
        if cluster_id_per_type_alt is not None:
            self.register_buffer("_cluster_id_per_type_alt", cluster_id_per_type_alt.long())
            self._cluster_to_indices_alt: dict[int, torch.Tensor] = {}
            _rep_alt = (
                _representable.to(cluster_id_per_type_alt.device)
                if _representable.shape[0] == cluster_id_per_type_alt.shape[0]
                else torch.ones_like(cluster_id_per_type_alt, dtype=torch.bool)
            )
            for cid in sorted(set(cluster_id_per_type_alt.cpu().tolist())):
                if cid < 0:
                    continue
                idx = torch.where((cluster_id_per_type_alt == cid) & _rep_alt)[0]
                if idx.numel() == 0:
                    continue  # cluster emptied by the max_sc filter
                self._cluster_to_indices_alt[int(cid)] = idx.long()
            self._n_clusters_alt = len(self._cluster_to_indices_alt)
        else:
            self._cluster_id_per_type_alt = None
            self._cluster_to_indices_alt = {}
            self._n_clusters_alt = 0
        if self.alt_sampling_prob > 0.0 and cluster_id_per_type_alt is None:
            raise ValueError("alt_sampling_prob > 0 requires cluster_id_per_type_alt to be supplied.")
        if is_canonical_per_type is not None:
            self.register_buffer("_is_canonical", is_canonical_per_type.bool())
            # Canonicals all have ≤10 sidechain atoms, so the max_sc filter is a no-op here for any
            # sane budget; AND it in anyway for correctness/consistency with the other pools.
            _rep_canon = (
                _representable.to(is_canonical_per_type.device)
                if _representable.shape[0] == is_canonical_per_type.shape[0]
                else torch.ones_like(is_canonical_per_type, dtype=torch.bool)
            )
            self.register_buffer("_canonical_indices", torch.where(is_canonical_per_type.bool() & _rep_canon)[0].long())
        else:
            self._is_canonical = None
            self._canonical_indices = None
        # ───────────────────────────────────────────────────────────────────

        self.register_buffer("_residue_database", residue_database)
        self.register_buffer("_residue_masks", residue_masks)

        # Store rotamer to amino acid type mapping for multi-rotamer databases
        # When provided, logits are aggregated by type for loss and accuracy
        if rotamer_to_type is not None:
            self.register_buffer("_rotamer_to_type", rotamer_to_type)
            self._num_types = int(rotamer_to_type.max().item()) + 1
        else:
            self._rotamer_to_type = None
            self._num_types = residue_database.shape[0]

        # Fail-loud shape guard: cluster/canonical tensors are per-TYPE and MUST match _num_types.
        # With separate 319 (train) and 325 (eval) DBs, a wiring slip would otherwise crash inside
        # gather or silently sample out-of-range type indices.
        for _name, _t in (
            ("cluster_id_per_type", self._cluster_id_per_type),
            ("cluster_id_per_type_alt", self._cluster_id_per_type_alt),
            ("is_canonical_per_type", self._is_canonical),
        ):
            if _t is not None and _t.shape[0] != self._num_types:
                raise ValueError(
                    f"{_name} has {_t.shape[0]} entries but the residue database has {self._num_types} "
                    f"types -- these must match (per-type tensors). Check the cluster CSV was loaded "
                    f"against the SAME (319-way train) DB passed as residue_database."
                )

        if backbone_indices is not None:
            self.register_buffer("_backbone_indices", backbone_indices)
        else:
            self._backbone_indices = None

        # Per-reference handedness for the chirality-mismatch penalty, shape (n_ROTAMER_ROWS,) in
        # {-1, 0, +1} where +1 = L, -1 = D, 0 = undeterminable (never penalized).
        # INDEXED BY ROTAMER, NOT BY TYPE. The database has one row per reference conformer
        # (~6487 on the production library) against ~691 residue types, so a type-indexed read
        # returns an unrelated residue's handedness with no error of any kind. Read it through
        # `chirality_signs_of_type`, which cannot be called without the rotamer->type map. Derived
        # GEOMETRICALLY here (the CCD codes / db dict aren't reachable at this construction site --
        # only the coord/mask/backbone-index tensors are) by delegating to
        # data_utils._chirality_from_reference per row: e3 = normalize((N-CA) x (C-CA)) and the sign
        # of e3 . (Cbeta - CA), where Cbeta is the SINGLE sidechain atom bonded to CA. Reference and
        # prediction use the identical single-Cbeta rule so a true L match shares the prediction's
        # sign (no penalty) while its D mirror flips sign (penalized). The Cbeta sign is rotamer-
        # invariant; the old mean-over-all-sidechain-atoms rule was NOT and rendered flexible chiral
        # residues (DAR among them) undeterminable across their rotamers. Computed unconditionally
        # (cheap, one-time) so the buffer is always present.
        # persistent=False: a pure function of construction args, always rebuilt -- keeping it out of
        # state_dict means older checkpoints (which never had this buffer) still load under strict=True.
        self.register_buffer(
            "_ref_chirality",
            self._build_ref_chirality(residue_database, residue_masks, backbone_indices),
            persistent=False,
        )

        if element_types is not None:
            self.register_buffer("_element_types", element_types)
            # Precompute element count histograms for each reference residue
            # Element types: 0=C, 1=N, 2=O, 3=S
            self._precompute_element_histograms(element_types, residue_masks, backbone_indices)
        else:
            self._element_types = None
            self._ref_element_histograms = None

        # Precompute reference data (NDM / normalized distance-matrix method only).
        self._init_normalized_method(residue_database, residue_masks)

    # ── InfoNCE-style stratified decoy sampling (2026-05-27 spec) ──────
    def set_epoch_progress(self, frac: float) -> None:
        """Update epoch_progress in [0, 1] for the in-cluster ramp.

        Called once per epoch from the LightningModule's on_train_epoch_start.
        Frac = current_epoch / max_epochs. Shared-memory float so forked
        DataLoader workers (if any read this) observe updates from rank 0.
        Has no effect unless decoy_sampling='stratified'.
        """
        self._epoch_progress.fill_(max(0.0, min(1.0, float(frac))))

    def _current_in_cluster_count(self) -> int:
        """In-cluster decoy count for the current epoch.

        Mean μ follows the linear ramp in_cluster_start -> in_cluster_end over
        [0, ramp_end_frac]. When ``in_cluster_std > 0`` the returned count is a
        per-call draw from Normal(μ, in_cluster_std), rounded and clamped to
        [0, n_decoys] (jitters the in/out-cluster split each batch so the model
        doesn't memorise a fixed decoy difficulty). std == 0 reproduces the
        deterministic rounded ramp exactly (backward-compatible default). Uses the
        torch global RNG so seeding is honored.
        """
        p = float(self._epoch_progress.item())
        ramp = 1.0 if self.ramp_end_frac <= 0.0 else min(p / self.ramp_end_frac, 1.0)
        mu = self.in_cluster_start + ramp * (self.in_cluster_end - self.in_cluster_start)
        if self.in_cluster_std > 0.0:
            mu = float(torch.normal(mean=torch.tensor(mu), std=torch.tensor(self.in_cluster_std)).item())
        return max(0, min(int(round(mu)), self.n_decoys))

    def _sample_decoy_subset(
        self,
        targets_flat: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample subset of (1 GT + n_decoys) candidate types per position.

        Returns
        -------
        subset_idx : torch.Tensor
            (n_pos, n_decoys+1) column indices into the (n_pos, n_types) full-logits matrix.
            Column 0 is the GT for the stratified path; for canonical-fallback the GT lives at
            its canonical-list position (varies by GT identity).
        gt_subset_pos : torch.Tensor
            (n_pos,) the column index (0..n_decoys) where the GT lives.
        """
        device = targets_flat.device
        n_pos = targets_flat.shape[0]
        n_decoys = self.n_decoys
        # CPU-side per-position branching; move to GPU at the end (B*L ~128, low overhead).
        targets_cpu = targets_flat.cpu()
        subset_idx_cpu = torch.zeros((n_pos, n_decoys + 1), dtype=torch.long)
        gt_subset_pos_cpu = torch.zeros((n_pos,), dtype=torch.long)

        # Per-batch clustering choice: primary (e.g. Qupid) vs alt (e.g. ECFP4). One coin flip
        # per call so all positions in this batch use the same clustering view.
        use_alt = (
            self._cluster_id_per_type_alt is not None
            and self.alt_sampling_prob > 0.0
            and bool(torch.rand(()).item() < self.alt_sampling_prob)
        )
        if use_alt:
            active_cluster_id = self._cluster_id_per_type_alt
            active_cluster_pools = self._cluster_to_indices_alt
            active_n_clusters = self._n_clusters_alt
        else:
            active_cluster_id = self._cluster_id_per_type
            active_cluster_pools = self._cluster_to_indices
            active_n_clusters = self._n_clusters

        n_in_target = self._current_in_cluster_count()
        n_out_target = n_decoys - n_in_target
        n_other_clusters = max(active_n_clusters - 1, 1)
        per_other_floor = n_out_target // n_other_clusters
        remainder = n_out_target - per_other_floor * n_other_clusters

        is_canonical_cpu = self._is_canonical.cpu() if self._is_canonical is not None else None
        canonical_indices_cpu = self._canonical_indices.cpu() if self._canonical_indices is not None else None
        cluster_id_cpu = active_cluster_id.cpu() if active_cluster_id is not None else None
        cluster_pools_cpu = {cid: pool.cpu() for cid, pool in active_cluster_pools.items()}
        valid_mask_cpu = valid_mask.cpu() if valid_mask is not None else None
        # Valid decoy universe = types that ARE clustered (cluster_id >= 0) AND representable by the
        # model's slot budget. ALL random padding (canonical-fallback + undersampled stratified) draws
        # ONLY from here, so cluster_id<0 holdouts (e.g. the NCAAs) and >max_sc references can never
        # re-enter the decoy pool via padding.
        representable_cpu = self._type_representable.cpu()
        valid_universe_cpu = (
            torch.where((cluster_id_cpu >= 0) & representable_cpu)[0].long()
            if cluster_id_cpu is not None
            else torch.where(representable_cpu)[0].long()
        )

        for i in range(n_pos):
            if valid_mask_cpu is not None and not bool(valid_mask_cpu[i].item()):
                # Invalid position: fill with anything; loss masks it out.
                subset_idx_cpu[i] = torch.arange(n_decoys + 1, dtype=torch.long)
                gt_subset_pos_cpu[i] = 0
                continue
            gt_idx = int(targets_cpu[i].item())
            # GT itself exceeds the model's slot budget (a rare >max_sc reference -- e.g. a 15/16-atom
            # NCAA that passed the DB-presence seq_mask): the model structurally cannot represent it, so
            # scoring it would train a meaningless gradient. Skip via ignore_index (-100), the SAME guard
            # used below for cluster_id<0 holdouts. No-op for the 16-slot model (all types representable).
            if not bool(representable_cpu[gt_idx].item()):
                subset_idx_cpu[i] = torch.arange(n_decoys + 1, dtype=torch.long)
                gt_subset_pos_cpu[i] = -100
                continue
            gt_is_canon = bool(is_canonical_cpu[gt_idx].item()) if is_canonical_cpu is not None else False
            gt_cluster = int(cluster_id_cpu[gt_idx].item()) if cluster_id_cpu is not None else -1

            # Canonical-fallback path: only when GT is canonical AND coin flip lands.
            if (
                gt_is_canon
                and self.canonical_fallback_prob > 0.0
                and canonical_indices_cpu is not None
                and torch.rand(()).item() < self.canonical_fallback_prob
            ):
                # GT ALWAYS at column 0 (gt_subset_pos=0 can never be wrong even if n_decoys+1 < #canon);
                # fill the rest with OTHER canonicals (excl GT), then pad from valid_universe (excl chosen).
                subset_idx_cpu[i, 0] = gt_idx
                others = canonical_indices_cpu[canonical_indices_cpu != gt_idx]
                k = min(others.shape[0], n_decoys)
                if k > 0:
                    perm = torch.randperm(others.shape[0])[:k]
                    subset_idx_cpu[i, 1 : 1 + k] = others[perm]
                filled = 1 + k
                if filled < n_decoys + 1:
                    chosen = set(subset_idx_cpu[i, :filled].tolist())
                    pad_avail = valid_universe_cpu[
                        torch.tensor([int(x) not in chosen for x in valid_universe_cpu.tolist()], dtype=torch.bool)
                    ]
                    if pad_avail.shape[0] > 0:
                        pp = torch.randperm(pad_avail.shape[0])[: (n_decoys + 1 - filled)]
                        subset_idx_cpu[i, filled : filled + pp.shape[0]] = pad_avail[pp]
                gt_subset_pos_cpu[i] = 0
                continue

            # Defensive: a holdout (cluster_id<0) that slipped past the caller's seq_mask filter has
            # no valid in/out-cluster contrast -- the stratified path below would draw in_picks=[] and
            # build a silently-wrong padding-only contrast. Mark it with ignore_index (-100) so
            # F.cross_entropy skips the position instead of training a meaningless gradient. (Holdouts
            # are excluded from the 319 train DB by construction, so this only fires on a data slip.)
            if gt_cluster < 0:
                subset_idx_cpu[i] = torch.arange(n_decoys + 1, dtype=torch.long)
                gt_subset_pos_cpu[i] = -100
                continue

            # Stratified path: GT + in-cluster + out-cluster decoys.
            in_picks: list[int] = []
            if gt_cluster >= 0 and gt_cluster in cluster_pools_cpu:
                same_pool = cluster_pools_cpu[gt_cluster]
                same_pool_excl_gt = same_pool[same_pool != gt_idx]
                k_in = min(n_in_target, same_pool_excl_gt.shape[0])
                if k_in > 0:
                    perm = torch.randperm(same_pool_excl_gt.shape[0])[:k_in]
                    in_picks = same_pool_excl_gt[perm].tolist()

            other_cids = [cid for cid in cluster_pools_cpu if cid != gt_cluster]
            if other_cids:
                shuf = torch.randperm(len(other_cids)).tolist()
                other_cids = [other_cids[k] for k in shuf]
            out_picks: list[int] = []
            for j, cid in enumerate(other_cids):
                pool = cluster_pools_cpu[cid]
                k_take = per_other_floor + (1 if j < remainder else 0)
                k_take = min(k_take, pool.shape[0])
                if k_take <= 0:
                    continue
                perm = torch.randperm(pool.shape[0])[:k_take]
                out_picks.extend(pool[perm].tolist())

            decoys = in_picks + out_picks
            decoys = decoys[:n_decoys]
            if len(decoys) < n_decoys:
                forbid = {gt_idx, *decoys}
                avail = valid_universe_cpu[
                    torch.tensor([int(x) not in forbid for x in valid_universe_cpu.tolist()], dtype=torch.bool)
                ]
                if avail.shape[0] > 0:
                    pad_perm = torch.randperm(avail.shape[0])[: (n_decoys - len(decoys))]
                    decoys.extend(avail[pad_perm].tolist())

            subset_idx_cpu[i, 0] = gt_idx
            subset_idx_cpu[i, 1 : 1 + len(decoys)] = torch.tensor(decoys[:n_decoys], dtype=torch.long)
            gt_subset_pos_cpu[i] = 0  # GT at column 0 by construction

        return subset_idx_cpu.to(device), gt_subset_pos_cpu.to(device)

    # ───────────────────────────────────────────────────────────────────────

    def _precompute_element_histograms(
        self,
        element_types: torch.Tensor,
        residue_masks: torch.Tensor,
        backbone_indices: torch.Tensor | None,
    ) -> None:
        """
        Precompute element count histograms for reference residues.

        This enables element-type-aware matching by comparing element distributions.
        For sidechain-only comparison, excludes backbone atoms (indices 0-3).

        Parameters
        ----------
        element_types : torch.Tensor
            Element type indices, shape (R, max_atoms). Values: 0=C, 1=N, 2=O, 3=S, -1=invalid
        residue_masks : torch.Tensor
            Valid atom masks, shape (R, max_atoms)
        backbone_indices : torch.Tensor | None
            Backbone atom indices, shape (R, 4) or None
        """
        from .diffusion import NUM_ELEMENT_TYPES

        num_residues = element_types.shape[0]
        # DB element types are 0-indexed (C=0, N=1, ...), so vocab minus PAD.
        # 2026-06-05: 4->9 to count P/F/Cl/Br/B in the composition penalty.
        num_element_types = NUM_ELEMENT_TYPES - 1  # C, N, O, S, P, F, Cl, Br, B

        # Compute histograms for sidechain atoms only (skip first 4 if backbone present)
        n_skip = 4 if backbone_indices is not None else 0
        sc_element_types = element_types[:, n_skip:]  # (R, max_sc)
        sc_masks = residue_masks[:, n_skip:]  # (R, max_sc)

        # Count each element type per residue
        histograms = torch.zeros(num_residues, num_element_types, device=element_types.device)
        for etype in range(num_element_types):
            type_matches = (sc_element_types == etype).float()
            matches = type_matches * sc_masks.float()
            histograms[:, etype] = matches.sum(dim=-1).float()

        self.register_buffer("_ref_element_histograms", histograms)  # (R, 4)

    # ── Chirality-mismatch penalty helpers (2026-08 spec) ──────────────
    @staticmethod
    def _build_ref_chirality(
        residue_database: torch.Tensor,
        residue_masks: torch.Tensor,
        backbone_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-ROTAMER-ROW handedness in {-1, 0, +1} (+1 = L, -1 = D, 0 = undeterminable).

        One entry per row of ``residue_database``, which is a rotamer, NOT a residue type; see
        ``chirality_signs_of_type`` before indexing the result of this.

        Derived from the SINGLE geometric C-beta (the one sidechain atom bonded to CA),
        by delegating to ``data_utils._chirality_from_reference`` per rotamer row so this buffer
        and ``build_chirality_lookup`` (the learned-readout gate) are guaranteed identical.

        WHY NOT the mean-over-all-sidechain-atoms projection this used to compute: handedness is
        fixed by the rigidly-placed C-beta alone, but a mean over the WHOLE side chain is dominated
        by distal atoms that swing to either side of the backbone plane as the chi angles rotate.
        On the production library that made flexible chiral residues read a DIFFERENT sign on
        different rotamers -- 56 of ~300 in-vocab types came out mixed {+1, -1}, D-arginine among
        them -- so their handedness collapsed to "undeterminable" and the penalty never fired.
        The single-C-beta triple product is rotamer-invariant: every DAR rotamer now reads -1.

        A reference with an incomplete backbone (any of N/CA/C index < 0), a non-alpha backbone
        (beta/gamma-amino acids), no C-beta (glycine), or an alpha,alpha-disubstituted CA yields
        0 and is therefore never penalized.
        """
        from atomweaver.joint_diffusion.data_utils import _chirality_from_reference

        num_residues = residue_database.shape[0]
        signs = torch.zeros(num_residues, dtype=torch.float32)
        if backbone_indices is None:
            return signs
        bi = backbone_indices.long()
        coords = residue_database.float()
        masks = residue_masks.bool()
        col_names = ("N", "CA", "C", "O")
        for row in range(num_residues):
            # Build the per-row backbone_indices dict, omitting absent (< 0) atoms so
            # _chirality_from_reference's own presence checks fire correctly.
            bb = {name: int(bi[row, col]) for col, name in enumerate(col_names) if int(bi[row, col]) >= 0}
            sign = _chirality_from_reference(coords[row], masks[row], bb)
            if sign is not None:
                signs[row] = float(sign)
        return signs

    @property
    def n_rotamer_rows(self) -> int:
        """Rows in the reference database, i.e. ROTAMERS -- not residue types."""
        return int(self._ref_chirality.shape[0])

    def chirality_signs_of_type(self, type_index: int, rotamer_to_type: torch.Tensor) -> set[float]:
        """
        Handedness signs of every ROTAMER belonging to one library TYPE, as a set.

        THIS IS THE ONLY SUPPORTED WAY TO READ ``_ref_chirality``, and it exists because the
        obvious way is wrong. That buffer is indexed by ROTAMER (one row per reference
        conformer, e.g. 6487) while callers overwhelmingly hold a TYPE index (one per residue,
        e.g. 691). ``_ref_chirality[type_index]`` is therefore a silent, plausible lie: it
        returns some unrelated residue's handedness, with no exception and no shape error,
        and feeds a penalty of ``chirality_mismatch_penalty`` (50.0 in the eval harness) --
        an order of magnitude larger than the entire spread of every other scoring term.

        Measured on the production library the naive read disagrees with the truth for
        316 of 691 residue types -- 46%. D-alanine reads +1 (L) when it is D, so the penalty
        that should isolate it never fires; glycine reads +1 (L) when its handedness is
        undeterminable (0), making a residue that must never be penalised penalisable; and
        multi-rotamer residues like KCR read a single sign where the rotamers actually carry
        both. Every one of those is a valid-looking value in {-1, 0, +1}.

        A comment would not have prevented it, so the shape of this API does: the rotamer map
        is a required argument rather than an optional one, and a map whose length does not
        match the buffer is refused. A type-indexed read is not expressible through it.

        Returns a SET because a flexible residue can carry more than one sign across its
        rotamers -- the geometric derivation is unstable when the atoms defining handedness
        sit far from CA. ``{0.0}`` means undeterminable (glycine, alpha,alpha-disubstituted
        references) and is never penalised.
        """
        # 1-D is required, not merely a matching length. A column-shaped (R, 1) map passes a
        # shape[0] check, and .nonzero().flatten() on it interleaves row AND column indices -- so
        # column 0 leaks in as row 0 and an unrelated rotamer's sign joins the returned set. For an
        # accessor whose whole purpose is making bad indexing inexpressible, silently accepting the
        # wrong shape is precisely the wrong failure. Raised in review of MR !82.
        if rotamer_to_type.ndim != 1 or rotamer_to_type.numel() != self.n_rotamer_rows:
            raise ValueError(
                f"rotamer_to_type must be a 1-D map of length {self.n_rotamer_rows} (one entry per "
                f"ROTAMER row); got shape {tuple(rotamer_to_type.shape)}. _ref_chirality is indexed by "
                "rotamer, and these differ only in a length nobody checks, so a mismatch here almost "
                "always means a TYPE index was about to be used as a ROTAMER index -- which silently "
                "returns an unrelated residue's handedness. Pass the rotamer_to_type built alongside "
                "this database, un-reshaped."
            )
        rows = (rotamer_to_type == int(type_index)).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            raise ValueError(f"no rotamer row maps to type {type_index}; the rotamer map does not match this database")
        return {float(self._ref_chirality[r]) for r in rows}

    def _compute_prediction_chirality(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        backbone_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position predicted handedness sign (B, L) and confidence |proj_cb| (B, L).

        predicted_coords/mask are the SIDECHAIN atoms (before any backbone prepend);
        backbone_coords is (B, L, 4, 3) with N=0, CA=1, C=2.

        Uses the SINGLE geometric C-beta (the valid predicted atom nearest CA), matching the
        reference rule in ``_build_ref_chirality`` / ``_chirality_from_reference``. Reference and
        prediction MUST share one convention for the penalty to be meaningful: a mean over the
        whole predicted side chain is dominated by distal atoms whose sign swings with the chi
        angles, so a correctly-placed L cloud could read the wrong sign and be spuriously
        penalized. The C-beta sign is rotamer-invariant. Positions with no valid predicted atom
        get sign 0 (never penalized).
        """
        n = backbone_coords[..., 0, :]
        ca = backbone_coords[..., 1, :]
        c = backbone_coords[..., 2, :]
        e3 = torch.linalg.cross(n - ca, c - ca, dim=-1)
        e3 = e3 / (e3.norm(dim=-1, keepdim=True) + 1e-8)  # (B, L, 3)
        rel = predicted_coords - ca.unsqueeze(-2)  # (B, L, max_sc, 3)
        # C-beta = the valid predicted atom nearest CA. Invalid slots pushed to +inf so they
        # never win the argmin; positions with no valid atom (all masked) fall back to sign 0.
        dist = rel.norm(dim=-1)  # (B, L, max_sc)
        valid = predicted_mask.bool()
        dist = dist.masked_fill(~valid, float("inf"))
        has_atom = valid.any(dim=-1)  # (B, L)
        cb_idx = dist.argmin(dim=-1, keepdim=True)  # (B, L, 1)
        proj = (rel * e3.unsqueeze(-2)).sum(-1)  # (B, L, max_sc)  signed handedness per atom
        proj_cb = proj.gather(-1, cb_idx).squeeze(-1)  # (B, L)  handedness at the C-beta
        proj_cb = proj_cb * has_atom.to(proj_cb.dtype)  # zero out atomless positions
        return torch.sign(proj_cb), proj_cb.abs()

    def _apply_chirality_penalty(
        self,
        logits: torch.Tensor,
        pred_sign: torch.Tensor | None,
        pred_conf: torch.Tensor | None,
    ) -> torch.Tensor:
        """Subtract a finite, confidence-weighted penalty from candidates of opposite handedness.

        mismatch = (pred_sign[..., None] * _ref_chirality[None, None, :] < 0)  # only both-nonzero-and-opposite
        logits   = logits - chirality_mismatch_penalty * conf[..., None] * mismatch

        Disabled (penalty <= 0) or missing prediction handedness => returns logits unchanged
        (byte-identical). Never -inf, so a wrong-chirality GT position can't NaN the CE.
        """
        if self.chirality_mismatch_penalty <= 0.0 or pred_sign is None or pred_conf is None:
            return logits
        ref = self._ref_chirality.to(device=logits.device, dtype=logits.dtype)  # (R,)
        mismatch = (pred_sign.unsqueeze(-1) * ref.view(1, 1, -1) < 0).to(logits.dtype)  # (B, L, R)
        return logits - self.chirality_mismatch_penalty * pred_conf.unsqueeze(-1) * mismatch

    def _init_normalized_method(self, residue_database: torch.Tensor, residue_masks: torch.Tensor) -> None:
        """Initialize precomputed data for normalized distance matrix method."""
        # Precompute and normalize reference distance matrices
        ref_dist_matrices = self._compute_distance_matrices(residue_database, residue_masks)

        # Get valid masks for normalization
        pair_masks = residue_masks.unsqueeze(-1) * residue_masks.unsqueeze(-2)
        triu_mask = torch.triu(torch.ones_like(pair_masks[0]), diagonal=1).bool()
        valid_masks = pair_masks * triu_mask.unsqueeze(0)  # (R, max_sc, max_sc)

        # Normalize each reference distance matrix (z-score on valid pairs)
        ref_normalized = self._normalize_distance_matrices(ref_dist_matrices, valid_masks)
        self.register_buffer("_ref_normalized", ref_normalized)
        self.register_buffer("_ref_valid_masks", valid_masks)

    def _compute_distance_matrices(self, coords: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrices."""
        sq_norms = (coords**2).sum(dim=-1, keepdim=True)
        dot_products = torch.matmul(coords, coords.transpose(-1, -2))
        sq_dists = sq_norms + sq_norms.transpose(-1, -2) - 2 * dot_products
        sq_dists = sq_dists.clamp(min=0)
        distances = torch.sqrt(sq_dists + 1e-8)
        pair_mask = masks.unsqueeze(-1) * masks.unsqueeze(-2)
        return distances * pair_mask

    def _normalize_distance_matrices(self, dist_matrices: torch.Tensor, valid_masks: torch.Tensor) -> torch.Tensor:
        """Z-score normalize distance matrices on valid pairs only."""
        valid_dists = dist_matrices * valid_masks
        valid_counts = valid_masks.sum(dim=(-1, -2), keepdim=True).clamp(min=1)
        means = valid_dists.sum(dim=(-1, -2), keepdim=True) / valid_counts

        centered = (dist_matrices - means) * valid_masks
        variances = (centered**2).sum(dim=(-1, -2), keepdim=True) / valid_counts
        stds = torch.sqrt(variances + 1e-8)

        normalized = centered / stds
        return normalized * valid_masks

    @staticmethod
    def _repack_prediction_dense(
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        predicted_element_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compact the predicted sidechain so masked-TRUE atoms occupy contiguous slots from 0.

        Predictions arrive in reserved_slot0 layout (slot 0 = ghost/masked, real atoms from
        slot 1, possibly with interior ghosts). The reference is dense-packed (sidechain
        contiguous from index 0), and the NDM / combined distance-matrix correlation is
        POSITIONAL -- cell [i, j] of the prediction is compared against cell [i, j] of the
        reference -- so a reserved-slot0 prediction is scored one slot off. This left-packs the
        TRUE atoms (dropping ghost slot 0 and any interior ghosts) while PRESERVING their order
        (ascending slot = radius/shell order), keeping coords / mask / element_types
        index-aligned. It is a no-op when the prediction is already dense (a stable sort of an
        already-left-packed mask is the identity permutation), so it is safe for all methods and
        for both reserved-slot0 and dense inputs.

        Parameters
        ----------
        predicted_coords : torch.Tensor
            Sidechain coords, shape (B, L, S, 3).
        predicted_mask : torch.Tensor
            Sidechain mask, shape (B, L, S).
        predicted_element_types : torch.Tensor, optional
            Sidechain element types, shape (B, L, S).

        Returns
        -------
        tuple
            (coords, mask, element_types) dense-packed, same shapes as inputs.
        """
        # Stable descending sort of the boolean mask: TRUE atoms move to the front in their
        # original relative order; FALSE (ghost) atoms fall to the back. argsort(..., stable=True)
        # guarantees the order-preserving permutation the positional comparison needs.
        order = torch.argsort(predicted_mask.to(torch.int64), dim=-1, descending=True, stable=True)  # (B, L, S)
        coords = torch.gather(predicted_coords, 2, order.unsqueeze(-1).expand(-1, -1, -1, 3))
        mask = torch.gather(predicted_mask, 2, order)
        etypes = torch.gather(predicted_element_types, 2, order) if predicted_element_types is not None else None
        return coords, mask, etypes

    def _prepend_backbone_to_predicted(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepend all backbone atoms (N, CA, C, O) to predicted sidechain coordinates.

        Centers everything on CA position so that backbone atoms define a local
        coordinate frame. This preserves chirality and provides a proper scaffold
        for alignment.

        Parameters
        ----------
        predicted_coords : torch.Tensor
            Predicted sidechain coords of shape (B, L, max_sc, 3).
        predicted_mask : torch.Tensor
            Mask of shape (B, L, max_sc).
        backbone_coords : torch.Tensor
            Backbone coords of shape (B, L, 4, 3) for N, CA, C, O.
        backbone_mask : torch.Tensor
            Backbone mask of shape (B, L, 4).

        Returns
        -------
        combined_coords : torch.Tensor
            Shape (B, L, 4 + max_sc, 3) with backbone atoms prepended.
        combined_mask : torch.Tensor
            Shape (B, L, 4 + max_sc).
        """
        # Extract CA position (index 1 in backbone_coords) for centering
        ca_pos = backbone_coords[:, :, 1:2, :]  # (B, L, 1, 3)

        # Center everything on CA
        centered_backbone = backbone_coords - ca_pos  # (B, L, 4, 3)
        centered_sidechain = predicted_coords - ca_pos  # (B, L, max_sc, 3)

        # Concatenate: backbone first (N, CA, C, O), then sidechain
        combined_coords = torch.cat([centered_backbone, centered_sidechain], dim=2)  # (B, L, 4+max_sc, 3)
        combined_mask = torch.cat([backbone_mask, predicted_mask], dim=2)  # (B, L, 4+max_sc)

        return combined_coords, combined_mask

    def _get_reference_with_backbone_first(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Get reference coords centered on CA, with backbone atoms (N, CA, C, O) first.

        Centers each reference residue on its CA position, then arranges atoms as:
        [N, CA, C, O, sidechain_atoms...]. This matches the predicted structure
        format from _prepend_backbone_to_predicted.

        Returns
        -------
        reordered_coords : torch.Tensor
            Shape (R, 4 + max_sidechain_atoms, 3) with backbone at indices 0-3.
        reordered_masks : torch.Tensor
            Shape (R, 4 + max_sidechain_atoms).
        reordered_element_types : torch.Tensor or None
            Shape (R, 4 + max_sidechain_atoms) if element_types available, else None.
        """
        # The reordered reference DB (CA-centered, backbone-first) is a pure function of the
        # static buffers (_residue_database/_masks/_backbone_indices/_element_types) and never
        # changes during training. Recomputing it every forward was a per-step hot loop of
        # ~R*max_atoms GPU-synced element writes (the dominant CPU cost on high-sync-latency
        # hardware). Compute once, cache; rebuild only if the module moved devices.
        _cached = getattr(self, "_ref_bbfirst_cache", None)
        if _cached is not None and _cached[0].device == self._residue_database.device:
            return _cached

        ref_coords = self._residue_database.clone()
        ref_masks = self._residue_masks.clone()
        backbone_indices = self._backbone_indices  # (R, 4) for N, CA, C, O
        has_element_types = self._element_types is not None

        num_residues, max_atoms, _ = ref_coords.shape
        device = ref_coords.device

        # Get CA index for each residue (index 1 in backbone_indices = CA)
        ca_indices = backbone_indices[:, 1]  # (R,)

        # Center each residue on its CA
        for r in range(num_residues):
            ca_idx = ca_indices[r].item()
            if ca_idx >= 0:
                ca_pos = ref_coords[r, ca_idx]  # (3,)
                ref_coords[r] = ref_coords[r] - ca_pos.unsqueeze(0)

        # For each residue, we need: 4 backbone + sidechain atoms (excluding backbone)
        # Max sidechain size = max_atoms - 4 (worst case: all non-backbone)
        max_sidechain = max_atoms
        new_max = 4 + max_sidechain
        reordered_coords = torch.zeros(num_residues, new_max, 3, device=device)
        reordered_masks = torch.zeros(num_residues, new_max, dtype=torch.bool, device=device)
        if has_element_types:
            reordered_etypes = torch.full((num_residues, new_max), -1, dtype=torch.long, device=device)
        else:
            reordered_etypes = None

        for r in range(num_residues):
            bb_set = set(backbone_indices[r].tolist())

            # Place backbone atoms at indices 0-3
            for bb_slot, bb_idx in enumerate(backbone_indices[r]):
                bb_idx = bb_idx.item()
                if bb_idx >= 0:
                    reordered_coords[r, bb_slot] = ref_coords[r, bb_idx]
                    reordered_masks[r, bb_slot] = True
                    if has_element_types:
                        reordered_etypes[r, bb_slot] = self._element_types[r, bb_idx]

            # Place sidechain atoms (non-backbone) starting at index 4
            sc_slot = 4
            for atom_idx in range(max_atoms):
                if ref_masks[r, atom_idx] and atom_idx not in bb_set:
                    reordered_coords[r, sc_slot] = ref_coords[r, atom_idx]
                    reordered_masks[r, sc_slot] = True
                    if has_element_types:
                        reordered_etypes[r, sc_slot] = self._element_types[r, atom_idx]
                    sc_slot += 1

        self._ref_bbfirst_cache = (reordered_coords, reordered_masks, reordered_etypes)
        return reordered_coords, reordered_masks, reordered_etypes

    def forward(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        target_indices: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        valid_residue_mask: torch.Tensor | None = None,
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
        predicted_element_types: torch.Tensor | None = None,
        return_atom_importance: bool = False,
        importance_top_k: int = 3,
    ) -> dict[str, torch.Tensor]:
        """
        Compute discretization loss.

        Parameters
        ----------
        predicted_coords : torch.Tensor
            Predicted side-chain coordinates of shape (B, L, max_sc, 3).
        predicted_mask : torch.Tensor
            Mask for predicted atoms of shape (B, L, max_sc).
        target_indices : torch.Tensor
            Ground truth residue indices of shape (B, L).
        seq_mask : torch.Tensor, optional
            Mask for valid residue positions of shape (B, L).
        backbone_coords : torch.Tensor, optional
            Backbone coordinates of shape (B, L, 4, 3) for N, CA, C, O.
            Used for CA-centering when include_backbone=True.
        backbone_mask : torch.Tensor, optional
            Backbone mask of shape (B, L, 4).
        predicted_element_types : torch.Tensor, optional
            Element types for predicted sidechain atoms, shape (B, L, max_sc).
            Values: 0=C, 1=N, 2=O, 3=S, -1=invalid. Used for element matching penalty.
        return_atom_importance : bool
            If True, return per-atom importance scores (M-weighted contribution to
            NDM correlation across top-k references). Shape (B, L, max_atoms).
        importance_top_k : int
            Number of top-scoring references to use for importance weighting.
            Default 3 avoids washing out signal across many low-scoring references.

        Returns
        -------
        dict
            Dictionary with 'loss', 'logits', 'accuracy', and optionally 'atom_importance'.
        """
        # Skip residues absent from the disc DB (e.g. dropped/non-alpha NCAAs): they fall back
        # to index 0 in collate, so without this they'd be mis-trained as residue-0. Coord/element
        # losses still see them via the unmasked seq_mask.
        if valid_residue_mask is not None and seq_mask is not None:
            seq_mask = seq_mask & valid_residue_mask

        # Per-prediction handedness for the chirality-mismatch penalty. Computed from the RAW
        # sidechain cloud (before the backbone prepend below) + backbone N/CA/C. When the penalty
        # is off OR no backbone_coords are supplied, both stay None => the penalty is a no-op and
        # the scoring path is byte-identical to the pre-chirality behavior.
        pred_sign = pred_conf = None
        if self.chirality_mismatch_penalty > 0.0 and backbone_coords is not None:
            pred_sign, pred_conf = self._compute_prediction_chirality(predicted_coords, predicted_mask, backbone_coords)

        # Slot off-by-one fix: dense-pack the predicted sidechain to match the dense-packed
        # reference BEFORE the (positional) distance-matrix comparison. No-op on already-dense
        # input; chirality above is order-invariant so its result is unaffected either way.
        if self.repack_prediction:
            predicted_coords, predicted_mask, predicted_element_types = self._repack_prediction_dense(
                predicted_coords, predicted_mask, predicted_element_types
            )

        # Optionally prepend backbone atoms to predicted coords for better matching
        if self.include_backbone and backbone_coords is not None and backbone_mask is not None:
            predicted_coords, predicted_mask = self._prepend_backbone_to_predicted(
                predicted_coords, predicted_mask, backbone_coords, backbone_mask
            )
            # Prepend backbone element types (N=1, CA=0, C=0, O=2) to match prepended coords
            if predicted_element_types is not None:
                b_size, l_size = predicted_element_types.shape[:2]
                bb_etypes = torch.tensor(
                    [1, 0, 0, 2], device=predicted_element_types.device, dtype=predicted_element_types.dtype
                )
                bb_etypes = bb_etypes.unsqueeze(0).unsqueeze(0).expand(b_size, l_size, 4)
                predicted_element_types = torch.cat([bb_etypes, predicted_element_types], dim=2)

        return self._forward_normalized(
            predicted_coords,
            predicted_mask,
            target_indices,
            seq_mask,
            predicted_element_types,
            return_atom_importance=return_atom_importance,
            importance_top_k=importance_top_k,
            pred_sign=pred_sign,
            pred_conf=pred_conf,
        )

    def _forward_normalized(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        target_indices: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        predicted_element_types: torch.Tensor | None = None,
        return_atom_importance: bool = False,
        importance_top_k: int = 3,
        pred_sign: torch.Tensor | None = None,
        pred_conf: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass using normalized distance matrix correlation.

        If use_backbone_for_alignment_only=True and include_backbone=True:
        - Backbone atoms (indices 0-3) are used for frame alignment
        - Distance matrix is computed on sidechain atoms only (indices 4+)
        - This makes the loss rotamer-invariant (chi1 rotation doesn't penalize)
        """
        batch_size, seq_len, max_atoms_pred, _ = predicted_coords.shape
        num_residues = self._residue_database.shape[0]

        # Determine if we're doing rotamer-invariant comparison
        rotamer_invariant = (
            self.include_backbone and self.use_backbone_for_alignment_only and self._backbone_indices is not None
        )
        n_backbone = 4 if rotamer_invariant else 0

        # Get reference data (optionally with backbone first for correspondence)
        if self.include_backbone and self._backbone_indices is not None:
            ref_coords, ref_masks, _ref_etypes = self._get_reference_with_backbone_first()
            max_atoms_ref = ref_coords.shape[1]

            if rotamer_invariant:
                # For rotamer-invariant: compute distance matrices on sidechain only (indices 4+)
                ref_sidechain_coords = ref_coords[:, n_backbone:, :]
                ref_sidechain_masks = ref_masks[:, n_backbone:]
                ref_dist = self._compute_distance_matrices(ref_sidechain_coords, ref_sidechain_masks)
                pair_masks = ref_sidechain_masks.unsqueeze(-1) * ref_sidechain_masks.unsqueeze(-2)
                max_sc_ref = ref_sidechain_coords.shape[1]
                triu_mask = torch.triu(torch.ones(max_sc_ref, max_sc_ref, device=ref_coords.device), diagonal=1).bool()
                ref_valid = pair_masks * triu_mask.unsqueeze(0)
                ref_normalized = self._normalize_distance_matrices(ref_dist, ref_valid)
            else:
                # Standard: compute on all atoms including backbone
                ref_dist = self._compute_distance_matrices(ref_coords, ref_masks)
                pair_masks = ref_masks.unsqueeze(-1) * ref_masks.unsqueeze(-2)
                triu_mask = torch.triu(
                    torch.ones(max_atoms_ref, max_atoms_ref, device=ref_coords.device), diagonal=1
                ).bool()
                ref_valid = pair_masks * triu_mask.unsqueeze(0)
                ref_normalized = self._normalize_distance_matrices(ref_dist, ref_valid)
        else:
            ref_normalized = self._ref_normalized
            ref_valid = self._ref_valid_masks
            max_atoms_ref = self._residue_database.shape[1]

        # For rotamer-invariant, we compare sidechain atoms only
        if rotamer_invariant:
            # Extract sidechain from predicted (indices 4+)
            pred_sidechain_coords = predicted_coords[:, :, n_backbone:, :]
            pred_sidechain_mask = predicted_mask[:, :, n_backbone:]
            # Also slice element types to sidechain-only (backbone types were prepended in forward())
            if predicted_element_types is not None:
                predicted_element_types = predicted_element_types[:, :, n_backbone:]
            max_sc_pred = pred_sidechain_coords.shape[2]
            max_sc_ref = ref_normalized.shape[1]  # Already sidechain-only
            max_sc = max(max_sc_pred, max_sc_ref)

            # Pad predicted sidechain if needed (coords, mask, and element_types)
            if max_sc_pred < max_sc:
                pad_size = max_sc - max_sc_pred
                pred_sidechain_coords = F.pad(pred_sidechain_coords, (0, 0, 0, pad_size))
                pred_sidechain_mask = F.pad(pred_sidechain_mask, (0, pad_size))
                if predicted_element_types is not None:
                    predicted_element_types = F.pad(predicted_element_types, (0, pad_size), value=-1)

            # Compute predicted sidechain distance matrices
            pred_dist = self._compute_distance_matrices(pred_sidechain_coords, pred_sidechain_mask)

            # Create valid mask for predicted sidechain
            pred_pair_mask = pred_sidechain_mask.unsqueeze(-1) * pred_sidechain_mask.unsqueeze(-2)
            triu_mask = torch.triu(torch.ones(max_sc, max_sc, device=predicted_coords.device), diagonal=1).bool()
            pred_valid = pred_pair_mask * triu_mask.unsqueeze(0).unsqueeze(0)

            # Normalize predicted distance matrices
            pred_normalized = self._normalize_distance_matrices(pred_dist, pred_valid)

            # Pad reference if needed
            if max_sc_ref < max_sc:
                pad_size = max_sc - max_sc_ref
                ref_normalized = F.pad(ref_normalized, (0, pad_size, 0, pad_size))
                ref_valid = F.pad(ref_valid, (0, pad_size, 0, pad_size))

            max_atoms = max_sc
        else:
            # Standard comparison on all atoms
            max_atoms = max(max_atoms_pred, max_atoms_ref)

            # Pad predicted if needed (coords, mask, and element_types)
            if max_atoms_pred < max_atoms:
                pad_size = max_atoms - max_atoms_pred
                predicted_coords = F.pad(predicted_coords, (0, 0, 0, pad_size))
                predicted_mask = F.pad(predicted_mask, (0, pad_size))
                if predicted_element_types is not None:
                    predicted_element_types = F.pad(predicted_element_types, (0, pad_size), value=-1)

            # Compute predicted distance matrices
            pred_dist = self._compute_distance_matrices(predicted_coords, predicted_mask)

            # Create valid mask for predicted
            pred_pair_mask = predicted_mask.unsqueeze(-1) * predicted_mask.unsqueeze(-2)
            triu_mask = torch.triu(torch.ones(max_atoms, max_atoms, device=predicted_coords.device), diagonal=1).bool()
            pred_valid = pred_pair_mask * triu_mask.unsqueeze(0).unsqueeze(0)

            # Normalize predicted distance matrices
            pred_normalized = self._normalize_distance_matrices(pred_dist, pred_valid)

            # Pad reference if needed
            if max_atoms_ref < max_atoms:
                pad_size = max_atoms - max_atoms_ref
                ref_normalized = F.pad(ref_normalized, (0, pad_size, 0, pad_size))
                ref_valid = F.pad(ref_valid, (0, pad_size, 0, pad_size))

        # Compute correlation-based similarity
        pred_exp = pred_normalized.unsqueeze(2)  # (B, L, 1, max_atoms, max_atoms)
        ref_exp = ref_normalized.unsqueeze(0).unsqueeze(0)  # (1, 1, R, max_atoms, max_atoms)

        # Joint valid mask
        combined_valid = pred_valid.unsqueeze(2) * ref_valid.unsqueeze(0).unsqueeze(0)

        # Correlation = mean product of normalized values
        products = pred_exp * ref_exp * combined_valid
        valid_counts = combined_valid.sum(dim=(-1, -2)).clamp(min=1)
        correlations = products.sum(dim=(-1, -2)) / valid_counts  # (B, L, R)

        # Per-atom importance: how much each atom contributes to NDM correlation,
        # averaged across top-k best-matching references weighted by match score M.
        # Each atom k's contribution = sum of all pair products involving k
        # (row contributions where k < j, plus column contributions where i < k).
        # Using top-k avoids washing out signal across many low-scoring references.
        atom_importance = None
        if return_atom_importance:
            # products: (B, L, R, max_atoms, max_atoms) upper-triangle masked
            row_contrib = products.sum(dim=-1)  # (B, L, R, max_atoms) -- pairs where atom is row (k < j)
            col_contrib = products.sum(dim=-2)  # (B, L, R, max_atoms) -- pairs where atom is col (i < k)
            atom_contrib_per_ref = (row_contrib + col_contrib) / valid_counts.unsqueeze(-1)  # (B, L, R, max_atoms)

            # Select top-k references by correlation score, weight by softmax over those k
            k = min(importance_top_k, correlations.shape[-1])
            topk_vals, topk_idx = correlations.topk(k, dim=-1)  # (B, L, k)
            m_weights = torch.softmax(topk_vals, dim=-1)  # (B, L, k)

            # Gather atom contributions for top-k references only
            max_atoms = atom_contrib_per_ref.shape[-1]
            topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, max_atoms)  # (B, L, k, max_atoms)
            topk_contrib = atom_contrib_per_ref.gather(2, topk_idx_exp)  # (B, L, k, max_atoms)

            atom_importance = (m_weights.unsqueeze(-1) * topk_contrib).sum(dim=2)  # (B, L, max_atoms)

        # Apply atom count mismatch penalty
        # This prevents smaller structures from getting artificially high correlations
        if self.atom_mismatch_penalty > 0:
            # Count atoms in predicted (sidechain only if rotamer_invariant)
            if rotamer_invariant:
                pred_atom_counts = pred_sidechain_mask.sum(dim=-1).float()  # (B, L)
                ref_atom_counts = ref_sidechain_masks.sum(dim=-1).float()  # (R,)
            elif self.include_backbone and self._backbone_indices is not None:
                # ref_masks was set in the include_backbone branch above
                pred_atom_counts = predicted_mask.sum(dim=-1).float()  # (B, L)
                ref_atom_counts = ref_masks.sum(dim=-1).float()  # (R,)
            else:
                # Fallback: use the registered buffer
                pred_atom_counts = predicted_mask.sum(dim=-1).float()  # (B, L)
                ref_atom_counts = self._residue_masks.sum(dim=-1).float()  # (R,)

            # Expand for broadcasting: (B, L, 1) and (1, 1, R)
            pred_counts_exp = pred_atom_counts.unsqueeze(-1)  # (B, L, 1)
            ref_counts_exp = ref_atom_counts.unsqueeze(0).unsqueeze(0)  # (1, 1, R)

            # Compute overlap ratio: min/max gives [0, 1]
            max_counts = torch.maximum(pred_counts_exp, ref_counts_exp).clamp(min=1)
            min_counts = torch.minimum(pred_counts_exp, ref_counts_exp)
            overlap_ratio = min_counts / max_counts  # (B, L, R)

            # Mismatch penalty: 0 when counts match, up to penalty_weight when no overlap
            mismatch = 1.0 - overlap_ratio
            correlations = correlations - self.atom_mismatch_penalty * mismatch

        # Apply element type mismatch penalty
        # This distinguishes residues with same atom count but different elements (e.g., SER vs CYS)
        if self.element_mismatch_penalty > 0 and predicted_element_types is not None:
            ref_histograms = self._ref_element_histograms  # (R, 4) for C, N, O, S
            if ref_histograms is not None:
                # Compute predicted element histograms (sidechain only)
                num_element_types = ref_histograms.shape[1]  # 4
                pred_histograms = torch.zeros(batch_size, seq_len, num_element_types, device=predicted_coords.device)
                for etype in range(num_element_types):
                    # predicted_element_types: (B, L, max_sc)
                    if rotamer_invariant:
                        # Sidechain only (use float multiplication for soft masks)
                        type_matches = (predicted_element_types == etype).float()
                        matches = type_matches * pred_sidechain_mask.float()
                    else:
                        # All atoms
                        if n_backbone > 0:
                            sc_etypes = predicted_element_types[:, :, n_backbone:]
                            sc_mask = predicted_mask[:, :, n_backbone:]
                        else:
                            sc_etypes = predicted_element_types
                            sc_mask = predicted_mask
                        type_matches = (sc_etypes == etype).float()
                        matches = type_matches * sc_mask.float()
                    pred_histograms[:, :, etype] = matches.sum(dim=-1).float()

                # Compute histogram difference penalty
                # pred_histograms: (B, L, 4), ref_histograms: (R, 4)
                # Manhattan distance normalized by total atoms
                pred_hist_exp = pred_histograms.unsqueeze(2)  # (B, L, 1, 4)
                ref_hist_exp = ref_histograms.unsqueeze(0).unsqueeze(0)  # (1, 1, R, 4)
                hist_diff = torch.abs(pred_hist_exp - ref_hist_exp).sum(dim=-1)  # (B, L, R)

                # Normalize by max(sum of histograms) to get [0, 2] range
                max_atoms_hist = torch.maximum(
                    pred_histograms.sum(dim=-1, keepdim=True),  # (B, L, 1)
                    ref_histograms.sum(dim=-1).unsqueeze(0).unsqueeze(0),  # (1, 1, R)
                ).clamp(min=1)
                normalized_diff = hist_diff / max_atoms_hist  # (B, L, R), range [0, 2]

                correlations = correlations - self.element_mismatch_penalty * normalized_diff

        logits = correlations / self.temperature

        # Chirality-mismatch penalty: demote candidates whose handedness disagrees with the
        # predicted cloud's handedness (no-op when disabled or pred_sign is None).
        logits = self._apply_chirality_penalty(logits, pred_sign, pred_conf)

        # Compute loss and accuracy
        result = self._compute_loss_and_accuracy(logits, target_indices, seq_mask, num_residues)
        if atom_importance is not None:
            result["atom_importance"] = atom_importance
        return result

    def _compute_loss_and_accuracy(
        self,
        logits: torch.Tensor,
        target_indices: torch.Tensor,
        seq_mask: torch.Tensor | None,
        num_residues: int,
    ) -> dict[str, torch.Tensor]:
        """
        Compute cross-entropy loss, top-1 and top-3 accuracy from logits.

        If rotamer_to_type mapping is set (for multi-rotamer databases), aggregates
        logits by amino acid type before computing loss and accuracy. This allows
        any rotamer of the correct type to be considered correct.
        """
        logits_flat = logits.view(-1, num_residues)
        targets_flat = target_indices.view(-1)

        # Handle multi-rotamer case: aggregate logits by amino acid type
        if self._rotamer_to_type is not None:
            # Aggregate logits by type using logsumexp (proper probability aggregation)
            # logits_flat: (N, num_rotamers=112), targets: AA types (0-19)
            num_types = self._num_types
            type_logits = torch.full(
                (logits_flat.shape[0], num_types),
                float("-inf"),
                device=logits_flat.device,
                dtype=logits_flat.dtype,
            )
            # Use scatter_reduce to aggregate logits by type
            type_idx = self._rotamer_to_type.unsqueeze(0).expand(logits_flat.shape[0], -1)
            type_logits = type_logits.scatter_reduce(
                dim=1,
                index=type_idx,
                src=logits_flat,
                reduce="amax",  # Use max for stability (approximates logsumexp)
            )
            logits_for_loss = type_logits
        else:
            logits_for_loss = logits_flat

        # Guard: target residue indices must index the discretizer's class space.
        # A mismatch (e.g. a 325-class eval-DB target reaching the 319-class train discretizer,
        # or dataset name_to_idx disagreeing with rotamer_to_type / num_types) otherwise surfaces
        # as a cryptic CUDA device-side assert deep inside F.cross_entropy. Fail loudly and early
        # with the offending value instead. (Ported from mainline Blocker-D hardening.)
        n_classes = logits_for_loss.shape[-1]
        if targets_flat.numel() > 0:
            t_min = int(targets_flat.min())
            t_max = int(targets_flat.max())
            if t_min < 0 or t_max >= n_classes:
                bad = targets_flat[(targets_flat < 0) | (targets_flat >= n_classes)]
                raise ValueError(
                    f"DiscretizationLoss: target residue index out of range -- targets span "
                    f"[{t_min}, {t_max}] but the discretizer has {n_classes} classes. "
                    f"Offending values: {bad.unique().tolist()[:10]}. The dataset's residue_indices "
                    f"(from name_to_idx) disagree with the disc-loss residue DB (rotamer_to_type / "
                    f"num_types) -- verify --residue-db matches the DB used to build name_to_idx."
                )

        # Use logits_for_loss (type-aggregated if rotamer mapping exists) for accuracy.
        # Stratified path (2026-05-27 spec): when decoy_sampling="stratified" AND in train()
        # mode, loss is computed over a per-position subset of (1 GT + n_decoys) classes. Accuracy
        # is reported on the SAME subset (training-time full-N argmax is wasted work; test-eval gives
        # the true full-N number). Validation always uses full-N CE so val metrics stay clean baselines.
        use_stratified = self.decoy_sampling == "stratified" and self.training and self._cluster_id_per_type is not None
        # Volumetric-decoy-CE reuse hook: when the stratified path samples decoys, stash the exact
        # (subset_idx, gt_subset_pos, flat valid indices) so a downstream consumer (the volumetric decoy CE
        # in training_step) can score the SAME decoys against the volumetric head's density field instead of
        # re-sampling. None on every non-stratified / no-decoy path (byte-identical there).
        decoy_subset_idx: torch.Tensor | None = None
        decoy_gt_pos: torch.Tensor | None = None
        decoy_valid_idx: torch.Tensor | None = None
        if seq_mask is not None:
            mask_flat = seq_mask.view(-1)
            valid_idx = torch.where(mask_flat)[0]

            if len(valid_idx) == 0:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                accuracy = torch.tensor(0.0, device=logits.device)
                accuracy_top3 = torch.tensor(0.0, device=logits.device)
            elif use_stratified:
                subset_idx, gt_subset_pos = self._sample_decoy_subset(targets_flat[valid_idx], valid_mask=None)
                decoy_subset_idx, decoy_gt_pos, decoy_valid_idx = subset_idx, gt_subset_pos, valid_idx
                subset_logits = torch.gather(logits_for_loss[valid_idx], dim=1, index=subset_idx)  # (N, n_decoys+1)
                loss = F.cross_entropy(subset_logits, gt_subset_pos)
                preds = subset_logits.argmax(dim=-1)
                accuracy = (preds == gt_subset_pos).float().mean()
                top3_preds = subset_logits.topk(min(3, subset_logits.shape[-1]), dim=-1).indices  # (N, <=3)
                accuracy_top3 = (top3_preds == gt_subset_pos.unsqueeze(-1)).any(dim=-1).float().mean()
            else:
                loss = F.cross_entropy(logits_for_loss[valid_idx], targets_flat[valid_idx])
                preds = logits_for_loss[valid_idx].argmax(dim=-1)
                accuracy = (preds == targets_flat[valid_idx]).float().mean()
                top3_preds = logits_for_loss[valid_idx].topk(3, dim=-1).indices  # (N, 3)
                targets_exp = targets_flat[valid_idx].unsqueeze(-1)  # (N, 1)
                accuracy_top3 = (top3_preds == targets_exp).any(dim=-1).float().mean()
        else:
            if use_stratified:
                subset_idx, gt_subset_pos = self._sample_decoy_subset(targets_flat, valid_mask=None)
                decoy_subset_idx, decoy_gt_pos = subset_idx, gt_subset_pos
                decoy_valid_idx = torch.arange(targets_flat.shape[0], device=logits.device)
                subset_logits = torch.gather(logits_for_loss, dim=1, index=subset_idx)
                loss = F.cross_entropy(subset_logits, gt_subset_pos)
                preds = subset_logits.argmax(dim=-1)
                accuracy = (preds == gt_subset_pos).float().mean()
                top3_preds = subset_logits.topk(min(3, subset_logits.shape[-1]), dim=-1).indices
                accuracy_top3 = (top3_preds == gt_subset_pos.unsqueeze(-1)).any(dim=-1).float().mean()
            else:
                loss = F.cross_entropy(logits_for_loss, targets_flat)
                preds = logits_for_loss.argmax(dim=-1)
                accuracy = (preds == targets_flat).float().mean()
                top3_preds = logits_for_loss.topk(3, dim=-1).indices
                targets_exp = targets_flat.unsqueeze(-1)
                accuracy_top3 = (top3_preds == targets_exp).any(dim=-1).float().mean()

        return {
            "loss": loss,
            "logits": logits,
            "accuracy": accuracy,
            "accuracy_top3": accuracy_top3,
            # Decoy reuse hook (None unless the stratified train path ran). subset_idx: (N, n_decoys+1) TYPE
            # indices with the GT at column 0; gt_pos: (N,) GT column (0, or -100 for masked/holdout/>slot-budget);
            # valid_idx: (N,) flat indices into the (B*L) position axis.
            "decoy_subset_idx": decoy_subset_idx,
            "decoy_gt_pos": decoy_gt_pos,
            "decoy_valid_idx": decoy_valid_idx,
        }

    def forward_multi_count(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        target_indices: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        valid_residue_mask: torch.Tensor | None = None,
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
        predicted_element_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Multi-count discretization.

        Try {count-1, count, count+1} atoms per residue, pick whichever count
        best matches GT. Atom count becomes emergent from coord quality.
        """
        if valid_residue_mask is not None and seq_mask is not None:
            seq_mask = seq_mask & valid_residue_mask  # skip residues absent from the disc DB
        hard_mask = predicted_mask > 0.5  # (B, L, max_sc)
        max_sc = hard_mask.shape[-1]
        counts = hard_mask.sum(dim=-1)  # (B, L)

        # mask_minus: drop last occupied atom
        last_idx = (counts - 1).clamp(min=0).unsqueeze(-1)  # (B, L, 1)
        mask_minus = hard_mask.clone()
        mask_minus.scatter_(dim=-1, index=last_idx, value=False)
        # If count was 0, last_idx=0 and mask[0] was already False -> no-op

        # mask_plus: add first unoccupied atom
        next_idx = counts.clamp(max=max_sc - 1).unsqueeze(-1)  # (B, L, 1)
        mask_plus = hard_mask.clone()
        mask_plus.scatter_(dim=-1, index=next_idx, value=True)
        # If count was max_sc, next_idx=max_sc-1 which was already True -> no-op

        mask_current = hard_mask

        # Stack 3 variants along batch dim: (B*3, L, max_sc)
        masks_3 = torch.cat([mask_minus.float(), mask_current.float(), mask_plus.float()], dim=0)
        coords_3 = predicted_coords.repeat(3, 1, 1, 1)
        targets_3 = target_indices.repeat(3, 1)
        seq_mask_3 = seq_mask.repeat(3, 1) if seq_mask is not None else None
        bb_coords_3 = backbone_coords.repeat(3, 1, 1, 1) if backbone_coords is not None else None
        bb_mask_3 = backbone_mask.repeat(3, 1, 1) if backbone_mask is not None else None
        elem_3 = predicted_element_types.repeat(3, 1, 1) if predicted_element_types is not None else None

        # Run discretization once on tripled batch
        # NOTE: keyword args are REQUIRED here. forward()'s signature is
        # (coords, mask, target_indices, seq_mask, valid_residue_mask, backbone_coords,
        # backbone_mask, predicted_element_types); the previous positional call was shifted
        # by one -- it passed backbone_coords into valid_residue_mask (crashing whenever a
        # backbone was supplied) and silently DROPPED predicted_element_types, so element
        # mismatch penalties never applied in multi-count disc.
        # valid_residue_mask is already folded into seq_mask at the top of this method.
        result_3 = self.forward(
            coords_3,
            masks_3,
            targets_3,
            seq_mask=seq_mask_3,
            backbone_coords=bb_coords_3,
            backbone_mask=bb_mask_3,
            predicted_element_types=elem_3,
        )
        logits_3 = result_3["logits"]  # (B*3, L, R)

        batch_sz = predicted_coords.shape[0]
        seq_len = predicted_coords.shape[1]
        n_ref = logits_3.shape[-1]

        # Reshape to (3, B, L, R)
        logits_3 = logits_3.view(3, batch_sz, seq_len, n_ref)

        # Use raw logits for variant selection, then delegate final loss/acc to
        # _compute_loss_and_accuracy which handles rotamer->type aggregation correctly.
        # For selection: compute per-residue NLL using raw rotamer logits with
        # rotamer-mapped targets (any rotamer of the correct type is valid).
        if self._rotamer_to_type is not None:
            # Map type targets to best-matching rotamer per variant for selection
            # Use max-logit rotamer of the correct type as proxy
            selection_losses = []
            for v in range(3):
                # For each position, find the max logit among rotamers of the GT type
                # This is equivalent to type-aggregated CE with amax
                v_logits = logits_3[v]  # (B, L, 112)
                # Mask out rotamers not belonging to GT type
                gt_type = target_indices.unsqueeze(-1)  # (B, L, 1)
                rotamer_types = self._rotamer_to_type.view(1, 1, -1).expand_as(v_logits)  # (B, L, 112)
                type_match = (rotamer_types == gt_type).float()  # 1 where rotamer belongs to GT type
                # Max logit among correct-type rotamers (others masked to -inf)
                masked_logits = v_logits + (1 - type_match) * (-1e9)
                best_gt_logit = masked_logits.max(dim=-1).values  # (B, L)
                # Log-sum-exp over all rotamers as denominator
                log_denom = torch.logsumexp(v_logits, dim=-1)  # (B, L)
                nll = log_denom - best_gt_logit  # per-residue NLL
                selection_losses.append(nll)
            stacked_losses = torch.stack(selection_losses, dim=0)  # (3, B, L)
        else:
            # Simple CE per variant
            losses_per_variant = []
            for v in range(3):
                loss_v = F.cross_entropy(logits_3[v].reshape(-1, n_ref), target_indices.reshape(-1), reduction="none")
                losses_per_variant.append(loss_v.reshape(batch_sz, seq_len))
            stacked_losses = torch.stack(losses_per_variant, dim=0)  # (3, B, L)

        # Mask out invalid positions before selection
        if seq_mask is not None:
            stacked_losses = stacked_losses + (~seq_mask).float().unsqueeze(0) * 1e9

        # Pick best variant per residue
        best_variant = stacked_losses.argmin(dim=0)  # (B, L) in {0, 1, 2}

        # Gather best raw logits
        best_logits = logits_3.gather(
            0, best_variant.unsqueeze(0).unsqueeze(-1).expand(1, batch_sz, seq_len, n_ref)
        ).squeeze(0)  # (B, L, n_ref)

        # Delegate loss/accuracy to standard method (handles rotamer->type aggregation)
        num_residues = self._residue_database.shape[0]
        result = self._compute_loss_and_accuracy(best_logits, target_indices, seq_mask, num_residues)

        # Log which variant won: 0=minus, 1=current, 2=plus
        valid_variants = best_variant[seq_mask].float() if seq_mask is not None else best_variant.float()
        best_variant_mean = valid_variants.mean() if valid_variants.numel() > 0 else torch.tensor(1.0)

        result["best_variant_mean"] = best_variant_mean
        return result

    @torch.no_grad()
    def select_best_count(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
        predicted_element_types: torch.Tensor | None = None,
        max_plus: int = 1,
        ghost_weight: float = 0.0,
        return_atom_importance: bool = False,
        importance_top_k: int = 3,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Score {count-1, count, count+1} variants and return best count per residue.

        Uses max logit across all reference residues (best geometric match) to select.
        No GT targets needed -- purely geometric scoring.

        Parameters
        ----------
        ghost_weight : float
            Weight for ghost (PAD) atoms in the NDM. When 0.0 (default), ghost atoms
            are hard-masked out. When > 0, ghost atoms contribute to the NDM at this
            weight, matching the training-time ghost_weight behavior.
        return_atom_importance : bool
            If True, also return per-atom importance scores from the NDM.

        Returns
        -------
        best_variant : torch.Tensor
            (B, L) tensor with values in {0, ..., n_variants-1} mapping to
            deltas {-1, 0, +1, ..., +max_plus}.
        atom_importance : torch.Tensor (only if return_atom_importance=True)
            (B, L, max_sc) per-atom importance scores from the winning variant's NDM.
        """
        hard_mask = predicted_mask > 0.5
        max_sc = hard_mask.shape[-1]
        counts = hard_mask.sum(dim=-1)

        # Build mask variants for deltas -1, 0, +1, ..., +max_plus
        # When ghost_weight > 0, PAD slots get soft weight instead of hard 0
        deltas = list(range(-1, max_plus + 1))
        n_variants = len(deltas)
        mask_variants = []
        for delta in deltas:
            if delta == 0:
                m_float = hard_mask.float()
            elif delta == -1:
                last_idx = (counts - 1).clamp(min=0).unsqueeze(-1)
                m = hard_mask.clone()
                m.scatter_(dim=-1, index=last_idx, value=False)
                m_float = m.float()
            else:
                m = hard_mask.clone()
                for _i in range(delta):
                    occupied = m.sum(dim=-1)
                    next_idx = occupied.long().clamp(max=max_sc - 1).unsqueeze(-1)
                    m.scatter_(dim=-1, index=next_idx, value=True)
                m_float = m.float()
            # Apply ghost_weight: PAD slots (mask=0) get ghost_weight instead of 0
            if ghost_weight > 0:
                m_float = m_float + ghost_weight * (1.0 - m_float)
            mask_variants.append(m_float)

        # Stack ALL variants along the batch dim: (n_variants*B, L, max_sc).
        # NOTE: this replaces a hard-coded 3-variant block that survived the
        # generalization to arbitrary max_plus and referenced names (mask_minus,
        # logits_3) that no longer exist -- select_best_count raised NameError on
        # every call until this was completed.
        masks_n = torch.cat(mask_variants, dim=0)
        coords_n = predicted_coords.repeat(n_variants, 1, 1, 1)
        batch_sz, seq_len = predicted_coords.shape[:2]
        # Dummy targets (not used for selection, only for forward() API)
        dummy_targets = torch.zeros(batch_sz * n_variants, seq_len, dtype=torch.long, device=predicted_coords.device)
        seq_mask_n = seq_mask.repeat(n_variants, 1) if seq_mask is not None else None
        bb_coords_n = backbone_coords.repeat(n_variants, 1, 1, 1) if backbone_coords is not None else None
        bb_mask_n = backbone_mask.repeat(n_variants, 1, 1) if backbone_mask is not None else None
        elem_n = predicted_element_types.repeat(n_variants, 1, 1) if predicted_element_types is not None else None

        # Keyword args required -- same one-off positional shift as in forward_multi_count.
        result_n = self.forward(
            coords_n,
            masks_n,
            dummy_targets,
            seq_mask=seq_mask_n,
            backbone_coords=bb_coords_n,
            backbone_mask=bb_mask_n,
            predicted_element_types=elem_n,
            return_atom_importance=return_atom_importance,
            importance_top_k=importance_top_k,
        )
        logits_n = result_n["logits"].view(n_variants, batch_sz, seq_len, -1)  # (n_variants, B, L, R)

        # Score each variant by best geometric match (max logit across all references)
        # If using rotamer database, aggregate by type first
        if self._rotamer_to_type is not None:
            n_types = self._num_types
            scores = []
            for v in range(n_variants):
                v_logits = logits_n[v]  # (B, L, R_rotamers)
                # Max logit per amino acid type, then max across types
                type_max = torch.full((batch_sz, seq_len, n_types), -1e9, device=v_logits.device)
                type_max.scatter_reduce_(
                    2,
                    self._rotamer_to_type.view(1, 1, -1).expand(batch_sz, seq_len, -1),
                    v_logits,
                    reduce="amax",
                )
                scores.append(type_max.max(dim=-1).values)  # (B, L)
            stacked_scores = torch.stack(scores, dim=0)  # (n_variants, B, L)
        else:
            stacked_scores = logits_n.max(dim=-1).values  # (n_variants, B, L)

        if seq_mask is not None:
            stacked_scores = stacked_scores + (~seq_mask).float().unsqueeze(0) * (-1e9)

        best_variant = stacked_scores.argmax(dim=0)  # (B, L) in {0, ..., n_variants-1}

        if return_atom_importance and "atom_importance" in result_n:
            # atom_importance from forward: (B*n_variants, L, max_atoms)
            # Extract importance for the winning variant per residue
            imp_all = result_n["atom_importance"].view(n_variants, batch_sz, seq_len, -1)
            # Gather the winning variant's importance per residue
            max_atoms = imp_all.shape[-1]
            atom_importance = imp_all.gather(
                0, best_variant.unsqueeze(0).unsqueeze(-1).expand(1, batch_sz, seq_len, max_atoms)
            ).squeeze(0)  # (B, L, max_atoms)
            return best_variant, atom_importance

        return best_variant
