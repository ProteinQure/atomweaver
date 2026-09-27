"""Normalized distance-matrix residue scoring for inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class GeometricMatcher(nn.Module):
    """Score side-chain clouds against reference rotamers and aggregate by residue type."""

    def __init__(
        self,
        residue_database: torch.Tensor,
        residue_masks: torch.Tensor,
        backbone_indices: torch.Tensor | None = None,
        element_types: torch.Tensor | None = None,
        rotamer_to_type: torch.Tensor | None = None,
        atom_mismatch_penalty: float = 0.5,
        element_mismatch_penalty: float = 0.3,
        chirality_mismatch_penalty: float = 50.0,
        repack_prediction: bool = True,
    ):
        super().__init__()
        self.include_backbone = backbone_indices is not None
        self.use_backbone_for_alignment_only = True
        self.temperature = 1.0
        self.atom_mismatch_penalty = atom_mismatch_penalty
        self.element_mismatch_penalty = element_mismatch_penalty
        self.chirality_mismatch_penalty = chirality_mismatch_penalty
        self.repack_prediction = repack_prediction
        self.register_buffer("_residue_database", residue_database)
        self.register_buffer("_residue_masks", residue_masks)
        self.register_buffer("_rotamer_to_type", rotamer_to_type)
        self._num_types = int(rotamer_to_type.max()) + 1 if rotamer_to_type is not None else len(residue_database)
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
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
        predicted_element_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores indexed by residue type, without target labels or training losses."""
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

        rotamer_scores = self._forward_normalized(
            predicted_coords,
            predicted_mask,
            predicted_element_types,
            pred_sign=pred_sign,
            pred_conf=pred_conf,
        )

        if self._rotamer_to_type is None:
            return rotamer_scores
        scores = torch.full(
            (*rotamer_scores.shape[:-1], self._num_types),
            float("-inf"),
            dtype=rotamer_scores.dtype,
            device=rotamer_scores.device,
        )
        indices = self._rotamer_to_type.view(1, 1, -1).expand_as(rotamer_scores)
        scores.scatter_reduce_(2, indices, rotamer_scores, reduce="amax", include_self=True)
        return scores

    def _forward_normalized(
        self,
        predicted_coords: torch.Tensor,
        predicted_mask: torch.Tensor,
        predicted_element_types: torch.Tensor | None = None,
        pred_sign: torch.Tensor | None = None,
        pred_conf: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        return logits
