"""Residue-level frame-aware stream + coarse side-chain latents (FIRST-CUT DRAFT).

  DRAFT / FIRST CUT -- needs architectural review before any production run.

Motivation
----------
The main model (:class:`~atomweaver.joint_diffusion.models.InverseFoldingDiffusion`) is
*atom-level*: an SE(3) transformer places side-chain atoms as a point cloud. This module bolts
on a small, parallel **residue-level** stream that reasons about each residue as a single node
carrying its local backbone frame, and predicts a *coarse* summary of the side chain it should
grow (centroid direction, radial extent, atom count). Two losses keep the stream honest:

* a **direct GT-supervision** loss (predicted coarse latents vs. the GT side-chain summaries),
  which keeps the stream a live, trained representation rather than dead decoration; and
* a **consistency** loss (predicted coarse latents vs. the SAME summaries recomputed from the
  atom cloud's clean-endpoint estimate ``x̂0``), which couples the two levels.

The residue latent is projected (zero-init) and added back into the per-residue features the
atom denoiser consumes -- exactly mirroring the ``burial_mlp`` / ``dihedral_mlp`` zero-init
additive grafts, so the model is **byte-identical when the feature is off** and resume-safe.

What is deliberately simplified (first cut -- flag for review)
-------------------------------------------------------------
* **NOT a full SE(3)-equivariant residue tower.** Equivariance is obtained the cheap way: all
  pairwise geometry is expressed in each residue's *local* backbone frame (an invariant), and
  attention runs on those invariants. There is no type-1 (vector) feature track and no
  re-derivation of the atom-level SE(3) machinery. Rotating the whole complex leaves the local
  frames' relative coordinates unchanged, so the stream is invariant -- but it cannot *emit* a
  vector that co-rotates without going back through the frame (which the centroid head does, by
  predicting in-frame and letting the loss rotate GT into-frame too).
* **Dense O(L²) residue attention**, masked by ``seq_mask``. Peptides are short (L ~ 15-40), so
  no kNN sparsification for the first cut.
* **Three geometry summaries only**: centroid vector (in local frame), mean radial extent, atom
  count. No per-atom or per-element structure.
* **Coarse count in the consistency loss is read off the atom track's predicted existence**
  (detached), so the consistency term trains the residue-stream count head without perturbing
  the fragile existence axis. Centroid/extent consistency *does* flow gradient into ``x̂0``.

Everything here is only constructed / run when ``use_residue_frame_stream`` is on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# N / CA / C / O ordering of the backbone atom axis (B, L, 4, 3). Matches the convention used
# throughout models.py (see BackboneEncoder._backbone_dihedral_features / _burial_features,
# which read N=index 0, CA=index 1, C=index 2).
_N_IDX, _CA_IDX, _C_IDX = 0, 1, 2

# -1 (graft-safety): strong-negative L-prior init for the stereochem head's OUTPUT bias so an
# UNTRAINED head emits P(D) = sigmoid(bias) ≈ sigmoid(-4) ≈ 0.018 ≈ 0 -> L-leaning from step 0. This is
# belt-and-suspenders for the gated cone-init: a fresh graft (random-init stereo head) sampled at
# inference (where the head-first ramp is at s=1, full gating) would otherwise flatten every cone
# in-plane (P(D)≈0.5). Only the final output bias is set; the weight stays at normal Linear init.
_STEREO_HEAD_LPRIOR_BIAS = -4.0


def build_local_frames(
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-residue orthonormal local backbone frame from N, CA, C (Gram-Schmidt).

    Convention
    ----------
    * ``e1 ∝ C - CA`` (first basis vector along the CA->C bond),
    * ``e2 ∝ (N - CA) - ((N - CA)·e1) e1`` (component of CA->N orthogonal to e1),
    * ``e3 = e1 × e2``.

    The returned rotation ``R`` has these basis vectors as its **columns**
    (``R[..., :, 0] = e1``), so ``R`` maps a *local* vector to the *global* frame
    (``global = R @ local``) and its transpose expresses a global vector in the local frame
    (``local = Rᵀ @ (global - CA)``). ``R`` is orthonormal: ``R @ Rᵀ ≈ I``.

    Parameters
    ----------
    backbone_coords : torch.Tensor
        Backbone atom coordinates of shape (B, L, 4, 3), N/CA/C/O ordering.
    backbone_mask : torch.Tensor, optional
        Validity mask of shape (B, L, 4). Residues missing any of N/CA/C get the identity frame
        (a clean, well-defined fallback; those residues are excluded from the losses anyway).
    eps : float
        Numerical floor for the normalisations.

    Returns
    -------
    R : torch.Tensor
        Per-residue rotation of shape (B, L, 3, 3), basis vectors as columns.
    origin : torch.Tensor
        Per-residue frame origin = CA, shape (B, L, 3).
    """
    n = backbone_coords[:, :, _N_IDX, :]
    ca = backbone_coords[:, :, _CA_IDX, :]
    c = backbone_coords[:, :, _C_IDX, :]

    e1 = c - ca
    e1 = e1 / (e1.norm(dim=-1, keepdim=True) + eps)

    v = n - ca
    v = v - (v * e1).sum(dim=-1, keepdim=True) * e1
    e2 = v / (v.norm(dim=-1, keepdim=True) + eps)

    e3 = torch.cross(e1, e2, dim=-1)

    # Columns = basis vectors -> R maps local->global; R^T maps global->local.
    R = torch.stack([e1, e2, e3], dim=-1)  # (B, L, 3, 3)

    if backbone_mask is not None:
        res_valid = backbone_mask[:, :, :3].all(dim=-1)  # (B, L) -- need N, CA, C
        ident = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)
        R = torch.where(res_valid[..., None, None], R, ident)

    return R, ca


def sidechain_summaries_in_frame(
    coords: torch.Tensor,
    atom_weight: torch.Tensor,
    R: torch.Tensor,
    ca: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarse side-chain summaries for a (weighted) atom cloud, expressed in the local frame.

    Used for BOTH the GT summaries (``atom_weight`` = GT ``sidechain_mask``) and the atom-cloud
    ``x̂0`` summaries (same mask) -- identical reduction, so the consistency loss compares
    like-with-like.

    Parameters
    ----------
    coords : torch.Tensor
        Atom coordinates (global), shape (B, L, K, 3).
    atom_weight : torch.Tensor
        Per-atom soft/hard weight (real=1, ghost/pad=0), shape (B, L, K). Non-negative.
    R : torch.Tensor
        Per-residue local frame, shape (B, L, 3, 3) (columns = basis vectors).
    ca : torch.Tensor
        Per-residue frame origin (CA), shape (B, L, 3).

    Returns
    -------
    centroid_local : torch.Tensor
        Weighted-mean side-chain position minus CA, rotated into the local frame, (B, L, 3).
    radial_extent : torch.Tensor
        Weighted-mean distance of side-chain atoms from CA (Å), (B, L).
    count : torch.Tensor
        Sum of ``atom_weight`` over atoms = (soft) atom count, (B, L).
    """
    w = atom_weight.clamp(min=0.0)
    wsum = w.sum(dim=-1)  # (B, L)
    denom = wsum.clamp(min=eps).unsqueeze(-1)  # (B, L, 1)

    offset = coords - ca.unsqueeze(2)  # (B, L, K, 3) atom - CA
    centroid_global = (offset * w.unsqueeze(-1)).sum(dim=2) / denom  # (B, L, 3)
    # Express in local frame: local = R^T @ global_offset (R columns are the basis vectors).
    centroid_local = torch.einsum("blij,blj->bli", R.transpose(-1, -2), centroid_global)

    dist = offset.norm(dim=-1)  # (B, L, K) per-atom |atom - CA|
    radial_extent = (dist * w).sum(dim=-1) / wsum.clamp(min=eps)  # (B, L)

    return centroid_local, radial_extent, wsum


class _FrameAttentionLayer(nn.Module):
    """One layer of frame-relative multi-head self-attention over residues (pre-norm).

    Standard scaled-dot-product attention with an additive, per-head bias derived from the
    neighbour's Cα position expressed in the query residue's local frame -- the geometry enters
    only through that invariant, keeping the block SE(3)-invariant.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        # Relative-position (in local frame) -> per-head scalar attention bias.
        self.rel_bias = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        rel_local: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Residue features, (B, L, hidden_dim).
        rel_local : torch.Tensor
            Neighbour Cα in query's local frame, (B, L, L, 3) indexed [b, i(query), j(key)].
        key_mask : torch.Tensor
            Valid-key (residue) mask, (B, L) bool.
        """
        b, length, _ = h.shape
        hn = self.norm1(h)
        q = self.q_proj(hn).view(b, length, self.n_heads, self.head_dim)
        k = self.k_proj(hn).view(b, length, self.n_heads, self.head_dim)
        v = self.v_proj(hn).view(b, length, self.n_heads, self.head_dim)

        # (B, i, j, heads)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(self.head_dim)
        attn = attn + self.rel_bias(rel_local)

        neg_inf = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(~key_mask[:, None, :, None], neg_inf)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(b, length, -1)
        attn_out = self.out_proj(out)
        # If a query's ENTIRE key set is masked, softmax over an all-(-inf) row returns a UNIFORM
        # distribution over invalid keys (not zeros/NaN), so the query would pool junk values.
        # Explicitly zero that query's attention contribution -> "no valid context, no update".
        # (The key set is identical for every query in a batch; zeroing is invariance-preserving.)
        has_key = key_mask.any(dim=-1)  # (B,)
        attn_out = attn_out * has_key[:, None, None].to(attn_out.dtype)
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h


class ResidueFrameStream(nn.Module):
    """Residue-level frame-aware transformer + coarse-latent heads + zero-init injection.

    FIRST-CUT DRAFT. See module docstring for the simplifications made vs. a full SE(3) tower.

    Forward returns the three predicted coarse latents (for the supervision + consistency losses,
    computed in ``InverseFoldingDiffusion.forward``) and a zero-init additive residual to inject
    into ``backbone_features``.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        rel_pos_scale: float = 10.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rel_pos_scale = float(rel_pos_scale)

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(_FrameAttentionLayer(hidden_dim, n_heads) for _ in range(n_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Coarse side-chain latent heads (predicted in the LOCAL frame).
        self.centroid_head = nn.Linear(hidden_dim, 3)  # centroid vector in local frame
        self.radial_head = nn.Linear(hidden_dim, 1)  # mean radial extent from CA (Å)
        self.count_head = nn.Linear(hidden_dim, 1)  # atom count

        # Zero-init injection back into the atom stream's per-residue features. Identity-at-init:
        # only this final projection is zeroed, so the added residual is exactly 0 at graft (byte-
        # identical resume) while gradient still reaches it (NOT a zero-init-both-ends deadlock).
        self.inject_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.inject_proj.weight)
        nn.init.zeros_(self.inject_proj.bias)

    def forward(
        self,
        backbone_features: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        seq_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        backbone_features : torch.Tensor
            Pocket-conditioned per-residue features from the encoder, (B, L, hidden_dim).
        backbone_coords : torch.Tensor
            Backbone atoms, (B, L, 4, 3).
        backbone_mask : torch.Tensor
            Backbone atom validity, (B, L, 4).
        seq_mask : torch.Tensor, optional
            Valid-residue mask, (B, L). None => all valid.

        Returns
        -------
        dict with keys:
            ``rf_centroid_pred`` (B, L, 3), ``rf_radial_pred`` (B, L),
            ``rf_count_pred`` (B, L), ``rf_injection`` (B, L, hidden_dim),
            ``rf_hidden`` (B, L, hidden_dim) -- the post-trunk, pre-output-head per-residue latent,
            exposed so the atom denoiser can OPTIONALLY do deep per-layer injection (each SE(3) layer
            adds a zero-init-projected residual of this latent, mapped residue->atom). This is an
            SE(3)-invariant tensor (the stream reasons only over local-frame invariants), which is why
            adding it to the atom transformer's type-0/invariant node features preserves equivariance.
        """
        b, length = backbone_features.shape[:2]
        device = backbone_features.device
        if seq_mask is None:
            key_mask = torch.ones(b, length, dtype=torch.bool, device=device)
        else:
            key_mask = seq_mask.bool()

        R, ca = build_local_frames(backbone_coords, backbone_mask)  # (B, L, 3, 3), (B, L, 3)

        # Neighbour Cα expressed in the query residue's local frame:
        # rel_local[b, i, j] = R_iᵀ @ (CA_j - CA_i).
        diff = ca[:, None, :, :] - ca[:, :, None, :]  # (B, i, j, 3): CA_j - CA_i
        rel_local = torch.einsum("blij,blmj->blmi", R.transpose(-1, -2), diff)  # (B, i, j, 3)
        rel_local = rel_local / self.rel_pos_scale

        h = self.input_norm(backbone_features)
        for layer in self.layers:
            h = layer(h, rel_local, key_mask)
        h = self.output_norm(h)

        return {
            "rf_centroid_pred": self.centroid_head(h),  # (B, L, 3)
            "rf_radial_pred": self.radial_head(h).squeeze(-1),  # (B, L)
            "rf_count_pred": self.count_head(h).squeeze(-1),  # (B, L)
            "rf_injection": self.inject_proj(h),  # (B, L, hidden_dim), exactly 0 at init
            "rf_hidden": h,  # (B, L, hidden_dim) post-trunk latent for optional deep per-layer inject
        }


# ==============================================================================================
# v2 -- residue-frame graph over BINDER + TARGET residues, ORIENTATION-ONLY heads.
# ==============================================================================================
#
# Chunk 3 rebuild of the residue-level stream. Two changes vs. v1 (:class:`ResidueFrameStream`),
# nothing else about the invariance argument moves:
#
# 1. **Target residues enter the graph as extra attention KEYS.** Each binder residue (query)
# now attends to *both* the other binder residues *and* the target residues. A target
# residue's "position" is its Cα and its feature is a learned embedding of its residue type
# (mirroring :class:`TargetEncoder`'s ``residue_embed``). This is still SE(3)-**invariant**,
# the exact same way v1 is: the only geometry that enters attention is the neighbour Cα
# expressed in the *query's local backbone frame* -- ``Rᵢᵀ(CAⱼ - CAᵢ)`` -- which is unchanged
# by any global rotation/translation of the whole complex. Adding target keys just extends
# ``j`` over the target residues; their relative-position bias is computed with the identical
# into-frame rotation, so the block stays invariant with target nodes added. Target nodes are
# **keys only** (one-way: binder attends to target); they are never updated.
#
# 2. **Orientation-only heads.** v2 predicts ONLY the side-chain *orientation*:
# * ``centroid_head`` -- the in-frame centroid vector (direction + magnitude), kept from v1;
# * ``chi1_head`` -- the χ1 rotation (Cβ->Cγ about the Cα-Cβ axis), represented as a
# singularity-free 2-vector ``(cos χ1, sin χ1)`` (2 logits, L2-normalized).
# The v1 ``radial_head`` and ``count_head`` are **dropped** -- extent/count are owned by the
# volumetric head, not this stream.
#
# Input source is a flag (``clean_input``):
# * ``True`` (default, the version we want to test): a **clean, self-contained** binder-residue
# embedding built ONLY from invariant backbone geometry (the N/C/O positions in each residue's
# own local frame) plus a single learned residue token -- it does NOT read the upstream
# pocket-conditioned ``backbone_features``. This isolates what the residue graph can learn from
# geometry + target alone.
# * ``False``: fall back to the detached pocket-conditioned ``backbone_features`` (v1 behaviour),
# for a safe/comparable ablation. Tradeoff: the clean input is a purer test of the graph but
# forgoes the encoder's learned pocket context; the pocket-conditioned input is stronger but
# entangles the stream with the base encoder.
#
# Everything is constructed / run only under ``use_residue_frame_stream_v2``; byte-identical off.


def chi1_cos_sin_from_coords(
    n: torch.Tensor,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cg: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """χ1 dihedral (N-Cα-Cβ-Cγ) as a unit ``(cos χ1, sin χ1)`` 2-vector.

    A dihedral angle is invariant to any global rigid motion of its four points, so this GT
    quantity is trivially "in-frame" -- no rotation into the local frame is needed, and it can be
    compared directly against the stream's in-frame ``chi1_head`` prediction.

    Parameters
    ----------
    n, ca, cb, cg : torch.Tensor
        The four dihedral atoms, each (..., 3): backbone N, backbone Cα, side-chain Cβ, side-chain
        Cγ. (In practice, under the reserved-slot0 layout, Cβ = side-chain slot 1 and Cγ ≈ slot 2 --
        slot 0 is the N-connecting atom, ghost for standard residues; see the χ1-loss docstring in
        ``models.py``.)

    Returns
    -------
    cos_sin : torch.Tensor
        ``stack([cos χ1, sin χ1], dim=-1)``, shape (..., 2), unit-norm where the geometry is
        non-degenerate.
    """
    b0 = ca - n
    b1 = cb - ca
    b2 = cg - cb

    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + eps)
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    n1 = n1 / (n1.norm(dim=-1, keepdim=True) + eps)
    n2 = n2 / (n2.norm(dim=-1, keepdim=True) + eps)
    m1 = torch.cross(n1, b1n, dim=-1)

    cos_chi = (n1 * n2).sum(dim=-1)
    sin_chi = (m1 * n2).sum(dim=-1)
    return torch.stack([cos_chi, sin_chi], dim=-1)


class _FrameCrossAttentionLayer(nn.Module):
    """One frame-relative attention layer where binder QUERIES attend to binder+target KEYS.

    Identical mechanism to :class:`_FrameAttentionLayer` -- standard scaled-dot-product attention
    with an additive, per-head bias derived from the neighbour Cα expressed in the query residue's
    local frame -- but the key/value set is the concatenation ``[binder residues, target residues]``.
    Binder query features are updated; the (static) target key features are passed through
    unchanged (target nodes are keys only). Geometry enters ONLY through the invariant in-frame
    relative position, so the block is SE(3)-invariant with target nodes added.
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_k = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rel_bias = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_heads),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h_query: torch.Tensor,
        key_feats: torch.Tensor,
        rel_local: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_query : torch.Tensor
            Binder residue (query) features, (B, L, hidden_dim).
        key_feats : torch.Tensor
            combined key features ``[binder, target]``, (B, Lk, hidden_dim) with Lk = L + L_t.
        rel_local : torch.Tensor
            Key Cα in query's local frame, (B, L, Lk, 3), indexed [b, i(query), j(key)].
        key_mask : torch.Tensor
            Valid-key mask over the combined key set, (B, Lk) bool.
        """
        b, length, _ = h_query.shape
        lk = key_feats.shape[1]
        qn = self.norm_q(h_query)
        kn = self.norm_k(key_feats)
        q = self.q_proj(qn).view(b, length, self.n_heads, self.head_dim)
        k = self.k_proj(kn).view(b, lk, self.n_heads, self.head_dim)
        v = self.v_proj(kn).view(b, lk, self.n_heads, self.head_dim)

        # (B, i(query), j(key), heads)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(self.head_dim)
        attn = attn + self.rel_bias(rel_local)

        neg_inf = torch.finfo(attn.dtype).min
        attn = attn.masked_fill(~key_mask[:, None, :, None], neg_inf)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(b, length, -1)
        attn_out = self.out_proj(out)
        # If a query's ENTIRE key set (binder+target) is masked, softmax over an all-(-inf) row
        # returns a UNIFORM distribution over invalid keys (not zeros/NaN), so the query would pool
        # junk values. Explicitly zero that query's attention contribution -> "no valid context, no
        # update". (The key set is identical for every query in a batch; zeroing preserves invariance.)
        has_key = key_mask.any(dim=-1)  # (B,)
        attn_out = attn_out * has_key[:, None, None].to(attn_out.dtype)
        h_query = h_query + attn_out
        h_query = h_query + self.ffn(self.norm2(h_query))
        return h_query


class ResidueFrameStreamV2(nn.Module):
    """v2 residue-frame graph -- binder+target residues, orientation-only heads (SE(3)-invariant).

    See the section banner above for the full design. Forward returns the two orientation
    predictions (centroid-in-frame + χ1 (cos,sin)) for the losses in
    ``InverseFoldingDiffusion.forward`` and a zero-init additive residual (byte-identical at graft).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 2,
        n_heads: int = 4,
        rel_pos_scale: float = 10.0,
        clean_input: bool = True,
        num_residue_types: int = 21,
        use_stereochem_head: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rel_pos_scale = float(rel_pos_scale)
        self.clean_input = bool(clean_input)
        self.use_stereochem_head = bool(use_stereochem_head)

        # Binder query input embedding.
        if self.clean_input:
            # Clean, self-contained: invariant backbone geometry (N/C/O in the local frame, 9 dims)
            # + a single learned residue token. Does NOT read pocket-conditioned backbone_features.
            self.binder_geom_mlp = nn.Sequential(
                nn.Linear(9, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.binder_token = nn.Parameter(torch.zeros(hidden_dim))
        # else: fall back to the (detached) pocket-conditioned backbone_features passed to forward.
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Target key features: learned residue-type embedding -> hidden (mirrors TargetEncoder).
        self.target_embed = nn.Embedding(num_residue_types, hidden_dim)
        self.num_residue_types = int(num_residue_types)

        self.layers = nn.ModuleList(_FrameCrossAttentionLayer(hidden_dim, n_heads) for _ in range(n_layers))
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Orientation-only heads (predicted in the LOCAL frame / as an invariant dihedral).
        self.centroid_head = nn.Linear(hidden_dim, 3)  # in-frame centroid vector
        self.chi1_head = nn.Linear(hidden_dim, 2)  # (cos χ1, sin χ1) logits, L2-normalized on output

        # supervised stereochemistry (e3-sign) head. Predicts a SINGLE logit per residue =
        # P(D) = P(e3>0): label 1 = D, canonical L (e3<0) -> 0. e3 is the out-of-plane component of Cβ in
        # the residue's local backbone frame (e3 = e1×e2 is the backbone-plane normal; L vs D flips which
        # face Cβ sits on). Trained by its own BCE, BUT (post-Chunk-4a) its P(D) is NO LONGER
        # generation-inert: it now feeds the SOURCE DISTRIBUTION via the gated cone-init
        # (`_stereochem_gated_pd` chooses the L vs D cone axis for the shell prior) AND the COORD FLOW via
        # the pieces-2a/2b t-resolution feedback (the evolving P(D)_t steers atoms toward the resolved
        # face). It still does NOT feed the element/existence tracks. It reads the post-trunk per-residue
        # latent AND (when the
        # volumetric head is on) a projection of the volumetric latent `vol_hidden` as an auxiliary input:
        # the volumetric head predicts occupancy on BOTH faces, so which face it filled is a stereochem
        # prior. Built ONLY when opted in => no params / no RNG advance when off (byte-identical). Head
        # input is ALWAYS [h, vol_feat] (2*hidden): vol_feat = vol_proj(vol_hidden) when a latent is
        # supplied, else zeros -- so the head runs from frame features alone when volumetric is off.
        if self.use_stereochem_head:
            self.stereo_vol_proj = nn.Linear(hidden_dim, hidden_dim)
            self.stereo_head = nn.Linear(2 * hidden_dim, 1)
            # L-prior init: strong-negative OUTPUT bias so an untrained head gives P(D)≈0 (L-leaning) at
            # step 0, keeping the gated cone-init's inference path (s=1) on the L cone on a FRESH graft
            # rather than flattening it (P(D)≈0.5). Weight stays at normal init (only the bias is set).
            nn.init.constant_(self.stereo_head.bias, _STEREO_HEAD_LPRIOR_BIAS)

        # Zero-init injection back into the atom stream's per-residue features (v1 convention).
        self.inject_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.inject_proj.weight)
        nn.init.zeros_(self.inject_proj.bias)

    def forward(
        self,
        backbone_features: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        seq_mask: torch.Tensor | None,
        target_backbone_coords: torch.Tensor | None = None,
        target_backbone_mask: torch.Tensor | None = None,
        target_residue_types: torch.Tensor | None = None,
        target_seq_mask: torch.Tensor | None = None,
        vol_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        backbone_features : torch.Tensor
            Pocket-conditioned per-residue features, (B, L, hidden_dim). Used as the binder input
            ONLY when ``clean_input=False``; otherwise ignored (clean geometry embedding is built
            from ``backbone_coords``).
        vol_hidden : torch.Tensor, optional
            Volumetric-head per-residue latent, (B, L, hidden_dim), read ONLY by the
            stereochemistry head as an auxiliary input (already DETACHED by the caller for
            graft-safety). None => the stereochem head falls back to frame features alone (its vol
            branch is zeros). Ignored entirely when ``use_stereochem_head`` is off.
        backbone_coords, backbone_mask : torch.Tensor
            Binder backbone atoms (B, L, 4, 3) and validity (B, L, 4).
        seq_mask : torch.Tensor, optional
            Valid binder-residue mask, (B, L). None => all valid.
        target_backbone_coords, target_backbone_mask : torch.Tensor, optional
            Target backbone atoms (B, L_t, 4, 3) and validity (B, L_t, 4). Cα (index 1) is the
            target residue "position". None => no target keys (self-attention over binder only).
        target_residue_types : torch.Tensor, optional
            Target residue type indices, (B, L_t) long. None => no target keys.
        target_seq_mask : torch.Tensor, optional
            Valid target-residue mask, (B, L_t). None => all target residues valid.

        Returns
        -------
        dict with keys:
            ``rfv2_centroid_pred`` (B, L, 3), ``rfv2_chi1_pred`` (B, L, 2) unit ``(cos,sin)``,
            ``rf_injection`` (B, L, hidden_dim) exactly 0 at init, ``rf_hidden`` (B, L, hidden_dim),
            ``rfv2_stereo_logit`` (B, L) one logit = P(D) = P(e3>0), label 1 = D, canonical L (e3<0) -> 0
            (None unless ``use_stereochem_head``).
        """
        b, length = backbone_coords.shape[:2]
        device = backbone_coords.device
        if seq_mask is None:
            binder_mask = torch.ones(b, length, dtype=torch.bool, device=device)
        else:
            binder_mask = seq_mask.bool()

        R, ca = build_local_frames(backbone_coords, backbone_mask)  # (B, L, 3, 3), (B, L, 3)

        # --- Binder query input embedding ---
        if self.clean_input:
            # N/C/O positions relative to CA, rotated into each residue's local frame -> invariant.
            nco = backbone_coords[:, :, [_N_IDX, _C_IDX, 3], :]  # (B, L, 3, 3): N, C, O
            nco_off = nco - ca.unsqueeze(2)  # (B, L, 3, 3) atom - CA (global)
            nco_local = torch.einsum("blij,blaj->blai", R.transpose(-1, -2), nco_off)  # (B, L, 3, 3)
            geom = nco_local.reshape(b, length, 9)
            h = self.binder_geom_mlp(geom) + self.binder_token
        else:
            h = backbone_features
        h = self.input_norm(h)

        # --- combined key set: binder residues + target residues ---
        # Binder self-keys: relative in-frame position Rᵢᵀ(CA_j - CA_i) for binder j.
        diff_bb = ca[:, None, :, :] - ca[:, :, None, :]  # (B, i, j, 3): CA_j^b - CA_i^b
        rel_bb = torch.einsum("blij,blmj->blmi", R.transpose(-1, -2), diff_bb)  # (B, i, j, 3)

        have_target = (
            target_backbone_coords is not None
            and target_residue_types is not None
            and target_backbone_coords.shape[1] > 0
        )
        if have_target:
            lt = target_backbone_coords.shape[1]
            target_ca = target_backbone_coords[:, :, _CA_IDX, :]  # (B, L_t, 3)
            # Rᵢᵀ(CA_t - CA_i^binder): target Cα in each binder query's local frame (invariant).
            diff_bt = target_ca[:, None, :, :] - ca[:, :, None, :]  # (B, i, t, 3)
            rel_bt = torch.einsum("blij,blmj->blmi", R.transpose(-1, -2), diff_bt)  # (B, i, t, 3)
            rel_local = torch.cat([rel_bb, rel_bt], dim=2)  # (B, L, L+L_t, 3)

            target_types = target_residue_types.long().clamp(0, self.num_residue_types - 1)
            target_feats = self.target_embed(target_types)  # (B, L_t, hidden)
            if target_seq_mask is not None:
                t_mask = target_seq_mask.bool()
            else:
                t_mask = torch.ones(b, lt, dtype=torch.bool, device=device)
            # A target residue can be present in target_seq_mask yet be missing its Cα (the atom
            # that defines its key "position"); such a residue is junk as an attention key. Gate the
            # key mask by per-atom backbone validity (CA = index 1, matching build_local_frames'
            # _CA_IDX). When target_backbone_mask is None, behaviour is unchanged.
            if target_backbone_mask is not None:
                t_mask = t_mask & target_backbone_mask[:, :, _CA_IDX].bool()
            target_feats = target_feats * t_mask.unsqueeze(-1).float()
            key_mask = torch.cat([binder_mask, t_mask], dim=1)  # (B, L+L_t)
        else:
            rel_local = rel_bb  # (B, L, L, 3)
            target_feats = h.new_zeros(b, 0, self.hidden_dim)
            key_mask = binder_mask

        rel_local = rel_local / self.rel_pos_scale

        for layer in self.layers:
            key_feats = torch.cat([h, target_feats], dim=1)  # (B, L+L_t, hidden); binder keys track h
            h = layer(h, key_feats, rel_local, key_mask)
        h = self.output_norm(h)

        chi1 = self.chi1_head(h)  # (B, L, 2)
        chi1 = F.normalize(chi1, dim=-1, eps=1e-8)  # unit (cos, sin)

        # supervised stereochemistry (e3-sign) head. Reads the post-trunk latent h plus a
        # projection of the (detached) volumetric latent when supplied; zeros otherwise (frame-only).
        stereo_logit = None
        if self.use_stereochem_head:
            if vol_hidden is not None:
                vol_feat = self.stereo_vol_proj(vol_hidden)
            else:
                vol_feat = h.new_zeros(b, length, self.hidden_dim)
            stereo_logit = self.stereo_head(torch.cat([h, vol_feat], dim=-1)).squeeze(-1)  # (B, L)

        return {
            "rfv2_centroid_pred": self.centroid_head(h),  # (B, L, 3)
            "rfv2_chi1_pred": chi1,  # (B, L, 2) unit (cos χ1, sin χ1)
            "rfv2_stereo_logit": stereo_logit,  # (B, L) P(D)=P(e3>0) logit (label 1=D), or None when head off
            "rf_injection": self.inject_proj(h),  # (B, L, hidden_dim), exactly 0 at init
            "rf_hidden": h,  # (B, L, hidden_dim) post-trunk latent for optional deep per-layer inject
        }
