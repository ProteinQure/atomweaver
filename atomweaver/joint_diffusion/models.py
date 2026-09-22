"""
Denoising models for SE(3)-equivariant diffusion.

This module implements the neural network architecture for denoising side-chain
atom clouds conditioned on fixed backbone coordinates and diffusion timestep.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Sequence

from .diffusion import ELEMENT_PAD, NUM_ELEMENT_TYPES, TimestepEmbedding
from .egnn import (
    EdgeTypeEmbedding,
    build_multi_type_radius_graph,
)
from .residue_frame_stream import (
    _STEREO_HEAD_LPRIOR_BIAS,
    ResidueFrameStreamV2,
    build_local_frames,
)

# number of in-frame e3 (out-of-plane) atom-cloud features the t-resolution head reads.
# The three moments of the real-atom-mask-weighted out-of-plane component e3 = eq3·(x_sc - CA):
# [signed mean, mean |e3|, mean e3²]. See SidechainDenoiser._forward_stereo_t_resolution.
_STEREO_T_ATOM_FEAT_DIM = 3
from .se3_transformer import SE3Transformer
from .volumetric_head import (
    VolumetricOccupancyHead,
    available_volume_cones,
)

# Volumetric-head parameter prefixes that are OPTIONAL / fresh grafts when warm-starting or freezing the
# head: a pretrained density head (e.g. no-sandclock or no-single-site) will NOT carry these, and they
# must be allowed to stay at their zero-init values (loader) and stay trainable (freeze). Scoped to EXACTLY
# these prefixes -- every OTHER head tensor stays strict-loaded and frozen as before. See the SANDCLOCK
# available-volume and SINGLE-SITE field features in volumetric_head.py.
_OPTIONAL_VOLUMETRIC_HEAD_KEY_PREFIXES = (
    "available_volume_proj.",
    "field_proj.",
    "field_to_hidden_proj.",
)


def _is_optional_volumetric_head_key(key: str) -> bool:
    """True iff ``key`` is an OPTIONAL/fresh-graft head parameter (may be absent from a pretrained head)."""
    return key.startswith(_OPTIONAL_VOLUMETRIC_HEAD_KEY_PREFIXES)


def build_kmask_ar_state(design_mask, seq_mask, max_sc: int, device) -> "torch.Tensor | None":
    """Per-residue denoiser state for K-mask inpainting (BUG-A fix, 2026-06-25).

    Tells the denoiser WHICH residues are designed (focus) vs clean revealed-context, so it conditions on
    the pinned clean context instead of "denoising" it (all residues share a single diffusion timestep t,
    and ``design_mask`` is not itself a denoiser feature). Reuses the existing ``ar_state`` channel.

    Returns a ``(B, L, max_sc)`` long tensor with: designed-in-a-partial-design-sample -> 1 (focus), valid
    context -> 2 (revealed clean), and **state 0 = inert sentinel** (no residual; see the residual site in the
    denoiser, which masks ``ar_state == 0``) for padding AND for every residue of a FULL-design sample.

    PER-SAMPLE parity (review HIGH): the "has context?" test is per-sample, not batch-wide. In a mixed batch
    (one full-design sample + one partial-design sample) the full-design sample's residues stay state 0 ->
    no residual -> identical to leg-A sampling (which passes no ``design_mask``/``ar_state``), regardless of
    its batch-mates. Returns ``None`` when NO sample has context (pure full-design batch) -- equivalent (all-0
    -> all masked) but skips the residual entirely.
    """
    dm = design_mask.to(device=device, dtype=torch.bool)  # (B, L) True = designed
    valid = seq_mask.to(device=device, dtype=torch.bool) if seq_mask is not None else torch.ones_like(dm)
    context = (~dm) & valid  # (B, L) True = clean revealed context
    if not bool(context.any()):
        return None  # no context anywhere (full-design batch): no residual needed
    has_context = context.any(dim=1, keepdim=True)  # (B, 1) True = partial-design sample
    ar = torch.zeros(dm.shape[0], dm.shape[1], max_sc, dtype=torch.long, device=device)  # 0 = inert sentinel
    # Focus ONLY for designed residues in partial-design samples; full-design samples stay 0 (per-sample parity).
    ar[((dm & valid) & has_context).unsqueeze(-1).expand(-1, -1, max_sc)] = 1  # designed (partial sample) -> focus
    ar[context.unsqueeze(-1).expand(-1, -1, max_sc)] = 2  # clean context -> revealed
    return ar


# Graft init for the glycine PAD gate (see InverseFoldingDiffusion.__init__ for WHY it is 0.0 and not
# -4.0). Shared by the fresh graft and the resume-time reset so the two can never drift apart.
GLYCINE_PAD_GATE_INIT = 0.0


def evc_from_element_state(element_types: torch.Tensor) -> torch.Tensor:
    """3-valued EVC existence from the discrete 2-track element state.

    real {C,N,O,X} -> 1.0 ; MASK (unknown/absorbing) -> 0.5 ; PAD (ghost) -> 0.0.
    This is the honest inference-time read: MASK is NOT conflated with PAD.

    Parameters
    ----------
    element_types : torch.Tensor
        Long tensor of discrete element ids (2-track), shape (..., max_sc).

    Returns
    -------
    torch.Tensor
        Float existence signal in {0.0, 0.5, 1.0} of the same shape as ``element_types``.
    """
    from .diffusion import ELEMENT_MASK, ELEMENT_PAD

    is_real = (element_types != ELEMENT_PAD) & (element_types != ELEMENT_MASK)
    is_mask = element_types == ELEMENT_MASK
    return is_real.float() + 0.5 * is_mask.float()


def apply_prefix_constraint(element_types: torch.Tensor, exempt_slot0: bool = False) -> torch.Tensor:
    """Enforce the hard prefix constraint: if slot i is PAD, all j>i are PAD.

    Real atoms must form a gap-free prefix. With reserved-slot0 the N-connecting slot 0 is
    usually PAD, which would wipe the whole residue; ``exempt_slot0=True`` treats slot 0 as an
    independent flag (contiguity applied over slots 1.. only), keeping slot 0's own value.
    """
    from .diffusion import ELEMENT_PAD

    is_not_pad = (element_types != ELEMENT_PAD).long()
    if exempt_slot0 and element_types.shape[-1] > 1:
        body_mask, _ = is_not_pad[..., 1:].cummin(dim=-1)
        prefix_mask = torch.cat([torch.ones_like(is_not_pad[..., :1]), body_mask], dim=-1)
    else:
        prefix_mask, _ = is_not_pad.cummin(dim=-1)
    return element_types * prefix_mask


# Number of standard amino acid types (20 canonical + unknown)
NUM_RESIDUE_TYPES = 21

RESIDUE_TO_IDX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
}


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors while avoiding division-by-zero."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def build_residue_frames(backbone_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-residue local frames from backbone atoms.

    Parameters
    ----------
    backbone_coords : torch.Tensor
        Backbone coordinates of shape (..., 4, 3) in atom order (N, CA, C, O).

    Returns
    -------
    ca_coords : torch.Tensor
        CA coordinates of shape (..., 3).
    frames : torch.Tensor
        Local orthonormal frames of shape (..., 3, 3), where columns are x/y/z axes.
    """
    n = backbone_coords[..., 0, :]
    ca = backbone_coords[..., 1, :]
    c = backbone_coords[..., 2, :]

    x_axis = _safe_normalize(c - ca)
    n_dir = _safe_normalize(n - ca)
    y_axis = _safe_normalize(n_dir - (n_dir * x_axis).sum(dim=-1, keepdim=True) * x_axis)
    z_axis = _safe_normalize(torch.cross(x_axis, y_axis, dim=-1))
    y_axis = _safe_normalize(torch.cross(z_axis, x_axis, dim=-1))

    frames = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return ca, frames


def compute_pseudo_cb_direction(
    backbone_coords: torch.Tensor,
    chirality: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute virtual CB direction from backbone geometry (N, CA, C).

    Uses the standard tetrahedral construction: the virtual CB sits opposite
    the N and C substituents around CA. The out-of-plane ``a`` term has a fixed
    handedness, so it must be flipped per residue chirality -- otherwise the cone
    hard-codes the L-amino-acid hemisphere and anti-aligns with the real CA->CB
    for every D-amino acid (cos ~ -0.04), actively mis-steering the donut prior.
    With ``chirality`` supplied the cone points to the correct hemisphere for
    both L and D residues; the construction still leaks no residue identity
    beyond the L/D sign.

    Parameters
    ----------
    backbone_coords : torch.Tensor
        Shape (..., 4, 3) with atoms in order (N, CA, C, O).
    chirality : torch.Tensor, optional
        Per-residue handedness sign, broadcastable to ``backbone_coords[..., 0, 0]``
        (i.e. shape ``(...)`` such as ``(B, L)``): ``+1`` for L, ``-1`` for D.
        When ``None`` (default) all residues are treated as L (+1), which is
        bit-identical to the historical chirality-blind behavior. For de-novo /
        unknown-chirality sampling (e.g. canonical evalA) the +1 default is the
        common case; a two-mode both-hemisphere prior is a possible future
        extension, not built here.

    Returns
    -------
    direction : torch.Tensor
        Normalized CA->pseudo-CB direction, shape (..., 3).
    """
    n = backbone_coords[..., 0, :]
    ca = backbone_coords[..., 1, :]
    c = backbone_coords[..., 2, :]
    b = _safe_normalize(n - ca)
    d = _safe_normalize(c - ca)
    # Cross product gives perpendicular to the N-CA-C plane (fixed handedness)
    a = _safe_normalize(torch.cross(b, d, dim=-1))
    if chirality is not None:
        # Flip the out-of-plane term to the residue's hemisphere (+1 L / -1 D).
        sign = chirality.to(dtype=a.dtype, device=a.device).unsqueeze(-1)
        a = sign * a
    # Virtual CB: opposite the N and C directions, with tetrahedral geometry
    return _safe_normalize(-0.58 * b - 0.58 * d + 0.58 * a)


def compute_residue_pair_geometry(
    query_backbone_coords: torch.Tensor,
    key_backbone_coords: torch.Tensor,
    num_rbf: int = 16,
    max_distance: float = 20.0,
) -> torch.Tensor:
    """
    Compute ProteinMPNN-style residue pair geometry features.

    Parameters
    ----------
    query_backbone_coords : torch.Tensor
        Query backbone coordinates of shape (B, L_q, 4, 3).
    key_backbone_coords : torch.Tensor
        Key backbone coordinates of shape (B, L_k, 4, 3).
    num_rbf : int
        Number of radial basis distance features.
    max_distance : float
        Maximum distance scale for the radial basis centers.

    Returns
    -------
    torch.Tensor
        Pair geometry features of shape (B, L_q, L_k, 3 + 3 + 9 + num_rbf).
    """
    q_ca, q_frames = build_residue_frames(query_backbone_coords)
    k_ca, k_frames = build_residue_frames(key_backbone_coords)

    delta = k_ca.unsqueeze(1) - q_ca.unsqueeze(2)
    q_rel = torch.einsum("blij,blkj->blki", q_frames.transpose(-2, -1), delta)
    k_rel = torch.einsum("bkij,bklj->bkli", k_frames.transpose(-2, -1), -delta.transpose(1, 2)).transpose(1, 2)
    orient = torch.einsum("blij,bkjm->blkim", q_frames.transpose(-2, -1), k_frames).reshape(
        query_backbone_coords.shape[0], query_backbone_coords.shape[1], key_backbone_coords.shape[1], 9
    )

    distances = delta.norm(dim=-1, keepdim=True)
    centers = torch.linspace(
        0.0,
        max_distance,
        num_rbf,
        device=query_backbone_coords.device,
        dtype=query_backbone_coords.dtype,
    )
    width = max_distance / max(num_rbf - 1, 1)
    rbf = torch.exp(-((distances - centers.view(1, 1, 1, -1)) ** 2) / (2 * width * width + 1e-8))

    return torch.cat([q_rel, k_rel, orient, rbf], dim=-1)


def _backbone_dihedral(p0, p1, p2, p3):
    """Signed dihedral angle (radians) about the p1-p2 bond for 4 points.

    Parameters shaped ``(..., 3)``. Returns ``(...)``. Uses atan2 for a stable,
    differentiable-a.e. angle. Degenerate (zero-length) bonds yield 0.
    """
    # Praxeolitic / IUPAC-signed dihedral: a right-handed alpha helix -> phi ~ -60.
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-8)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _backbone_bond_angle(p0, p1, p2):
    """Interior angle (radians) at ``p1`` for the p0-p1-p2 triple. Shapes ``(..., 3)``."""
    v1 = p0 - p1
    v2 = p2 - p1
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    cos = (v1 * v2).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


class BackboneEncoder(nn.Module):
    """
    Encoder for fixed backbone coordinates.

    Produces per-residue features from backbone atom positions (N, CA, C, O).
    These features condition the side-chain denoising.

    Parameters
    ----------
    hidden_dim : int
        Output feature dimension.
    """

    def __init__(self, hidden_dim: int = 128, use_backbone_dihedral: bool = False, use_burial_feature: bool = False):
        super().__init__()

        self.use_backbone_dihedral = use_backbone_dihedral
        self.use_burial_feature = use_burial_feature

        # Each backbone atom gets embedded, then aggregated
        # 4 atoms * 3 coords = 12 input features per residue
        self.mlp = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Parallel backbone-dihedral pathway (opt-in). Featurizes per-residue
        # phi/psi/omega + N-CA-C bond angle as [sin, cos] (8 features), embeds
        # them, and ADDS the result to the coord-MLP output. IDENTITY-AT-INIT
        # graft: only the FINAL Linear (weight AND bias) is zero-init, so the
        # added term is exactly 0 at init (bit-exact match to the base model),
        # while the input/hidden Linear stays live so gradient flows into the
        # zeroed output layer -> NOT the EVC zero-init-both-ends deadlock.
        if use_backbone_dihedral:
            self._n_dihedral_feat = 8  # (phi, psi, omega, tau) x [sin, cos]
            self.dihedral_mlp = nn.Sequential(
                nn.Linear(self._n_dihedral_feat, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.dihedral_mlp[-1].weight)
            nn.init.zeros_(self.dihedral_mlp[-1].bias)

        # Parallel Cbeta-burial pathway (opt-in). Featurizes a per-residue
        # backbone-only solvent-burial proxy (pseudo-Cbeta neighbour density at a
        # few radii + a Calpha coordination count), embeds it, and ADDS the result
        # to the coord-MLP output. Same IDENTITY-AT-INIT graft as the dihedral
        # pathway: only the FINAL Linear (weight AND bias) is zero-init so the
        # added term is exactly 0 at init (bit-exact match to the base model) while
        # the input/hidden Linear stays live so gradient flows in -> NOT a deadlock.
        # Leak-free: uses ONLY backbone atoms (no sidechain, no GT identity) so it is
        # available unchanged at inference.
        if use_burial_feature:
            self._burial_radii = (8.0, 10.0, 12.0)  # Cbeta-Cbeta neighbour-count radii (Angstrom)
            self._burial_ca_radius = 10.0  # Calpha coordination-number radius (Angstrom)
            self._burial_norm = 10.0  # neighbour-count normaliser (keeps feature ~O(1))
            self._burial_cb_bond_len = 1.53  # Angstrom, CA->pseudo-CB placement
            self._n_burial_feat = len(self._burial_radii) + 1
            self.burial_mlp = nn.Sequential(
                nn.Linear(self._n_burial_feat, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.burial_mlp[-1].weight)
            nn.init.zeros_(self.burial_mlp[-1].bias)

    def forward(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode backbone coordinates.

        Parameters
        ----------
        backbone_coords : torch.Tensor
            Backbone atom coordinates of shape (B, L, 4, 3).
        backbone_mask : torch.Tensor
            Mask for valid backbone atoms of shape (B, L, 4).

        Returns
        -------
        features : torch.Tensor
            Per-residue backbone features of shape (B, L, hidden_dim).
        """
        batch_size, seq_len = backbone_coords.shape[:2]

        # Flatten backbone atoms per residue: (B, L, 4, 3) -> (B, L, 12)
        backbone_flat = backbone_coords.view(batch_size, seq_len, -1)

        # Zero out invalid atoms
        mask_expanded = backbone_mask.unsqueeze(-1).expand_as(backbone_coords)
        backbone_flat = backbone_coords.masked_fill(~mask_expanded, 0.0).view(batch_size, seq_len, -1)

        # Encode
        features = self.mlp(backbone_flat)

        # Backbone-dihedral graft (adds exactly 0 at init; ramps in as it trains).
        if self.use_backbone_dihedral:
            dihedral_feats = self._backbone_dihedral_features(backbone_coords, backbone_mask)
            features = features + self.dihedral_mlp(dihedral_feats)

        # Cbeta-burial graft (adds exactly 0 at init; ramps in as it trains).
        if self.use_burial_feature:
            burial_feats = self._burial_features(backbone_coords, backbone_mask)
            features = features + self.burial_mlp(burial_feats)

        return features

    def _backbone_dihedral_features(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-residue backbone dihedral features, computed on the fly.

        From backbone atoms (N, CA, C, O; slot order 0,1,2,3) computes phi, psi,
        omega and the N-CA-C bond angle per residue, each encoded as [sin, cos].
        Dihedrals that reach into a neighbouring residue (phi/omega need the
        previous residue, psi needs the next) are zeroed at chain ends and at
        masked/padded residues (a zero pad -> sin=cos=0).

        Parameters
        ----------
        backbone_coords : torch.Tensor
            Backbone atom coordinates of shape (B, L, 4, 3).
        backbone_mask : torch.Tensor
            Mask for valid backbone atoms of shape (B, L, 4).

        Returns
        -------
        torch.Tensor
            Dihedral features of shape (B, L, 8).
        """
        n = backbone_coords[:, :, 0, :]
        ca = backbone_coords[:, :, 1, :]
        c = backbone_coords[:, :, 2, :]

        # A residue is usable if its N, CA, C are all present.
        res_valid = backbone_mask[:, :, :3].all(dim=-1)  # (B, L)

        # Neighbour tensors via shift along the residue axis.
        c_prev = torch.roll(c, shifts=1, dims=1)
        ca_prev = torch.roll(ca, shifts=1, dims=1)
        n_next = torch.roll(n, shifts=-1, dims=1)

        valid_prev = torch.roll(res_valid, shifts=1, dims=1)
        valid_next = torch.roll(res_valid, shifts=-1, dims=1)
        # Chain ends: roll wraps around, so explicitly invalidate the boundary.
        valid_prev = valid_prev.clone()
        valid_prev[:, 0] = False
        valid_next = valid_next.clone()
        valid_next[:, -1] = False

        phi = _backbone_dihedral(c_prev, n, ca, c)  # C(i-1)-N-CA-C
        psi = _backbone_dihedral(n, ca, c, n_next)  # N-CA-C-N(i+1)
        omega = _backbone_dihedral(ca_prev, c_prev, n, ca)  # CA(i-1)-C(i-1)-N-CA
        tau = _backbone_bond_angle(n, ca, c)  # N-CA-C bond angle

        phi_ok = (res_valid & valid_prev).to(phi.dtype)
        psi_ok = (res_valid & valid_next).to(psi.dtype)
        omega_ok = (res_valid & valid_prev).to(omega.dtype)
        tau_ok = res_valid.to(tau.dtype)

        return torch.stack(
            [
                torch.sin(phi) * phi_ok,
                torch.cos(phi) * phi_ok,
                torch.sin(psi) * psi_ok,
                torch.cos(psi) * psi_ok,
                torch.sin(omega) * omega_ok,
                torch.cos(omega) * omega_ok,
                torch.sin(tau) * tau_ok,
                torch.cos(tau) * tau_ok,
            ],
            dim=-1,
        )

    def _burial_features(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-residue backbone-only burial proxy, computed on the fly.

        Reuses the chirality-aware direction-cone pseudo-Cbeta
        (:func:`compute_pseudo_cb_direction`) to place a virtual Cbeta, then counts
        neighbouring residues' pseudo-Cbeta within a few radii (a coordination-number /
        neighbour-density burial estimate) plus a Calpha coordination count. Buried
        residues sit in a dense neighbourhood (high counts); exposed residues do not.
        Uses ONLY backbone atoms (N, CA, C) -> leak-free and available at inference.
        Chain-end / masked residues contribute nothing and are zeroed.

        Parameters
        ----------
        backbone_coords : torch.Tensor
            Backbone atom coordinates of shape (B, L, 4, 3).
        backbone_mask : torch.Tensor
            Mask for valid backbone atoms of shape (B, L, 4).

        Returns
        -------
        torch.Tensor
            Burial features of shape (B, L, n_burial_feat).
        """
        batch_size, seq_len = backbone_coords.shape[:2]
        # A residue is usable if its N, CA, C are all present.
        res_valid = backbone_mask[:, :, :3].all(dim=-1)  # (B, L)
        ca = backbone_coords[:, :, 1, :]  # (B, L, 3)
        # Reuse the direction-cone pseudo-CB (L-default; identity-of-hemisphere is
        # immaterial to a neighbour-density count) rather than recomputing a Cbeta.
        cb_dir = compute_pseudo_cb_direction(backbone_coords)  # (B, L, 3)
        cb = ca + self._burial_cb_bond_len * cb_dir  # (B, L, 3)

        # Pairwise neighbour distances. Invalid residues are pushed to +inf so they
        # never count as neighbours, and self-pairs are excluded.
        big = torch.finfo(cb.dtype).max / 4
        eye = torch.eye(seq_len, dtype=torch.bool, device=cb.device).unsqueeze(0)  # (1, L, L)
        pair_valid = res_valid.unsqueeze(1) & res_valid.unsqueeze(2) & ~eye  # (B, L, L)
        d_cb = torch.cdist(cb, cb)  # (B, L, L)
        d_ca = torch.cdist(ca, ca)  # (B, L, L)
        d_cb = d_cb.masked_fill(~pair_valid, big)
        d_ca = d_ca.masked_fill(~pair_valid, big)

        feats = []
        for r in self._burial_radii:
            feats.append((d_cb < r).sum(dim=-1).to(cb.dtype) / self._burial_norm)  # (B, L)
        feats.append((d_ca < self._burial_ca_radius).sum(dim=-1).to(cb.dtype) / self._burial_norm)
        burial = torch.stack(feats, dim=-1)  # (B, L, n_burial_feat)
        # Zero out invalid / padded residues (sin/cos-style clean pad).
        return burial * res_valid.unsqueeze(-1).to(burial.dtype)


class TargetEncoder(nn.Module):
    """
    Encoder for target chain structure and sequence.

    Encodes the target protein that the binder interacts with,
    combining backbone structure and residue identity.

    Parameters
    ----------
    hidden_dim : int
        Output feature dimension.
    num_residue_types : int
        Number of residue types for embedding.
    """

    def __init__(self, hidden_dim: int = 128, num_residue_types: int = NUM_RESIDUE_TYPES):
        super().__init__()

        # Residue type embedding: always use learned embedding
        self.residue_embed = nn.Embedding(num_residue_types, hidden_dim // 2)

        # Backbone structure encoder (same as BackboneEncoder)
        self.backbone_mlp = nn.Sequential(
            nn.Linear(12, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )

        # Combine structure + sequence
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        residue_types: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode target chain.

        Parameters
        ----------
        backbone_coords : torch.Tensor
            Target backbone coordinates of shape (B, L_target, 4, 3).
        backbone_mask : torch.Tensor
            Mask for valid backbone atoms of shape (B, L_target, 4).
        residue_types : torch.Tensor
            Residue type indices of shape (B, L_target).
        seq_mask : torch.Tensor, optional
            Mask for valid target residues of shape (B, L_target).

        Returns
        -------
        features : torch.Tensor
            Per-residue target features of shape (B, L_target, hidden_dim).
        """
        batch_size, seq_len = backbone_coords.shape[:2]

        # Encode backbone structure
        mask_expanded = backbone_mask.unsqueeze(-1).expand_as(backbone_coords)
        backbone_flat = backbone_coords.masked_fill(~mask_expanded, 0.0).view(batch_size, seq_len, -1)
        struct_features = self.backbone_mlp(backbone_flat)  # (B, L, hidden/2)

        # Encode residue types: learned embedding
        seq_features = self.residue_embed(residue_types)  # (B, L, hidden/2)

        # Combine
        combined = torch.cat([struct_features, seq_features], dim=-1)
        features = self.combine(combined)

        # Zero out invalid positions
        if seq_mask is not None:
            features = features * seq_mask.unsqueeze(-1).float()

        return features


class CrossAttention(nn.Module):
    """
    Cross-attention layer for binder-to-target attention.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute cross-attention.

        Parameters
        ----------
        query : torch.Tensor
            Query features of shape (B, L_q, hidden_dim).
        key : torch.Tensor
            Key features of shape (B, L_k, hidden_dim).
        value : torch.Tensor
            Value features of shape (B, L_k, hidden_dim).
        key_mask : torch.Tensor, optional
            Mask for valid key positions of shape (B, L_k).
        attn_bias : torch.Tensor, optional
            Additive attention bias of shape (B, heads, L_q, L_k) or (B, L_q, L_k).

        Returns
        -------
        output : torch.Tensor
            Attended features of shape (B, L_q, hidden_dim).
        """
        bsz, len_q, _ = query.shape
        _, len_k, _ = key.shape

        # Project to heads
        q = self.q_proj(query).view(bsz, len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, heads, L_q, L_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            attn = attn + attn_bias.unsqueeze(1) if attn_bias.dim() == 3 else attn + attn_bias

        # Mask invalid keys
        if key_mask is not None:
            attn_mask = ~key_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # Handle all-masked case
        attn = self.attn_dropout(attn)

        # Apply attention: (B, heads, L_q, head_dim)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, len_q, -1)

        return self.out_proj(out)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Modulates input features via learned scale and shift parameters
    derived from conditioning input. This is stronger than additive
    conditioning because it can multiplicatively gate features.

    Parameters
    ----------
    hidden_dim : int
        Dimension of features to modulate.
    cond_dim : int
        Dimension of conditioning input.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, hidden_dim)
        self.shift_proj = nn.Linear(cond_dim, hidden_dim)

        # Initialize to identity: scale=0, shift=0 at start
        # This ensures FiLM doesn't disrupt training initially
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Parameters
        ----------
        x : torch.Tensor
            Features to modulate of shape (..., hidden_dim).
        cond : torch.Tensor
            Conditioning input of shape (..., cond_dim).

        Returns
        -------
        torch.Tensor
            Modulated features of shape (..., hidden_dim).
        """
        scale = self.scale_proj(cond)  # (..., hidden_dim)
        shift = self.shift_proj(cond)  # (..., hidden_dim)
        return x * (1 + scale) + shift  # Residual-style: scale=0, shift=0 -> identity


class IntraResidueBondAttention(nn.Module):
    """Lightweight attention within each residue's slots, biased by predicted bond probabilities.

    At each denoising step, predicts a max_sc × max_sc bond probability matrix from
    current slot features (hidden + coordinates), then uses it to bias slot-to-slot
    attention within each residue.

    The bond graph is emergent (predicted, not looked up) -- works for any chemistry
    including NCAAs. Ghost slots naturally have zero bond probability.
    """

    def __init__(
        self, hidden_dim: int, max_sc: int = 16, num_heads: int = 4, dropout: float = 0.0, zero_init_out: bool = False
    ):
        super().__init__()
        self.max_sc = max_sc
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        # Bond probability prediction: pairwise from slot features + relative coords
        self.bond_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Standard multi-head self-attention within residue
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head_dim = head_dim

        # Identity-start (opt-in): forward is `h_out = residual + out_proj(out)`, so zeroing out_proj makes
        # this module an exact no-op at init -- matching FiLMLayer / CrossResiduePackingAttention / slot-
        # attention, which all zero-init their residual out-projections. Lets bond-attention be retrofitted
        # onto a checkpoint that never had it WITHOUT perturbing mature features; it grows from zero as it
        # earns loss. bond_proj still trains via the bond-denoising loss.
        # Default False preserves legacy/from-scratch behaviour (trained standard-init).
        if zero_init_out:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        noised_bond_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : (n_residues, max_sc, hidden_dim) -- per-slot features
        coords : (n_residues, max_sc, 3) -- per-slot coordinates
        mask : (n_residues, max_sc) -- True for valid/real slots
        noised_bond_probs : (n_residues, max_sc, max_sc), optional
            Noised GT bond probabilities for co-diffusion mode. When provided,
            attention uses these (interpolated GT<->uniform) instead of predicted bonds.
            The model still predicts bond_logits for the denoising loss.

        Returns
        -------
        h_out : (n_residues, max_sc, hidden_dim) -- refined features
        bond_logits : (n_residues, max_sc, max_sc) -- predicted bond log-probs
        """
        n_res, S, D = h.shape

        # Predict pairwise bond probabilities from concatenated features + distance
        hi = h.unsqueeze(2).expand(-1, -1, S, -1)  # (n_res, S, S, D)
        hj = h.unsqueeze(1).expand(-1, S, -1, -1)  # (n_res, S, S, D)
        dist = (coords.unsqueeze(2) - coords.unsqueeze(1)).norm(dim=-1, keepdim=True)  # (n_res, S, S, 1)
        pair_feat = torch.cat([hi, hj, dist], dim=-1)  # (n_res, S, S, 2D+1)
        bond_logits = self.bond_proj(pair_feat).squeeze(-1)  # (n_res, S, S)

        # Mask: invalid slots can't bond
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (n_res, S, S)
        bond_logits = bond_logits.masked_fill(~pair_mask, -1e9)

        # Bond-biased self-attention
        residual = h
        h_norm = self.norm(h)
        Q = self.q_proj(h_norm).view(n_res, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h_norm).view(n_res, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h_norm).view(n_res, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)  # (n_res, heads, S, S)
        # For co-diffusion: use noised GT bond probs for attention bias (self-correcting input)
        # For standard mode: use predicted bonds
        if noised_bond_probs is not None:
            bond_bias = noised_bond_probs.unsqueeze(1)  # (n_res, 1, S, S) -- already probabilities
        else:
            bond_bias = torch.sigmoid(bond_logits).unsqueeze(1)  # (n_res, 1, S, S)
        attn = attn + bond_bias * 2.0  # Bond pairs get attention boost

        # Mask invalid slots
        slot_mask = mask.unsqueeze(1).unsqueeze(3).expand_as(attn)  # (n_res, heads, S, S)
        attn = attn.masked_fill(~slot_mask, -1e9)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(n_res, S, -1)
        h_out = residual + self.out_proj(out)

        return h_out, bond_logits


#: Width of the per-slot neighbour-x0 packing context vector produced by
#: :func:`compute_neighbor_x0_features`. Fixed so a checkpoint's ``neighbor_x0_proj``
#: shape is stable across runs.
NEIGHBOR_X0_FEAT_DIM = 6

#: Default per-batch distribution over the TOTAL recycle count ``N`` when
#: ``neighbor_x0_packing_random_recycles`` is on. Index i holds P(N = i + 2), i.e.
#: ``{2: 0.70, 3: 0.15, 4: 0.10, 5: 0.05}`` -> E[N] = 2.5.
#:
#: Why mass at low N with only a thin right tail (2026-07-24):
#:
#: * Gradient flows ONLY through the FINAL pass (earlier passes run under ``no_grad``), so the
#: model is trained on the recycle-index encoding only at ``j = N``. The distribution over N
#: IS the distribution of trained ``j`` values -- intermediate ``j`` occur but produce no
#: gradient.
#: * The encoding SATURATES: ``1 - 1/j`` is 0.50 / 0.67 / 0.75 / 0.80 at j = 2/3/4/5. The
#: j=4 -> j=5 gap is only 0.05, so a model trained on j in {2,3,4} extrapolates to j >= 5
#: essentially for free. A thin tail is therefore enough; heavy coverage of large j buys
#: nothing and costs wall-clock.
#:
#: Cost, with backward ~ 2x forward (a full graded pass ~ 3 units, a ``no_grad`` pass ~ 1 unit):
#: N=2 -> 4 units, N=3 -> 5, N=4 -> 6, N=5 -> 7. Under these weights E[cost] = 4.5 units, i.e.
#: **+12.5% wall-clock vs. always-N=2**. Retune the weights with that number in view.
NEIGHBOR_X0_RECYCLE_WEIGHTS_DEFAULT: tuple[float, ...] = (0.70, 0.15, 0.10, 0.05)

#: Smallest randomisable recycle count. N=1 is the unconditioned single pass, which the firing
#: PROBABILITY ramp already covers, so the random-N draw starts at 2.
NEIGHBOR_X0_MIN_RANDOM_RECYCLES = 2


def parse_neighbor_x0_recycle_weights(
    weights: str | Sequence[float] | None,
    max_recycles: int,
) -> tuple[float, ...]:
    """Normalise the per-batch recycle-count distribution to a probability tuple over N=2..max.

    Accepts a comma-separated string (so the same value can travel through Typer, a distributed
    dataclass and a checkpoint hparam dict unchanged) or any float sequence. ``None`` selects
    :data:`NEIGHBOR_X0_RECYCLE_WEIGHTS_DEFAULT`.

    Parameters
    ----------
    weights : str or sequence of float, optional
        Relative (unnormalised) weights, index i == N of ``i + 2``.
    max_recycles : int
        Largest N with non-zero weight. Must equal ``len(weights) + 1``.

    Returns
    -------
    tuple of float
        Probabilities summing to 1.0, of length ``max_recycles - 1``.

    Raises
    ------
    ValueError
        If the length disagrees with ``max_recycles``, or the weights are negative / all zero.
        Fails loud rather than silently re-shaping: a mismatched tail is exactly the sort of
        wall-clock blow-up this distribution exists to bound.
    """
    if isinstance(weights, str) and weights.strip():
        parsed = [float(w) for w in weights.replace(" ", "").split(",") if w]
    elif weights is None or isinstance(weights, str):
        # "" / " " are how an unset value travels through Typer and a distributed dataclass (neither
        # round-trips None cleanly), so treat blank exactly as "use the default".
        parsed = list(NEIGHBOR_X0_RECYCLE_WEIGHTS_DEFAULT)
    else:
        parsed = [float(w) for w in weights]
    n_slots = int(max_recycles) - NEIGHBOR_X0_MIN_RANDOM_RECYCLES + 1
    if n_slots < 1:
        raise ValueError(
            f"neighbor_x0_packing_max_recycles={max_recycles} is below the minimum randomisable "
            f"recycle count ({NEIGHBOR_X0_MIN_RANDOM_RECYCLES}). Use "
            f"neighbor_x0_packing_random_recycles=False for a fixed single-pass regime."
        )
    if len(parsed) != n_slots:
        raise ValueError(
            f"neighbor_x0_packing_recycle_weights has {len(parsed)} entries but "
            f"neighbor_x0_packing_max_recycles={max_recycles} needs {n_slots} "
            f"(one per N in {NEIGHBOR_X0_MIN_RANDOM_RECYCLES}..{max_recycles}). The default "
            f"{NEIGHBOR_X0_RECYCLE_WEIGHTS_DEFAULT} pairs with max_recycles=5; pass explicit "
            f"weights whenever you change the tail length."
        )
    if any(w < 0.0 for w in parsed):
        raise ValueError(f"neighbor_x0_packing_recycle_weights must be non-negative, got {parsed}")
    total = float(sum(parsed))
    if total <= 0.0:
        raise ValueError("neighbor_x0_packing_recycle_weights sum to 0 -- no recycle count could ever be drawn")
    return tuple(w / total for w in parsed)


def compute_neighbor_x0_features(
    query_coords: torch.Tensor,
    query_residue_idx: torch.Tensor,
    neighbor_coords: torch.Tensor,
    neighbor_valid: torch.Tensor,
    neighbor_trust: torch.Tensor,
    radius: float = 8.0,
) -> torch.Tensor:
    """Per-slot packing context read off OTHER residues' clean-endpoint (x0) estimates.

    Each query slot (an atom slot of residue ``i``, at its current noisy position) is
    described by how the *predicted clean* atoms of the residues around it are arranged:
    a nearest-neighbour distance plus trust-weighted occupancy in three concentric
    shells. All features are SE(3)-invariant scalars, so they can be concatenated into
    the equivariant transformer's node features without breaking equivariance.

    **Anti-self (load-bearing).** Every neighbour atom belonging to the query slot's own
    residue is excluded. Feeding a residue its own x0 estimate back in is plain
    self-conditioning, which entrenches the residue's own error (the v32/v33 pathology);
    excluding self is what makes this conditioning carry genuinely new information.

    **Trust.** ``neighbor_trust[j]`` in ``[0, 1]`` is how denoised residue ``j`` is
    (1.0 = clean ground-truth context, ~0 = pure noise). Occupancy counts are weighted
    by it, so the model can learn to discount a high-noise neighbour's estimate.

    Parameters
    ----------
    query_coords : torch.Tensor
        Current coordinates of the query slots, shape (N_q, 3).
    query_residue_idx : torch.Tensor
        Residue index of each query slot, shape (N_q,), dtype long.
    neighbor_coords : torch.Tensor
        Per-residue neighbour atom positions (x0 estimates / clean GT), shape (L, S, 3).
    neighbor_valid : torch.Tensor
        Bool mask of which neighbour slots hold a real atom, shape (L, S).
    neighbor_trust : torch.Tensor
        Per-residue trust in ``[0, 1]``, shape (L,).
    radius : float
        Outer shell radius in Angstrom. Shells are ``0.5 * radius``, ``0.75 * radius``
        and ``radius``.

    Returns
    -------
    torch.Tensor
        Features of shape (N_q, :data:`NEIGHBOR_X0_FEAT_DIM`), finite everywhere,
        and exactly zero-ish/neutral for query slots that have no valid neighbour.
    """
    device = query_coords.device
    dtype = query_coords.dtype
    n_res, n_slots, _ = neighbor_coords.shape
    n_q = query_coords.shape[0]

    far = 2.0 * radius
    if n_q == 0 or n_res == 0:
        return torch.zeros(n_q, NEIGHBOR_X0_FEAT_DIM, device=device, dtype=dtype)

    flat_coords = neighbor_coords.reshape(-1, 3)  # (L*S, 3)
    flat_res = (
        torch.arange(n_res, device=device).unsqueeze(-1).expand(-1, n_slots).reshape(-1)
    )  # (L*S,) residue owning each neighbour slot
    weight = neighbor_valid.reshape(-1).to(dtype) * neighbor_trust.to(dtype)[flat_res]  # (L*S,)

    # ANTI-SELF: a neighbour slot only counts for a query slot from a DIFFERENT residue.
    other = query_residue_idx.unsqueeze(1) != flat_res.unsqueeze(0)  # (N_q, L*S)
    usable = other & (neighbor_valid.reshape(-1).unsqueeze(0))  # (N_q, L*S)

    dist = torch.cdist(query_coords.unsqueeze(0), flat_coords.unsqueeze(0)).squeeze(0)  # (N_q, L*S)
    dist_masked = torch.where(usable, dist, torch.full_like(dist, far))
    d_min = dist_masked.min(dim=1).values.clamp(min=0.0, max=far)  # (N_q,)

    w = weight.unsqueeze(0) * usable.to(dtype)  # (N_q, L*S)
    shells = (0.5 * radius, 0.75 * radius, radius)
    counts = [((dist < r).to(dtype) * w).sum(dim=1) / 8.0 for r in shells]

    in_outer = (dist < radius).to(dtype) * usable.to(dtype)  # (N_q, L*S)
    n_outer = in_outer.sum(dim=1)
    mean_trust = (in_outer * neighbor_trust.to(dtype)[flat_res].unsqueeze(0)).sum(dim=1) / n_outer.clamp(min=1.0)

    feats = torch.stack(
        [
            torch.exp(-d_min / 3.0),
            d_min / far,
            counts[0],
            counts[1],
            counts[2],
            mean_trust,
        ],
        dim=-1,
    )
    return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


#: Small isotropic jitter (Angstrom) added to a phantom (ADD-ed) atom so it does not land exactly
#: on the cloud ray. Fixed small constant -- see :func:`corrupt_neighbor_x0`.
_NX0_ADD_JITTER_STD = 0.3


class CrossResiduePackingAttention(nn.Module):
    """Attention between atoms of neighboring binder residues for inter-residue packing.

    Predicts a sparse contact matrix between atoms of *paired* residues, then uses the
    predicted contacts to bias inter-residue attention. Supervised from GT inter-residue
    distances < 3.5Å.

    Two pairing modes:

    ``spatial=False`` (default, historical)
        Pairs are sequence-consecutive only (``i`` with ``i±1``). Note that the inpainting path
        evaluation defines *Buried* tip-packing over neighbours with ``|q - p| > 1``,
        i.e. it explicitly EXCLUDES ``i±1`` -- so in this mode the module structurally
        cannot contribute to the Buried metric.

    ``spatial=True``
        Pairs are the ``k`` spatially nearest residues by pseudo-Cbeta distance, with a
        minimum sequence separation (default 2, i.e. ``i±1`` excluded) and an optional
        radius cutoff. Neighbour selection uses only backbone-derived pseudo-Cbeta
        positions, which are fixed model INPUTS -- no ground truth is consumed.

    Memory: the pair tensor is ``O(n_res * k * S^2 * hidden)``, i.e. roughly ``k``x the
    consecutive-pair mode. Keep ``k`` small (4 is the default).
    """

    def __init__(
        self,
        hidden_dim: int,
        max_sc: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        spatial: bool = False,
        k_neighbors: int = 4,
        radius: float = 10.0,
        min_seq_sep: int = 2,
    ):
        super().__init__()
        self.max_sc = max_sc
        self.num_heads = num_heads
        self.spatial = spatial
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.min_seq_sep = min_seq_sep
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim

        # Contact prediction: pairwise from slot features of neighboring residues + distance
        self.contact_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cross-residue attention (residue i attends to residue i±1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Zero-init output so this is identity at start (backward compatible with pre-packing checkpoints)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Parameters
        ----------
        h : (n_residues, max_sc, hidden_dim) -- per-slot features
        coords : (n_residues, max_sc, 3) -- per-slot coordinates
        mask : (n_residues, max_sc) -- True for valid slots
        residue_pos : (n_residues, 3), optional
            Backbone-derived per-residue position (pseudo-Cbeta) used for spatial
            neighbour selection. Required when ``spatial=True``.

        Returns
        -------
        h_out : (n_residues, max_sc, hidden_dim) -- refined features
        contact_logits : predicted inter-residue contact log-probs.
            ``(n_residues-1, max_sc, max_sc)`` in consecutive mode,
            ``(n_residues, k, max_sc, max_sc)`` in spatial mode.
        neighbor_idx : (n_residues, k) long, or None in consecutive mode.
        neighbor_valid : (n_residues, k) bool, or None in consecutive mode.
        """
        n_res, S, D = h.shape
        device = h.device

        if n_res < 2:
            return h, torch.zeros(0, S, S, device=device), None, None

        if self.spatial:
            return self._forward_spatial(h, coords, mask, residue_pos)

        # Build pairs of consecutive residues (i, i+1)
        h_i = h[:-1]  # (n_res-1, S, D)
        h_j = h[1:]  # (n_res-1, S, D)
        c_i = coords[:-1]  # (n_res-1, S, 3)
        c_j = coords[1:]  # (n_res-1, S, 3)
        m_i = mask[:-1]  # (n_res-1, S)
        m_j = mask[1:]  # (n_res-1, S)

        # Predict pairwise contacts between slots of neighboring residues
        # hi_exp: (n_res-1, S, S, D), hj_exp: (n_res-1, S, S, D)
        hi_exp = h_i.unsqueeze(2).expand(-1, -1, S, -1)
        hj_exp = h_j.unsqueeze(1).expand(-1, S, -1, -1)
        dist = (c_i.unsqueeze(2) - c_j.unsqueeze(1)).norm(dim=-1, keepdim=True)  # (n_res-1, S, S, 1)
        pair_feat = torch.cat([hi_exp, hj_exp, dist], dim=-1)
        contact_logits = self.contact_proj(pair_feat).squeeze(-1)  # (n_res-1, S, S)

        # Mask invalid pairs
        pair_mask = m_i.unsqueeze(2) & m_j.unsqueeze(1)  # (n_res-1, S, S)
        contact_logits = contact_logits.masked_fill(~pair_mask, -1e9)

        # Contact-biased cross-residue attention: each residue attends to its neighbors
        # For efficiency, each slot in residue i attends to all slots in residues i-1 and i+1
        residual = h
        h_norm = self.norm(h)

        Q = self.q_proj(h_norm).view(n_res, S, self.num_heads, self.head_dim).transpose(1, 2)  # (n_res, H, S, d)
        K = self.k_proj(h_norm).view(n_res, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h_norm).view(n_res, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Attend forward (i -> i+1) and backward (i -> i-1), sum contributions
        out = torch.zeros_like(Q)

        # Forward: residue i attends to residue i+1
        attn_fwd = (Q[:-1] @ K[1:].transpose(-2, -1)) / (self.head_dim**0.5)
        contact_bias_fwd = torch.sigmoid(contact_logits).unsqueeze(1)  # (n_res-1, 1, S, S)
        attn_fwd = attn_fwd + contact_bias_fwd * 2.0
        fwd_mask = m_j.unsqueeze(1).unsqueeze(2).expand_as(attn_fwd)
        attn_fwd = attn_fwd.masked_fill(~fwd_mask, -1e9)
        attn_fwd = torch.softmax(attn_fwd, dim=-1)
        attn_fwd = self.dropout(attn_fwd)
        out[:-1] = out[:-1] + attn_fwd @ V[1:]

        # Backward: residue i attends to residue i-1
        attn_bwd = (Q[1:] @ K[:-1].transpose(-2, -1)) / (self.head_dim**0.5)
        # Transpose contact logits for backward direction (j->i becomes i->j)
        contact_bias_bwd = torch.sigmoid(contact_logits).transpose(-2, -1).unsqueeze(1)
        attn_bwd = attn_bwd + contact_bias_bwd * 2.0
        bwd_mask = m_i.unsqueeze(1).unsqueeze(2).expand_as(attn_bwd)
        attn_bwd = attn_bwd.masked_fill(~bwd_mask, -1e9)
        attn_bwd = torch.softmax(attn_bwd, dim=-1)
        attn_bwd = self.dropout(attn_bwd)
        out[1:] = out[1:] + attn_bwd @ V[:-1]

        out = out.transpose(1, 2).reshape(n_res, S, -1)
        h_out = residual + self.out_proj(out)

        return h_out, contact_logits, None, None

    def select_spatial_neighbors(
        self,
        residue_pos: torch.Tensor,
        residue_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick the ``k`` nearest residues by pseudo-Cbeta distance, ``|i-j| >= min_seq_sep``.

        Uses only backbone-derived positions (a fixed model input), so nothing about the
        designed side chains -- predicted or ground truth -- enters the selection.

        Parameters
        ----------
        residue_pos : torch.Tensor
            Per-residue position of shape (n_res, 3).
        residue_valid : torch.Tensor
            Bool mask of usable residues, shape (n_res,).

        Returns
        -------
        neighbor_idx : torch.Tensor
            Long tensor (n_res, k) of neighbour residue indices (padded with 0).
        neighbor_valid : torch.Tensor
            Bool tensor (n_res, k), False where the slot is padding.
        """
        n_res = residue_pos.shape[0]
        device = residue_pos.device
        idx = torch.arange(n_res, device=device)
        sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        allowed = (sep >= self.min_seq_sep) & residue_valid.unsqueeze(0) & residue_valid.unsqueeze(1)
        dist = torch.cdist(residue_pos.unsqueeze(0), residue_pos.unsqueeze(0)).squeeze(0)
        if self.radius > 0:
            allowed = allowed & (dist <= self.radius)
        big = torch.finfo(dist.dtype).max / 4
        dist = torch.where(allowed, dist, torch.full_like(dist, big))
        k_eff = max(1, min(self.k_neighbors, n_res))
        nb_dist, nb_idx = torch.topk(dist, k_eff, dim=-1, largest=False)
        return nb_idx, nb_dist < (big / 2)

    def _forward_spatial(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_pos: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spatial-neighbour variant of :meth:`forward` (see the class docstring)."""
        if residue_pos is None:
            raise ValueError(
                "CrossResiduePackingAttention(spatial=True) needs residue_pos (backbone pseudo-Cbeta positions)."
            )
        n_res, S, D = h.shape
        res_valid = mask.any(dim=-1)
        nb_idx, nb_ok = self.select_spatial_neighbors(residue_pos[:n_res], res_valid)
        k = nb_idx.shape[1]

        h_j = h[nb_idx]  # (n_res, k, S, D)
        c_j = coords[nb_idx]  # (n_res, k, S, 3)
        m_j = mask[nb_idx] & nb_ok.unsqueeze(-1)  # (n_res, k, S)

        hi_exp = h.unsqueeze(1).unsqueeze(3).expand(n_res, k, S, S, D)
        hj_exp = h_j.unsqueeze(2).expand(n_res, k, S, S, D)
        dist = (coords.unsqueeze(1).unsqueeze(3) - c_j.unsqueeze(2)).norm(dim=-1, keepdim=True)
        contact_logits = self.contact_proj(torch.cat([hi_exp, hj_exp, dist], dim=-1)).squeeze(-1)
        pair_mask = mask.unsqueeze(1).unsqueeze(3) & m_j.unsqueeze(2)  # (n_res, k, S, S)
        contact_logits = contact_logits.masked_fill(~pair_mask, -1e9)

        residual = h
        h_norm = self.norm(h)
        H, dh = self.num_heads, self.head_dim
        Q = self.q_proj(h_norm).view(n_res, S, H, dh).transpose(1, 2)  # (n_res, H, S, dh)
        K_res = self.k_proj(h_norm).view(n_res, S, H, dh)
        V_res = self.v_proj(h_norm).view(n_res, S, H, dh)
        Kg = K_res[nb_idx].permute(0, 3, 1, 2, 4).reshape(n_res, H, k * S, dh)
        Vg = V_res[nb_idx].permute(0, 3, 1, 2, 4).reshape(n_res, H, k * S, dh)

        attn = (Q @ Kg.transpose(-2, -1)) / (dh**0.5)  # (n_res, H, S, k*S)
        bias = torch.sigmoid(contact_logits).permute(0, 2, 1, 3).reshape(n_res, S, k * S)
        attn = attn + bias.unsqueeze(1) * 2.0
        key_mask = m_j.reshape(n_res, k * S).unsqueeze(1).unsqueeze(2)  # (n_res, 1, 1, k*S)
        attn = attn.masked_fill(~key_mask.expand_as(attn), -1e9)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ Vg  # (n_res, H, S, dh)
        # Residues with no usable neighbour softmax over an all-masked row -> uniform garbage; drop them.
        has_nb = nb_ok.any(dim=-1).view(n_res, 1, 1, 1).to(out.dtype)
        out = out * has_nb

        out = out.transpose(1, 2).reshape(n_res, S, -1)
        h_out = residual + self.out_proj(out)
        return h_out, contact_logits, nb_idx, nb_ok


#: "bond-inject" descriptor: one [w-mean, w-std, soft-hist(n_bins)] block for internal ANGLES and one for
#: bonded-scale LENGTHS, computed from the same predicted x0 -> BOND_GEOM_DESC_DIM = 2 * (2 + BOND_GEOM_HIST_BINS).
BOND_GEOM_HIST_BINS = 6
BOND_ANGLE_DESC_DIM = 2 + BOND_GEOM_HIST_BINS  # width of a single (angle OR length) block
BOND_GEOM_DESC_DIM = 2 * BOND_ANGLE_DESC_DIM  # angle block ++ length block
BOND_LENGTH_HIST_MIN = 0.8  # Å: soft-hist range for bonded-scale distances (covers C-C/C-N/C-S bonds ~1.5)
BOND_LENGTH_HIST_MAX = 3.0  # Å


class SidechainDenoiser(nn.Module):
    """Released AtomWeaver inference architecture; module construction order preserves seeded sampling."""

    _nx0_corrupt_announced: bool = False

    def __init__(self):
        hidden_dim = 432
        num_layers = 10
        time_embed_dim = 128
        max_sidechain_atoms = 14
        num_cross_attn_layers = 3
        num_cross_attn_heads = 8
        target_condition_scale = 1.0
        cluster_target_condition_scale = 1.0
        edge_embed_dim = 16
        intra_residue_cutoff = 8.0
        inter_residue_cutoff = 8.0
        sidechain_target_cutoff = 15.0
        backbone_target_cutoff = 15.0
        rbf_span_to_cutoff = False
        ca_ca_prefilter = 20.0
        dropout = 0.1
        num_element_classes = 6
        num_timesteps = 250
        use_ca_dist_element_feature = False
        zero_init_bond_attention = False
        cross_residue_packing_spatial = False
        cross_residue_packing_k = 4
        cross_residue_packing_radius = 10.0
        cross_residue_packing_min_seq_sep = 2
        neighbor_x0_packing_radius = 8.0
        neighbor_x0_corrupt_add_prob = 0.35
        neighbor_x0_corrupt_drop_prob = 0.35
        neighbor_x0_corrupt_noise_prob = 0.9
        neighbor_x0_corrupt_coord_noise = 0.75
        neighbor_x0_corrupt_disconnect_prob = 0.0
        shape_prior_n_anchors = 4
        interaction_intent_num_classes = 4
        num_edge_types = 3
        graph_num_edge_types = None
        use_backbone_dihedral = False
        use_burial_feature = True
        residue_frame_v2_layers = 2
        frame_v2_clean_input = True
        activation_checkpointing = False
        activation_checkpoint_stride = 1.0
        graft_init_std = 0.0
        super().__init__()
        _graft_std = float(graft_init_std)
        _graft_gen = torch.Generator()
        _graft_gen.manual_seed(27159)

        def _init_graft_weight(w: torch.Tensor) -> None:
            if _graft_std > 0.0:
                with torch.no_grad():
                    w.normal_(0.0, _graft_std, generator=_graft_gen)
            else:
                nn.init.zeros_(w)

        self.hidden_dim = hidden_dim
        self._graph_num_edge_types: int | None = graph_num_edge_types
        self.neighbor_x0_packing_radius = neighbor_x0_packing_radius
        self.neighbor_x0_corrupt_add_prob = float(neighbor_x0_corrupt_add_prob)
        self.neighbor_x0_corrupt_drop_prob = float(neighbor_x0_corrupt_drop_prob)
        self.neighbor_x0_corrupt_noise_prob = float(neighbor_x0_corrupt_noise_prob)
        self.neighbor_x0_corrupt_coord_noise = float(neighbor_x0_corrupt_coord_noise)
        self.neighbor_x0_corrupt_disconnect_prob = float(neighbor_x0_corrupt_disconnect_prob)
        self.num_element_classes = num_element_classes
        self.num_timesteps = num_timesteps
        self.max_sidechain_atoms = max_sidechain_atoms
        self.num_cross_attn_layers = num_cross_attn_layers
        self.target_condition_scale = target_condition_scale
        self.cluster_target_condition_scale = cluster_target_condition_scale
        self.intra_residue_cutoff = intra_residue_cutoff
        self.inter_residue_cutoff = inter_residue_cutoff
        self.sidechain_target_cutoff = sidechain_target_cutoff
        self.backbone_target_cutoff = backbone_target_cutoff
        self.ca_ca_prefilter = ca_ca_prefilter
        self.backbone_encoder = BackboneEncoder(
            hidden_dim, use_backbone_dihedral=use_backbone_dihedral, use_burial_feature=use_burial_feature
        )
        self.residue_frame_stream_v2 = ResidueFrameStreamV2(
            hidden_dim=hidden_dim,
            n_layers=residue_frame_v2_layers,
            clean_input=frame_v2_clean_input,
            num_residue_types=NUM_RESIDUE_TYPES,
            use_stereochem_head=True,
        )
        self.stereo_t_atom_head = nn.Sequential(
            nn.Linear(_STEREO_T_ATOM_FEAT_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.stereo_t_atom_head[-1].bias, _STEREO_HEAD_LPRIOR_BIAS)
        self.stereo_feedback_embed = nn.Linear(1, hidden_dim)
        self.stereo_feedback_deep_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )
        for _proj in self.stereo_feedback_deep_proj:
            _init_graft_weight(_proj.weight)
        self.stereo_feedback_face_scale = nn.Parameter(torch.zeros(1))
        self.residue_frame_v2_deep_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )
        for _proj in self.residue_frame_v2_deep_proj:
            _init_graft_weight(_proj.weight)
        self.volumetric_deep_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )
        for _proj in self.volumetric_deep_proj:
            _init_graft_weight(_proj.weight)
        self.target_deep_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
        )
        for _proj in self.target_deep_proj:
            _init_graft_weight(_proj.weight)
        self.volumetric_existence_proj = nn.Linear(hidden_dim, max_sidechain_atoms)
        _init_graft_weight(self.volumetric_existence_proj.weight)
        _init_graft_weight(self.volumetric_existence_proj.bias)
        self.target_encoder = TargetEncoder(hidden_dim)
        pair_geom_dim = 3 + 3 + 9 + 16
        self.residue_pair_bias_proj = nn.Sequential(
            nn.Linear(pair_geom_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_cross_attn_heads)
        )
        self.sidechain_pair_bias_proj = nn.Sequential(
            nn.Linear(pair_geom_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_cross_attn_heads)
        )
        self.cross_attention_layers = nn.ModuleList(
            [
                CrossAttention(hidden_dim, num_heads=num_cross_attn_heads, dropout=dropout)
                for _ in range(num_cross_attn_layers)
            ]
        )
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cross_attn_layers)])
        self.sidechain_target_attention = CrossAttention(hidden_dim, num_heads=num_cross_attn_heads, dropout=dropout)
        self.sidechain_target_norm = nn.LayerNorm(hidden_dim)
        self.sidechain_target_film = FiLMLayer(hidden_dim, hidden_dim)
        self.cluster_target_context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        nn.init.zeros_(self.cluster_target_context_proj[0].weight)
        nn.init.zeros_(self.cluster_target_context_proj[0].bias)
        nn.init.zeros_(self.cluster_target_context_proj[2].weight)
        nn.init.zeros_(self.cluster_target_context_proj[2].bias)
        self.target_film = FiLMLayer(hidden_dim, hidden_dim)
        self.time_embed = TimestepEmbedding(time_embed_dim, hidden_dim)
        self.sc_atom_embed = nn.Embedding(max_sidechain_atoms, hidden_dim // 4)
        self.element_type_embed = nn.Embedding(num_element_classes, hidden_dim // 4)
        self.bb_atom_embed = nn.Embedding(4, hidden_dim // 4)
        self.target_atom_type_embed = nn.Embedding(18, hidden_dim // 4)
        self.target_element_type_embed = nn.Embedding(NUM_ELEMENT_TYPES, hidden_dim // 4)
        self.target_residue_type_embed = nn.Embedding(21, hidden_dim // 4)
        self.target_is_backbone_embed = nn.Embedding(2, hidden_dim // 8)
        target_proj_in = 3 * (hidden_dim // 4) + hidden_dim // 8
        self.target_atom_proj = nn.Linear(target_proj_in, hidden_dim)
        self.noised_count_embed = nn.Linear(1, hidden_dim // 8)
        self.self_cond_element_proj = nn.Linear(num_element_classes, hidden_dim // 8)
        self.cluster_id_embed = nn.Embedding(max_sidechain_atoms, hidden_dim // 8)
        self.self_cond_cluster_proj = nn.Linear(max_sidechain_atoms, hidden_dim // 8)
        self.max_residue_idx = 256
        self.residue_idx_embed = nn.Embedding(self.max_residue_idx, hidden_dim // 8)
        self.ar_state_embed = nn.Embedding(3, hidden_dim // 8)
        nn.init.normal_(self.ar_state_embed.weight, std=0.02)
        self.ar_state_proj = nn.Linear(hidden_dim // 8, hidden_dim, bias=False)
        nn.init.zeros_(self.ar_state_proj.weight)
        sc_proj_in = (
            hidden_dim // 4
            + hidden_dim // 4
            + hidden_dim // 8
            + hidden_dim // 8
            + hidden_dim // 8
            + hidden_dim // 8
            + hidden_dim // 8
            + hidden_dim
            + hidden_dim
        )
        self.sc_node_proj = nn.Linear(sc_proj_in, hidden_dim)
        self.bb_node_proj = nn.Linear(hidden_dim // 4 + hidden_dim + hidden_dim, hidden_dim)
        self.num_edge_types = num_edge_types
        self.edge_embed = EdgeTypeEmbedding(num_types=num_edge_types, embed_dim=edge_embed_dim)
        self.transformer = SE3Transformer(
            node_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            edge_dim=edge_embed_dim,
            num_layers=num_layers,
            num_heads=num_cross_attn_heads,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
            activation_checkpoint_stride=activation_checkpoint_stride,
            rbf_max_dist=backbone_target_cutoff if rbf_span_to_cutoff else None,
        )
        self.evc_film = FiLMLayer(hidden_dim, hidden_dim)
        self.evc_state_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.SiLU(), nn.Linear(hidden_dim // 4, hidden_dim)
        )
        nn.init.normal_(self.evc_state_proj[-1].weight, std=0.02)
        nn.init.zeros_(self.evc_state_proj[-1].bias)
        self.output_proj = nn.Linear(hidden_dim, 3)
        element_head_input_dim = hidden_dim + (1 if use_ca_dist_element_feature else 0)
        self.element_type_head = nn.Sequential(
            nn.Linear(element_head_input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_element_classes),
        )
        self.cluster_assignment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max_sidechain_atoms),
        )
        self.occupancy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.use_pocket_contact_prediction = getattr(self, "use_pocket_contact_prediction", False)
        self.pocket_contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.SiLU(), nn.Linear(hidden_dim // 4, 1)
        )
        self.bond_attention = IntraResidueBondAttention(
            hidden_dim=hidden_dim,
            max_sc=max_sidechain_atoms,
            num_heads=4,
            dropout=dropout,
            zero_init_out=zero_init_bond_attention,
        )
        self.shape_prior_n_anchors = shape_prior_n_anchors
        self.shape_prior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.shape_prior_n_anchors * 3)
        )
        nn.init.zeros_(self.shape_prior_head[-1].weight)
        nn.init.zeros_(self.shape_prior_head[-1].bias)
        self.shape_prior_bias_scale = nn.Parameter(torch.tensor(0.1))
        self.cross_residue_packing = CrossResiduePackingAttention(
            hidden_dim=hidden_dim,
            max_sc=max_sidechain_atoms,
            num_heads=4,
            dropout=dropout,
            spatial=cross_residue_packing_spatial,
            k_neighbors=cross_residue_packing_k,
            radius=cross_residue_packing_radius,
            min_seq_sep=cross_residue_packing_min_seq_sep,
        )
        self.neighbor_x0_proj = nn.Linear(NEIGHBOR_X0_FEAT_DIM, hidden_dim, bias=False)
        _init_graft_weight(self.neighbor_x0_proj.weight)
        self.t_cond_delta_proj = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
        _init_graft_weight(self.t_cond_delta_proj.weight)
        self.recycle_index_proj = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
        _init_graft_weight(self.recycle_index_proj.weight)
        self.interaction_intent_head = nn.Linear(hidden_dim, interaction_intent_num_classes)
        nn.init.zeros_(self.interaction_intent_head.weight)
        nn.init.zeros_(self.interaction_intent_head.bias)
        self.interaction_bias_scale = nn.Parameter(torch.tensor(0.05))
        self.residue_mixture_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.residue_centroid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 3)
        )
        self.residue_cloud_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.residue_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.normal_(self.residue_count_head[2].weight, std=0.01)
        nn.init.constant_(self.residue_count_head[2].bias, 4.0)
        self.residue_lrt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.zeros_(self.residue_lrt_head[2].weight)
        nn.init.zeros_(self.residue_lrt_head[2].bias)

    def _t_resolution_weight(self, t: torch.Tensor) -> torch.Tensor:
        """per-TIMESTEP blend weight w(t) for the atom-informed stereochem resolution.

        A smooth function of ``t_norm = t / (T - 1)`` (T = ``num_timesteps``):

            w(t) = clip((0.5 - t_norm) / 0.5, 0, 1)

        so w = 0 for t_norm ≥ 0.5 (high noise -> the STATIC P(D)_prior dominates, matching the cone init
        which used the pure prior) and rises LINEARLY to 1 at t_norm = 0 (clean end -> the atom readout
        fully takes over). This is deliberately NOT the per-epoch training_progress ramp; it depends only
        on the current reverse-step noise level, so it is meaningful at BOTH training and sampling.

        Parameters
        ----------
        t : torch.Tensor
            Diffusion timesteps, shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            w(t) in ``[0, 1]``, shape ``(B,)``.
        """
        t_norm = (t.float() / max(self.num_timesteps - 1, 1)).clamp(0.0, 1.0)
        return ((0.5 - t_norm) / 0.5).clamp(0.0, 1.0)

    def _forward_stereo_t_resolution(
        self,
        sidechain_coords: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        t: torch.Tensor,
        real_weight: torch.Tensor,
        stereo_prior_logit: torch.Tensor,
    ) -> torch.Tensor:
        """Evolving P(D)_t from the current atom cloud blended with the static prior.

        Reads the CURRENT (noised) side-chain atoms' out-of-plane configuration in each residue's local
        backbone frame -- the real-atom-mask-weighted moments of the e3 component
        ``e3 = eq3 · (x_sc - CA)`` (``eq3`` = the frame's third basis vector = backbone-plane normal;
        the same axis whose SIGN of Cβ defines L vs D in Chunk 4a). Three moments are read: the signed
        mean, the mean magnitude, and the mean square -- how the atoms currently sit relative to (and how
        far off) the backbone plane. A small MLP maps them to a scalar ``atom_logit`` per residue, and

            P(D)_t = (1 - w(t)) · sigmoid(stereo_prior_logit) + w(t) · sigmoid(atom_logit)

        with the per-timestep schedule ``w(t)`` (see :meth:`_t_resolution_weight`). At high noise w = 0
        so P(D)_t is EXACTLY the static prior (atoms ignored); by mid-trajectory the atoms take over.

        Parameters
        ----------
        sidechain_coords : torch.Tensor
            Current (noised) side-chain coordinates, ``(B, L, K, 3)``.
        backbone_coords, backbone_mask : torch.Tensor
            Binder backbone atoms ``(B, L, 4, 3)`` and validity ``(B, L, 4)`` -- for the local frames.
        t : torch.Tensor
            Diffusion timesteps ``(B,)``.
        real_weight : torch.Tensor
            Per-slot real-atom weight ``(B, L, K)`` (1 = real, 0 = ghost/PAD).
        stereo_prior_logit : torch.Tensor
            The static Chunk-4a ``rfv2_stereo_logit`` ``(B, L)``; ``sigmoid`` of it = P(D)_prior.

        Returns
        -------
        torch.Tensor
            ``P(D)_t`` in ``[0, 1]``, shape ``(B, L)``.
        """
        eps = 1e-08
        R, ca = build_local_frames(backbone_coords, backbone_mask)
        offset = sidechain_coords - ca.unsqueeze(2)
        eq3 = R[..., :, 2]
        e3 = (offset * eq3.unsqueeze(2)).sum(dim=-1)
        w = real_weight.clamp(min=0.0)
        wsum = w.sum(dim=-1).clamp(min=eps)
        mean_e3 = (e3 * w).sum(dim=-1) / wsum
        mean_abs_e3 = (e3.abs() * w).sum(dim=-1) / wsum
        mean_sq_e3 = (e3**2 * w).sum(dim=-1) / wsum
        feats = torch.stack([mean_e3, mean_abs_e3, mean_sq_e3], dim=-1)
        atom_logit = self.stereo_t_atom_head(feats).squeeze(-1)
        pd_prior = torch.sigmoid(stereo_prior_logit)
        pd_atom = torch.sigmoid(atom_logit)
        wt = self._t_resolution_weight(t).view(-1, 1)
        return (1.0 - wt) * pd_prior + wt * pd_atom

    def forward(
        self,
        sidechain_coords: torch.Tensor,
        seq_mask: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        t: torch.Tensor,
        noised_element_types: torch.Tensor | None = None,
        noised_cluster_ids: torch.Tensor | None = None,
        noised_count: torch.Tensor | None = None,
        prev_element_pred: torch.Tensor | None = None,
        prev_cluster_pred: torch.Tensor | None = None,
        cluster_feature_scale: float = 1.0,
        target_coords: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_backbone_coords: torch.Tensor | None = None,
        target_backbone_mask: torch.Tensor | None = None,
        target_residue_types: torch.Tensor | None = None,
        target_seq_mask: torch.Tensor | None = None,
        target_atom_type: torch.Tensor | None = None,
        target_atom_element_type: torch.Tensor | None = None,
        target_atom_residue_type: torch.Tensor | None = None,
        target_atom_is_backbone: torch.Tensor | None = None,
        gt_sidechain_mask: torch.Tensor | None = None,
        ar_state: torch.Tensor | None = None,
        element_velocity_conditioning: torch.Tensor | None = None,
        count_velocity_conditioning: torch.Tensor | None = None,
        soft_element_probs: torch.Tensor | None = None,
        noised_bond_probs: torch.Tensor | None = None,
        bond_prev_x0: torch.Tensor | None = None,
        bond_prev_mask: torch.Tensor | None = None,
        neighbor_x0_coords: torch.Tensor | None = None,
        neighbor_x0_mask: torch.Tensor | None = None,
        neighbor_x0_trust: torch.Tensor | None = None,
        neighbor_x0_apply: torch.Tensor | None = None,
        t_original_res: torch.Tensor | None = None,
        t_conditioning_res: torch.Tensor | None = None,
        recycle_index: int | None = None,
        vol_hidden: torch.Tensor | None = None,
        vol_density_inject: torch.Tensor | None = None,
        stereo_feedback_ramp: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Predict noise for side-chain coordinates and element types.

        Processes ALL max_sc atom positions for each valid residue (determined by seq_mask).
        The atom mask is derived from element types (PAD=0 means absent).

        Parameters
        ----------
        sidechain_coords : torch.Tensor
            Noised side-chain coordinates of shape (B, L, max_sc, 3).
        seq_mask : torch.Tensor
            Mask for valid residue positions of shape (B, L). Determines which residues to process.
        backbone_coords : torch.Tensor
            Fixed backbone coordinates of shape (B, L, 4, 3).
        backbone_mask : torch.Tensor
            Mask for valid backbone atoms of shape (B, L, 4).
        t : torch.Tensor
            Diffusion timesteps of shape (B,).
        noised_element_types : torch.Tensor, optional
            Noised element types of shape (B, L, max_sc). Values in [0, 4] where
            PAD=0, C=1, N=2, O=3, S=4.
        noised_count : torch.Tensor, optional
            Noised atom count per residue of shape (B, L). Used as INPUT FEATURE.
        prev_mask_pred : torch.Tensor, optional
            Previous mask prediction (P(not PAD)) for self-conditioning.
            Shape (B, L, max_sc). Values in [0, 1]. If None, uses zeros.
        prev_element_pred : torch.Tensor, optional
            Previous element prediction (softmax output) for self-conditioning.
            Shape (B, L, max_sc, NUM_ELEMENT_TYPES). Values in [0, 1]. If None, uses zeros.
        target_coords : torch.Tensor, optional
            Target atom coordinates of shape (B, N_target, 3).
        target_mask : torch.Tensor, optional
            Mask for valid target atoms of shape (B, N_target).
        target_backbone_coords : torch.Tensor, optional
            Target backbone coordinates of shape (B, L_t, 4, 3) for cross-attention.
        target_backbone_mask : torch.Tensor, optional
            Mask for target backbone atoms of shape (B, L_t, 4).
        target_residue_types : torch.Tensor, optional
            Target residue type indices of shape (B, L_t).
        target_seq_mask : torch.Tensor, optional
            Mask for valid target residues of shape (B, L_t).
        target_atom_type : torch.Tensor, optional
            Per-atom type indices of shape (B, N_target). 0-3=backbone (N,CA,C,O), 4-17=sidechain slots.
        target_atom_element_type : torch.Tensor, optional
            Per-atom element type indices of shape (B, N_target). 0=C, 1=N, 2=O, 3=S.
        target_atom_residue_type : torch.Tensor, optional
            Per-atom residue type indices of shape (B, N_target). 0-19=amino acids, 20=unknown.
        target_atom_is_backbone : torch.Tensor, optional
            Per-atom boolean indicating backbone vs sidechain of shape (B, N_target).
        neighbor_x0_coords : torch.Tensor, optional
            Neighbour clean-endpoint estimates of shape (B, L, max_sc, 3), used only when
            ``use_neighbor_x0_packing``. Each residue reads these from OTHER residues only.
        neighbor_x0_mask : torch.Tensor, optional
            Bool mask of which neighbour slots hold a real atom, shape (B, L, max_sc).
        neighbor_x0_trust : torch.Tensor, optional
            Per-residue trust in ``[0, 1]`` of shape (B, L): 1.0 for clean ground-truth
            context residues, ``1 - t_original/T`` for residues still being denoised.
        neighbor_x0_apply : torch.Tensor, optional
            Per-sample gate of shape (B,) in ``[0, 1]``. 0 means the neighbour-x0 residual is
            switched off for that batch item (used by the firing-probability curriculum), which
            makes the item bit-identical to a model without the feature.
        t_original_res : torch.Tensor, optional
            Per-residue ORIGINAL noise level of shape (B, L) -- the noise level the residue's own
            state is actually at.
        t_conditioning_res : torch.Tensor, optional
            Per-residue CONDITIONING noise level of shape (B, L) -- the nominal noise level of the
            neighbour coordinates being supplied. Equal to ``t_original_res`` reproduces the
            historical single-timestep behaviour exactly.
        recycle_index : int, optional
            1-based ABSOLUTE index of the recycle pass this call is (1 = the unconditioned first
            pass). Deliberately absolute rather than ``j/N``: the conditioning a pass receives is a
            function of how many refinements already happened, not of how many are still to come, so
            pass 2-of-2 and pass 2-of-5 must be encoded identically. ``1`` (or ``None``) contributes
            exactly zero.

        Returns
        -------
        outputs : dict[str, torch.Tensor]
            Dictionary containing:
            - 'noise_pred': Predicted noise of shape (B, L, max_sc, 3)
            - 'element_logits': Predicted element type logits of shape (B, L, max_sc, 5)
        """
        batch_size, seq_len, max_sc, _ = sidechain_coords.shape
        if "none" in ("cross_attention", "film"):
            if noised_element_types is not None:
                occ_gate = (noised_element_types != 0).float()
                occ_gate = torch.clamp(occ_gate, min=0.3)
                occ_gate.mean(dim=-1, keepdim=True)
            else:
                torch.ones(batch_size, seq_len, 1, device=sidechain_coords.device)
        backbone_features = self.backbone_encoder(backbone_coords, backbone_mask)
        skip_residue_target = self.num_cross_attn_layers == 0 and False
        if target_backbone_coords is not None and (not skip_residue_target):
            target_features = self.target_encoder(
                target_backbone_coords, target_backbone_mask, target_residue_types, target_seq_mask
            )
            residue_pair_geometry = compute_residue_pair_geometry(backbone_coords, target_backbone_coords)
            residue_pair_bias = self.residue_pair_bias_proj(residue_pair_geometry).permute(0, 3, 1, 2)
            for cross_attn, norm in zip(self.cross_attention_layers, self.cross_attn_norms, strict=False):
                cross_attn_out = cross_attn(
                    query=backbone_features,
                    key=target_features,
                    value=target_features,
                    key_mask=target_seq_mask,
                    attn_bias=residue_pair_bias,
                )
                backbone_features = norm(backbone_features + cross_attn_out)
        else:
            target_features = None
            residue_pair_geometry = None
        if target_backbone_coords is not None and target_features is not None:
            if target_seq_mask is not None:
                target_mask_expanded = target_seq_mask.unsqueeze(-1).float()
                target_global = (target_features * target_mask_expanded).sum(dim=1) / (
                    target_mask_expanded.sum(dim=1) + 1e-08
                )
            else:
                target_global = target_features.mean(dim=1)
            target_global_expanded = target_global.unsqueeze(1).expand_as(backbone_features)
            film_out = self.target_film(backbone_features, target_global_expanded)
            backbone_features = film_out
        residue_frame_outputs = None
        residue_frame_v2_outputs = None
        residue_frame_v2_outputs = self.residue_frame_stream_v2(
            backbone_features.detach(),
            backbone_coords,
            backbone_mask,
            seq_mask,
            target_backbone_coords=target_backbone_coords,
            target_backbone_mask=target_backbone_mask,
            target_residue_types=target_residue_types,
            target_seq_mask=target_seq_mask,
            vol_hidden=vol_hidden.detach() if vol_hidden is not None else None,
        )
        backbone_features = backbone_features + residue_frame_v2_outputs["rf_injection"]
        rfv2_stereo_pd_t = None
        if residue_frame_v2_outputs is not None and residue_frame_v2_outputs.get("rfv2_stereo_logit") is not None:
            if noised_element_types is not None:
                stereo_t_real_w = (noised_element_types != 0).float()
            else:
                stereo_t_real_w = torch.ones(
                    batch_size, seq_len, max_sc, device=sidechain_coords.device, dtype=sidechain_coords.dtype
                )
            rfv2_stereo_pd_t = self._forward_stereo_t_resolution(
                sidechain_coords,
                backbone_coords,
                backbone_mask,
                t,
                stereo_t_real_w,
                residue_frame_v2_outputs["rfv2_stereo_logit"],
            )
        stereo_feedback_latent = None
        stereo_feedback_e3 = None
        if rfv2_stereo_pd_t is not None:
            face_pref = 2.0 * rfv2_stereo_pd_t.detach() - 1.0
            stereo_feedback_latent = self.stereo_feedback_embed(face_pref.unsqueeze(-1))
            stereo_R_fb, _ = build_local_frames(backbone_coords, backbone_mask)
            stereo_e3_axis = stereo_R_fb[..., :, 2]
            stereo_feedback_e3 = face_pref.unsqueeze(-1) * stereo_e3_axis
        residue_frame_hidden_deep = None
        residue_frame_v2_hidden_deep = None
        if residue_frame_v2_outputs is not None:
            residue_frame_v2_hidden_deep = residue_frame_v2_outputs["rf_hidden"]
        vol_hidden_deep = None
        if vol_hidden is not None:
            vol_hidden_deep = vol_hidden.detach()
        vol_density_inject_deep = None
        if vol_density_inject is not None:
            vol_density_inject_deep = vol_density_inject
        vol_hidden_exist = None
        if vol_hidden is not None:
            vol_hidden_exist = vol_hidden.detach()
        global_latent_z_pred = None
        global_latent_conf = None
        latent_clean_summary = None
        clean_per_layer = None
        residue_count_pred = self.residue_count_head(backbone_features).squeeze(-1)
        shape_prior_anchors = None
        K = self.shape_prior_n_anchors
        anchor_offsets = self.shape_prior_head(backbone_features)
        anchor_offsets = anchor_offsets.view(batch_size, seq_len, K, 3)
        ca_coords = backbone_coords[:, :, 1, :]
        shape_prior_anchors = ca_coords.unsqueeze(2) + anchor_offsets
        plan_latent = None
        lrt_input = backbone_features.detach() if getattr(self, "_dlrt_detach", False) else backbone_features
        residue_lrt_delta = self.residue_lrt_head(lrt_input).squeeze(-1)
        ema_decay = getattr(self, "_dlrt_ema_decay", 0.0)
        if ema_decay > 0 and hasattr(self, "_lrt_ema_shadow"):
            with torch.no_grad():
                residue_lrt_delta = self._lrt_ema_shadow(lrt_input.detach()).squeeze(-1)
        time_features = self.time_embed(t.float())
        neighbor_x0_highnoise_factor_all = None
        bb_combined_mask = backbone_mask * seq_mask.unsqueeze(-1) if seq_mask is not None else backbone_mask
        if noised_count is None:
            noised_count = torch.full(
                (batch_size, seq_len), max_sc, dtype=torch.float32, device=sidechain_coords.device
            )
        if prev_element_pred is None:
            prev_element_pred = torch.zeros(
                batch_size, seq_len, max_sc, self.num_element_classes, device=sidechain_coords.device
            )
        noise_pred = torch.zeros_like(sidechain_coords)
        element_logits = torch.zeros(
            batch_size, seq_len, max_sc, self.num_element_classes, device=sidechain_coords.device
        )
        cluster_logits = torch.zeros(batch_size, seq_len, max_sc, max_sc, device=sidechain_coords.device)
        occupancy_logits = torch.zeros(batch_size, seq_len, max_sc, device=sidechain_coords.device)
        residue_centroid = torch.zeros(batch_size, seq_len, 3, device=sidechain_coords.device)
        residue_cloud_logvar = torch.zeros(batch_size, seq_len, device=sidechain_coords.device)
        packing_logits_per_sample: list[torch.Tensor | None] = []
        packing_nb_idx_per_sample: list[torch.Tensor | None] = []
        packing_nb_valid_per_sample: list[torch.Tensor | None] = []
        for b in range(batch_size):
            seq_mask_b = (
                seq_mask[b]
                if seq_mask is not None
                else torch.ones(seq_len, dtype=torch.bool, device=sidechain_coords.device)
            )
            outputs_b = self._forward_single(
                sidechain_coords=sidechain_coords[b],
                seq_mask=seq_mask_b,
                backbone_coords=backbone_coords[b],
                backbone_mask=bb_combined_mask[b],
                backbone_features=backbone_features[b],
                time_features=time_features[b],
                neighbor_x0_highnoise_factor=neighbor_x0_highnoise_factor_all[b]
                if neighbor_x0_highnoise_factor_all is not None
                else None,
                noised_element_types=noised_element_types[b] if noised_element_types is not None else None,
                noised_cluster_ids=noised_cluster_ids[b] if noised_cluster_ids is not None else None,
                noised_count=noised_count[b],
                prev_element_pred=prev_element_pred[b],
                prev_cluster_pred=prev_cluster_pred[b] if prev_cluster_pred is not None else None,
                cluster_feature_scale=cluster_feature_scale,
                target_coords=target_coords[b] if target_coords is not None else None,
                target_mask=target_mask[b] if target_mask is not None else None,
                target_features=target_features[b] if target_features is not None else None,
                target_seq_mask=target_seq_mask[b] if target_seq_mask is not None else None,
                target_residue_pair_geometry=residue_pair_geometry[b] if residue_pair_geometry is not None else None,
                target_atom_type=target_atom_type[b] if target_atom_type is not None else None,
                target_atom_element_type=target_atom_element_type[b] if target_atom_element_type is not None else None,
                target_atom_residue_type=target_atom_residue_type[b] if target_atom_residue_type is not None else None,
                target_atom_is_backbone=target_atom_is_backbone[b] if target_atom_is_backbone is not None else None,
                gt_sidechain_mask=gt_sidechain_mask[b] if gt_sidechain_mask is not None else None,
                ar_state=ar_state[b] if ar_state is not None else None,
                element_velocity_conditioning=element_velocity_conditioning[b]
                if element_velocity_conditioning is not None
                else None,
                count_velocity_conditioning=count_velocity_conditioning[b]
                if count_velocity_conditioning is not None
                else None,
                soft_element_probs=soft_element_probs[b] if soft_element_probs is not None else None,
                noised_bond_probs=noised_bond_probs[b] if noised_bond_probs is not None else None,
                shape_prior_anchors=shape_prior_anchors[b] if shape_prior_anchors is not None else None,
                prev_coord_pred=None,
                bond_prev_x0=bond_prev_x0[b] if bond_prev_x0 is not None else None,
                bond_prev_mask=bond_prev_mask[b] if bond_prev_mask is not None else None,
                plan_latent=plan_latent[b] if plan_latent is not None else None,
                target_coords_for_intent=target_coords[b] if target_coords is not None else None,
                target_mask_for_intent=target_mask[b] if target_mask is not None else None,
                neighbor_x0_coords=neighbor_x0_coords[b] if neighbor_x0_coords is not None else None,
                neighbor_x0_mask=neighbor_x0_mask[b] if neighbor_x0_mask is not None else None,
                neighbor_x0_trust=neighbor_x0_trust[b] if neighbor_x0_trust is not None else None,
                neighbor_x0_apply=neighbor_x0_apply[b] if neighbor_x0_apply is not None else None,
                t_original_res=t_original_res[b] if t_original_res is not None else None,
                t_conditioning_res=t_conditioning_res[b] if t_conditioning_res is not None else None,
                recycle_index=recycle_index,
                latent_clean_summary=latent_clean_summary[b] if latent_clean_summary is not None else None,
                latent_clean_per_layer=[cl[b] for cl in clean_per_layer] if clean_per_layer is not None else None,
                residue_frame_hidden=residue_frame_hidden_deep[b] if residue_frame_hidden_deep is not None else None,
                residue_frame_v2_hidden=residue_frame_v2_hidden_deep[b]
                if residue_frame_v2_hidden_deep is not None
                else None,
                volumetric_hidden=vol_hidden_deep[b] if vol_hidden_deep is not None else None,
                volumetric_density_inject=vol_density_inject_deep[b] if vol_density_inject_deep is not None else None,
                volumetric_hidden_exist=vol_hidden_exist[b] if vol_hidden_exist is not None else None,
                stereo_feedback_hidden=stereo_feedback_latent[b] if stereo_feedback_latent is not None else None,
                stereo_feedback_e3=stereo_feedback_e3[b] if stereo_feedback_e3 is not None else None,
                stereo_feedback_ramp=stereo_feedback_ramp,
            )
            noise_pred[b] = outputs_b["noise_pred"]
            element_logits[b] = outputs_b["element_logits"]
            cluster_logits[b] = outputs_b["cluster_logits"]
            occupancy_logits[b] = outputs_b["occupancy_logits"]
            residue_centroid[b] = outputs_b["residue_centroid"]
            residue_cloud_logvar[b] = outputs_b["residue_cloud_logvar"]
            packing_logits_per_sample.append(outputs_b.get("packing_contact_logits"))
            packing_nb_idx_per_sample.append(outputs_b.get("packing_neighbor_idx"))
            packing_nb_valid_per_sample.append(outputs_b.get("packing_neighbor_valid"))
            if b == 0:
                bond_logits_all = outputs_b.get("bond_logits")
                cl = outputs_b.get("contact_logits")
                contact_logits_all = cl.unsqueeze(0) if cl is not None else None
                packing_logits_all = outputs_b.get("packing_contact_logits")
                packing_nb_idx_all = outputs_b.get("packing_neighbor_idx")
                packing_nb_valid_all = outputs_b.get("packing_neighbor_valid")
                valence_logits_all = outputs_b.get("valence_logits")
                interaction_intent_all = outputs_b.get("interaction_intent_logits")
        return {
            "noise_pred": noise_pred,
            "element_logits": element_logits,
            "cluster_logits": cluster_logits,
            "occupancy_logits": occupancy_logits,
            "residue_centroid": residue_centroid,
            "residue_cloud_logvar": residue_cloud_logvar,
            "residue_count_pred": residue_count_pred,
            "residue_lrt_delta": residue_lrt_delta,
            "backbone_features": backbone_features,
            "rf_centroid_pred": residue_frame_outputs["rf_centroid_pred"]
            if residue_frame_outputs is not None
            else None,
            "rf_radial_pred": residue_frame_outputs["rf_radial_pred"] if residue_frame_outputs is not None else None,
            "rf_count_pred": residue_frame_outputs["rf_count_pred"] if residue_frame_outputs is not None else None,
            "rfv2_centroid_pred": residue_frame_v2_outputs["rfv2_centroid_pred"]
            if residue_frame_v2_outputs is not None
            else None,
            "rfv2_chi1_pred": residue_frame_v2_outputs["rfv2_chi1_pred"]
            if residue_frame_v2_outputs is not None
            else None,
            "rfv2_stereo_logit": residue_frame_v2_outputs["rfv2_stereo_logit"]
            if residue_frame_v2_outputs is not None
            else None,
            "rfv2_stereo_pd_t": rfv2_stereo_pd_t,
            "rfv2_stereo_w_t": self._t_resolution_weight(t) if rfv2_stereo_pd_t is not None else None,
            "bond_logits": bond_logits_all if batch_size > 0 else None,
            "contact_logits": contact_logits_all if batch_size > 0 else None,
            "packing_contact_logits": packing_logits_all if batch_size > 0 else None,
            "packing_neighbor_idx": packing_nb_idx_all if batch_size > 0 else None,
            "packing_neighbor_valid": packing_nb_valid_all if batch_size > 0 else None,
            "packing_contact_logits_per_sample": packing_logits_per_sample,
            "packing_neighbor_idx_per_sample": packing_nb_idx_per_sample,
            "packing_neighbor_valid_per_sample": packing_nb_valid_per_sample,
            "valence_logits": valence_logits_all if batch_size > 0 else None,
            "shape_prior_anchors": shape_prior_anchors,
            "interaction_intent_logits": interaction_intent_all if batch_size > 0 else None,
            "plan_latent": plan_latent,
            "global_latent_z_pred": global_latent_z_pred,
            "global_latent_conf": global_latent_conf,
        }

    def _pool_cluster_nodes(
        self,
        coords_valid: torch.Tensor,
        node_features: torch.Tensor,
        cluster_condition: torch.Tensor,
        residue_idx_valid: torch.Tensor,
        cluster_ids_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool sidechain particles into one SE(3) node per active cluster within each residue."""
        if cluster_ids_valid is None or len(coords_valid) == 0:
            identity = torch.arange(len(coords_valid), device=coords_valid.device)
            return (coords_valid, node_features, cluster_condition, residue_idx_valid, identity)
        cluster_keys = residue_idx_valid * self.max_sidechain_atoms + cluster_ids_valid
        unique_keys, inverse = torch.unique(cluster_keys, sorted=False, return_inverse=True)
        n_clusters = unique_keys.numel()
        pooled_counts = torch.bincount(inverse, minlength=n_clusters).to(coords_valid.dtype).unsqueeze(-1)
        pooled_coords = torch.zeros(
            n_clusters, coords_valid.shape[-1], device=coords_valid.device, dtype=coords_valid.dtype
        )
        pooled_features = torch.zeros(
            n_clusters, node_features.shape[-1], device=node_features.device, dtype=node_features.dtype
        )
        pooled_cluster_condition = torch.zeros(
            n_clusters, cluster_condition.shape[-1], device=cluster_condition.device, dtype=cluster_condition.dtype
        )
        pooled_coords.index_add_(0, inverse, coords_valid)
        pooled_features.index_add_(0, inverse, node_features)
        pooled_cluster_condition.index_add_(0, inverse, cluster_condition)
        pooled_coords = pooled_coords / pooled_counts.clamp_min(1.0)
        pooled_features = pooled_features / pooled_counts.clamp_min(1.0)
        pooled_cluster_condition = pooled_cluster_condition / pooled_counts.clamp_min(1.0)
        pooled_residue_idx = torch.div(unique_keys, self.max_sidechain_atoms, rounding_mode="floor")
        return (pooled_coords, pooled_features, pooled_cluster_condition, pooled_residue_idx, inverse)

    def _forward_single(
        self,
        sidechain_coords: torch.Tensor,
        seq_mask: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        backbone_features: torch.Tensor,
        time_features: torch.Tensor,
        noised_element_types: torch.Tensor | None = None,
        noised_cluster_ids: torch.Tensor | None = None,
        noised_count: torch.Tensor | None = None,
        prev_element_pred: torch.Tensor | None = None,
        prev_cluster_pred: torch.Tensor | None = None,
        cluster_feature_scale: float = 1.0,
        target_coords: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
        target_seq_mask: torch.Tensor | None = None,
        target_residue_pair_geometry: torch.Tensor | None = None,
        target_atom_type: torch.Tensor | None = None,
        target_atom_element_type: torch.Tensor | None = None,
        target_atom_residue_type: torch.Tensor | None = None,
        target_atom_is_backbone: torch.Tensor | None = None,
        gt_sidechain_mask: torch.Tensor | None = None,
        ar_state: torch.Tensor | None = None,
        element_velocity_conditioning: torch.Tensor | None = None,
        count_velocity_conditioning: torch.Tensor | None = None,
        soft_element_probs: torch.Tensor | None = None,
        noised_bond_probs: torch.Tensor | None = None,
        shape_prior_anchors: torch.Tensor | None = None,
        prev_coord_pred: torch.Tensor | None = None,
        bond_prev_x0: torch.Tensor | None = None,
        bond_prev_mask: torch.Tensor | None = None,
        plan_latent: torch.Tensor | None = None,
        target_coords_for_intent: torch.Tensor | None = None,
        target_mask_for_intent: torch.Tensor | None = None,
        neighbor_x0_highnoise_factor: torch.Tensor | None = None,
        neighbor_x0_coords: torch.Tensor | None = None,
        neighbor_x0_mask: torch.Tensor | None = None,
        neighbor_x0_trust: torch.Tensor | None = None,
        neighbor_x0_apply: torch.Tensor | None = None,
        t_original_res: torch.Tensor | None = None,
        t_conditioning_res: torch.Tensor | None = None,
        recycle_index: int | None = None,
        latent_clean_summary: torch.Tensor | None = None,
        latent_clean_per_layer: list[torch.Tensor] | None = None,
        residue_frame_hidden: torch.Tensor | None = None,
        residue_frame_v2_hidden: torch.Tensor | None = None,
        volumetric_hidden: torch.Tensor | None = None,
        volumetric_density_inject: torch.Tensor | None = None,
        volumetric_hidden_exist: torch.Tensor | None = None,
        stereo_feedback_hidden: torch.Tensor | None = None,
        stereo_feedback_e3: torch.Tensor | None = None,
        stereo_feedback_ramp: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for a single sample.

        Processes ALL max_sc atom positions for each valid residue (determined by seq_mask).

        Parameters
        ----------
        sidechain_coords : torch.Tensor
            Shape (L, max_sc, 3).
        seq_mask : torch.Tensor
            Shape (L,). Determines which residues are valid (not padding).
        backbone_coords : torch.Tensor
            Shape (L, 4, 3).
        backbone_mask : torch.Tensor
            Shape (L, 4).
        backbone_features : torch.Tensor
            Shape (L, hidden).
        time_features : torch.Tensor
            Shape (hidden,).
        noised_element_types : torch.Tensor, optional
            Shape (L, max_sc). Values in [0, 4] where PAD=0, C=1, N=2, O=3, S=4.
        prev_mask_pred : torch.Tensor, optional
            Shape (L, max_sc). Previous mask prediction for self-conditioning.
        prev_element_pred : torch.Tensor, optional
            Shape (L, max_sc, NUM_ELEMENT_TYPES). Previous element prediction for self-conditioning.
        target_coords : torch.Tensor, optional
            Shape (N_target, 3).
        target_mask : torch.Tensor, optional
            Shape (N_target,).
        target_features : torch.Tensor, optional
            Shape (L_target, hidden) per-residue target features after residue-level conditioning.
        target_seq_mask : torch.Tensor, optional
            Shape (L_target,) mask for valid target residues.
        target_residue_pair_geometry : torch.Tensor, optional
            Shape (L, L_target, pair_geom_dim) ProteinMPNN-style peptide-target
            residue geometry features.
        target_atom_type : torch.Tensor, optional
            Shape (N_target,). Per-atom type: 0-3=backbone (N,CA,C,O), 4-17=sidechain slots.
        target_atom_element_type : torch.Tensor, optional
            Shape (N_target,). Per-atom element type: 0=C, 1=N, 2=O, 3=S.
        target_atom_residue_type : torch.Tensor, optional
            Shape (N_target,). Per-atom residue type: 0-19=amino acids, 20=unknown.
        target_atom_is_backbone : torch.Tensor, optional
            Shape (N_target,). Boolean indicating backbone (True) vs sidechain (False).

        Returns
        -------
        outputs : dict[str, torch.Tensor]
            Dictionary containing:
            - 'noise_pred': Shape (L, max_sc, 3)
            - 'element_logits': Shape (L, max_sc, 5)
        """
        seq_len, max_sc, _ = sidechain_coords.shape
        device = sidechain_coords.device
        residue_mask = seq_mask.unsqueeze(-1).expand(-1, max_sc)
        sc_valid_idx = torch.where(residue_mask.reshape(-1))[0]
        if len(sc_valid_idx) == 0:
            return {
                "noise_pred": torch.zeros_like(sidechain_coords),
                "element_logits": torch.zeros(seq_len, max_sc, self.num_element_classes, device=device),
                "cluster_logits": torch.zeros(seq_len, max_sc, max_sc, device=device),
            }
        sc_cluster_condition = torch.zeros(len(sc_valid_idx), self.hidden_dim, device=device)
        sc_coords_flat = sidechain_coords.view(-1, 3)
        sc_coords_valid = sc_coords_flat[sc_valid_idx]
        sc_residue_idx_flat = torch.arange(seq_len, device=device).unsqueeze(-1).expand(-1, max_sc).reshape(-1)
        sc_residue_idx_valid = sc_residue_idx_flat[sc_valid_idx]
        sc_atom_idx_flat = torch.arange(max_sc, device=device).unsqueeze(0).expand(seq_len, -1).reshape(-1)
        sc_atom_idx_valid = sc_atom_idx_flat[sc_valid_idx]
        if noised_count is not None:
            noised_count_valid = noised_count[sc_residue_idx_valid].unsqueeze(-1)
        else:
            noised_count_valid = torch.full((len(sc_valid_idx), 1), float(max_sc), device=device)
        sc_noised_count_features = self.noised_count_embed(noised_count_valid)
        sc_residue_idx_clamped = sc_residue_idx_valid.clamp(0, self.max_residue_idx - 1)
        sc_residue_idx_features = self.residue_idx_embed(sc_residue_idx_clamped)
        if prev_element_pred is not None:
            prev_element_pred_flat = prev_element_pred.reshape(-1, self.num_element_classes)
            prev_element_pred_valid = prev_element_pred_flat[sc_valid_idx]
        else:
            prev_element_pred_valid = torch.zeros(len(sc_valid_idx), self.num_element_classes, device=device)
        sc_self_cond_element_features = self.self_cond_element_proj(prev_element_pred_valid)
        if noised_cluster_ids is not None:
            cluster_ids_flat = noised_cluster_ids.reshape(-1)
            cluster_ids_valid = cluster_ids_flat[sc_valid_idx].clamp(0, max_sc - 1)
        else:
            cluster_ids_valid = sc_atom_idx_valid
        sc_cluster_features = self.cluster_id_embed(cluster_ids_valid)
        if prev_cluster_pred is not None:
            prev_cluster_pred_flat = prev_cluster_pred.reshape(-1, max_sc)
            prev_cluster_pred_valid = prev_cluster_pred_flat[sc_valid_idx]
        else:
            prev_cluster_pred_valid = F.one_hot(sc_atom_idx_valid, num_classes=max_sc).float()
        sc_self_cond_cluster_features = self.self_cond_cluster_proj(prev_cluster_pred_valid.float())
        sc_cluster_features = sc_cluster_features * cluster_feature_scale
        sc_self_cond_cluster_features = sc_self_cond_cluster_features * cluster_feature_scale
        if soft_element_probs is not None:
            probs_flat = soft_element_probs.reshape(-1, soft_element_probs.shape[-1])
            probs_valid = probs_flat[sc_valid_idx]
            sc_element_features = probs_valid @ self.element_type_embed.weight
        elif noised_element_types is not None:
            element_types_flat = noised_element_types.view(-1)
            element_types_valid = element_types_flat[sc_valid_idx]
            element_types_valid = element_types_valid.clamp(0, self.num_element_classes - 1)
            sc_element_features = self.element_type_embed(element_types_valid)
        else:
            sc_element_features = self.element_type_embed(
                torch.zeros(len(sc_valid_idx), device=device, dtype=torch.long)
            )
        sc_atom_features = self.sc_atom_embed(sc_atom_idx_valid)
        sc_backbone_context = backbone_features[sc_residue_idx_valid]
        sc_time_features = time_features.unsqueeze(0).expand(len(sc_valid_idx), -1)
        cat_features = [
            sc_atom_features,
            sc_element_features,
            sc_noised_count_features,
            sc_self_cond_element_features,
            sc_cluster_features,
            sc_self_cond_cluster_features,
            sc_residue_idx_features,
            sc_backbone_context,
            sc_time_features,
        ]
        sc_node_features = self.sc_node_proj(torch.cat(cat_features, dim=-1))
        if ar_state is not None:
            ar_state_flat = ar_state.reshape(-1)
            ar_state_valid = ar_state_flat[sc_valid_idx]
            ar_state_residual = self.ar_state_proj(self.ar_state_embed(ar_state_valid))
            ar_state_apply = (ar_state_valid > 0).unsqueeze(-1).to(ar_state_residual.dtype)
            sc_node_features = sc_node_features + ar_state_residual * ar_state_apply
        gate = 1.0 if neighbor_x0_apply is None else neighbor_x0_apply.to(sc_node_features.dtype)
        if neighbor_x0_coords is not None:
            nb_mask = (
                neighbor_x0_mask.bool()
                if neighbor_x0_mask is not None
                else torch.ones(seq_len, max_sc, dtype=torch.bool, device=device)
            )
            nb_trust = (
                neighbor_x0_trust.to(sc_node_features.dtype)
                if neighbor_x0_trust is not None
                else torch.ones(seq_len, device=device, dtype=sc_node_features.dtype)
            )
            _nx0_coords = neighbor_x0_coords.to(sc_coords_valid.dtype)
            _nb_valid = nb_mask & seq_mask.bool().unsqueeze(-1)
            _c_add = self.neighbor_x0_corrupt_add_prob
            _c_drop = self.neighbor_x0_corrupt_drop_prob
            _c_noise_prob = self.neighbor_x0_corrupt_noise_prob
            _c_noise_std = self.neighbor_x0_corrupt_coord_noise
            _c_disc = self.neighbor_x0_corrupt_disconnect_prob
            _gate_on = neighbor_x0_apply is None or bool((neighbor_x0_apply != 0).any())
            nb_feats = compute_neighbor_x0_features(
                query_coords=sc_coords_valid,
                query_residue_idx=sc_residue_idx_valid,
                neighbor_coords=_nx0_coords,
                neighbor_valid=_nb_valid,
                neighbor_trust=nb_trust,
                radius=self.neighbor_x0_packing_radius,
            )
            nb_residual = self.neighbor_x0_proj(nb_feats.to(sc_node_features.dtype))
            if neighbor_x0_highnoise_factor is not None:
                nb_residual = nb_residual * neighbor_x0_highnoise_factor.to(nb_residual.dtype)
            sc_node_features = sc_node_features + nb_residual * gate
        if t_original_res is not None and t_conditioning_res is not None:
            t_o = t_original_res.to(sc_node_features.dtype).reshape(-1)
            t_c = t_conditioning_res.to(sc_node_features.dtype).reshape(-1)
            delta_embed = self.time_embed(t_c) - self.time_embed(t_o)
            delta_scalar = ((t_c - t_o) / max(self.num_timesteps, 1)).unsqueeze(-1)
            t_delta_res = self.t_cond_delta_proj(torch.cat([delta_embed, delta_scalar], dim=-1))
            sc_node_features = sc_node_features + t_delta_res[sc_residue_idx_valid] * gate
        if recycle_index is not None:
            j = float(max(1, int(recycle_index)))
            j_t = torch.tensor([j], device=device, dtype=sc_node_features.dtype)
            j_embed = self.time_embed(j_t) - self.time_embed(torch.ones_like(j_t))
            j_scalar = (1.0 - 1.0 / j_t).unsqueeze(-1)
            rc_res = self.recycle_index_proj(torch.cat([j_embed, j_scalar], dim=-1))
            sc_node_features = sc_node_features + rc_res * gate
        sc_target_attn = None
        if target_features is not None:
            sc_target_bias = None
            if target_residue_pair_geometry is not None:
                sc_pair_geometry = target_residue_pair_geometry[sc_residue_idx_valid]
                sc_target_bias = self.sidechain_pair_bias_proj(sc_pair_geometry).permute(2, 0, 1).unsqueeze(0)
            sc_target_attn = (
                self.sidechain_target_attention(
                    query=sc_node_features.unsqueeze(0),
                    key=target_features.unsqueeze(0),
                    value=target_features.unsqueeze(0),
                    key_mask=target_seq_mask.unsqueeze(0) if target_seq_mask is not None else None,
                    attn_bias=sc_target_bias,
                ).squeeze(0)
                * self.target_condition_scale
            )
            sc_cluster_condition = self.cluster_target_context_proj(
                sc_target_attn * self.cluster_target_condition_scale
            )
            sc_node_features = self.sidechain_target_film(sc_node_features, sc_target_attn)
            sc_node_features = self.sidechain_target_norm(sc_node_features + sc_target_attn)
        sc_graph_coords, sc_graph_features, sc_graph_cluster_condition, sc_graph_residue_idx, sc_cluster_inverse = (
            self._pool_cluster_nodes(
                sc_coords_valid,
                sc_node_features,
                sc_cluster_condition,
                sc_residue_idx_valid,
                cluster_ids_valid if noised_cluster_ids is not None else None,
            )
        )
        n_binder_sc = len(sc_graph_coords)
        bb_valid_idx = torch.where(backbone_mask.view(-1))[0]
        bb_coords_flat = backbone_coords.view(-1, 3)
        bb_coords_valid = bb_coords_flat[bb_valid_idx]
        bb_residue_idx_flat = torch.arange(seq_len, device=device).unsqueeze(-1).expand(-1, 4).reshape(-1)
        bb_residue_idx_valid = bb_residue_idx_flat[bb_valid_idx]
        bb_atom_idx_flat = torch.arange(4, device=device).unsqueeze(0).expand(seq_len, -1).reshape(-1)
        bb_atom_idx_valid = bb_atom_idx_flat[bb_valid_idx]
        bb_atom_features = self.bb_atom_embed(bb_atom_idx_valid)
        bb_backbone_context = backbone_features[bb_residue_idx_valid]
        bb_time_features = time_features.unsqueeze(0).expand(len(bb_valid_idx), -1)
        bb_node_features = self.bb_node_proj(
            torch.cat([bb_atom_features, bb_backbone_context, bb_time_features], dim=-1)
        )
        n_binder_bb = len(bb_valid_idx)
        target_coords_valid = None
        target_node_features = None
        n_target = 0
        if target_coords is not None and target_mask is not None:
            target_valid_idx = torch.where(target_mask)[0]
            if len(target_valid_idx) > 0:
                target_coords_valid = target_coords[target_valid_idx]
                n_target = len(target_valid_idx)
                if (
                    target_atom_type is not None
                    and target_atom_element_type is not None
                    and (target_atom_residue_type is not None)
                    and (target_atom_is_backbone is not None)
                ):
                    atom_type_valid = target_atom_type[target_valid_idx]
                    element_type_valid = target_atom_element_type[target_valid_idx]
                    residue_type_valid = target_atom_residue_type[target_valid_idx]
                    is_backbone_valid = target_atom_is_backbone[target_valid_idx].long()
                    atom_type_valid = atom_type_valid.clamp(0, 17)
                    element_type_valid = element_type_valid.clamp(0, NUM_ELEMENT_TYPES - 1)
                    residue_type_valid = residue_type_valid.clamp(0, 20)
                    atom_type_feat = self.target_atom_type_embed(atom_type_valid)
                    element_type_feat = self.target_element_type_embed(element_type_valid)
                    residue_type_feat = self.target_residue_type_embed(residue_type_valid)
                    is_backbone_feat = self.target_is_backbone_embed(is_backbone_valid)
                    target_cat_features = torch.cat(
                        [atom_type_feat, element_type_feat, residue_type_feat, is_backbone_feat], dim=-1
                    )
                    target_node_features = self.target_atom_proj(target_cat_features)
                else:
                    target_node_features = torch.zeros(n_target, self.hidden_dim, device=device)
        all_coords_list = [sc_graph_coords]
        all_features_list = [sc_graph_features]
        if n_binder_bb > 0:
            all_coords_list.append(bb_coords_valid)
            all_features_list.append(bb_node_features)
        if n_target > 0:
            all_coords_list.append(target_coords_valid)
            all_features_list.append(target_node_features)
        all_coords = torch.cat(all_coords_list, dim=0)
        all_features = torch.cat(all_features_list, dim=0)
        binder_ca_coords = backbone_coords[:, 1, :]
        ca_mask = bb_atom_idx_valid == 1
        binder_ca_only_coords = bb_coords_valid[ca_mask]
        ca_bb_indices = torch.where(ca_mask)[0]
        edge_index, edge_type, _, _, _ = build_multi_type_radius_graph(
            binder_sc_coords=sc_graph_coords,
            binder_sc_residue_idx=sc_graph_residue_idx,
            binder_bb_coords=bb_coords_valid if n_binder_bb > 0 else None,
            binder_bb_residue_idx=bb_residue_idx_valid if n_binder_bb > 0 else None,
            target_coords=target_coords_valid,
            intra_residue_cutoff=self.intra_residue_cutoff,
            inter_residue_cutoff=self.inter_residue_cutoff,
            sidechain_target_cutoff=self.sidechain_target_cutoff,
            backbone_target_cutoff=self.backbone_target_cutoff,
            ca_ca_prefilter=self.ca_ca_prefilter,
            binder_ca_coords=binder_ca_coords,
            target_ca_coords=None,
            binder_ca_only_coords=binder_ca_only_coords if n_binder_bb > 0 else None,
            binder_ca_only_bb_indices=ca_bb_indices if n_binder_bb > 0 else None,
            num_edge_types=self.num_edge_types,
        )
        edge_attr = self.edge_embed(edge_type)
        layer_conditioning = None
        if residue_frame_v2_hidden is not None:
            n_total = all_features.shape[0]
            rfv2_latent = residue_frame_v2_hidden.to(all_features.dtype)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.residue_frame_v2_deep_proj)
            for li, proj in enumerate(self.residue_frame_v2_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(rfv2_latent[sc_graph_residue_idx])
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(rfv2_latent[bb_residue_idx_valid])
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )
        if volumetric_density_inject is not None or volumetric_hidden is not None:
            n_total = all_features.shape[0]
            _vol_src = volumetric_density_inject if volumetric_density_inject is not None else volumetric_hidden
            vol_latent = _vol_src.to(all_features.dtype)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.volumetric_deep_proj)
            for li, proj in enumerate(self.volumetric_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(vol_latent[sc_graph_residue_idx])
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(vol_latent[bb_residue_idx_valid])
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )
        if sc_target_attn is not None and n_binder_sc > 0:
            n_total = all_features.shape[0]
            tgt_latent = sc_target_attn.detach().to(all_features.dtype)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.target_deep_proj)
            for li, proj in enumerate(self.target_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                node_cond[:n_binder_sc] = proj(tgt_latent)
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )
        if stereo_feedback_hidden is not None and stereo_feedback_ramp > 0.0:
            n_total = all_features.shape[0]
            st_latent = stereo_feedback_hidden.to(all_features.dtype)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.stereo_feedback_deep_proj)
            for li, proj in enumerate(self.stereo_feedback_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(st_latent[sc_graph_residue_idx])
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(st_latent[bb_residue_idx_valid])
                node_cond = node_cond * stereo_feedback_ramp
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )
        out_features, out_coords = self.transformer(
            all_features, all_coords, edge_index, edge_attr, layer_conditioning=layer_conditioning
        )
        _ = out_coords[:n_binder_sc]
        sc_out_features = out_features[:n_binder_sc]
        sc_out_features = sc_out_features[sc_cluster_inverse]
        if element_velocity_conditioning is not None:
            evc_flat = element_velocity_conditioning[:seq_len].reshape(-1)
            evc_valid = evc_flat[sc_valid_idx].unsqueeze(-1)
            evc_embed = self.evc_state_proj(evc_valid)
            sc_out_features = self.evc_film(sc_out_features, evc_embed)
        bond_logits_out = None
        n_valid_res_ba = int(sc_residue_idx_valid.max().item()) + 1
        h_3d = torch.zeros(n_valid_res_ba, max_sc, sc_out_features.shape[-1], device=device)
        coords_3d = torch.zeros(n_valid_res_ba, max_sc, 3, device=device)
        mask_3d = torch.zeros(n_valid_res_ba, max_sc, dtype=torch.bool, device=device)
        slot_within_res = sc_valid_idx % max_sc
        h_3d[sc_residue_idx_valid, slot_within_res] = sc_out_features
        coords_3d[sc_residue_idx_valid, slot_within_res] = sc_coords_valid
        mask_3d[sc_residue_idx_valid, slot_within_res] = True
        noised_bp_3d = None
        if noised_bond_probs is not None:
            noised_bp_3d = noised_bond_probs[:n_valid_res_ba]
        h_3d, bond_logits_3d = self.bond_attention(h_3d, coords_3d, mask_3d, noised_bond_probs=noised_bp_3d)
        bond_logits_out = bond_logits_3d
        sc_out_features = h_3d[sc_residue_idx_valid, slot_within_res]
        packing_contact_logits_out = None
        packing_neighbor_idx_out = None
        packing_neighbor_valid_out = None
        n_valid_res_crp = int(sc_residue_idx_valid.max().item()) + 1
        h_3d_crp = torch.zeros(n_valid_res_crp, max_sc, sc_out_features.shape[-1], device=device)
        coords_3d_crp = torch.zeros(n_valid_res_crp, max_sc, 3, device=device)
        mask_3d_crp = torch.zeros(n_valid_res_crp, max_sc, dtype=torch.bool, device=device)
        slot_within_res_crp = sc_valid_idx % max_sc
        h_3d_crp[sc_residue_idx_valid, slot_within_res_crp] = sc_out_features
        coords_3d_crp[sc_residue_idx_valid, slot_within_res_crp] = sc_coords_valid
        mask_3d_crp[sc_residue_idx_valid, slot_within_res_crp] = True
        crp_residue_pos = None
        h_3d_crp, packing_contact_logits_out, packing_neighbor_idx_out, packing_neighbor_valid_out = (
            self.cross_residue_packing(h_3d_crp, coords_3d_crp, mask_3d_crp, residue_pos=crp_residue_pos)
        )
        sc_out_features = h_3d_crp[sc_residue_idx_valid, slot_within_res_crp]
        valence_logits_valid = None
        mixture_features = sc_out_features
        occupancy_logits_valid = self.occupancy_head(mixture_features)
        contact_logits_valid = self.pocket_contact_head(sc_out_features)
        noise_valid = self.output_proj(sc_out_features)
        if stereo_feedback_e3 is not None and stereo_feedback_ramp > 0.0:
            e3_per_slot = stereo_feedback_e3[sc_residue_idx_valid].to(noise_valid.dtype)
            noise_valid = noise_valid + stereo_feedback_ramp * self.stereo_feedback_face_scale * e3_per_slot
        if shape_prior_anchors is not None:
            slot_anchors = shape_prior_anchors[sc_residue_idx_valid]
            slot_coords = sc_coords_valid.unsqueeze(1)
            anchor_dists = (slot_coords - slot_anchors).norm(dim=-1)
            nearest_idx = anchor_dists.argmin(dim=-1)
            nearest_anchor = slot_anchors[torch.arange(len(nearest_idx), device=device), nearest_idx]
            direction = nearest_anchor - sc_coords_valid
            dist_to_anchor = direction.norm(dim=-1, keepdim=True).clamp(min=0.1)
            shape_bias = direction / dist_to_anchor * self.shape_prior_bias_scale
            noise_valid = noise_valid + shape_bias
        interaction_intent_logits_valid = None
        interaction_intent_logits_valid = self.interaction_intent_head(sc_out_features)
        if target_coords_for_intent is not None and target_mask_for_intent is not None:
            p_interact = 1.0 - torch.softmax(interaction_intent_logits_valid, dim=-1)[:, 0]
            tgt_valid = target_coords_for_intent[target_mask_for_intent]
            if tgt_valid.shape[0] > 0:
                dists = torch.cdist(sc_coords_valid, tgt_valid)
                nearest_tgt = tgt_valid[dists.argmin(dim=-1)]
                intent_direction = nearest_tgt - sc_coords_valid
                intent_dist = intent_direction.norm(dim=-1, keepdim=True).clamp(min=0.1)
                intent_bias = intent_direction / intent_dist * p_interact.unsqueeze(-1) * self.interaction_bias_scale
                noise_valid = noise_valid + intent_bias
        element_input = sc_out_features
        element_logits_valid = self.element_type_head(element_input)
        n_valid_res = seq_mask.sum().item() if seq_mask is not None else seq_len
        residue_pooled = torch.zeros(n_valid_res, mixture_features.shape[-1], device=device)
        unique_res, slot_to_res = torch.unique(sc_residue_idx_valid, return_inverse=True)
        residue_pooled.index_add_(0, slot_to_res, mixture_features)
        res_counts = torch.zeros(n_valid_res, device=device)
        res_counts.index_add_(0, slot_to_res, torch.ones(len(slot_to_res), device=device))
        residue_pooled = residue_pooled / res_counts.unsqueeze(-1).clamp(min=1)
        residue_pooled = self.residue_mixture_pool(residue_pooled)
        residue_centroid_valid = self.residue_centroid_head(residue_pooled)
        residue_cloud_logvar_valid = self.residue_cloud_logvar_head(residue_pooled).squeeze(-1)
        cluster_logits_valid = self.cluster_assignment_head(
            sc_out_features + sc_graph_cluster_condition[sc_cluster_inverse]
        )
        noise_flat = torch.zeros(seq_len * max_sc, 3, device=device, dtype=noise_valid.dtype)
        noise_flat[sc_valid_idx] = noise_valid
        noise_pred = noise_flat.view(seq_len, max_sc, 3)
        element_logits_flat = torch.zeros(
            seq_len * max_sc, self.num_element_classes, device=device, dtype=element_logits_valid.dtype
        )
        element_logits_flat[sc_valid_idx] = element_logits_valid
        element_logits = element_logits_flat.view(seq_len, max_sc, self.num_element_classes)
        if volumetric_hidden_exist is not None:
            exist_bias = self.volumetric_existence_proj(volumetric_hidden_exist.to(element_logits.dtype))
            pad_bias = torch.zeros_like(element_logits)
            pad_bias[..., ELEMENT_PAD] = -exist_bias
            element_logits = element_logits + pad_bias
        cluster_logits_flat = torch.zeros(seq_len * max_sc, max_sc, device=device, dtype=cluster_logits_valid.dtype)
        cluster_logits_flat[sc_valid_idx] = cluster_logits_valid
        cluster_logits = cluster_logits_flat.view(seq_len, max_sc, max_sc)
        occupancy_logits_flat = torch.zeros(seq_len * max_sc, 1, device=device, dtype=occupancy_logits_valid.dtype)
        occupancy_logits_flat[sc_valid_idx] = occupancy_logits_valid
        occupancy_logits = occupancy_logits_flat.view(seq_len, max_sc)
        contact_logits_flat = torch.zeros(seq_len * max_sc, 1, device=device, dtype=contact_logits_valid.dtype)
        contact_logits_flat[sc_valid_idx] = contact_logits_valid
        contact_logits = contact_logits_flat.view(seq_len, max_sc)
        residue_centroid = torch.zeros(seq_len, 3, device=device, dtype=residue_centroid_valid.dtype)
        residue_centroid[unique_res] = residue_centroid_valid
        residue_cloud_logvar = torch.zeros(seq_len, device=device, dtype=residue_cloud_logvar_valid.dtype)
        residue_cloud_logvar[unique_res] = residue_cloud_logvar_valid
        valence_logits = None
        if valence_logits_valid is not None:
            valence_logits_flat = torch.zeros(seq_len * max_sc, 5, device=device, dtype=valence_logits_valid.dtype)
            valence_logits_flat[sc_valid_idx] = valence_logits_valid
            valence_logits = valence_logits_flat.view(seq_len, max_sc, 5)
        interaction_intent_logits = None
        if interaction_intent_logits_valid is not None:
            interaction_intent_logits = torch.zeros(
                seq_len * max_sc, 4, device=device, dtype=interaction_intent_logits_valid.dtype
            )
            interaction_intent_logits[sc_valid_idx] = interaction_intent_logits_valid
            interaction_intent_logits = interaction_intent_logits.view(seq_len, max_sc, 4)
        return {
            "noise_pred": noise_pred,
            "element_logits": element_logits,
            "cluster_logits": cluster_logits,
            "occupancy_logits": occupancy_logits,
            "residue_centroid": residue_centroid,
            "residue_cloud_logvar": residue_cloud_logvar,
            "bond_logits": bond_logits_out,
            "contact_logits": contact_logits,
            "valence_logits": valence_logits,
            "packing_contact_logits": packing_contact_logits_out,
            "packing_neighbor_idx": packing_neighbor_idx_out,
            "packing_neighbor_valid": packing_neighbor_valid_out,
            "shape_prior_anchors": shape_prior_anchors,
            "interaction_intent_logits": interaction_intent_logits,
            "plan_latent": plan_latent,
        }


class InverseFoldingDiffusion(nn.Module):
    """Released AtomWeaver inference architecture; module construction order preserves seeded sampling."""

    _ba_corrupt_announced: bool = False

    def __init__(self):
        hidden_dim = 432
        timesteps = 250
        schedule = "cosine"
        max_sidechain_atoms = 14
        prediction_type = "v"
        coord_process_type = "flow_matching"
        flow_ghost_power = 0.5
        flow_real_proximal_power = 2.5
        flow_real_distal_power = 1.0
        flow_use_conditional_groupwise = True
        flow_noise_scale = 1.0
        pad_sampling_init = "bare"
        decoupled_count = False
        count_ramp_threshold = 0.9
        count_overdispersion = 1.5
        ghost_var_floor = 0.3
        mixture_loss_weight = 0.0
        all_carbon_sampling = False
        mixture_lr_threshold = 1.5
        non_pad_element_sampling = False
        late_element_resolution = False
        occupancy_match_timestep_floor = 0.0
        volumetric_ss_context_p_max = 0.8
        volumetric_context_radius = 10.0
        volumetric_n_query = 384
        volumetric_sigma = 0.6
        volumetric_per_element_sigma = True
        volumetric_sigma_element_scale = None
        volumetric_empty_weight = 1.75
        volumetric_fourier_frequencies = 64
        volumetric_fourier_scale = 10.0
        volumetric_dropout = 0.1
        volumetric_use_softplus = True
        volumetric_per_query_context = True
        volumetric_context_k = 128
        volumetric_context_heads = 4
        volumetric_context_chunk = 32
        volumetric_atom_anchored_queries = False
        self_consistency_ramp_start = 0.0
        self_consistency_ramp_end = 0.001
        mixture_head_dropout = 0.0
        dlrt_detach = False
        dlrt_ema_decay = 0.0
        sharpen_temperature_min = 1.0
        use_donut_source = True
        donut_thickness_ratio = 0.3
        use_empirical_shell_thickness = True
        shell_target_var_scale = 1.0
        donut_element_init = "mask"
        evc_velocity_blend = False
        evc_ss_noised_element_prob = 0.5
        neighbor_x0_packing_recycles = 2
        neighbor_x0_packing_max_recycles = 5
        neighbor_x0_packing_recycle_weights = None
        occupancy_weighted_source = False
        t_resolution_feedback_ramp_start = 0.0
        t_resolution_feedback_ramp_end = 0.05
        distal_threshold_cap = 0.0
        distal_jitter_cap = 0.0
        distal_shell_ramp_epochs = 0
        super().__init__()
        from .diffusion import ClassWeightedDiscreteDiffusion, GaussianDiffusion, GhostRealFlowMatching

        if coord_process_type not in {"ddpm", "flow_matching"}:
            raise ValueError(f"Unknown coord_process_type={coord_process_type}")
        from .diffusion import ELEMENT_MASK

        self.num_element_classes = NUM_ELEMENT_TYPES + (1 if donut_element_init == "mask" else 0)
        self.t_resolution_feedback_ramp_start = float(t_resolution_feedback_ramp_start)
        self.t_resolution_feedback_ramp_end = float(t_resolution_feedback_ramp_end)
        self.denoiser = SidechainDenoiser()
        n_real_classes = self.num_element_classes - 1
        self.polarity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, n_real_classes)
        )
        nn.init.zeros_(self.polarity_head[-1].weight)
        nn.init.zeros_(self.polarity_head[-1].bias)
        self.neighbor_x0_packing_recycles = max(1, int(neighbor_x0_packing_recycles))
        self.distal_threshold_cap = float(distal_threshold_cap)
        self.distal_shell_ramp_epochs = int(distal_shell_ramp_epochs)
        self.neighbor_x0_packing_max_recycles = int(neighbor_x0_packing_max_recycles)
        self.neighbor_x0_packing_recycle_weights = (
            parse_neighbor_x0_recycle_weights(
                neighbor_x0_packing_recycle_weights, self.neighbor_x0_packing_max_recycles
            )
            if False
            else None
        )
        self.evc_velocity_blend = evc_velocity_blend
        self.evc_ss_noised_element_prob = evc_ss_noised_element_prob
        if self.denoiser is not None:
            self.denoiser._mixture_head_dropout = mixture_head_dropout
            self.denoiser._dlrt_detach = dlrt_detach
            self.denoiser._dlrt_ema_decay = dlrt_ema_decay
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps, schedule=schedule, noise_scale=4.0, prediction_type=prediction_type
        )
        self.coord_flow = GhostRealFlowMatching(
            timesteps=timesteps,
            noise_scale=flow_noise_scale,
            ghost_power=flow_ghost_power,
            real_proximal_power=flow_real_proximal_power,
            real_distal_power=flow_real_distal_power,
            use_conditional_groupwise=flow_use_conditional_groupwise,
            use_donut_source=use_donut_source,
            donut_thickness_ratio=donut_thickness_ratio,
            use_empirical_shell_thickness=use_empirical_shell_thickness,
            shell_target_var_scale=shell_target_var_scale,
            occupancy_weighted_source=occupancy_weighted_source,
            max_sidechain_atoms=max_sidechain_atoms,
            distal_jitter_cap=distal_jitter_cap,
            distal_shell_ramp_epochs=distal_shell_ramp_epochs,
        )
        custom_prior = None
        bare_target = ELEMENT_MASK
        elem_prior = custom_prior
        elem_prior_schedule = pad_sampling_init
        from .diffusion import DEFAULT_ELEMENT_PRIOR

        data_prior_5 = custom_prior if custom_prior is not None else DEFAULT_ELEMENT_PRIOR[:NUM_ELEMENT_TYPES].clone()
        elem_prior = torch.zeros(self.num_element_classes)
        elem_prior[ELEMENT_MASK] = 1.0
        elem_prior_schedule = None
        self.element_diffusion = ClassWeightedDiscreteDiffusion(
            timesteps=timesteps,
            schedule=schedule,
            num_classes=self.num_element_classes,
            prior=elem_prior,
            prior_schedule=elem_prior_schedule,
            bare_target=bare_target,
            decoupled_count=decoupled_count,
            count_ramp_threshold=count_ramp_threshold,
            count_overdispersion=count_overdispersion,
            max_sidechain_atoms=max_sidechain_atoms,
        )
        data_prior_6 = torch.cat([data_prior_5, torch.ones(1)])
        data_weights = (1.0 / data_prior_6.clamp(min=0.01)).clamp(max=20.0)
        _anchor = True
        if _anchor:
            from .diffusion import DEFAULT_ELEMENT_PRIOR

            _canon = torch.cat([DEFAULT_ELEMENT_PRIOR[:5].clone(), torch.ones(1)])
            _norm = (1.0 / _canon.clamp(min=0.01)).clamp(max=20.0).mean()
        else:
            _norm = data_weights.mean()
        data_weights = data_weights / _norm
        _pad_scale = 1.0
        os.environ.get("ATOMWEAVER_VERBOSE") and print(
            f"[element-loss] vocab={data_prior_6.numel() - 1} anchor={_anchor} pad_scale={_pad_scale} -> PAD class weight {float(data_weights[ELEMENT_PAD]):.4f}"
        )
        self.element_diffusion.class_weights.copy_(data_weights)
        _occ_floor = float(occupancy_match_timestep_floor)
        if not (math.isfinite(_occ_floor) and 0.0 <= _occ_floor <= 1.0):
            raise ValueError(
                f"occupancy_match_timestep_floor must be finite in [0.0, 1.0] (got {_occ_floor!r}); >1 reverses the interpolation, nan poisons the loss, <0 silently disables"
            )
        self.max_sidechain_atoms = int(max_sidechain_atoms)
        self.volumetric_density_inject_proj = nn.Linear(volumetric_n_query, hidden_dim, bias=False)
        nn.init.zeros_(self.volumetric_density_inject_proj.weight)
        self.volumetric_ss_context_p_max = float(volumetric_ss_context_p_max)
        if not 0.0 <= self.volumetric_ss_context_p_max <= 1.0:
            raise ValueError(
                f"volumetric_ss_context_p_max={self.volumetric_ss_context_p_max} must be in [0, 1] (it is a per-site Bernoulli probability of using the predicted x0 instead of the GT clean x0)."
            )
        if self._neighbor_x0_effective_recycles() <= 1:
            raise ValueError(
                f"volumetric_ss_context_p_max={self.volumetric_ss_context_p_max} (>0) requires use_neighbor_x0_packing=True (currently {True}) AND effective recycles>1 (currently {self._neighbor_x0_effective_recycles()}): the predicted-x0 pocket context is stashed ONLY inside the neighbour-x0 recycle loop (2-pass), so without it ss_pred_coords is never populated, the option-(ii) recompute never fires, and the scheduled-sampling ramp silently no-ops to pure GT teacher forcing for the ENTIRE run (vol_hidden keeps the GT context). Enable use_neighbor_x0_packing with neighbor_x0_packing_recycles>=2, or set volumetric_ss_context_p_max=0."
            )
        self.volumetric_head = VolumetricOccupancyHead(
            hidden_dim=hidden_dim,
            num_element_types=NUM_ELEMENT_TYPES,
            context_radius=volumetric_context_radius,
            n_query=volumetric_n_query,
            sigma=volumetric_sigma,
            empty_weight=volumetric_empty_weight,
            use_available_volume=True,
            per_element_sigma=bool(volumetric_per_element_sigma),
            sigma_element_scale=volumetric_sigma_element_scale,
            use_single_site_context=True,
            fourier_frequencies=int(volumetric_fourier_frequencies),
            fourier_scale=float(volumetric_fourier_scale),
            dropout=float(volumetric_dropout),
            use_softplus=bool(volumetric_use_softplus),
            use_per_query_context=bool(volumetric_per_query_context),
            context_k=int(volumetric_context_k),
            context_heads=int(volumetric_context_heads),
            context_chunk=int(volumetric_context_chunk),
            atom_anchored_queries=bool(volumetric_atom_anchored_queries),
        )
        missing_optional_keys = {
            name for name, _ in self.volumetric_head.named_parameters() if _is_optional_volumetric_head_key(name)
        }
        n_frozen = 0
        n_exempt = 0
        for name, p in self.volumetric_head.named_parameters():
            if _is_optional_volumetric_head_key(name) and name in missing_optional_keys:
                n_exempt += 1
                continue
            p.requires_grad = False
            n_frozen += 1
        os.environ.get("ATOMWEAVER_VERBOSE") and print(
            f"[volumetric-head] FROZE {n_frozen} volumetric_head tensors (requires_grad=False); they are excluded from all optimizer param groups."
            + (f" EXEMPTED {n_exempt} trainable fresh-graft tensors absent from the checkpoint." if n_exempt else "")
        )
        self.self_consistency_ramp_start = float(self_consistency_ramp_start)
        self.self_consistency_ramp_end = float(self_consistency_ramp_end)
        self.mixture_lr_threshold = mixture_lr_threshold
        self.flow_real_proximal_power = flow_real_proximal_power
        self.flow_real_distal_power = flow_real_distal_power
        self.ghost_var_floor = ghost_var_floor
        self.mixture_loss_weight = mixture_loss_weight
        self.sharpen_temperature_min = sharpen_temperature_min
        if not 0.0 <= self.self_consistency_ramp_start < self.self_consistency_ramp_end <= 1.0:
            raise ValueError(
                f"self_consistency_ramp_start/self_consistency_ramp_end must satisfy 0 <= start < end <= 1 (epoch fractions) when use_volumetric_self_consistency is on, got start={self.self_consistency_ramp_start}, end={self.self_consistency_ramp_end}."
            )
        if not 0.0 <= self.t_resolution_feedback_ramp_start < self.t_resolution_feedback_ramp_end <= 1.0:
            raise ValueError(
                f"t_resolution_feedback_ramp_start/t_resolution_feedback_ramp_end must satisfy 0 <= start < end <= 1 (epoch fractions) when use_stereochem_t_resolution_feedback is on, got start={self.t_resolution_feedback_ramp_start}, end={self.t_resolution_feedback_ramp_end}."
            )
        sampling_mode_count = sum(
            (int(flag) for flag in (all_carbon_sampling, late_element_resolution, non_pad_element_sampling))
        )
        if sampling_mode_count > 1:
            raise ValueError(
                "all_carbon_sampling, late_element_resolution, and non_pad_element_sampling are mutually exclusive"
            )
        self.register_buffer("_slot_fill_rate", torch.zeros(max_sidechain_atoms))
        self.register_buffer("_slot_coord_mean", torch.zeros(max_sidechain_atoms, 3))
        self.register_buffer("_slot_coord_var", torch.ones(max_sidechain_atoms))
        self.timesteps = timesteps

    def _compute_available_volume(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        target_coords: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """GT-free "sandclock" available-volume descriptor for the volumetric head (t-INDEPENDENT).

        Returns ``None`` unless ``use_available_volume`` is effective (already AND-gated with
        ``use_volumetric_head``), so callers can pass the result to the head unconditionally -- the head
        ignores it when its own flag is off. Computed from the binder BACKBONE + TARGET atoms ONLY (no
        side chains), so it is valid at inference. See :func:`available_volume_cones`.
        """
        R, ca = build_local_frames(backbone_coords, backbone_mask)
        return available_volume_cones(backbone_coords, backbone_mask, target_coords, target_mask, R, ca)

    def _element_slot_powers(self, sidechain_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-slot element noise powers matching the coordinate flow schedule.

        Real atoms get the same proximal->distal power interpolation as coordinates.
        Ghost atoms (PAD) get power=1.0 (default schedule, fast transition to PAD).
        Higher power = stays original longer = slower corruption.

        Parameters
        ----------
        sidechain_mask : torch.Tensor
            GT mask of shape (B, L, max_sc). True = real atom.

        Returns
        -------
        slot_powers : torch.Tensor
            Per-slot powers of shape (B, L, max_sc).
        """
        batch_size, seq_len, max_sc = sidechain_mask.shape
        device = sidechain_mask.device
        slot_idx = torch.arange(max_sc, device=device, dtype=torch.float32)
        depth = slot_idx / (max_sc - 1) if max_sc > 1 else torch.zeros_like(slot_idx)
        real_power = self.flow_real_proximal_power + depth * (
            self.flow_real_distal_power - self.flow_real_proximal_power
        )
        real_power = real_power.view(1, 1, max_sc).expand(batch_size, seq_len, max_sc)
        ghost_power = torch.ones(batch_size, seq_len, max_sc, device=device)
        mask_f = sidechain_mask.float()
        return mask_f * real_power + (1.0 - mask_f) * ghost_power

    def compute_mixture_posterior(
        self,
        noised_coords: torch.Tensor,
        ca_coords: torch.Tensor,
        t: torch.Tensor,
        sidechain_mask: torch.Tensor | None = None,
        learned_centroid: torch.Tensor | None = None,
        learned_cloud_logvar: torch.Tensor | None = None,
        residue_lrt_delta: torch.Tensor | None = None,
        residue_count_pred: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute analytic P(real | x_t, t, slot) from two-component Gaussian mixture.

        Ghost component: N(CA, [s²σ² + σ²_floor] I)
        Real component:  N(CA + (1-s)·μ_res, [(1-s)²·σ²_res + s²·(σ² + real_var_floor)] I)

        The real component uses a **shared per-residue centroid** rather than independent
        per-slot means. All slots within a residue see the same sidechain cloud center.

        Parameters
        ----------
        noised_coords : (B, L, max_sc, 3)
        ca_coords : (B, L, 3)
        t : (B,) integer timesteps
        sidechain_mask : (B, L, max_sc) optional, unused but kept for API consistency
        learned_centroid : (B, L, 3) optional Stage 2 per-residue centroid offset from CA
        learned_cloud_logvar : (B, L) optional Stage 2 per-residue cloud log-variance
        residue_lrt_delta : (B, L) optional per-residue LRT offset from backbone head
        residue_count_pred : (B, L) optional count head prediction for analytical threshold

        Returns
        -------
        posterior : (B, L, max_sc) P(real | x_t) in [0, 1]
        """
        batch_size, seq_len, max_sc, _ = noised_coords.shape
        device = noised_coords.device
        if self.coord_flow.use_donut_source:
            sigma_sq = self.coord_flow._shell_source_var[:max_sc].view(1, 1, max_sc, 1)
        else:
            noise_scale = self.coord_flow.noise_scale
            sigma_sq = noise_scale**2
        tau = self.coord_flow._tau(t)
        s = tau.view(batch_size, 1, 1)
        if learned_centroid is not None:
            mu_res = learned_centroid
        else:
            mu_res = torch.zeros(batch_size, seq_len, 3, device=device)
        if learned_cloud_logvar is not None:
            var_res = learned_cloud_logvar.exp()
        else:
            var_res = self._slot_coord_var.mean().expand(batch_size, seq_len)
        mu_slot = mu_res.unsqueeze(2).expand(-1, -1, max_sc, -1)
        var_slot = var_res.unsqueeze(2).expand(-1, -1, max_sc)
        pi_real = self._slot_fill_rate.view(1, 1, max_sc).expand(batch_size, seq_len, -1).clamp(0.01, 0.99)
        ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, max_sc, -1)
        x_rel = noised_coords - ca_expanded
        ghost_var = s.unsqueeze(-1) ** 2 * sigma_sq + self.ghost_var_floor
        ghost_var = ghost_var.squeeze(-1)
        ghost_mahal = (x_rel**2).sum(dim=-1) / ghost_var
        ghost_log_norm = 3.0 * torch.log(ghost_var)
        log_p_ghost = -0.5 * (ghost_mahal + ghost_log_norm)
        one_minus_s = 1.0 - s
        real_var = one_minus_s.unsqueeze(-1) ** 2 * var_slot.unsqueeze(-1) + s.unsqueeze(-1) ** 2 * sigma_sq
        real_var = real_var.squeeze(-1).clamp(min=0.0001)
        real_mean = one_minus_s.unsqueeze(-1) * mu_slot
        diff = x_rel - real_mean
        real_mahal = (diff**2).sum(dim=-1) / real_var
        real_log_norm = 3.0 * torch.log(real_var)
        log_p_real = -0.5 * (real_mahal + real_log_norm)
        log_prior_ratio = torch.log(pi_real / (1.0 - pi_real))
        log_likelihood_ratio = log_p_real - log_p_ghost
        threshold = self.mixture_lr_threshold
        logit = log_likelihood_ratio + log_prior_ratio - threshold
        if temperature != 1.0:
            logit = logit / temperature
        return torch.sigmoid(logit)

    def _apply_expert_logit_biases(self, element_logits, backbone_features, backbone_coords, backbone_mask):
        """Apply the supervised expert logit modulations to `element_logits`.

        Two experts, applied in order: (1) polarity product-of-experts, which reweights WHICH real
        element while keeping P(PAD) EXACTLY invariant; (2) glycine, which raises P(PAD) where the
        backbone dihedrals indicate glycine. Called from BOTH ``forward`` (training) and ``sample``
        (inference) via this single helper, so the element distribution the model is TRAINED on is the
        SAME one it is SAMPLED from -- otherwise the heads shape the training loss but never touch the
        de-novo eval. Returns ``(element_logits, aux)`` where ``aux`` carries the head outputs the
        training losses consume (empty when neither head is on). Glycine is guarded off in split-existence
        mode at ``__init__`` (class 0 = Carbon there, not PAD).
        """
        aux = {}
        polarity_logits = self.polarity_head(backbone_features)
        log_pol = F.log_softmax(polarity_logits, dim=-1)
        pad_logit = element_logits[..., :1]
        real = element_logits[..., 1:]
        lse_before = torch.logsumexp(real, dim=-1, keepdim=True)
        real_biased = real + log_pol.unsqueeze(2)
        lse_after = torch.logsumexp(real_biased, dim=-1, keepdim=True)
        real_renorm = real_biased - (lse_after - lse_before)
        element_logits = torch.cat([pad_logit, real_renorm], dim=-1)
        aux["polarity_logits"] = polarity_logits
        aux["log_pol"] = log_pol
        return (element_logits, aux)

    def _x0_from_model_output(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invert the denoiser output to a clean-endpoint (x0) estimate.

        Handles every coordinate process the repo supports (flow matching --
        plus the legacy ``v`` / ``epsilon`` / ``x0`` DDPM parameterisations).

        Parameters
        ----------
        model_output : torch.Tensor
            Denoiser ``noise_pred`` of shape (B, L, max_sc, 3).
        x_t : torch.Tensor
            Current noisy coordinates of shape (B, L, max_sc, 3).
        t : torch.Tensor
            Timesteps of shape (B,).
        ca_coords : torch.Tensor
            Calpha coordinates of shape (B, L, 3).
        mask_probs : torch.Tensor, optional
            Per-slot P(real) of shape (B, L, max_sc); selects the per-slot shell power.

        Returns
        -------
        torch.Tensor
            x0 estimate of shape (B, L, max_sc, 3).
        """
        b = model_output.shape[0]
        x_flat = x_t.reshape(b, -1, 3)
        v_flat = model_output.reshape(b, -1, 3)
        mp = mask_probs.reshape(b, -1, 1).float() if mask_probs is not None else None
        x0 = self.coord_flow.predict_x0_from_velocity(x_flat, t, v_flat, ca_coords=ca_coords, mask_probs=mp)
        return x0.reshape_as(x_t)

    def _distal_read_threshold(self, shell_radii: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        """Per-slot MASK-slot existence read-threshold, with the FEATURE 4 distal cap (warmup).

        A slot's atom reads as PRESENT when ``dist_to_ca >= threshold``. The historical threshold is
        ``0.5 * shell_radii`` (~4.6 A for the farthest slots) -- so HIGH for mid/distal slots that a
        genuine-but-under-extended atom (drifted inward of half its shell) is read as ABSENT, i.e.
        ghost-collapsed -> undercount. ``distal_threshold_cap > 0`` LOWERS the threshold for those
        slots so those under-read atoms are recovered as present (the Diagnostic-1 recovery direction):

            threshold(slot) = lerp(0.5*r, min(0.5*r, cap), ramp_progress)

        This is NOT a tightening of the read -- it is a LOOSENING (a lower bar to count as present) for
        every slot whose ``0.5*r`` exceeds the cap. With the default 14-slot radii and cap ~1.3, that is
        the MID + DISTAL slots (roughly slot 2 onward), NOT just the far tail. PROXIMAL slots
        (``0.5*r <= cap``) are UNCHANGED at every epoch (``min`` = 0.5*r), which keeps the tight bar that
        protects Exposed positions. (Contrast the FEATURE 5 jitter cap, which genuinely tightens the
        source SPREAD.)

        ``epoch`` drives the ramp on the (epoch-aware) forward site. The sample sites pass ``epoch=None``
        -> fully-ramped ``min(0.5*r, cap)`` (correct for any ep>=ramp_epochs checkpoint; early-checkpoint
        eval is thus slightly off-ramp -- both sample sub-sites and the forward site share this one
        helper, so the two are never on different conventions). Bit-exact ``0.5*r`` when cap == 0.0.
        """
        base = shell_radii * 0.5
        cap = float(self.distal_threshold_cap)
        if cap <= 0.0:
            return base
        ramp = int(self.distal_shell_ramp_epochs)
        prog = 1.0 if epoch is None or ramp <= 0 else min(float(epoch) / ramp, 1.0)
        capped = torch.minimum(base, base.new_full((), cap))
        return torch.lerp(base, capped, base.new_tensor(prog))

    def _neighbor_x0_effective_recycles(self) -> int:
        """Static "will an EXTRA neighbour-x0 pass actually run?" count for this config.

        Identical predicate to ``validate_training_flag_coherence`` check (4) (the transition-weighted
        loss prerequisite): with ``neighbor_x0_packing_random_recycles`` the per-batch N is DRAWN from
        ``{2 .. max_recycles}`` (never 1), so the static upper bound stands in for "an extra pass will
        run"; otherwise it is the fixed ``neighbor_x0_packing_recycles``. ``> 1`` means the recycle loop
        supplies neighbour-x0 context -- the precondition for self-dropout to have anything to fall back
        on.
        """
        return int(self.neighbor_x0_packing_recycles)

    def _predicted_existence_probs(self, denoiser_out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Detached, MODEL-OWNED per-slot P(atom exists) read off one denoiser pass.

        The neighbour-x0 context must be model-owned on BOTH channels. Using the noised GT
        existence (``noised_mask``) here would teacher-force the occupancy/count half of the
        context -- worst at low t, on the single most fragile channel in this model -- and would
        manufacture exactly the exposure bias the whole design exists to avoid. Ground truth
        re-enters only for pinned clean-context residues, in
        :meth:`_build_neighbor_x0_inputs`.

        In the 2-track default, existence is read straight off the element track (the GHOST/PAD
        symbol) rather than from the auxiliary occupancy head -- that is the repo's load-bearing
        "existence is not a separate track" convention. The RAW denoiser logits are used
        deliberately: the occupancy gate and the expert (polarity/glycine) modulations are applied
        later in ``forward``/``sample``, and the recycle context only needs the denoiser's own
        belief, not the fully post-processed distribution.

        Parameters
        ----------
        denoiser_out : dict of str to torch.Tensor
            Output dict of a :class:`SidechainDenoiser` pass.

        Returns
        -------
        torch.Tensor
            Detached P(real) of shape (B, L, max_sc), clamped to ``[0, 1]``.
        """
        probs = torch.softmax(denoiser_out["element_logits"], dim=-1)
        p_real = 1.0 - probs[..., ELEMENT_PAD]
        if self.num_element_classes > NUM_ELEMENT_TYPES:
            from .diffusion import ELEMENT_MASK

            p_real = p_real - probs[..., ELEMENT_MASK]
        return p_real.detach().clamp(0.0, 1.0)

    def _build_neighbor_x0_inputs(
        self,
        x0_estimate: torch.Tensor,
        existence_probs: torch.Tensor,
        t_original_res: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
        clean_context_coords: torch.Tensor | None = None,
        clean_context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the (coords, mask, trust) triple the denoiser reads neighbour context from.

        Designed positions contribute the model's OWN **detached** x0 estimate. This is
        deliberate: teacher-forcing ground-truth x0 here would be an input leak and would
        recreate exposure bias (great in training, collapse at sampling). Ground truth
        enters only through ``clean_context_coords``, i.e. the non-designed / pinned
        positions of the inpainting path -- coordinates that ARE available at inference in
        single-site design, which is exactly what makes leg-B the clean limit of the
        mechanism.

        Parameters
        ----------
        x0_estimate : torch.Tensor
            Detached x0 estimate of shape (B, L, max_sc, 3).
        existence_probs : torch.Tensor
            MODEL-PREDICTED per-slot P(real) of shape (B, L, max_sc), values in [0, 1] --
            see :meth:`_predicted_existence_probs`. Must NOT be the noised GT mask: the
            occupancy/count half of the context is as much of a leak as the coordinates.
        t_original_res : torch.Tensor
            Per-residue ORIGINAL noise level of shape (B, L). Drives the per-neighbour trust
            ``1 - t_original/T``: a clean pinned context residue reads as fully trustworthy,
            a co-generated neighbour at high noise as barely so.
        keep_mask : torch.Tensor, optional
            (B, L) bool, True for non-designed / clean-context residues.
        clean_context_coords : torch.Tensor, optional
            (B, L, max_sc, 3) clean side-chain coordinates for the context positions.
        clean_context_mask : torch.Tensor, optional
            (B, L, max_sc) existence mask for the context positions.

        Returns
        -------
        tuple of torch.Tensor
            ``(coords (B, L, S, 3), mask (B, L, S) bool, trust (B, L) float)``.
        """
        coords = x0_estimate.detach()
        mask = existence_probs.detach() > 0.5
        trust = (1.0 - t_original_res.float() / max(self.timesteps, 1)).clamp(0.0, 1.0)
        if keep_mask is not None:
            keep = keep_mask.to(device=coords.device, dtype=torch.bool)
            if clean_context_coords is not None:
                coords = torch.where(keep[:, :, None, None], clean_context_coords.detach(), coords)
            if clean_context_mask is not None:
                mask = torch.where(keep[:, :, None], clean_context_mask.detach().bool(), mask)
            trust = torch.where(keep, torch.ones_like(trust), trust)
        return (coords, mask, trust)

    def _pin_inpaint(
        self,
        x,
        element_types,
        noised_mask,
        evc_sampling,
        *,
        design_mask,
        inpaint_gt_coords,
        inpaint_gt_elements,
        inpaint_gt_mask,
        inpaint_mode,
        t,
        ca_coords,
        leak_direction,
    ):
        """Replacement-inpaint pin: set non-designed binder positions to GT (coords/elements/mask/EVC).

        Called BEFORE every denoiser call (incl. once pre-loop) so sampling matches training, which pins
        before the forward -- otherwise the first reverse step would denoise with un-pinned high-noise
        context. Also called after each reverse update so the final output stays pinned. No-op when not
        inpainting. Returns (x, element_types, noised_mask, evc_sampling).
        """
        from .diffusion import ELEMENT_PAD

        if design_mask is None or inpaint_gt_coords is None:
            return (x, element_types, noised_mask, evc_sampling)
        batch_size, seq_len = (design_mask.shape[0], design_mask.shape[1])
        keep = (~design_mask.bool()).view(batch_size, seq_len, 1)
        if inpaint_mode == "clean":
            pin_coords, pin_elems = (inpaint_gt_coords, inpaint_gt_elements)
        elif inpaint_mode == "noised":
            raise NotImplementedError(
                "inpaint_mode='noised' is not supported (post-update pin re-noises GT to the wrong step). Use inpaint_mode='clean' (the validated path); fix the in-loop step index to re-enable."
            )
        else:
            raise ValueError(f"inpaint_mode must be 'clean' or 'noised', got {inpaint_mode!r}")
        x = torch.where(keep.unsqueeze(-1), pin_coords, x)
        if pin_elems is not None:
            element_types = torch.where(keep, pin_elems, element_types)
            noised_mask = (element_types != ELEMENT_PAD).float()
            if evc_sampling is not None and inpaint_gt_mask is not None:
                evc_sampling = torch.where(keep, inpaint_gt_mask.float(), evc_sampling)
        return (x, element_types, noised_mask, evc_sampling)

    @torch.no_grad()
    def sample(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        sidechain_mask: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        target_backbone_coords: torch.Tensor | None = None,
        target_backbone_mask: torch.Tensor | None = None,
        target_residue_types: torch.Tensor | None = None,
        target_seq_mask: torch.Tensor | None = None,
        target_coords: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_atom_type: torch.Tensor | None = None,
        target_atom_element_type: torch.Tensor | None = None,
        target_atom_residue_type: torch.Tensor | None = None,
        target_atom_is_backbone: torch.Tensor | None = None,
        return_intermediates: bool = False,
        return_coord_trajectory: bool = False,
        element_sampling_temp_max: float = 25.0,
        element_sampling_temp_power: float = 5.0,
        reserved_slot0_prefix_exempt: bool = False,
        design_mask: torch.Tensor | None = None,
        inpaint_gt_coords: torch.Tensor | None = None,
        inpaint_gt_elements: torch.Tensor | None = None,
        inpaint_gt_mask: torch.Tensor | None = None,
        inpaint_mode: str = "clean",
        chirality: torch.Tensor | None = None,
        neighbor_x0_packing_recycles: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample side-chain coordinates and element types given backbone.

        The model jointly denoises coordinates and element types (PAD=0, C=1, N=2,
        O=3, S=4). Atom existence is determined by element_type != PAD. A hard prefix
        constraint is enforced: if slot i is PAD, all j>i are PAD.

        Parameters
        ----------
        backbone_coords : torch.Tensor
            Fixed backbone coordinates of shape (B, L, 4, 3).
        backbone_mask : torch.Tensor
            Mask for backbone atoms of shape (B, L, 4).
        sidechain_mask : torch.Tensor
            Maximum mask for sidechain atoms of shape (B, L, max_sc). During sampling,
            this is typically all True to allow the model to predict which atoms exist.
        seq_mask : torch.Tensor, optional
            Mask for valid residues of shape (B, L).
        num_steps : int, optional
            Number of sampling steps. Defaults to all timesteps.
        target_backbone_coords : torch.Tensor, optional
            Target backbone coordinates of shape (B, L_t, 4, 3).
        target_backbone_mask : torch.Tensor, optional
            Mask for target backbone atoms of shape (B, L_t, 4).
        target_residue_types : torch.Tensor, optional
            Target residue type indices of shape (B, L_t).
        target_seq_mask : torch.Tensor, optional
            Mask for valid target residues of shape (B, L_t).
        use_ddim : bool, optional
            Whether to use DDIM sampling for coordinates. Default False.
        ddim_eta : float, optional
            Stochasticity parameter for DDIM. Only used when use_ddim=True.
        return_intermediates : bool, optional
            If True, return intermediate atom counts at each sampling step. Default False.

        Returns
        -------
        dict
            Dictionary with:
            - 'sidechain_coords': Sampled coordinates of shape (B, L, max_sc, 3)
            - 'element_types': Predicted element types of shape (B, L, max_sc)
            - 'predicted_mask': Predicted atom mask of shape (B, L, max_sc)
            - 'intermediate_atom_counts': (optional) List of atom counts per step if return_intermediates=True
        """
        batch_size, seq_len, max_sc = sidechain_mask.shape
        device = backbone_coords.device
        num_steps = num_steps or self.timesteps
        _sample_recycles = max(1, neighbor_x0_packing_recycles or 1)
        _sample_absorbing = True
        ca_coords = backbone_coords[:, :, 1, :]
        ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, max_sc, -1)
        leak_direction = None
        _src_chir = chirality
        leak_direction = compute_pseudo_cb_direction(backbone_coords, chirality=_src_chir)
        x, _ = self.coord_flow.sample_prior(
            (batch_size, seq_len * max_sc, 3),
            ca_coords=ca_coords,
            per_atom_mask=None,
            direction_override=leak_direction,
        )
        x = x.view(batch_size, seq_len, max_sc, 3)
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()
        from .diffusion import ELEMENT_MASK, ELEMENT_PAD

        element_types = torch.full((batch_size, seq_len, max_sc), ELEMENT_MASK, device=device, dtype=torch.long)
        noised_mask = (element_types != ELEMENT_PAD).float()
        if hasattr(self.coord_flow, "_shell_radii"):
            from .diffusion import ELEMENT_MASK

            is_mask = element_types == ELEMENT_MASK
            if is_mask.any():
                ca_exp = ca_coords.unsqueeze(2).expand(-1, -1, max_sc, -1)
                dist_to_ca = (x - ca_exp).norm(dim=-1)
                shell_radii = self.coord_flow._shell_radii
                threshold = self._distal_read_threshold(shell_radii, epoch=None).view(1, 1, max_sc)
                noised_mask = torch.where(is_mask, (dist_to_ca >= threshold).float(), noised_mask)
        prev_element_pred = None
        prev_cluster_pred = None
        final_residue_centroid = None
        final_residue_cloud_logvar = None
        final_mixture_posterior = None
        intermediate_atom_counts = [] if return_intermediates else None
        coord_traj = [] if return_coord_trajectory else None
        elem_traj = [] if return_coord_trajectory else None
        mask_traj = [] if return_coord_trajectory else None
        intermediate_mixture_counts = [] if return_intermediates else None
        intermediate_per_res_soft_pre = [] if return_intermediates else None
        intermediate_per_res_hard = [] if return_intermediates else None
        intermediate_per_res_ca_dist_real = [] if return_intermediates else None
        intermediate_per_res_ca_dist_ghost = [] if return_intermediates else None
        intermediate_slot_pad_logit_pre = [] if return_intermediates else None
        intermediate_slot_non_pad_post = [] if return_intermediates else None
        if return_intermediates:
            init_counts = noised_mask.float().view(batch_size, -1).sum(dim=-1).tolist()
            intermediate_atom_counts.append(init_counts)
        evc_sampling = None
        from .diffusion import ELEMENT_MASK, ELEMENT_PAD

        if getattr(self, "evc_ss_noised_element_prob", 0.0) > 0:
            evc_sampling = evc_from_element_state(element_types)
        else:
            is_resolved_real = ((element_types != ELEMENT_PAD) & (element_types != ELEMENT_MASK)).float()
            evc_sampling = is_resolved_real
        if design_mask is not None and inpaint_gt_coords is not None:
            _t0 = torch.full((batch_size,), timesteps[0].item(), device=device, dtype=torch.long)
            x, element_types, noised_mask, evc_sampling = self._pin_inpaint(
                x,
                element_types,
                noised_mask,
                evc_sampling,
                design_mask=design_mask,
                inpaint_gt_coords=inpaint_gt_coords,
                inpaint_gt_elements=inpaint_gt_elements,
                inpaint_gt_mask=inpaint_gt_mask,
                inpaint_mode=inpaint_mode,
                t=_t0,
                ca_coords=ca_coords,
                leak_direction=leak_direction,
            )
        inpaint_ar_state = None
        if design_mask is not None and inpaint_gt_coords is not None:
            inpaint_ar_state = build_kmask_ar_state(design_mask, seq_mask, x.shape[2], device)
        if return_coord_trajectory:
            coord_traj.append(x.detach().to("cpu", torch.float32).clone())
            elem_traj.append(element_types.detach().to("cpu").clone())
            mask_traj.append((element_types != ELEMENT_PAD).detach().to("cpu").clone())
        _vol_consumer_active = True
        _avail_vol = self._compute_available_volume(backbone_coords, backbone_mask, target_coords, target_mask)
        x0_prev_coords: torch.Tensor | None = None
        x0_prev_mask: torch.Tensor | None = None
        x0_prev_element: torch.Tensor | None = None
        vol_hidden_sample = None
        vol_density_inject_sample = None
        for step_i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx.item(), device=device, dtype=torch.long)
            mix_temperature = 1.0
            noised_count = noised_mask.sum(dim=-1)
            _denoise_kwargs = {
                "noised_element_types": element_types,
                "noised_cluster_ids": None,
                "noised_count": noised_count,
                "prev_element_pred": prev_element_pred,
                "prev_cluster_pred": prev_cluster_pred,
                "cluster_feature_scale": 1.0,
                "target_backbone_coords": target_backbone_coords,
                "target_backbone_mask": target_backbone_mask,
                "target_residue_types": target_residue_types,
                "target_seq_mask": target_seq_mask,
                "target_coords": target_coords,
                "target_mask": target_mask,
                "target_atom_type": target_atom_type,
                "target_atom_element_type": target_atom_element_type,
                "target_atom_residue_type": target_atom_residue_type,
                "target_atom_is_backbone": target_atom_is_backbone,
                "element_velocity_conditioning": evc_sampling,
                "count_velocity_conditioning": None,
                "soft_element_probs": None,
                "ar_state": inpaint_ar_state,
                "vol_hidden": vol_hidden_sample,
                "vol_density_inject": vol_density_inject_sample,
            }
            nx0_coords = nx0_mask = nx0_trust = nx0_apply = None
            t_orig_s = t_cond_s = None
            nx0_recycle_index = None
            _geom_src_x0 = None
            _geom_src_mask = None
            _n_rc = self.neighbor_x0_packing_recycles if _sample_recycles is None else _sample_recycles
            _n_rc = max(1, int(_n_rc))
            _keep_s = ~design_mask.to(device=device, dtype=torch.bool) if design_mask is not None else None
            t_orig_s = t.float().view(-1, 1).expand(batch_size, seq_len).clone()
            if _keep_s is not None:
                t_orig_s = torch.where(_keep_s, torch.zeros_like(t_orig_s), t_orig_s)
            t_cond_s = t_orig_s.clone()
            nx0_recycle_index = 1
            if _n_rc > 1:
                nx0_apply = torch.ones(batch_size, device=device, dtype=t_orig_s.dtype)
                t_cond_s = torch.zeros_like(t_orig_s)
                nx0_recycle_index = _n_rc
                for _rc_i in range(_n_rc - 1):
                    _rc_out = self.denoiser(
                        x,
                        seq_mask
                        if seq_mask is not None
                        else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
                        backbone_coords,
                        backbone_mask,
                        t,
                        neighbor_x0_coords=nx0_coords,
                        neighbor_x0_mask=nx0_mask,
                        neighbor_x0_trust=nx0_trust,
                        neighbor_x0_apply=nx0_apply,
                        t_original_res=t_orig_s,
                        t_conditioning_res=t_cond_s if nx0_coords is not None else t_orig_s,
                        recycle_index=_rc_i + 1,
                        **_denoise_kwargs,
                    )
                    _rc_pexist = self._predicted_existence_probs(_rc_out)
                    _rc_x0 = self._x0_from_model_output(_rc_out["noise_pred"], x, t, ca_coords, mask_probs=_rc_pexist)
                    _geom_src_x0 = _rc_x0
                    _geom_src_mask = _rc_pexist.detach() > 0.5
                    nx0_coords, nx0_mask, nx0_trust = self._build_neighbor_x0_inputs(
                        _rc_x0,
                        _rc_pexist,
                        t_orig_s,
                        keep_mask=_keep_s,
                        clean_context_coords=inpaint_gt_coords,
                        clean_context_mask=inpaint_gt_mask,
                    )
            if x0_prev_coords is None:
                _vh_boot_out = self.volumetric_head(
                    backbone_coords=backbone_coords,
                    backbone_mask=backbone_mask,
                    seq_mask=seq_mask,
                    target_coords=target_coords,
                    target_mask=target_mask,
                    target_element=target_atom_element_type,
                    target_is_backbone=target_atom_is_backbone,
                    available_volume=_avail_vol,
                )
                _vh_bootstrap = _vh_boot_out["vol_hidden"]
                _vd_bootstrap = self.volumetric_density_inject_proj(_vh_boot_out["vol_density_pred"].detach())
                _boot_out = self.denoiser(
                    x,
                    seq_mask
                    if seq_mask is not None
                    else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
                    backbone_coords,
                    backbone_mask,
                    t,
                    neighbor_x0_coords=nx0_coords,
                    neighbor_x0_mask=nx0_mask,
                    neighbor_x0_trust=nx0_trust,
                    neighbor_x0_apply=nx0_apply,
                    t_original_res=t_orig_s,
                    t_conditioning_res=t_cond_s,
                    recycle_index=nx0_recycle_index,
                    **{**_denoise_kwargs, "vol_hidden": _vh_bootstrap, "vol_density_inject": _vd_bootstrap},
                )
                _ctx_coords = self._x0_from_model_output(
                    _boot_out["noise_pred"], x, t, ca_coords, mask_probs=noised_mask
                ).detach()
                _ctx_mask = noised_mask.detach()
                _ctx_element = element_types.detach()
            else:
                _ctx_coords, _ctx_mask, _ctx_element = (x0_prev_coords, x0_prev_mask, x0_prev_element)
            _vh_ss_out = self.volumetric_head(
                backbone_coords=backbone_coords,
                backbone_mask=backbone_mask,
                seq_mask=seq_mask,
                target_coords=target_coords,
                target_mask=target_mask,
                target_element=target_atom_element_type,
                target_is_backbone=target_atom_is_backbone,
                available_volume=_avail_vol,
                context_sidechain_coords=_ctx_coords,
                context_sidechain_mask=_ctx_mask,
                context_sidechain_element=_ctx_element,
            )
            vol_hidden_sample = _vh_ss_out["vol_hidden"]
            _denoise_kwargs["vol_hidden"] = vol_hidden_sample
            _denoise_kwargs["vol_density_inject"] = self.volumetric_density_inject_proj(
                _vh_ss_out["vol_density_pred"].detach()
            )
            denoiser_outputs = self.denoiser(
                x,
                seq_mask if seq_mask is not None else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device),
                backbone_coords,
                backbone_mask,
                t,
                neighbor_x0_coords=nx0_coords,
                neighbor_x0_mask=nx0_mask,
                neighbor_x0_trust=nx0_trust,
                neighbor_x0_apply=nx0_apply,
                bond_prev_x0=None,
                bond_prev_mask=None,
                t_original_res=t_orig_s,
                t_conditioning_res=t_cond_s,
                recycle_index=nx0_recycle_index,
                **_denoise_kwargs,
            )
            model_output = denoiser_outputs["noise_pred"]
            element_logits = denoiser_outputs["element_logits"]
            element_logits, _ = self._apply_expert_logit_biases(
                element_logits, denoiser_outputs["backbone_features"], backbone_coords, backbone_mask
            )
            cluster_logits = denoiser_outputs["cluster_logits"]
            if return_intermediates:
                _lm = denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None
                _lv = denoiser_outputs.get("residue_cloud_logvar") if self.mixture_loss_weight > 0 else None
                pre_mix_post = self.compute_mixture_posterior(
                    noised_coords=x,
                    ca_coords=ca_coords,
                    t=t,
                    learned_centroid=_lm,
                    learned_cloud_logvar=_lv,
                    residue_lrt_delta=None,
                    residue_count_pred=None,
                    temperature=mix_temperature,
                )
                if seq_mask is not None:
                    pre_mix_post = pre_mix_post * seq_mask.unsqueeze(-1).float()
                intermediate_per_res_soft_pre.append(pre_mix_post.sum(dim=-1).detach().cpu())
                slot_dist_pre = (x - ca_expanded).norm(dim=-1)
                p_real = pre_mix_post.detach()
                p_ghost = 1.0 - p_real
                real_denom = p_real.sum(dim=-1).clamp(min=1e-06)
                intermediate_per_res_ca_dist_real.append(
                    (slot_dist_pre * p_real).sum(dim=-1).div(real_denom).detach().cpu()
                )
                ghost_denom = p_ghost.sum(dim=-1).clamp(min=1e-06)
                intermediate_per_res_ca_dist_ghost.append(
                    (slot_dist_pre * p_ghost).sum(dim=-1).div(ghost_denom).detach().cpu()
                )
            if return_intermediates:
                intermediate_slot_pad_logit_pre.append(element_logits[..., ELEMENT_PAD].detach().cpu())
            if element_sampling_temp_max != 1.0:
                frac = t_idx.float() / max(self.timesteps - 1, 1)
                elem_temp = 1.0 + (element_sampling_temp_max - 1.0) * frac**element_sampling_temp_power
            else:
                elem_temp = 1.0
            elem_sched_kwargs = {}
            mask_for_powers = torch.ones(batch_size, seq_len, max_sc, dtype=torch.bool, device=device)
            elem_slot_powers = self._element_slot_powers(mask_for_powers)
            sab, sabp, sb = self.element_diffusion.compute_slot_schedule(t, elem_slot_powers)
            elem_sched_kwargs = {"slot_alpha_bar": sab, "slot_alpha_bar_prev": sabp, "slot_beta": sb}
            effective_squash = 1.0
            element_types_new = self.element_diffusion.p_sample(
                element_types,
                t,
                element_logits,
                temperature=elem_temp,
                posterior_pad_squash=effective_squash,
                absorbing_mask=True,
                **elem_sched_kwargs,
            )
            element_types_new = apply_prefix_constraint(element_types_new, exempt_slot0=reserved_slot0_prefix_exempt)
            element_types = element_types_new
            noised_mask = (element_types != ELEMENT_PAD).float()
            if evc_sampling is not None:
                1.0 - step_i / max(num_steps - 1, 1)
                if getattr(self, "evc_ss_noised_element_prob", 0.0) > 0:
                    evc_sampling = evc_from_element_state(element_types)
                else:
                    evc_sampling = ((element_types != ELEMENT_PAD) & (element_types != ELEMENT_MASK)).float()
            prev_element_pred = torch.softmax(element_logits, dim=-1).detach()
            prev_cluster_pred = torch.softmax(cluster_logits, dim=-1).detach()
            x0_prev_coords = self._x0_from_model_output(model_output, x, t, ca_coords, mask_probs=noised_mask).detach()
            x0_prev_mask = noised_mask.detach()
            x0_prev_element = element_types.detach()
            x_flat = x.view(batch_size, -1, 3)
            model_output_flat = model_output.view(batch_size, -1, 3)
            if evc_sampling is not None and getattr(self, "evc_velocity_blend", False):
                v_ghost = self.coord_flow.analytical_ghost_velocity(x_flat, t, ca_coords)
                p_real_flat = evc_sampling.reshape(batch_size, -1, 1)
                model_output_flat = p_real_flat * model_output_flat + (1.0 - p_real_flat) * v_ghost
            if t_idx > 0:
                t_prev_idx = timesteps[step_i + 1]
                t_prev = torch.full((batch_size,), t_prev_idx.item(), device=device, dtype=torch.long)
                x = self.coord_flow.flow_step(x_flat, t, t_prev, model_output_flat).view_as(x)
            else:
                x = self.coord_flow.predict_x0_from_velocity(
                    x_flat, t, model_output_flat, ca_coords=ca_coords, mask_probs=noised_mask.view(batch_size, -1, 1)
                ).view_as(x)
            x, element_types, noised_mask, evc_sampling = self._pin_inpaint(
                x,
                element_types,
                noised_mask,
                evc_sampling,
                design_mask=design_mask,
                inpaint_gt_coords=inpaint_gt_coords,
                inpaint_gt_elements=inpaint_gt_elements,
                inpaint_gt_mask=inpaint_gt_mask,
                inpaint_mode=inpaint_mode,
                t=t,
                ca_coords=ca_coords,
                leak_direction=leak_direction,
            )
            if return_coord_trajectory:
                coord_traj.append(x.detach().to("cpu", torch.float32).clone())
                elem_traj.append(element_types.detach().to("cpu").clone())
                mask_traj.append((element_types != ELEMENT_PAD).detach().to("cpu").clone())
            if return_intermediates:
                current_counts = noised_mask.float().view(batch_size, -1).sum(dim=-1).tolist()
                intermediate_atom_counts.append(current_counts)
                mix_post = self.compute_mixture_posterior(
                    noised_coords=x,
                    ca_coords=ca_coords,
                    t=t,
                    learned_centroid=denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None,
                    learned_cloud_logvar=denoiser_outputs.get("residue_cloud_logvar")
                    if self.mixture_loss_weight > 0
                    else None,
                    residue_lrt_delta=None,
                    residue_count_pred=None,
                    temperature=mix_temperature,
                )
                if seq_mask is not None:
                    mix_post = mix_post * seq_mask.unsqueeze(-1).float()
                mix_counts = mix_post.view(batch_size, -1).sum(dim=-1).tolist()
                intermediate_mixture_counts.append(mix_counts)
                per_res_hard = (element_types != ELEMENT_PAD).float().sum(dim=-1)
                intermediate_per_res_hard.append(per_res_hard.detach().cpu())
                intermediate_slot_non_pad_post.append((element_types != ELEMENT_PAD).detach().cpu())
        from .diffusion import ELEMENT_MASK

        is_mask = element_types == ELEMENT_MASK
        n_mask = is_mask.sum().item()
        n_total = element_types.numel()
        if is_mask.any():
            ca_exp = ca_coords.unsqueeze(2).expand_as(x)
            dist_to_ca = (x - ca_exp).norm(dim=-1)
            shell_radii = self.coord_flow._shell_radii
            threshold = self._distal_read_threshold(shell_radii, epoch=None).view(1, 1, max_sc).expand_as(dist_to_ca)
            collapse_to_pad = dist_to_ca < threshold
            n_to_pad = (is_mask & collapse_to_pad).sum().item()
            n_to_carbon = (is_mask & ~collapse_to_pad).sum().item()
            print(
                f"  MASK collapse: {n_mask}/{n_total} ({100 * n_mask / n_total:.1f}%) remaining -> {n_to_pad} PAD + {n_to_carbon} Carbon"
            )
            element_types = torch.where(is_mask & collapse_to_pad, torch.zeros_like(element_types), element_types)
            element_types = torch.where(is_mask & ~collapse_to_pad, torch.ones_like(element_types), element_types)
        else:
            print(f"  MASK collapse: 0/{n_total} remaining (all resolved)")
        predicted_mask = element_types != ELEMENT_PAD
        raw_coords = x.clone()
        x = x * predicted_mask.unsqueeze(-1).float()
        result = {
            "sidechain_coords": x,
            "element_types": element_types,
            "predicted_mask": predicted_mask,
            "raw_coords": raw_coords,
        }
        if return_coord_trajectory:
            result["coord_trajectory"] = coord_traj
            result["elem_trajectory"] = elem_traj
            result["mask_trajectory"] = mask_traj
        if return_intermediates:
            result["intermediate_atom_counts"] = intermediate_atom_counts
            result["intermediate_mixture_counts"] = intermediate_mixture_counts
            if intermediate_per_res_soft_pre:
                result["intermediate_per_res_soft_pre"] = intermediate_per_res_soft_pre
                result["intermediate_per_res_hard"] = intermediate_per_res_hard
                result["intermediate_per_res_ca_dist_real"] = intermediate_per_res_ca_dist_real
                result["intermediate_per_res_ca_dist_ghost"] = intermediate_per_res_ca_dist_ghost
            if intermediate_slot_pad_logit_pre:
                result["intermediate_slot_pad_logit_pre"] = intermediate_slot_pad_logit_pre
            if intermediate_slot_non_pad_post:
                result["intermediate_slot_non_pad_post"] = intermediate_slot_non_pad_post
        if return_intermediates and final_mixture_posterior is None:
            t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
            diag_centroid = denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None
            diag_logvar = denoiser_outputs.get("residue_cloud_logvar") if self.mixture_loss_weight > 0 else None
            if diag_centroid is not None:
                final_residue_centroid = diag_centroid
            if diag_logvar is not None:
                final_residue_cloud_logvar = diag_logvar
            final_mixture_posterior = self.compute_mixture_posterior(
                noised_coords=x,
                ca_coords=ca_coords,
                t=t_zero,
                learned_centroid=diag_centroid,
                learned_cloud_logvar=diag_logvar,
                residue_lrt_delta=None,
                temperature=self.sharpen_temperature_min,
            )
            if seq_mask is not None:
                final_mixture_posterior = final_mixture_posterior * seq_mask.unsqueeze(-1).float()
        if final_mixture_posterior is not None:
            result["mixture_posterior"] = final_mixture_posterior.detach()
        if final_residue_centroid is not None:
            result["residue_centroid"] = final_residue_centroid.detach()
        if final_residue_cloud_logvar is not None:
            result["residue_cloud_logvar"] = final_residue_cloud_logvar.detach()
        return result
