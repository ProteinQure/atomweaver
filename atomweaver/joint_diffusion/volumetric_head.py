"""Volumetric self-occupancy head (self-occupancy-style) -- head + supervision loss only.

  ADDITIVE GRAFT / FIRST CHUNK. No flow injection is wired here (a later chunk deep-injects
``vol_hidden``); this module is byte-identical when its flag is off and resume-safe.

Motivation
----------
The main model (:class:`~atomweaver.joint_diffusion.models.InverseFoldingDiffusion`) places
side-chain atoms as a point cloud. 's *self-occupancy* competitor instead learns a **neural
occupancy field**: given a target-aware local context, it predicts the *density* a masked
residue's own side chain should occupy at arbitrary query points. This module ports that idea
as a small, parallel, **t-independent** per-residue head:

* **Self-occupancy.** For binder residue ``i`` it predicts the density of the side chain it
  should grow -- evaluated at a fixed lattice of query points in ``i``'s local backbone frame --
  from a context that DELIBERATELY EXCLUDES that side chain (so there is nothing to copy).
* **Target-aware context.** Context = all atoms within ``context_radius`` of ``CA_i`` -- binder
  BACKBONE atoms (N/CA/C/O of every residue) + TARGET atoms -- expressed in ``i``'s local frame
  (an SE(3) invariant). Binder side chains are never in the context (neither ``i``'s own, which
  is the whole point, nor its neighbours', which would leak generated/GT structure), so the
  head is a clean function of the fixed backbone + fixed target.
* **Anti-leak.** The GT density is a Gaussian splat of ``i``'s GT side-chain atoms at the query
  points. Query points far from any GT atom (the backbone hemisphere + an explicit far shell)
  get ~0 GT -- regressing the field toward 0 there is the reference's anti-leak term. Those "empty"
  query points are up-weighted by ``empty_weight`` (self-occupancy default 1.75).

Deviations from self-occupancy (flagged for review)
-------------------------------------------
* **Fixed canonical query lattice** (radial Fibonacci shells in the local frame + a far shell),
  evaluated per residue, instead of the reference's atom-relative sampling (masked-center /
  masked-around / distance-banded empties). Simpler, fully deterministic, batched, invariant.
  The "empty" bucket is derived at run time from where the GT splat is ~0, rather than sampled.
* **Å units + Gaussian sigma** matching the existing ``occupancy_match_loss`` (self-occupancy works in
  nm with a per-atom vdW-derived sigma). One scalar ``sigma`` here; no per-atom vdW radii.
* **Sum composition** of per-atom Gaussians (the reference's default), unit height.
* **Element vocab.** Per-atom element embedding sized to the active vocab (5 or 12); backbone
  atom elements are assigned N/C/C/O via the shared ``ELEMENT_*`` ids (vocab-agnostic).

Everything here is only constructed / run when ``use_volumetric_head`` is on.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint

from .diffusion import ELEMENT_C, ELEMENT_N, ELEMENT_O, NUM_ELEMENT_TYPES
from .residue_frame_stream import build_local_frames

# Backbone atom axis ordering (B, L, 4, 3): N / CA / C / O -- matches models.py throughout.
_N_IDX, _CA_IDX, _C_IDX, _O_IDX = 0, 1, 2, 3
# Per-backbone-atom element ids (vocab-agnostic: C/N/O share ids across the 5- and 12-vocabs).
_BACKBONE_ELEMENTS = (ELEMENT_N, ELEMENT_C, ELEMENT_C, ELEMENT_O)

# "Sandclock" available-volume descriptor width (see ``available_volume_cones``): a per-lobe radial
# first-hit histogram over ``AVAILABLE_VOLUME_N_BINS`` bins for BOTH e3 lobes => 2 * n_bins channels.
AVAILABLE_VOLUME_N_BINS = 8
AVAILABLE_VOLUME_DIM = 2 * AVAILABLE_VOLUME_N_BINS


# --- Per-element vdW-derived occupancy splat sigma (self-occupancy-style) -------------------------------------
# The uniform-sigma splat gives EVERY atom the same fuzzy ball, so the head can hit low occupancy-MSE with
# a generic non-discriminative blob. Self-occupancy instead splats each atom with its vdW radius, injecting
# atom-SIZE structure (bulky S/X vs small O) that distinguishes similar-heavy-atom side chains. Our element
# vocab is tiny, so no AMBER/GAFF atom typing is needed: C/N/O get their own Bondi radius and every non-CNO
# heavy atom (vocab5 X=id 4; vocab12 S/P/F/Cl/Br/I/Se/B=ids 4..11 -- X is dominated by S, Bondi(S)=1.80)
# uses the single X radius. PAD/ghost slots contribute NO density (splat weight 0), so their radius is
# never read. Keyed by the MODEL element ids (``ELEMENT_C/N/O`` from diffusion.py); DB rotamer atoms are
# mapped into these model ids before lookup (see eval_volumetric_recovery.build_reference_densities), so
# there is ONE element->id scheme, not a second one.
_BONDI_RADIUS_C = 1.70
_BONDI_RADIUS_N = 1.55
_BONDI_RADIUS_O = 1.52
_BONDI_RADIUS_X = 1.80  # any non-CNO heavy element (dominated by S)
_SIGMA_NORM_RADII = (_BONDI_RADIUS_C, _BONDI_RADIUS_N, _BONDI_RADIUS_O, _BONDI_RADIUS_X)

# Self-occupancy density convention (self-occupancy/ml/density.py::evaluate_atom_density, config alpha=0.35, min_sigma=0.03 nm):
# sigma_atom = max(vdw_radius * alpha, min_sigma). This is ~1.7x SHARPER than the legacy mean-normalized-to-1.0
# scale (0.609) -- carbon lands at 1.70 * 0.35 = 0.595 Å vs the old ~1.03 Å. A broad splat gives every side chain
# a similar fuzzy blob (the decoy-CE cannot separate them); the sharp self-occupancy sigma injects the spatial structure
# that distinguishes one residue's occupancy from another's. Å units here (self-occupancy is nm; 0.03 nm floor -> 0.3 Å).
VOLUMETRIC_SIGMA_ALPHA = 0.35
VOLUMETRIC_MIN_SIGMA_ANGSTROM = (
    0.3  # self-occupancy min_sigma 0.03 nm; inert for C/N/O/X (smallest = O: 1.52*0.35=0.53 Å)
)

# Fourier query-lift + non-negative-field defaults (see FourierFeatureEncoder + VolumetricOccupancyHead).
# Fourier is ON by default (deterministic: it changes the arch but never the bit-for-bit forward given a fixed
# b_matrix, so it is safe for the resume/byte-identical invariants). DROPOUT, by contrast, is stochastic and its
# RNG position shifts when the head is invoked early (deep-inject) vs in the loss section, so it defaults to 0.0
# here (byte-identical) and is turned on to the self-occupancy 0.10 by the CLI for production runs.
DEFAULT_FOURIER_FREQUENCIES = 64  # -> 128-d lift (self-occupancy pos_encoding_dim=128 = num_frequencies*2)
DEFAULT_FOURIER_SCALE = 10.0  # nm-frequency scale; = the reference's, since query/radius is the Å->nm conversion
PRODUCTION_VOLUMETRIC_DROPOUT = 0.1  # self-occupancy trunk/attention dropout (the CLI default; module default is 0.0)
# Final-Linear bias init when softplus is on. Self-occupancy never re-inits its final occ-head bias (default
# ``nn.Linear`` zero bias), so the field starts at ``softplus(0)=0.69`` -- a live, mid-range gradient. The
# earlier ``-2.0`` init pinned the field in the empty basin (softplus(-2)≈0.13) with a 4× weaker gradient
# (σ(-2)≈0.12 vs σ(0)=0.5); the collapsed run6 head's trained final bias sat at -1.98, i.e. it never
# escaped that basin. Set to 0.0 (faithful) UNCONDITIONALLY. This only affects FRESH init; a loaded
# checkpoint overrides this bias in ``load_state_dict``, so resume/byte-identical-on-resume is unaffected.
_DENSITY_SOFTPLUS_BIAS_INIT = 0.0


def default_sigma_element_scale() -> float:
    """Default per-element sigma scale = the reference's ``alpha`` (``sigma = Bondi_radius * alpha``).

    Returns the reference's ``alpha=0.35`` (:data:`VOLUMETRIC_SIGMA_ALPHA`), so the per-element table is the faithful
    ``vdw * 0.35`` (floored at :data:`VOLUMETRIC_MIN_SIGMA_ANGSTROM`), NOT the legacy mean-normalized-to-1.0 scale.
    Only the ABSOLUTE sharpness changed (blurry ~1.0 Å -> sharp ~0.6 Å); the relative per-element order is
    unchanged. A checkpoint that stored an explicit scale keeps its own value; this default only fills the
    ``None`` (unspecified) case.
    """
    return VOLUMETRIC_SIGMA_ALPHA


def legacy_default_sigma_element_scale() -> float:
    """Pre-Fourier per-element sigma scale = ``1 / mean(Bondi C/N/O/X)`` so the table's mean sigma == 1.0 Å.

    This was :func:`default_sigma_element_scale` BEFORE the self-occupancy-alpha (0.6 Å) change. Kept ONLY so the
    eval-rebuild can faithfully score a checkpoint that was TRAINED with the blurry ~1.0 Å convention and did
    not store an explicit scale (``volumetric_sigma_element_scale=None``). Do NOT use for new runs.
    """
    return 1.0 / (sum(_SIGMA_NORM_RADII) / len(_SIGMA_NORM_RADII))


def _element_bondi_radius(element_id: int) -> float:
    """Bondi vdW radius (Å) for a MODEL element id (vocab5/vocab12). Any non-CNO id -> X radius."""
    if element_id == ELEMENT_C:
        return _BONDI_RADIUS_C
    if element_id == ELEMENT_N:
        return _BONDI_RADIUS_N
    if element_id == ELEMENT_O:
        return _BONDI_RADIUS_O
    return _BONDI_RADIUS_X  # X / any non-CNO heavy element (and PAD, whose splat weight is 0)


def build_element_sigma_table(num_element_types: int = NUM_ELEMENT_TYPES, scale: float | None = None) -> torch.Tensor:
    """``(num_element_types,)`` per-element-id Gaussian sigma = ``Bondi_radius * scale`` (MODEL-id indexed).

    ``scale`` defaults to :func:`default_sigma_element_scale` (mean over C/N/O/X normalized to 1.0). The same
    table is used by the head's GT-target splat, the self-occupancy-target splat, and the eval-gate reference splat,
    so a checkpoint trained with per-element sigma is scored against per-element references.
    """
    s = float(scale) if scale is not None else default_sigma_element_scale()
    sigmas = [max(_element_bondi_radius(i) * s, VOLUMETRIC_MIN_SIGMA_ANGSTROM) for i in range(int(num_element_types))]
    return torch.tensor(sigmas, dtype=torch.float32)


def per_atom_sigma_from_ids(element_ids: torch.Tensor, sigma_table: torch.Tensor) -> torch.Tensor:
    """Gather a per-atom sigma from per-atom MODEL element ids via ``sigma_table``. Out-of-range ids clamp."""
    n = sigma_table.shape[0]
    ids = element_ids.long().clamp(0, n - 1)
    return sigma_table.to(device=element_ids.device)[ids]


def splat_gaussian_kernel(d2: torch.Tensor, sigma: float, per_atom_sigma: torch.Tensor | None = None) -> torch.Tensor:
    """Isotropic Gaussian occupancy kernel over squared distances ``d2`` ``(..., Q, M)``.

    ``per_atom_sigma is None`` (default) => the LEGACY scalar-sigma path, ``exp(-0.5 d2 / sigma²)``, BYTE-
    IDENTICAL to the pre-existing splat. When a ``(..., M)`` per-atom sigma is given, atom ``m`` uses its own
    width (broadcast over the query axis ``Q``): ``exp(-0.5 d2 / sigma_m²)``.
    """
    if per_atom_sigma is None:
        return torch.exp(-0.5 * d2 / (float(sigma) ** 2))
    inv_two_sigma2 = (0.5 / (per_atom_sigma.to(d2.dtype) ** 2)).unsqueeze(-2)  # (..., 1, M) -- broadcast over Q
    return torch.exp(-d2 * inv_two_sigma2)


class FourierFeatureEncoder(nn.Module):
    """Random Fourier feature lift of a coordinate (self-occupancy ``FourierFeatureEncoder``, ported byte-faithfully).

    Lifts a ``(..., input_dim)`` coordinate to ``(..., 2 * num_frequencies)`` via
    ``[sin(2π·x·B), cos(2π·x·B)]`` with a FIXED random projection ``B ~ N(0, scale²)`` of shape
    ``(input_dim, num_frequencies)``. Matches self-occupancy exactly (``the reference encoder``): NO raw
    coordinate is concatenated, and the output width is ``2 * num_frequencies`` (NOT ``2·input_dim·num_freq``).

    Why this is the key lever
    -------------------------
    An MLP on the raw 3-vector has a strong spectral bias -- it can only fit smooth/blurry functions of the
    query point, so every residue's predicted occupancy collapses to a similar blob and the decoy-CE (neg_mse
    to references) is near-uniform. Lifting the query through high-frequency Fourier features lets the trunk
    represent the SHARP spatial structure that distinguishes side chains.

    Scale convention (nm vs our normalized coords)
    ----------------------------------------------
    ``B`` entries ~ ``N(0, scale²)`` set the finest representable spatial wavelength to ``~1/scale`` in the
    encoder's INPUT units. Self-occupancy feeds coords in **nm** (``context_radius_nm=1.0``) with ``scale=10.0`` =>
    finest wavelength ``0.1 nm = 1 Å``. The head feeds ``query / context_radius`` (Å ÷ ``radius``); with the
    default ``radius=10 Å`` that division is exactly the Å->nm conversion, so our normalized coord numerically
    EQUALS the reference's nm coord and ``scale=10.0`` transfers directly (finest wavelength ``= radius/scale ≈ 1 Å``).

    ``b_matrix`` is a PERSISTENT buffer: it is random at init, so it must be saved with the checkpoint and
    reloaded verbatim (the trained trunk is tied to this specific projection). It is a buffer (not a Parameter),
    so it is frozen and follows the module's device/dtype.
    """

    def __init__(self, input_dim: int = 3, num_frequencies: int = 64, scale: float = 10.0) -> None:
        super().__init__()
        if num_frequencies <= 0:
            raise ValueError(f"num_frequencies must be > 0 (got {num_frequencies})")
        self.input_dim = int(input_dim)
        self.num_frequencies = int(num_frequencies)
        self.scale = float(scale)
        # Persistent (default) so the RANDOM projection is checkpointed + reloaded exactly.
        self.register_buffer("b_matrix", torch.randn(self.input_dim, self.num_frequencies) * self.scale)

    @property
    def output_dim(self) -> int:
        """Encoded width = ``2 * num_frequencies`` (sin ++ cos), no raw-coord concat (self-occupancy convention)."""
        return self.num_frequencies * 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """``(..., input_dim) -> (..., 2*num_frequencies)`` random Fourier features."""
        proj = 2.0 * math.pi * (coords @ self.b_matrix.to(coords.dtype))
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# Per-query kNN local-context defaults (self-occupancy ContextEncoder; see configs/baseline_June6_2026.yaml).
DEFAULT_CONTEXT_K = 128  # k_neighbors
DEFAULT_CONTEXT_HEADS = 4  # NeighborhoodAttention num_heads
# Residue-axis chunk for the per-query ContextEncoder. The dominant tensor is (G, Q, k, hidden); at G=B*L
# residues in parallel it OOMs (~tens of GB at B=8). Each residue's context is INDEPENDENT, so processing G
# in chunks of this size and concatenating is NUMERICALLY EXACT (peak set by ``chunk``, not B*L). Only active
# when ``use_per_query_context`` is on; chunk >= G => a single pass, identical to the unchunked call.
DEFAULT_CONTEXT_CHUNK = 32

# Number of self-occupancy region buckets (center / around / context / near / medium / far-empty). Mirrors
# ``volumetric_supervision.REGION_NAMES`` but is duplicated here to avoid a circular import (volumetric_supervision
# imports from this module). Used as the default segment count for the per-region scale-anchor loss.
NUM_VOLUMETRIC_REGIONS = 6


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-query neighbor values (self-occupancy ``encoders._gather_neighbors``, ported verbatim).

    ``values`` (G, N, feat), ``indices`` (G, M, k) -> (G, M, k, feat): for each of the ``M`` queries,
    gather the ``k`` selected context rows out of the ``N`` context atoms.
    """
    bsz, num_queries, k = indices.shape
    _, num_context, feat_dim = values.shape
    expanded = values.unsqueeze(1).expand(bsz, num_queries, num_context, feat_dim)
    idx = indices.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
    return torch.gather(expanded, 2, idx)


class NeighborhoodAttention(nn.Module):
    """Dot-product attention of a query over its per-query neighborhood (self-occupancy ``layers.NeighborhoodAttention``).

    Ported byte-faithfully: ``num_heads`` heads, ``1/sqrt(head_dim)`` scaling, separate
    ``query_proj``/``key_proj``/``value_proj``; masked-neighbor softmax over the k axis. ``query`` is
    ``(G, M, query_dim)``, ``neighbors`` is ``(G, M, K, neighbor_dim)`` -> ``(G, M, hidden_dim)``.
    """

    def __init__(
        self,
        query_dim: int,
        neighbor_dim: int,
        hidden_dim: int,
        num_heads: int = DEFAULT_CONTEXT_HEADS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(neighbor_dim, hidden_dim)
        self.value_proj = nn.Linear(neighbor_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``query`` (G, M, query_dim), ``neighbors`` (G, M, K, neighbor_dim) -> (G, M, hidden_dim)."""
        bsz, num_queries, num_neighbors, _ = neighbors.shape
        q = self.query_proj(query).view(bsz, num_queries, self.num_heads, self.head_dim)
        k = self.key_proj(neighbors).view(bsz, num_queries, num_neighbors, self.num_heads, self.head_dim)
        v = self.value_proj(neighbors).view(bsz, num_queries, num_neighbors, self.num_heads, self.head_dim)
        q = q.unsqueeze(2)  # (G, M, 1, H, D)
        attn_scores = (q * k).sum(dim=-1) * self.scale  # (G, M, K, H)
        if neighbor_mask is not None:
            # ``finfo(dtype).min`` (not a hard-coded ``-1e9``, which OVERFLOWS to -inf in fp16 and yields a
            # NaN softmax): masks the neighbor without breaking any precision. Mirrors the pooled-attention
            # sibling in ``VolumetricOccupancyHead.forward``.
            attn_scores = attn_scores.masked_fill(~neighbor_mask.unsqueeze(-1), torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores, dim=2)
        attn_weights = self.dropout(attn_weights)
        context = (attn_weights.unsqueeze(-1) * v).sum(dim=2)  # (G, M, H, D)
        context = context.reshape(bsz, num_queries, self.num_heads * self.head_dim)
        if neighbor_mask is not None:
            # A query whose ENTIRE neighborhood is masked has an all-(min) score row => softmax returns a
            # UNIFORM distribution over invalid neighbors (whose projected values are non-zero because
            # ``neighbor_proj`` has a bias), i.e. a spurious non-zero context. Zero those rows so an
            # all-masked neighborhood => a zero context vector, never NaN, under any precision. (Rows with at
            # least one valid neighbor keep their softmax => byte-identical.)
            has_neighbor = neighbor_mask.any(dim=-1)  # (G, M)
            context = context * has_neighbor.unsqueeze(-1).to(context.dtype)
        return context


class ContextEncoder(nn.Module):
    """Per-query local-context encoder via kNN + neighborhood attention (self-occupancy ``encoders.ContextEncoder``).

    Ported faithfully (the AMBER-charge branch is dropped -- this repo has no partial charges). For each
    query point it selects its ``k_neighbors`` nearest CONTEXT atoms, builds per-neighbor features
    ``[distance(1), relative_xyz(3) = query - neighbor, initiator_bit(1)]``, projects them
    (``neighbor_proj``), lets the query's embedding attend over them (:class:`NeighborhoodAttention`), and
    projects the pooled result (``out_proj``). This gives EACH query point its own neighborhood -- the
    discrimination signal the shared attention-pooled ``vol_hidden`` broadcast cannot provide.

    Frame convention (SE(3) invariance)
    ------------------------------------
    The head is rotation/translation invariant by construction. The caller therefore passes
    ``query_pos``/``context_pos`` already expressed in each residue's LOCAL frame (``R_iᵀ (x - CA_i)``),
    so ``distance`` and ``relative_xyz`` are invariants -- unlike self-occupancy, which works in a merely centered
    global frame. The math is otherwise byte-faithful to self-occupancy.
    """

    NEIGHBOR_DIM = 1 + 3 + 1  # distance + relative_xyz + initiator_bit

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        k_neighbors: int = DEFAULT_CONTEXT_K,
        num_heads: int = DEFAULT_CONTEXT_HEADS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        self.neighbor_proj = nn.Linear(self.NEIGHBOR_DIM, hidden_dim)
        self.attn = NeighborhoodAttention(query_dim, hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _knn(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(G, M, 3), (G, N, 3) -> (dists, indices) each (G, M, k). Invalid context atoms masked to +inf."""
        distances = torch.cdist(query, context)  # (G, M, N)
        if context_mask is not None:
            distances = distances.masked_fill(~context_mask.unsqueeze(1), float("inf"))
        k = min(self.k_neighbors, context.shape[1])
        dists, indices = torch.topk(distances, k=k, largest=False)
        return dists, indices

    def forward(
        self,
        query_pos: torch.Tensor,
        query_embed: torch.Tensor,
        context_pos: torch.Tensor,
        context_initiator: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``query_pos`` (G, M, 3), ``query_embed`` (G, M, query_dim), ``context_pos`` (G, N, 3),
        ``context_initiator``/``context_mask`` (G, N) -> per-query context vector (G, M, hidden_dim).
        """
        dtype = context_pos.dtype
        dists, indices = self._knn(query_pos, context_pos, context_mask)
        neighbor_pos = _gather_neighbors(context_pos, indices)  # (G, M, k, 3)
        initiator_float = context_initiator.to(dtype).unsqueeze(-1)  # (G, N, 1)
        neighbor_initiator = _gather_neighbors(initiator_float, indices).squeeze(-1)  # (G, M, k)

        neighbor_mask = None
        if context_mask is not None:
            neighbor_mask = _gather_neighbors(context_mask.unsqueeze(-1).to(dtype), indices).squeeze(-1) > 0.5
            dists = dists.masked_fill(~neighbor_mask, 0.0)
            neighbor_initiator = neighbor_initiator.masked_fill(~neighbor_mask, 0.0)

        rel = query_pos.unsqueeze(2) - neighbor_pos  # (G, M, k, 3) query - neighbor
        neighbor_features = torch.cat([dists.unsqueeze(-1), rel, neighbor_initiator.unsqueeze(-1)], dim=-1)
        if neighbor_mask is not None:
            neighbor_features = neighbor_features * neighbor_mask.unsqueeze(-1).to(dtype)
        neighbor_features = self.neighbor_proj(neighbor_features)  # (G, M, k, hidden)
        context = self.attn(query_embed, neighbor_features, neighbor_mask=neighbor_mask)  # (G, M, hidden)
        return self.out_proj(context)


def _fibonacci_sphere(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """``n`` roughly-uniform unit vectors on S² (deterministic Fibonacci spiral). (n, 3)."""
    if n <= 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=dtype)
    # z in (-1, 1); golden-angle azimuth.
    z = 1.0 - 2.0 * (idx + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    phi = idx * (math.pi * (3.0 - math.sqrt(5.0)))  # golden angle
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)


def build_query_points(
    n_query: int,
    near_min_radius: float = 1.5,
    near_max_radius: float = 6.0,
    far_radius: float = 9.0,
    far_fraction: float = 0.25,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Fixed local-frame query lattice for the occupancy field. (Q, 3), origin = CA.

    ``near`` points form a volumetric Fibonacci spiral spanning the side-chain shell
    ``[near_min_radius, near_max_radius]`` Å from CA; ``far`` points sit on a single shell at
    ``far_radius`` Å (guaranteed ~0 GT => explicit anti-leak). Deterministic given ``n_query``.
    """
    if n_query <= 0:
        raise ValueError(f"n_query must be > 0 (got {n_query})")
    n_far = round(far_fraction * n_query)
    n_far = max(0, min(n_query - 1, n_far)) if n_query > 1 else 0
    n_near = n_query - n_far

    near_dirs = _fibonacci_sphere(n_near, device=device, dtype=dtype)  # (n_near, 3)
    if n_near > 1:
        frac = torch.arange(n_near, device=device, dtype=dtype) / (n_near - 1)
    else:
        frac = torch.zeros(n_near, device=device, dtype=dtype)
    near_radii = near_min_radius + (near_max_radius - near_min_radius) * frac  # (n_near,)
    near_pts = near_dirs * near_radii.unsqueeze(-1)

    if n_far > 0:
        far_pts = _fibonacci_sphere(n_far, device=device, dtype=dtype) * far_radius
        return torch.cat([near_pts, far_pts], dim=0)  # (Q, 3)
    return near_pts


# =====================================================================================================
# VOLUMETRIC-FAITHFUL ATOM-ANCHORED QUERY SAMPLING (gated behind ``atom_anchored_queries``)
# -----------------------------------------------------------------------------------------------------
# The fixed Fibonacci lattice above re-labels a shared CA-centered point cloud into the reference's region
# buckets, which leaves the per-residue target mostly EMPTY (only ~2-4 of 128 fixed points land near a GT
# atom; zero on small residues Ser/Ala/Thr) and drives an all-empty collapse. Self-occupancy instead SAMPLES
# atom-anchored queries per example -- one at each GT own-atom center, a cluster around each, negatives at
# context atoms, and distance-banded empties -- so the positive buckets are guaranteed populated. We do the
# same here, ON-THE-FLY per forward, in each residue's LOCAL frame (origin = CA), so the sampled queries are
# SE(3) invariants directly comparable to the head's fixed-lattice queries.
#
# Query-type codes: these integers MUST equal ``volumetric_supervision.REGION_*`` (and hence the reference's
# ``QUERY_TYPE_*``) so the sampled ``query_type`` tensor plugs straight into ``region_occupancy_loss`` /
# ``OccupancyLossWeights`` (center/around/context @1.0, near/medium/far_empty @empty_weight). Duplicated
# here (not imported) to avoid a circular import (volumetric_supervision imports from this module).
QUERY_TYPE_CENTER = 0
QUERY_TYPE_AROUND = 1
QUERY_TYPE_CONTEXT = 2
QUERY_TYPE_NEAR_EMPTY = 3
QUERY_TYPE_MEDIUM_EMPTY = 4
QUERY_TYPE_FAR_EMPTY = 5

# Self-occupancy defaults (config baseline_June6_2026.yaml), converted nm -> Å (self-occupancy works in nm; ×10). These are
# hard self-occupancy constants (not model hparams), so they live as sampler defaults rather than CLI knobs.
VOLUMETRIC_MASKED_AROUND_PER_ATOM = 8  # masked_around_per_atom
VOLUMETRIC_MASKED_AROUND_RADIUS_A = 1.2  # masked_around_radius 0.12 nm
VOLUMETRIC_CONTEXT_NEGATIVES = 96  # context_negatives_per_example
VOLUMETRIC_CENTER_JITTER_STD_A = 0.2  # ~0.02 nm light isotropic jitter on masked-center
VOLUMETRIC_NEAR_EMPTY_BAND_A = (1.2, 2.5)  # near_empty 0.12-0.25 nm
VOLUMETRIC_MEDIUM_EMPTY_BAND_A = (2.5, 3.5)  # medium_empty 0.25-0.35 nm
VOLUMETRIC_FAR_EMPTY_MIN_A = 3.5  # far_empty 0.35 nm+ (capped at context_radius by the caller)
VOLUMETRIC_EMPTY_FRACTIONS = (0.35, 0.35, 0.30)  # near / medium / far split of the leftover empty slots


def _uniform_in_ball(shape: tuple[int, ...], radius: float, device, dtype, generator=None) -> torch.Tensor:
    """``shape + (3,)`` points sampled uniformly inside a ball of ``radius`` (Å). Direction ⊥ radius."""
    dirs = torch.randn(*shape, 3, device=device, dtype=dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u = torch.rand(*shape, device=device, dtype=dtype, generator=generator)
    r = radius * u.clamp_min(0.0) ** (1.0 / 3.0)  # uniform-in-volume radius
    return dirs * r.unsqueeze(-1)


def _uniform_in_shell(
    shape: tuple[int, ...], r_min: float, r_max: float, device, dtype, generator=None
) -> torch.Tensor:
    """``shape + (3,)`` points sampled in a spherical shell ``[r_min, r_max]`` (Å), uniform-in-volume radius."""
    dirs = torch.randn(*shape, 3, device=device, dtype=dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u = torch.rand(*shape, device=device, dtype=dtype, generator=generator)
    r = (r_min**3 + u * (r_max**3 - r_min**3)).clamp_min(0.0) ** (1.0 / 3.0)
    return dirs * r.unsqueeze(-1)


def sample_atom_anchored_queries(
    own_local: torch.Tensor,
    own_mask: torch.Tensor,
    context_local: torch.Tensor,
    context_mask: torch.Tensor,
    n_query: int,
    *,
    masked_around_per_atom: int = VOLUMETRIC_MASKED_AROUND_PER_ATOM,
    masked_around_radius: float = VOLUMETRIC_MASKED_AROUND_RADIUS_A,
    context_negatives: int = VOLUMETRIC_CONTEXT_NEGATIVES,
    center_jitter_std: float = VOLUMETRIC_CENTER_JITTER_STD_A,
    near_band: tuple[float, float] = VOLUMETRIC_NEAR_EMPTY_BAND_A,
    medium_band: tuple[float, float] = VOLUMETRIC_MEDIUM_EMPTY_BAND_A,
    far_min: float = VOLUMETRIC_FAR_EMPTY_MIN_A,
    far_max: float = 10.0,
    empty_fractions: tuple[float, float, float] = VOLUMETRIC_EMPTY_FRACTIONS,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faithful atom-anchored per-residue query sampler (local frame, origin = CA).

    Returns a FIXED-``n_query`` set of queries per residue, in a CANONICAL slot layout so
    ``field_to_hidden_proj = Linear(n_query, hidden)`` keeps consistent slot semantics:

    ``[ per-own-atom block (K·(1+masked_around_per_atom)) | context (context_negatives) | empties ]``

    where each own-atom ``k`` owns ``1`` masked-CENTER slot (its GT center + light jitter) followed by
    ``masked_around_per_atom`` masked-AROUND slots (uniform within ``masked_around_radius`` of the center).
    Own-atom slots are marked INVALID (via the returned mask) where the residue has no atom in slot ``k``
    (padded), so the layout is fixed even though residues have different atom counts. Context slots take the
    ``context_negatives`` NEAREST valid context atoms (backbone + target + other residues, self-excluded by
    the caller); a shortfall pads with invalid slots. The leftover slots are distance-banded EMPTIES
    (near/medium/far by ``empty_fractions``), always valid and guaranteed ~0 GT own-density.

    DDP-safe: every residue gets exactly ``n_query`` slots (no variable length), sampling is per-example
    independent (no cross-rank sync), and runs on the input device/dtype. Reproducibility is optional via
    ``generator``; plain global RNG is fine (this is epoch-to-epoch augmentation, not a fixed target).

    Parameters
    ----------
    own_local : (B, L, K, 3) -- the residue's OWN GT side-chain atoms in ITS local frame.
    own_mask : (B, L, K) -- own real-atom mask.
    context_local : (B, L, N, 3) -- context atoms (backbone + target + other side chains) in the local frame.
    context_mask : (B, L, N) -- context validity.
    n_query : int -- total queries per residue (must equal the head's ``n_query``).

    Returns
    -------
    queries : (B, L, n_query, 3) local-frame query points.
    query_type : (B, L, n_query) ``long`` ``QUERY_TYPE_*`` code per slot.
    valid : (B, L, n_query) ``bool`` per-slot validity (masked own/context slots are False).
    """
    b, length, k = own_local.shape[:3]
    device, dtype = own_local.device, own_local.dtype
    a = int(masked_around_per_atom)
    n_own_block = k * (1 + a)
    n_ctx = int(context_negatives)
    n_empty = int(n_query) - n_own_block - n_ctx
    if n_empty < 0:
        raise ValueError(
            f"atom-anchored layout does not fit n_query={n_query}: own block K·(1+around)={n_own_block} + "
            f"context {n_ctx} > {n_query}. Set --volumetric-n-query >= {n_own_block + n_ctx} (canonical 256), "
            f"or lower masked_around_per_atom / context_negatives."
        )
    n_near = round(empty_fractions[0] * n_empty)
    n_medium = round(empty_fractions[1] * n_empty)
    n_far = n_empty - n_near - n_medium

    # --- per-own-atom block: 1 center + ``a`` around, grouped per atom (canonical layout) ---
    center = own_local.unsqueeze(3) + center_jitter_std * torch.randn(
        b, length, k, 1, 3, device=device, dtype=dtype, generator=generator
    )  # (B, L, K, 1, 3)
    around = own_local.unsqueeze(3) + _uniform_in_ball(
        (b, length, k, a), masked_around_radius, device, dtype, generator=generator
    )  # (B, L, K, a, 3)
    own_block = torch.cat([center, around], dim=3).reshape(b, length, n_own_block, 3)  # (B, L, K·(1+a), 3)
    own_type_per_atom = torch.tensor(
        [QUERY_TYPE_CENTER] + [QUERY_TYPE_AROUND] * a, device=device, dtype=torch.long
    )  # (1+a,)
    own_type = own_type_per_atom.view(1, 1, 1, 1 + a).expand(b, length, k, 1 + a).reshape(b, length, n_own_block)
    own_valid = own_mask.bool().unsqueeze(-1).expand(b, length, k, 1 + a).reshape(b, length, n_own_block)

    # --- context block: the ``n_ctx`` NEAREST valid context atoms (CA is the local-frame origin) ---
    n_ctx_avail = context_local.shape[2]
    d_ctx = context_local.norm(dim=-1).masked_fill(~context_mask.bool(), float("inf"))  # (B, L, N)
    k_take = min(n_ctx, n_ctx_avail)
    if k_take > 0:
        top_d, top_idx = torch.topk(d_ctx, k=k_take, largest=False)  # (B, L, k_take)
        ctx_sel = torch.gather(context_local, 2, top_idx.unsqueeze(-1).expand(b, length, k_take, 3))
        ctx_valid_sel = torch.isfinite(top_d)  # invalid where fewer than k_take real atoms exist
    else:
        ctx_sel = own_local.new_zeros(b, length, 0, 3)
        ctx_valid_sel = own_local.new_zeros(b, length, 0, dtype=torch.bool)
    if k_take < n_ctx:  # pad the shortfall with invalid zero slots (keeps the fixed layout)
        pad = n_ctx - k_take
        ctx_sel = torch.cat([ctx_sel, ctx_sel.new_zeros(b, length, pad, 3)], dim=2)
        ctx_valid_sel = torch.cat([ctx_valid_sel, ctx_valid_sel.new_zeros(b, length, pad, dtype=torch.bool)], dim=2)
    ctx_type = torch.full((b, length, n_ctx), QUERY_TYPE_CONTEXT, device=device, dtype=torch.long)

    # --- distance-banded empties (always valid; GT own-density is ~0 there) ---
    near = _uniform_in_shell((b, length, n_near), near_band[0], near_band[1], device, dtype, generator=generator)
    medium = _uniform_in_shell(
        (b, length, n_medium), medium_band[0], medium_band[1], device, dtype, generator=generator
    )
    far = _uniform_in_shell((b, length, n_far), far_min, max(far_min, far_max), device, dtype, generator=generator)
    empty = torch.cat([near, medium, far], dim=2)  # (B, L, n_empty, 3)
    empty_type = torch.cat(
        [
            torch.full((b, length, n_near), QUERY_TYPE_NEAR_EMPTY, device=device, dtype=torch.long),
            torch.full((b, length, n_medium), QUERY_TYPE_MEDIUM_EMPTY, device=device, dtype=torch.long),
            torch.full((b, length, n_far), QUERY_TYPE_FAR_EMPTY, device=device, dtype=torch.long),
        ],
        dim=2,
    )
    empty_valid = torch.ones(b, length, n_empty, device=device, dtype=torch.bool)

    queries = torch.cat([own_block, ctx_sel, empty], dim=2)  # (B, L, n_query, 3)
    query_type = torch.cat([own_type, ctx_type, empty_type], dim=2)  # (B, L, n_query)
    valid = torch.cat([own_valid, ctx_valid_sel, empty_valid], dim=2)  # (B, L, n_query)
    return queries, query_type, valid


def splat_own_density_on_queries(
    query_local: torch.Tensor,
    own_local: torch.Tensor,
    own_mask: torch.Tensor,
    sigma: float,
    per_atom_sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian-splat a residue's OWN atoms onto PER-RESIDUE query points (both in the local frame).

    The atom-anchored analogue of :meth:`VolumetricOccupancyHead._splat_density`, but the query lattice is
    per-residue ``(B, L, Q, 3)`` (sampled) rather than the shared fixed ``(Q, 3)``. Unit-height Gaussians,
    sum composition -- identical convention, so the target is directly comparable to ``vol_density_pred``.

    Parameters
    ----------
    query_local : (B, L, Q, 3) per-residue query points (local frame).
    own_local : (B, L, K, 3) own atoms (local frame).
    own_mask : (B, L, K) own real-atom mask.
    sigma : float -- scalar Gaussian width (Å); used when ``per_atom_sigma`` is None.
    per_atom_sigma : (B, L, K), optional -- per-atom width (broadcast over Q).
    """
    d2 = ((query_local.unsqueeze(3) - own_local.unsqueeze(2)) ** 2).sum(dim=-1)  # (B, L, Q, K)
    kernel = splat_gaussian_kernel(d2, sigma, per_atom_sigma)  # (B, L, Q, K)
    w = own_mask.to(kernel.dtype).unsqueeze(2)  # (B, L, 1, K)
    return (kernel * w).sum(dim=-1)  # (B, L, Q)


def _fibonacci_cone(n: int, half_angle_deg: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """``n`` roughly-uniform-in-solid-angle unit vectors inside a cone of ``half_angle_deg`` about +z.

    Deterministic Fibonacci-in-cone (golden-angle azimuth; ``cos θ`` swept uniformly over
    ``[cos(half_angle), 1]`` so directions tile the cone's solid angle evenly). No RNG. (n, 3).
    """
    if n <= 0:
        return torch.zeros(0, 3, device=device, dtype=dtype)
    idx = torch.arange(n, device=device, dtype=dtype)
    cos_max = math.cos(math.radians(half_angle_deg))
    cos_theta = 1.0 - (idx + 0.5) / n * (1.0 - cos_max)  # in (cos_max, 1)
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))
    phi = idx * (math.pi * (3.0 - math.sqrt(5.0)))  # golden angle
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    return torch.stack([x, y, cos_theta], dim=-1)  # (n, 3), all within half_angle of +z


def available_volume_cones(
    backbone_coords: torch.Tensor,
    backbone_mask: torch.Tensor,
    target_coords: torch.Tensor | None,
    target_mask: torch.Tensor | None,
    R: torch.Tensor,
    ca: torch.Tensor,
    *,
    half_angle_deg: float = 45.0,
    n_dirs: int = 16,
    max_radius: float = 8.0,
    n_bins: int = AVAILABLE_VOLUME_N_BINS,
    atom_radius: float = 1.7,
) -> torch.Tensor:
    """GT-FREE "sandclock" available-volume descriptor on the two e3 (out-of-plane / L-D) faces.

    For each binder residue this measures how much clear room a side chain has on EACH of the two
    lobes of a 2-cone "hourglass" aligned with the local frame's out-of-plane axis ``e3 = R[..., :, 2]``
    (origin = ``ca``). One lobe points along ``+e3``, the other along ``-e3``. The wall of TARGET atoms
    typically occludes one lobe while backbone/solvent leaves the other open -- an asymmetry the
    volumetric head can weigh to decide "which face has room".

    **GT-free by construction.** Occluders are the binder BACKBONE atoms (N/CA/C/O of every residue) +
    TARGET heavy atoms ONLY. The function NEVER reads the residue's own or any neighbour's side-chain
    coordinates (unknown at inference), so it is valid at sampling time.

    Method (fully vectorised, deterministic, no RNG)
    ------------------------------------------------
    * ``n_dirs`` Fibonacci-in-cone directions fill each lobe's cone (half-angle ``half_angle_deg``).
    * Along each direction ray from the origin, a simple **ray-vs-sphere first hit** finds the nearest
      occluder: for occluder ``p`` with along-ray projection ``t = (p-o)·d`` (in front, ``t>0``) and
      perpendicular distance ``< atom_radius``, the entry distance is ``t - sqrt(atom_radius² - perp²)``;
      the ray's first hit is the min such positive entry over occluders, or "open" if none within
      ``max_radius``. (An occluder closer than ``atom_radius`` to the origin gives a negative entry and
      is skipped -- this naturally drops the residue's own in-cone backbone atoms / the CA at the origin.)

    Descriptor (fixed width, documented)
    ------------------------------------
    Per lobe: an ``n_bins`` radial histogram of first-hit distances over ``[0, max_radius)`` whose values
    are the **fraction of the lobe's ``n_dirs`` rays** hitting in each bin (a ray that finds no occluder
    contributes to NO bin). So a fully-open lobe is an all-zero block and a walled lobe carries mass in
    the near bins; the open fraction of a lobe is recoverable as ``1 - block.sum()``. Output ``(B, L, D)``
    with ``D = 2 * n_bins`` -- the first ``n_bins`` channels are the ``+e3`` lobe, the next ``n_bins`` the
    ``-e3`` lobe. Rows for residues with an invalid backbone frame (N/CA/C missing) are zeroed.

    Parameters
    ----------
    backbone_coords : (B, L, 4, 3) -- binder backbone N/CA/C/O.
    backbone_mask : (B, L, 4) -- backbone atom validity.
    target_coords : (B, N_t, 3), optional -- target heavy-atom coords (occluders).
    target_mask : (B, N_t), optional -- target atom validity.
    R : (B, L, 3, 3) -- per-residue local frame (columns e1,e2,e3), from ``build_local_frames``.
    ca : (B, L, 3) -- per-residue frame origin (CA).
    """
    b, length = backbone_coords.shape[:2]
    device = backbone_coords.device
    dtype = backbone_coords.dtype

    # --- occluders: binder BACKBONE atoms + TARGET heavy atoms (NO side chains) ---
    bb_coords = backbone_coords.reshape(b, length * 4, 3)
    bb_mask = backbone_mask.reshape(b, length * 4).to(torch.bool)
    if target_coords is not None and target_mask is not None:
        occ_coords = torch.cat([bb_coords, target_coords.to(dtype)], dim=1)  # (B, N_occ, 3)
        occ_mask = torch.cat([bb_mask, target_mask.to(torch.bool)], dim=1)  # (B, N_occ)
    else:
        occ_coords, occ_mask = bb_coords, bb_mask

    # --- cone directions per lobe, mapped into the global frame (local->global = R @ local) ---
    cone = _fibonacci_cone(n_dirs, half_angle_deg, device=device, dtype=dtype)  # (n_dirs, 3) about +z
    reflect_z = torch.tensor([1.0, 1.0, -1.0], device=device, dtype=dtype)
    cone_two = torch.stack([cone, cone * reflect_z], dim=0)  # (2, n_dirs, 3): [+e3 lobe, -e3 lobe] local
    dirs = torch.einsum("blij,pkj->blpki", R, cone_two)  # (B, L, 2, n_dirs, 3) global unit dirs

    # --- ray-vs-sphere first hit (never materialises the per-direction 3-vector difference) ---
    m = occ_coords[:, None, :, :] - ca[:, :, None, :]  # (B, L, N_occ, 3): occluder - origin
    m2 = (m * m).sum(dim=-1)  # (B, L, N_occ) = |occluder - origin|²
    t_ca = torch.einsum("blnc,blpkc->blpkn", m, dirs)  # (B, L, 2, n_dirs, N_occ) along-ray projection
    perp2 = m2[:, :, None, None, :] - t_ca * t_ca  # (B, L, 2, n_dirs, N_occ) perpendicular dist²
    disc = atom_radius * atom_radius - perp2  # >0 => ray line passes within atom_radius of the sphere
    s_hit = t_ca - torch.sqrt(torch.clamp(disc, min=0.0))  # sphere entry distance along the ray
    valid_hit = (
        (disc > 0.0)  # ray line intersects the atom sphere
        & (t_ca > 0.0)  # occluder in front of the origin
        & (s_hit > 0.0)  # entry ahead of the origin (drops occluders enclosing the origin, e.g. own CA)
        & (s_hit < max_radius)  # within the sampling window
        & occ_mask[:, None, None, None, :]  # real (non-padded) occluder
    )  # (B, L, 2, n_dirs, N_occ)
    s_masked = torch.where(valid_hit, s_hit, torch.full_like(s_hit, float(max_radius)))
    first_hit = s_masked.amin(dim=-1)  # (B, L, 2, n_dirs); == max_radius when the ray is fully open
    hit = first_hit < max_radius  # rays that actually struck an occluder

    # --- per-lobe radial first-hit histogram: fraction of the lobe's rays hitting in each bin ---
    bin_w = max_radius / n_bins
    bin_idx = torch.clamp((first_hit / bin_w).floor().long(), 0, n_bins - 1)  # (B, L, 2, n_dirs)
    onehot = F.one_hot(bin_idx, num_classes=n_bins).to(dtype) * hit.unsqueeze(-1).to(dtype)
    hist = onehot.sum(dim=3) / float(n_dirs)  # (B, L, 2, n_bins) -- fraction of ALL n_dirs rays per bin
    desc = hist.reshape(b, length, 2 * n_bins)  # (B, L, 2*n_bins): [+e3 block | -e3 block]

    # Zero the descriptor for residues whose backbone frame is undefined (N/CA/C missing).
    res_valid = backbone_mask[:, :, :3].all(dim=-1)  # (B, L)
    return desc * res_valid.unsqueeze(-1).to(dtype)


def volumetric_occupancy_loss(
    density_pred: torch.Tensor,
    density_gt: torch.Tensor,
    seq_mask: torch.Tensor | None,
    empty_weight: float = 1.75,
    empty_threshold: float = 0.05,
) -> torch.Tensor:
    """Empty-weighted MSE between predicted and GT self-occupancy fields.

    Mirrors the reference's per-bucket weighting: query points whose GT density is ~0 (the anti-leak
    "empty" bucket) are up-weighted by ``empty_weight``. Masked to valid residues by ``seq_mask``.

    Parameters
    ----------
    density_pred, density_gt : torch.Tensor
        (B, L, Q) predicted / GT occupancy at the query points.
    seq_mask : torch.Tensor, optional
        (B, L) valid-residue mask. None => all valid.
    """
    b, length, _q = density_pred.shape
    device = density_pred.device
    sq = (density_pred - density_gt) ** 2  # (B, L, Q)
    qweight = torch.where(
        density_gt < empty_threshold,
        torch.full_like(density_gt, float(empty_weight)),
        torch.ones_like(density_gt),
    )  # (B, L, Q)
    if seq_mask is not None:
        res_mask = seq_mask.to(sq.dtype)[:, :, None]  # (B, L, 1)
    else:
        res_mask = torch.ones(b, length, 1, device=device, dtype=sq.dtype)
    w = qweight * res_mask
    return (sq * w).sum() / w.sum().clamp(min=1.0)


def volumetric_self_consistency_loss(
    flow_density: torch.Tensor,
    head_density: torch.Tensor,
    seq_mask: torch.Tensor | None,
) -> torch.Tensor:
    """self-consistency: MSE(flow-x0 density, volumetric-head density), masked to residues.

    The FLOW's predicted-x0 side-chain cloud (soft-P(real)-weighted; see
    :meth:`VolumetricOccupancyHead.splat_flow_density`) and the volumetric HEAD's predicted density
    live on the SAME query lattice + sigma, so their densities are directly comparable. The caller
    passes ``head_density`` ALREADY DETACHED, so the flow chases the head one-way (the head stays
    anchored by its own Chunk-1 GT loss; bidirectional feedback is a deliberate later step).

    Complementary to ``occupancy_match`` in models.py: that term is the low-t TEACHER-FORCING guard
    (flow-x0 vs GT density, GT-existence-masked, tau-gated). This term is the FREE-SAMPLING anchor --
    it uses the model's OWN predicted P(real) and the head's OWN prediction (both exist at inference,
    GT does not), and is ramped in by epoch rather than tau-gated.

    Parameters
    ----------
    flow_density, head_density : torch.Tensor
        (B, L, Q) flow-x0 / head predicted occupancy at the query points. ``head_density`` must be
        detached by the caller.
    seq_mask : torch.Tensor, optional
        (B, L) valid-residue mask. None => all valid.
    """
    sq = (flow_density - head_density) ** 2  # (B, L, Q)
    per_res = sq.mean(dim=-1)  # (B, L)
    if seq_mask is not None:
        m = seq_mask.to(per_res.dtype)
    else:
        m = torch.ones_like(per_res)
    return (per_res * m).sum() / m.sum().clamp(min=1.0)


def volumetric_scale_anchor_loss(
    density_pred: torch.Tensor,
    density_gt: torch.Tensor,
    seq_mask: torch.Tensor | None,
    region_labels: torch.Tensor | None = None,
    num_regions: int = NUM_VOLUMETRIC_REGIONS,
) -> torch.Tensor:
    """Mass-match anchor pinning the predicted field's magnitude to the GT splat's, per residue.

    The z-scored decoy-CE is scale-INVARIANT, so it leaves the predicted density's absolute magnitude
    unconstrained and the mass drifts (neg_mse recovery collapses while cosine rises). This term pins that
    magnitude so the decoy-CE can sharpen SHAPE while scale stays calibrated. Uses the SAME ``density_gt`` the
    occupancy loss consumes (the head's GT splat); gradient flows into ``density_pred``. Returned UNWEIGHTED --
    the caller scales by ``volumetric_scale_anchor_weight`` and only adds it when that weight is > 0 (so it is
    byte-identical off).

    Per-REGION vs per-residue-TOTAL
    -------------------------------
    When ``region_labels`` is given, the match is per self-occupancy REGION bucket (center / around / context /
    near / medium / far-empty): ``sum_regions (sum_{Q in region} pred - sum_{Q in region} gt)²``. This pins
    the mass DISTRIBUTION across regions, so the head can no longer redistribute density center<->empty at a
    FIXED per-residue total and evade the anchor. When ``region_labels`` is ``None`` it falls back to the
    per-residue TOTAL mass ``(sum_Q pred - sum_Q gt)²`` -- a documented weaker limitation (the total-mass
    loophole), used only where the labels are not readily available at the call site.

    Parameters
    ----------
    density_pred, density_gt : torch.Tensor
        (B, L, Q) predicted / GT occupancy at the query points.
    seq_mask : torch.Tensor, optional
        (B, L) valid-residue mask. None => all valid.
    region_labels : torch.Tensor, optional
        (B, L, Q) self-occupancy region id in ``[0, num_regions)`` per query point (from
        :func:`~atomweaver.joint_diffusion.volumetric_supervision.classify_query_regions`). None => total-mass fallback.
    num_regions : int
        Number of region buckets (defaults to the self-occupancy region count).
    """
    if region_labels is not None:
        # PER-REGION mass: segment-sum pred/gt over the Q axis within each region bucket, then sum the squared
        # per-region mass diffs. scatter_add over the region id is exact and cheap (num_regions is small).
        b, length, _q = density_pred.shape
        idx = region_labels.long()  # (B, L, Q)
        pred_mass = density_pred.new_zeros(b, length, num_regions).scatter_add_(2, idx, density_pred)  # (B, L, R)
        gt_mass = density_gt.new_zeros(b, length, num_regions).scatter_add_(2, idx, density_gt)  # (B, L, R)
        sq = ((pred_mass - gt_mass) ** 2).sum(dim=-1)  # (B, L) -- summed over region buckets
    else:
        # FALLBACK: per-residue TOTAL mass (documented limitation -- evadable by within-residue redistribution).
        pred_mass = density_pred.sum(dim=-1)  # (B, L)
        gt_mass = density_gt.sum(dim=-1)  # (B, L)
        sq = (pred_mass - gt_mass) ** 2  # (B, L)
    if seq_mask is not None:
        m = seq_mask.to(sq.dtype)
    else:
        m = torch.ones_like(sq)
    return (sq * m).sum() / m.sum().clamp(min=1.0)


class VolumetricOccupancyHead(nn.Module):
    """Per-residue neural self-occupancy field over a target-aware local-frame context.

    t-INDEPENDENT. Byte-identical / not built unless ``use_volumetric_head``. predicts
    the field and (optionally) computes the GT splat; NO injection into the flow yet. A later
    chunk deep-injects the exposed ``vol_hidden``.

    Forward returns ``vol_hidden`` (B, L, hidden) [exposed for the future injection chunk],
    ``vol_density_pred`` (B, L, Q), and -- when GT side chains are supplied -- ``vol_density_gt``
    (B, L, Q). The loss is assembled by :func:`volumetric_occupancy_loss`.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_element_types: int = NUM_ELEMENT_TYPES,
        context_radius: float = 10.0,
        n_query: int = 128,
        sigma: float = 1.0,
        empty_weight: float = 1.75,
        atom_feat_dim: int = 64,
        use_available_volume: bool = False,
        available_volume_dim: int = AVAILABLE_VOLUME_DIM,
        per_element_sigma: bool = False,
        sigma_element_scale: float | None = None,
        use_single_site_context: bool = False,
        fourier_frequencies: int = DEFAULT_FOURIER_FREQUENCIES,
        fourier_scale: float = DEFAULT_FOURIER_SCALE,
        dropout: float = 0.0,  # module default 0.0 = byte-identical; CLI drives PRODUCTION_VOLUMETRIC_DROPOUT
        use_softplus: bool = True,
        use_per_query_context: bool = False,
        context_k: int = DEFAULT_CONTEXT_K,
        context_heads: int = DEFAULT_CONTEXT_HEADS,
        context_chunk: int = DEFAULT_CONTEXT_CHUNK,
        atom_anchored_queries: bool = False,
    ):
        super().__init__()
        if context_radius <= 0:
            raise ValueError(f"context_radius must be > 0 (got {context_radius})")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0 (got {sigma})")
        if fourier_frequencies < 0:
            raise ValueError(f"fourier_frequencies must be >= 0 (0 disables the lift; got {fourier_frequencies})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1) (got {dropout})")
        self.hidden_dim = hidden_dim
        self.fourier_frequencies = int(fourier_frequencies)
        self.fourier_scale = float(fourier_scale)
        self.dropout = float(dropout)
        self.use_softplus = bool(use_softplus)
        # PER-QUERY kNN CONTEXT (self-occupancy ContextEncoder). Off (default) => the density trunk reads the shared
        # attention-pooled ``vol_hidden`` broadcast to every query point (byte-identical; no new params/module).
        # On => EACH query point gets its own kNN neighborhood over the SINGLE-SITE context atoms (backbone +
        # target + OTHER residues' side chains, self-excluded), giving the per-point discrimination the pooled
        # broadcast cannot. Built only when on (see below), so off carries no extra parameters.
        self.use_per_query_context = bool(use_per_query_context)
        self.context_k = int(context_k)
        self.context_heads = int(context_heads)
        # Residue-axis chunk for the per-query ContextEncoder (OOM guard; numerically exact). Only read on the
        # per-query path. <= 0 => a single unchunked pass (== chunk >= G). One-time fallback-warning latch below.
        self.context_chunk = int(context_chunk)
        self._warned_per_query_fallback = False
        self.num_element_types = int(num_element_types)
        self.context_radius = float(context_radius)
        self.n_query = int(n_query)
        self.sigma = float(sigma)
        self.empty_weight = float(empty_weight)
        # VOLUMETRIC-FAITHFUL ATOM-ANCHORED QUERIES. Off (default) => the fixed Fibonacci lattice (byte-identical).
        # On => the caller samples per-residue atom-anchored queries (:func:`sample_atom_anchored_queries`) and
        # passes them to ``forward(..., query_local_override=...)``; the head evaluates its density field at
        # those queries instead of the fixed lattice. This flag builds NO parameters (the sampler is a stateless
        # function and the override is threaded per-call), so the module state_dict is UNCHANGED whether it is on
        # or off -- resume-safe, and a checkpoint transfers between the two modes. It is stored only for
        # provenance / an at-a-glance config check; behaviour is driven purely by ``query_local_override``.
        self.atom_anchored_queries = bool(atom_anchored_queries)

        # SINGLE-SITE POCKET CONTEXT. Off (default) => byte-identical to the pre-single-site head (no field is
        # built, no projection, no conditioning). On => the head is conditioned on a per-query NEIGHBOR-OCCUPANCY
        # FIELD: for residue i, the OTHER residues' side-chain atoms (+ binder backbone + target atoms) are
        # Gaussian-splatted onto i's OWN query lattice (:meth:`_neighbor_occupancy_field`), with residue i's own
        # side chain SELF-EXCLUDED (a site never occludes itself). That (B, L, Q) "which query points are blocked"
        # signal is fed into the density MLP through a ZERO-INIT projection (``field_proj`` below), so the head
        # predicts site i's own occupancy KNOWING which points the neighbour pocket blocks. The raw neighbour
        # atoms NEVER enter the head's attention/feature set (``_gather_context`` stays backbone + target); only
        # the aggregated field conditions the head. Zero-init => ON-at-init is byte-identical too (resume-safe).
        self.use_single_site_context = bool(use_single_site_context)

        # PER-ELEMENT vdW-derived splat sigma. Off (default) => scalar ``self.sigma`` everywhere =>
        # BYTE-IDENTICAL to the uniform-sigma splat. On => the GT-target / self-occupancy-target / self-consistency
        # splats use a per-atom Bondi-radius-derived sigma (small O vs bulky S/X); the head LEARNS to match.
        # ``sigma_element_scale`` stored so the eval gate rebuilds the exact same table from the checkpoint.
        self.per_element_sigma = bool(per_element_sigma)
        self.sigma_element_scale = (
            float(sigma_element_scale) if sigma_element_scale is not None else default_sigma_element_scale()
        )
        if self.per_element_sigma:
            self.register_buffer(
                "_element_sigma_table",
                build_element_sigma_table(self.num_element_types, self.sigma_element_scale),
                persistent=False,
            )

        # Fixed local-frame query lattice (buffer => device-follows, saved for reproducibility).
        self.register_buffer("query_local", build_query_points(self.n_query), persistent=False)
        # Per-backbone-atom element ids (N/CA/C/O), buffer so it follows device.
        self.register_buffer(
            "_backbone_element_ids",
            torch.tensor(_BACKBONE_ELEMENTS, dtype=torch.long),
            persistent=False,
        )

        # Per-atom (i-independent) categorical features.
        self.element_embed = nn.Embedding(self.num_element_types, atom_feat_dim)
        self.is_target_embed = nn.Embedding(2, atom_feat_dim)
        self.is_backbone_embed = nn.Embedding(2, atom_feat_dim)

        # Per-(residue, context-atom) token: [rel_local(3), dist(1)] ++ summed atom embedding.
        self.atom_mlp = nn.Sequential(
            nn.Linear(4 + atom_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Attention-pool context atoms -> per-residue latent.
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # FOURIER QUERY LIFT (the key expressivity lever; see FourierFeatureEncoder). fourier_frequencies==0
        # disables it (legacy raw-coord path, 3-d). >0 => the normalized query coord is lifted to
        # ``2*fourier_frequencies`` random Fourier features before the density trunk.
        if self.fourier_frequencies > 0:
            self.query_encoder = FourierFeatureEncoder(
                input_dim=3, num_frequencies=self.fourier_frequencies, scale=self.fourier_scale
            )
            query_feat_dim = self.query_encoder.output_dim
        else:
            self.query_encoder = None
            query_feat_dim = 3

        # PER-QUERY kNN CONTEXT ENCODER (self-occupancy ContextEncoder). Only built when ``use_per_query_context`` is on
        # (off => no module, no params => byte-identical). Its query dim == the query lift width (Fourier or raw
        # coord), so the SAME density trunk consumes either the per-query context vector or the pooled broadcast
        # (both are ``hidden_dim``); the trunk's input width is UNCHANGED. NB: the neighborhood attention needs
        # ``hidden_dim % context_heads == 0`` (validated in NeighborhoodAttention).
        if self.use_per_query_context:
            self.context_encoder = ContextEncoder(
                query_dim=query_feat_dim,
                hidden_dim=hidden_dim,
                k_neighbors=self.context_k,
                num_heads=self.context_heads,
                dropout=self.dropout,
            )

        # Conditional density field: (latent, fourier(query_local/radius)) -> scalar occupancy. Softplus at the
        # end (:attr:`use_softplus`) makes the predicted field NON-NEGATIVE, aligning it with the ≥0 unit-height
        # Gaussian GT/reference splats (a raw signed output lives on a different sign/scale => a uniform neg_mse
        # offset kills the decoy-CE). Dropout mirrors the reference's trunk (0.10).
        self.density_mlp = nn.Sequential(
            nn.Linear(hidden_dim + query_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, 1),
        )
        if self.use_softplus:
            # Start the field near 0 in empty space (softplus(-2)≈0.13) while keeping a live gradient.
            nn.init.constant_(self.density_mlp[-1].bias, _DENSITY_SOFTPLUS_BIAS_INIT)

        # SANDCLOCK available-volume input (GT-free 2-cone descriptor; see ``available_volume_cones``).
        # Projected into the per-residue latent `vol_hidden` and ADDED there so downstream consumers
        # (deep-inject / existence coupling / stereo head) can weigh "which e3 face has room". ZERO-INIT
        # => a no-op at graft/init: the model is byte-identical when the flag is off (proj not built) AND
        # at the first step when on (proj outputs 0), so it is resume-safe. Only built when the flag is on.
        # NB: added AFTER the density field is read (see forward), so the standalone volumetric pretrain --
        # which trains ONLY the density MLP -- never sees it and stays inert; it earns gradient only through
        # the full-arch consumers of `vol_hidden`.
        self.use_available_volume = bool(use_available_volume)
        self.available_volume_dim = int(available_volume_dim)
        if self.use_available_volume:
            self.available_volume_proj = nn.Linear(self.available_volume_dim, hidden_dim)
            nn.init.zeros_(self.available_volume_proj.weight)
            nn.init.zeros_(self.available_volume_proj.bias)

        # SINGLE-SITE neighbour-occupancy FIELD conditioning. Projects the per-query scalar field (which of
        # site i's query points are blocked by neighbours) into the density MLP's per-query latent, ADDED there.
        # ZERO-INIT => adds exactly 0 at graft/init: byte-identical when the flag is off (proj not built) AND at
        # the first step when on (proj outputs 0), so it is resume-safe. Only built when the flag is on. A
        # pretrained (no-single-site) head does not carry it -> it is an OPTIONAL graft key on warm-start (see
        # models._OPTIONAL_VOLUMETRIC_HEAD_KEY_PREFIXES) and stays trainable under freeze.
        if self.use_single_site_context:
            self.field_proj = nn.Linear(1, hidden_dim)
            nn.init.zeros_(self.field_proj.weight)
            nn.init.zeros_(self.field_proj.bias)
            # OPTION (ii): route the SAME per-query neighbour-occupancy field into the RETURNED per-residue
            # `vol_hidden` (so it reaches the deep-inject / stereo / existence consumers => GENERATION), IN
            # ADDITION to the per-query field_proj -> density-MLP path above. Learned pool over the Q-point
            # lattice (Linear(n_query, hidden)), mirroring available_volume_proj. ZERO-INIT => +0 at init:
            # byte-identical when off (proj not built) AND at the first step when on (proj outputs 0),
            # resume-safe. OPTIONAL graft key on warm-start (see models._OPTIONAL_VOLUMETRIC_HEAD_KEY_PREFIXES).
            self.field_to_hidden_proj = nn.Linear(self.n_query, hidden_dim)
            nn.init.zeros_(self.field_to_hidden_proj.weight)
            nn.init.zeros_(self.field_to_hidden_proj.bias)

    def _assemble_context_atoms(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        target_coords: torch.Tensor | None,
        target_mask: torch.Tensor | None,
        target_element: torch.Tensor | None,
        sidechain_coords: torch.Tensor,
        sidechain_mask: torch.Tensor,
        sidechain_element: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The SINGLE-SITE context/occluder atom set: binder backbone + target + binder side chains.

        Returns (coords, weight, element, res, anchor_owner), each ``(B, M, ...)`` in this fixed atom order:
        ``[backbone (L*4) | target (N_t) | side chains (L*K)]``. Shared by :meth:`_neighbor_occupancy_field`
        (Gaussian-splat occluders) and :meth:`_per_query_context` (per-query kNN), so both consume the EXACT
        same atoms -- the per-query path never rebuilds its own context.

        * ``res`` : per-atom owner-residue index for SELF-EXCLUSION -- side-chain slots carry their owning
          residue (residue ``r`` owns ``[r*K, (r+1)*K)``); shared backbone + target carry ``-1`` (kept for
          every query residue). A residue never occludes / neighbors its OWN side chain.
        * ``anchor_owner`` : per-atom owner-residue index used to flag the INITIATOR bit -- a backbone
          ``N``/``CA``/``C`` atom carries its owning residue; backbone ``O``, target, and side-chain atoms
          carry ``-1``. For query residue ``i``, ``anchor_owner == i`` marks ``i``'s own backbone anchor.
        """
        b, length = backbone_coords.shape[:2]
        device = backbone_coords.device

        bb_coords = backbone_coords.reshape(b, length * 4, 3)
        bb_w = backbone_mask.reshape(b, length * 4).to(dtype)
        bb_element = self._backbone_element_ids.repeat(length).unsqueeze(0).expand(b, -1)  # (B, L*4)
        bb_res = torch.full((b, length * 4), -1, dtype=torch.long, device=device)
        # Initiator anchor owner: residue r's N/CA/C backbone atoms flag its own anchor; O/other => -1.
        bb_owner = torch.arange(length, device=device).repeat_interleave(4)  # (L*4,) : [0,0,0,0,1,1,1,1,...]
        anchor_is = torch.tensor([True, True, True, False], device=device).repeat(length)  # N,CA,C anchors (not O)
        bb_anchor = torch.where(anchor_is, bb_owner, torch.full_like(bb_owner, -1)).unsqueeze(0).expand(b, -1)

        occ_coords = [bb_coords]
        occ_w = [bb_w]
        occ_element = [bb_element]
        occ_res = [bb_res]
        occ_anchor = [bb_anchor]

        if target_coords is not None and target_mask is not None:
            n_t = target_coords.shape[1]
            occ_coords.append(target_coords.to(dtype))
            occ_w.append(target_mask.to(dtype))
            if target_element is not None:
                occ_element.append(target_element.long().clamp(0, self.num_element_types - 1))
            else:
                occ_element.append(torch.zeros(b, n_t, dtype=torch.long, device=device))
            occ_res.append(torch.full((b, n_t), -1, dtype=torch.long, device=device))
            occ_anchor.append(torch.full((b, n_t), -1, dtype=torch.long, device=device))

        k = sidechain_coords.shape[2]
        occ_coords.append(sidechain_coords.reshape(b, length * k, 3).to(dtype))
        occ_w.append(sidechain_mask.reshape(b, length * k).to(dtype))
        if sidechain_element is not None:
            occ_element.append(sidechain_element.long().clamp(0, self.num_element_types - 1).reshape(b, length * k))
        else:
            occ_element.append(torch.zeros(b, length * k, dtype=torch.long, device=device))
        # Per-atom binder residue index: residue r owns slots [r*K, (r+1)*K) => self-exclusion tag.
        occ_res.append(
            torch.arange(length, device=device).view(1, length, 1).expand(b, length, k).reshape(b, length * k)
        )
        occ_anchor.append(torch.full((b, length * k), -1, dtype=torch.long, device=device))

        return (
            torch.cat(occ_coords, dim=1),  # (B, M, 3)
            torch.cat(occ_w, dim=1),  # (B, M)
            torch.cat(occ_element, dim=1),  # (B, M)
            torch.cat(occ_res, dim=1),  # (B, M)
            torch.cat(occ_anchor, dim=1),  # (B, M)
        )

    def _per_query_context(
        self,
        R: torch.Tensor,
        ca: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        target_coords: torch.Tensor | None,
        target_mask: torch.Tensor | None,
        target_element: torch.Tensor | None,
        sidechain_coords: torch.Tensor,
        sidechain_mask: torch.Tensor,
        sidechain_element: torch.Tensor | None,
        query_feat: torch.Tensor,
        query_local_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-query kNN local-context vector (self-occupancy ContextEncoder). Returns ``(B, L, Q, hidden)``.

        For each binder residue ``i`` and each of its query points, does kNN over the SINGLE-SITE context
        atoms (:meth:`_assemble_context_atoms`: backbone + target + OTHER residues' side chains, ``i``'s own
        side chain self-excluded) and encodes the neighborhood. Positions are expressed in ``i``'s LOCAL
        frame (``R_iᵀ (x - CA_i)``), keeping the head SE(3)-invariant; ``query_feat`` is the (already-lifted)
        per-query embedding the encoder attends WITH. Replaces the shared pooled-``vol_hidden`` broadcast as
        the density trunk's per-query latent when ``use_per_query_context`` is on.

        SOURCE SEAM: mirrors :meth:`_neighbor_occupancy_field` -- the pretrain path passes GT side chains as
        ``sidechain_*``; an FT/inference caller swaps to the flow's predicted x0 without touching this method.
        """
        b, length = backbone_coords.shape[:2]
        device = backbone_coords.device
        dtype = ca.dtype
        q = query_local_override.shape[2] if query_local_override is not None else self.query_local.shape[0]

        coords_m, weight_m, _element_m, res_m, anchor_owner = self._assemble_context_atoms(
            backbone_coords,
            backbone_mask,
            target_coords,
            target_mask,
            target_element,
            sidechain_coords,
            sidechain_mask,
            sidechain_element,
            dtype,
        )  # each (B, M, ...)
        m = coords_m.shape[1]

        # Context atoms in each residue's LOCAL frame (invariant): R_iᵀ (x_a - CA_i). (B, L, M, 3)
        rel = coords_m[:, None, :, :] - ca[:, :, None, :]
        ctx_local = torch.einsum("blij,blmj->blmi", R.transpose(-1, -2), rel)
        # Query points in the SAME local frame: the per-residue sampled queries (atom-anchored override) or the
        # fixed lattice broadcast to every residue. (B, L, Q, 3). Match ``ctx_local``'s dtype so the shared
        # ``cdist`` in ``_knn`` never mixes dtypes under autocast (einsum lowers to fp16/bf16). No-op in fp32.
        if query_local_override is not None:
            query_local = query_local_override.to(ctx_local.dtype)  # (B, L, Q, 3)
        else:
            query_local = self.query_local.to(ctx_local.dtype).view(1, 1, q, 3).expand(b, length, q, 3)
        # SCALE FIX: the ContextEncoder computes its neighbor geometry (distance, relative_xyz) from THESE
        # positions, so divide BOTH the query points and the context atoms by ``context_radius`` to put that
        # geometry at the SAME nm scale as the Fourier query (which is fed ``query / context_radius``). Without
        # this the neighbor features sit at raw-Å scale while the query embedding is nm-scale -- a silent scale
        # mismatch. Correctness fix, applied on every per-query-context call (fixed lattice or override).
        inv_radius = 1.0 / self.context_radius
        query_local = query_local * inv_radius
        ctx_local = ctx_local * inv_radius

        res_idx = torch.arange(length, device=device).view(1, length, 1)  # (1, L, 1)
        # Per-residue context mask: real atom AND not this residue's own side chain (backbone/target res==-1 kept).
        ctx_mask = (weight_m[:, None, :] > 0.5) & (res_m[:, None, :] != res_idx)  # (B, L, M) bool
        # Initiator bit: this residue's OWN backbone anchor atoms (N/CA/C).
        initiator = (anchor_owner[:, None, :] == res_idx).to(dtype)  # (B, L, M)

        # Flatten (B, L) -> G for the faithful ContextEncoder (query batch = residues).
        g = b * length
        query_local_g = query_local.reshape(g, q, 3)
        query_feat_g = query_feat.reshape(g, q, query_feat.shape[-1])
        ctx_local_g = ctx_local.reshape(g, m, 3)
        initiator_g = initiator.reshape(g, m)
        ctx_mask_g = ctx_mask.reshape(g, m)

        # RESIDUE-AXIS CHUNKING (OOM guard). The encoder materializes a (chunk, Q, k, hidden) neighbor tensor;
        # at chunk=G=B*L this is ~tens of GB (B=8, L=30 fp32). Each residue is INDEPENDENT (no cross-residue
        # interaction), so looping the encoder over G-chunks and concatenating the (G, Q, hidden) outputs is
        # NUMERICALLY EXACT -- peak memory is set by ``context_chunk``, not B*L. chunk >= G => the exact single
        # pass (no ``cat``), so the unchunked result is recovered byte-for-byte.
        chunk = self.context_chunk if self.context_chunk and self.context_chunk > 0 else g
        if chunk >= g:
            ctx = self.context_encoder(
                query_local_g, query_feat_g, ctx_local_g, initiator_g, context_mask=ctx_mask_g
            )  # (G, Q, hidden)
        else:
            # GRADIENT CHECKPOINTING (load-bearing). Chunking the FORWARD alone does NOT bound BACKWARD memory:
            # autograd saves every chunk's (chunk, Q, k, hidden) activations, so the backward peak is still the
            # full B*L tensor (this OOM'd rank0 at B=8 fp32). Checkpointing recomputes each chunk in backward
            # instead of storing it, collapsing peak memory to ~one chunk. preserve_rng_state (default) keeps the
            # recompute's dropout mask consistent. Only meaningful (and only applied) when grad is enabled.
            def _enc(ql, qf, cl, ini, cm):
                return self.context_encoder(ql, qf, cl, ini, context_mask=cm)

            use_ckpt = torch.is_grad_enabled()
            pieces = []
            for s in range(0, g, chunk):
                args = (
                    query_local_g[s : s + chunk],
                    query_feat_g[s : s + chunk],
                    ctx_local_g[s : s + chunk],
                    initiator_g[s : s + chunk],
                    ctx_mask_g[s : s + chunk],
                )
                if use_ckpt:
                    pieces.append(torch.utils.checkpoint.checkpoint(_enc, *args, use_reentrant=False))
                else:
                    pieces.append(_enc(*args))
            ctx = torch.cat(pieces, dim=0)  # (G, Q, hidden)
        return ctx.reshape(b, length, q, self.hidden_dim)

    def _neighbor_occupancy_field(
        self,
        R: torch.Tensor,
        ca: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        target_coords: torch.Tensor | None,
        target_mask: torch.Tensor | None,
        target_element: torch.Tensor | None,
        sidechain_coords: torch.Tensor,
        sidechain_mask: torch.Tensor,
        sidechain_element: torch.Tensor | None,
        query_local_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-residue NEIGHBOR-OCCUPANCY field on the head's query lattice. Returns ``(B, L, Q)``.

        For each binder residue ``i`` the occluder atoms -- every OTHER residue's side-chain atoms (residue
        ``i``'s own side chain SELF-EXCLUDED, since a site never occludes itself) PLUS the shared binder
        BACKBONE + TARGET atoms -- are Gaussian-splatted onto ``i``'s OWN query points (``query_local`` mapped
        into ``i``'s local frame). The result marks which of ``i``'s query points are blocked/occupied by its
        neighbour pocket. Reuses the head's splat kernel + per-element sigma (:func:`splat_gaussian_kernel` /
        :meth:`_per_atom_sigma`), so the field is consistent with the GT/self-occupancy splats when ``per_element_sigma``
        is on.

        SOURCE SEAM: the pretrain path passes the GT side chains as ``sidechain_*`` (the head already receives
        them for its own-only target). An FT/inference caller later swaps this source to the flow's predicted x0
        (scheduled sampling) WITHOUT touching this method -- only the tensors passed in change.
        """
        b, length = backbone_coords.shape[:2]
        device = backbone_coords.device
        q = query_local_override.shape[2] if query_local_override is not None else self.query_local.shape[0]
        dtype = ca.dtype

        # --- assemble context/occluder atoms: binder backbone + target + binder side chains (shared helper) ---
        coords_m, weight_m, element_m, res_m, _anchor_owner = self._assemble_context_atoms(
            backbone_coords,
            backbone_mask,
            target_coords,
            target_mask,
            target_element,
            sidechain_coords,
            sidechain_mask,
            sidechain_element,
            dtype,
        )  # each (B, M, ...)
        m = coords_m.shape[1]

        # --- query points in the GLOBAL frame per residue: CA_i + R_i @ q_local ---
        # ATOM-ANCHORED: per-residue local queries (B, L, Q, 3) map with a per-residue einsum; the fixed lattice
        # (Q, 3) maps residue-independently. Both land in the global frame for the occluder splat.
        if query_local_override is not None:
            query_global = ca.unsqueeze(2) + torch.einsum("blij,blqj->blqi", R, query_local_override.to(dtype))
        else:
            query_local = self.query_local.to(dtype)  # (Q, 3)
            query_global = ca.unsqueeze(2) + torch.einsum("blij,qj->blqi", R, query_local)  # (B, L, Q, 3)

        # --- Gaussian splat of occluders at the query points (reuse the shared kernel + per-element sigma) ---
        diff = query_global.unsqueeze(3) - coords_m[:, None, None, :, :]  # (B, L, Q, M, 3)
        d2 = (diff * diff).sum(dim=-1)  # (B, L, Q, M)
        per_atom_sigma = self._per_atom_sigma(element_m)  # (B, M) or None
        if per_atom_sigma is not None:
            per_atom_sigma = per_atom_sigma.unsqueeze(1).expand(b, length, m)  # (B, L, M) => broadcasts over Q
        kernel = splat_gaussian_kernel(d2, self.sigma, per_atom_sigma)  # (B, L, Q, M)

        # SELF-EXCLUSION: residue i never occludes itself (its own side chain, res == i). Shared backbone +
        # target occluders carry res == -1 and are kept for every residue. combined with the real-atom weight.
        query_idx = torch.arange(length, device=device).view(1, length, 1)  # (1, L, 1)
        keep = (res_m[:, None, :] != query_idx).to(dtype)  # (B, L, M)
        w = weight_m[:, None, :] * keep  # (B, L, M)
        return (kernel * w.unsqueeze(2)).sum(dim=-1)  # (B, L, Q) -- sum composition

    def _gather_context(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        target_coords: torch.Tensor | None,
        target_mask: torch.Tensor | None,
        target_element: torch.Tensor | None,
        target_is_backbone: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the candidate context atom set: binder backbone + target atoms.

        Returns (coords, mask, element, is_target, is_backbone), each (B, N_cand, ...).

        NB: this is the head's FEATURE context (attention-pooled per-residue latent). It deliberately
        carries NO binder side chains -- the single-site pocket signal is injected instead as a separate
        NEIGHBOR-OCCUPANCY FIELD (:meth:`_neighbor_occupancy_field`), not as raw atoms in this set.
        """
        b, length = backbone_coords.shape[:2]
        device = backbone_coords.device

        bb_coords = backbone_coords.reshape(b, length * 4, 3)  # (B, L*4, 3)
        bb_mask = backbone_mask.reshape(b, length * 4).bool()  # (B, L*4)
        bb_element = self._backbone_element_ids.repeat(length).unsqueeze(0).expand(b, -1)  # (B, L*4)
        bb_is_target = torch.zeros(b, length * 4, dtype=torch.long, device=device)
        bb_is_backbone = torch.ones(b, length * 4, dtype=torch.long, device=device)

        if target_coords is None or target_mask is None:
            return bb_coords, bb_mask, bb_element, bb_is_target, bb_is_backbone

        n_t = target_coords.shape[1]
        t_mask = target_mask.bool()
        if target_element is not None:
            t_element = target_element.long().clamp(0, self.num_element_types - 1)
        else:
            t_element = torch.zeros(b, n_t, dtype=torch.long, device=device)
        t_is_target = torch.ones(b, n_t, dtype=torch.long, device=device)
        if target_is_backbone is not None:
            t_is_backbone = target_is_backbone.long().clamp(0, 1)
        else:
            t_is_backbone = torch.zeros(b, n_t, dtype=torch.long, device=device)

        coords = torch.cat([bb_coords, target_coords], dim=1)
        mask = torch.cat([bb_mask, t_mask], dim=1)
        element = torch.cat([bb_element, t_element], dim=1)
        is_target = torch.cat([bb_is_target, t_is_target], dim=1)
        is_backbone = torch.cat([bb_is_backbone, t_is_backbone], dim=1)
        return coords, mask, element, is_target, is_backbone

    def _per_atom_sigma(self, element_ids: torch.Tensor | None) -> torch.Tensor | None:
        """Per-atom Gaussian sigma from per-atom MODEL element ids, or ``None`` (=> scalar-sigma path).

        Returns ``None`` (byte-identical scalar path) unless ``per_element_sigma`` is on AND element ids are
        supplied; otherwise a same-shape tensor of Bondi-radius-derived widths from ``_element_sigma_table``.
        """
        if not self.per_element_sigma or element_ids is None:
            return None
        return per_atom_sigma_from_ids(element_ids, self._element_sigma_table)

    def _splat_density(
        self,
        sidechain_coords: torch.Tensor,
        sidechain_weights: torch.Tensor,
        R: torch.Tensor,
        ca: torch.Tensor,
        element_ids: torch.Tensor | None = None,
        query_local_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gaussian-splat a per-residue side-chain cloud onto the local-frame query lattice.

        Shared by the GT self-occupancy target (``forward``, weights = GT real-atom mask) and the
        self-consistency term (:meth:`splat_flow_density`, weights = soft predicted P(real)).
        Both use the identical ``query_local`` lattice + ``sigma``, so the resulting densities are
        directly comparable. Unit-height Gaussians, sum composition, matching the GT splat exactly.

        Parameters
        ----------
        sidechain_coords : (B, L, K, 3) -- side-chain atom coords (global frame).
        sidechain_weights : (B, L, K) -- per-atom weight (GT 0/1 mask or soft P(real)).
        R, ca : (B, L, 3, 3), (B, L, 3) -- per-residue local frame + CA origin.
        element_ids : (B, L, K), optional -- per-atom MODEL element ids. Only consulted when
            ``per_element_sigma`` is on; ``None`` (or the flag off) => the scalar-sigma path (byte-identical).
        """
        rel = sidechain_coords - ca.unsqueeze(2)  # (B, L, K, 3)
        local = torch.einsum("blij,blkj->blki", R.transpose(-1, -2), rel)  # (B, L, K, 3)
        # ATOM-ANCHORED: per-residue sampled queries (B, L, Q, 3) vs the shared fixed lattice (Q, 3). Either way
        # ``d2`` is (B, L, Q, K) = squared distance from each query point to each own atom (local frame).
        if query_local_override is not None:
            ql = query_local_override.to(sidechain_coords.dtype).unsqueeze(3)  # (B, L, Q, 1, 3)
        else:
            q = self.query_local.shape[0]
            ql = self.query_local.to(sidechain_coords.dtype).view(1, 1, q, 1, 3)
        d2 = ((ql - local.unsqueeze(2)) ** 2).sum(dim=-1)  # (B, L, Q, K)
        kernel = splat_gaussian_kernel(d2, self.sigma, self._per_atom_sigma(element_ids))  # (B, L, Q, K)
        w = sidechain_weights.to(kernel.dtype).unsqueeze(2)  # (B, L, 1, K)
        return (kernel * w).sum(dim=-1)  # (B, L, Q) -- sum composition

    def splat_flow_density(
        self,
        sidechain_coords: torch.Tensor,
        sidechain_weights: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        element_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Splat a FLOW-predicted x0 side-chain cloud onto the head's query lattice.

        Builds the same local frames the head uses (``build_local_frames``) and reuses the head's
        GT splat routine (:meth:`_splat_density`) with ``sidechain_weights`` = the model's OWN soft
        P(real) (NOT the GT mask). The result lives on the identical lattice + sigma as
        ``vol_density_pred``, so it is directly comparable for the self-consistency MSE. Gradient
        flows through ``sidechain_coords`` (the flow's x0) into the denoiser; the frame construction
        touches only fixed backbone data and no head parameters.
        """
        R, ca = build_local_frames(backbone_coords, backbone_mask)  # (B, L, 3, 3), (B, L, 3)
        return self._splat_density(sidechain_coords, sidechain_weights, R, ca, element_ids=element_ids)  # (B, L, Q)

    def forward(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        target_coords: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_element: torch.Tensor | None = None,
        target_is_backbone: torch.Tensor | None = None,
        gt_sidechain_coords: torch.Tensor | None = None,
        gt_sidechain_mask: torch.Tensor | None = None,
        gt_sidechain_element: torch.Tensor | None = None,
        available_volume: torch.Tensor | None = None,
        context_sidechain_coords: torch.Tensor | None = None,
        context_sidechain_mask: torch.Tensor | None = None,
        context_sidechain_element: torch.Tensor | None = None,
        query_local_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        query_local_override : (B, L, Q, 3), optional -- VOLUMETRIC-FAITHFUL ATOM-ANCHORED path. When given, the
            head evaluates its density field (and, when supplied, the GT / neighbour splats) at these
            PER-RESIDUE local-frame query points instead of the shared fixed ``query_local`` lattice. The
            caller samples them via :func:`sample_atom_anchored_queries` (one per GT own-atom + around cluster
            + context negatives + banded empties) so the positive buckets are populated. ``None`` (default) =>
            the fixed lattice (byte-identical). ``Q`` must equal ``self.n_query`` (the layout is fixed).
        context_sidechain_coords : (B, L, K, 3), optional -- SINGLE-SITE neighbour-occupancy source: the binder's
            per-residue side-chain atoms Gaussian-splatted (with per-residue SELF-EXCLUSION) into a per-query
            NEIGHBOR-OCCUPANCY FIELD that CONDITIONS the density head via a zero-init projection. NOT fed as raw
            atoms into the head's attention/feature set. Consulted ONLY when ``use_single_site_context`` is on;
            ``None`` (or the flag off) => no field, no conditioning (byte-identical). SOURCE SEAM: the pretrain
            path passes the GT side chains here (same tensors as ``gt_sidechain_*``); an FT/inference caller later
            swaps this to the flow's predicted x0 (scheduled sampling) by passing different tensors.
        context_sidechain_mask : (B, L, K), optional -- real-atom mask for ``context_sidechain_coords``.
        context_sidechain_element : (B, L, K), optional -- MODEL element ids for ``context_sidechain_coords``
            (used for the field's per-element splat sigma when ``per_element_sigma`` is on).
        available_volume : (B, L, available_volume_dim), optional -- GT-free "sandclock" descriptor
            (:func:`available_volume_cones`). When ``use_available_volume`` is on it is zero-init-projected
            and ADDED into the returned per-residue latent ``vol_hidden`` (a no-op at init). Ignored when the
            flag is off or ``None``.
        backbone_coords : (B, L, 4, 3) -- binder backbone N/CA/C/O.
        backbone_mask : (B, L, 4) -- backbone atom validity.
        seq_mask : (B, L), optional -- valid residues.
        target_coords : (B, N_t, 3), optional -- target atom coords (SE(3)-graph source).
        target_mask : (B, N_t), optional -- target atom validity.
        target_element : (B, N_t), optional -- target atom element ids (active vocab).
        target_is_backbone : (B, N_t), optional -- 1=target backbone, 0=target sidechain.
        gt_sidechain_coords : (B, L, K, 3), optional -- GT clean side-chain coords (self-occ GT).
        gt_sidechain_mask : (B, L, K), optional -- GT real-atom mask.
        gt_sidechain_element : (B, L, K), optional -- GT per-slot MODEL element ids for the per-element
            splat sigma. Only consulted when ``per_element_sigma`` is on; ``None`` => scalar-sigma GT splat.

        Returns
        -------
        dict with ``vol_hidden`` (B, L, hidden), ``vol_density_pred`` (B, L, Q), and
        ``vol_density_gt`` (B, L, Q) when GT side chains are provided (else None).
        """
        b, length = backbone_coords.shape[:2]
        device = backbone_coords.device
        radius = self.context_radius

        R, ca = build_local_frames(backbone_coords, backbone_mask)  # (B, L, 3, 3), (B, L, 3)

        coords, cand_mask, element, is_target, is_backbone = self._gather_context(
            backbone_coords, backbone_mask, target_coords, target_mask, target_element, target_is_backbone
        )  # (B, N_cand, ...)

        # Context atom positions in each residue's local frame (invariant): R_iᵀ (x_a - CA_i).
        rel = coords[:, None, :, :] - ca[:, :, None, :]  # (B, L, N_cand, 3)
        rel_local = torch.einsum("blij,blnj->blni", R.transpose(-1, -2), rel)  # (B, L, N_cand, 3)
        dist = rel.norm(dim=-1)  # (B, L, N_cand)
        within = (dist <= radius) & cand_mask[:, None, :]  # (B, L, N_cand)

        # Per-atom (i-independent) categorical features -> broadcast over residues.
        atom_feat = (
            self.element_embed(element) + self.is_target_embed(is_target) + self.is_backbone_embed(is_backbone)
        )  # (B, N_cand, atom_feat_dim)
        n_cand = coords.shape[1]
        atom_feat_b = atom_feat.unsqueeze(1).expand(b, length, n_cand, -1)  # (B, L, N_cand, ef)

        # ``rel_local`` comes from an einsum (autocast lowers it to fp16/bf16) while ``dist`` comes from
        # ``.norm()`` (kept fp32 under autocast) -- cast ``dist`` to ``rel_local``'s dtype so this ``cat`` never
        # mixes dtypes. No-op in pure fp32 (both fp32).
        geom = torch.cat(
            [rel_local / radius, (dist / radius).unsqueeze(-1).to(rel_local.dtype)], dim=-1
        )  # (B, L, N_cand, 4)
        token = self.atom_mlp(torch.cat([geom, atom_feat_b], dim=-1))  # (B, L, N_cand, hidden)

        # Masked attention-pool over context atoms -> per-residue latent.
        score = self.score_proj(token).squeeze(-1)  # (B, L, N_cand)
        neg_inf = torch.finfo(score.dtype).min
        score = score.masked_fill(~within, neg_inf)
        attn = F.softmax(score, dim=-1)
        value = self.value_proj(token)  # (B, L, N_cand, hidden)
        vol_hidden = self.output_norm((attn.unsqueeze(-1) * value).sum(dim=2))  # (B, L, hidden)
        # A residue with NO context atom in range has an all-(-inf) score row -> softmax returns a
        # UNIFORM distribution over invalid atoms (not zeros), and output_norm of the pooled vector
        # is the (nonzero) LayerNorm bias. Explicitly zero the latent so "no context -> zero latent"
        # actually holds. (Zeroing is invariance-preserving.)
        has_context = within.any(dim=-1)  # (B, L)
        vol_hidden = vol_hidden * has_context.unsqueeze(-1).to(vol_hidden.dtype)

        # SINGLE-SITE POCKET CONTEXT: build the per-query NEIGHBOR-OCCUPANCY field (which of residue i's query
        # points are blocked by its neighbours' side chains + backbone + target, residue i's own side chain
        # self-excluded) and CONDITION the density MLP on it via the zero-init ``field_proj``. Only when the flag
        # is on AND a side-chain source is supplied (pretrain = GT; sampling passes None => no field, the
        # documented predicted-x0 seam). Off / no source => field_feat stays None and the density path is
        # byte-identical; zero-init projection => byte-identical at init even when on.
        vol_density_neighbor = None
        field_feat = None  # (B, L, Q, hidden) zero-init projection of the field, or None
        if self.use_single_site_context and context_sidechain_coords is not None and context_sidechain_mask is not None:
            vol_density_neighbor = self._neighbor_occupancy_field(
                R,
                ca,
                backbone_coords,
                backbone_mask,
                target_coords,
                target_mask,
                target_element,
                context_sidechain_coords,
                context_sidechain_mask,
                context_sidechain_element,
                query_local_override=query_local_override,
            )  # (B, L, Q)
            field_feat = self.field_proj(vol_density_neighbor.to(vol_hidden.dtype).unsqueeze(-1))  # (B, L, Q, hidden)

        # SANDCLOCK (rerouted 2026-08-13, run10): condition the DENSITY TRUNK, not vol_hidden. Its old route
        # (added into the returned vol_hidden) reached no loss on the per-query path (latent_b comes from the
        # kNN ContextEncoder, not vol_hidden), so with the pooled mix off it trained nothing. Conditioning the
        # trunk makes it train on ordinary per-query steps; its information reaches generation through the
        # density field (which run10's deep-inject projects). Per-residue (B,L,D) -> (B,L,1,hidden), broadcast
        # over the query axis. Zero-init proj => exactly +0 at init (byte-identical when the sandclock is off).
        av_feat = None
        if self.use_available_volume and available_volume is not None:
            av_feat = self.available_volume_proj(available_volume.to(vol_hidden.dtype)).unsqueeze(2)  # (B,L,1,hidden)

        # Conditional density field at the query lattice. ATOM-ANCHORED: when ``query_local_override`` is given,
        # the field is evaluated at the per-residue sampled queries (B, L, Q, 3); else at the shared fixed
        # lattice broadcast to every residue. Either way ``query_b`` is (B, L, Q, 3) and the rest is identical.
        if query_local_override is not None:
            query_b = query_local_override.to(vol_hidden.dtype)  # (B, L, Q, 3)
            q = query_b.shape[2]
        else:
            q = self.query_local.shape[0]
            query_b = self.query_local.to(vol_hidden.dtype).view(1, 1, q, 3).expand(b, length, q, 3)
        query_norm = query_b / radius  # normalized query coord (Å ÷ radius; = nm at radius=10 Å)
        query_feat = self.query_encoder(query_norm) if self.query_encoder is not None else query_norm

        # PER-QUERY kNN CONTEXT: replace the shared pooled-`vol_hidden` broadcast with each query point's own
        # kNN neighborhood encoding over the SINGLE-SITE context atoms (self-occupancy ContextEncoder). Only when the
        # flag is on AND a side-chain context source is supplied (pretrain = GT; sampling passes None => the
        # pooled broadcast, the documented predicted-x0 seam). Off / no source => the exact pooled-latent path
        # (byte-identical). The pooled `vol_hidden` is still computed above and returned unchanged for the
        # downstream consumers (deep-inject / stereo / existence); only the density trunk's latent changes.
        if self.use_per_query_context and context_sidechain_coords is not None and context_sidechain_mask is not None:
            latent_b = self._per_query_context(
                R,
                ca,
                backbone_coords,
                backbone_mask,
                target_coords,
                target_mask,
                target_element,
                context_sidechain_coords,
                context_sidechain_mask,
                context_sidechain_element,
                query_feat,
                query_local_override=query_local_override,
            )  # (B, L, Q, hidden)
        else:
            # SILENT-DEGRADATION SEAM: a per-query-ON head with NO context source falls back to the pooled
            # `vol_hidden` broadcast, a DIFFERENTLY-normalized latent. This is intended for the vol_hidden-only
            # sampling path (which ignores vol_density_pred), but a density CONSUMER here would silently read the
            # pooled field. Warn once so a future density-consuming path can't degrade unnoticed. (Does not break
            # the sampling path -- it only emits a warning and still returns vol_hidden.)
            if self.use_per_query_context and not self._warned_per_query_fallback:
                warnings.warn(
                    "VolumetricOccupancyHead: use_per_query_context is ON but no context side chains were "
                    "supplied, so vol_density_pred falls back to the pooled vol_hidden broadcast (a "
                    "differently-normalized latent). Safe for the vol_hidden-only sampling path; a density "
                    "consumer would silently read the pooled field. Pass context_sidechain_* to use the "
                    "per-query kNN context.",
                    stacklevel=2,
                )
                self._warned_per_query_fallback = True
            latent_b = vol_hidden.unsqueeze(2).expand(b, length, q, self.hidden_dim)  # (B, L, Q, hidden)
        if field_feat is not None:
            latent_b = latent_b + field_feat  # neighbour-occupancy conditioning (zero-init => +0 at init)
        if av_feat is not None:
            latent_b = latent_b + av_feat  # sandclock available-volume conditioning (zero-init => +0 at init)
        # ``query_feat`` may be lowered by autocast (Fourier matmul) while ``latent_b`` (LayerNorm'd pooled path
        # or the context out_proj) is fp32 -- cast to ``latent_b``'s dtype so this ``cat`` never mixes dtypes.
        # No-op in pure fp32.
        density_pred = self.density_mlp(torch.cat([latent_b, query_feat.to(latent_b.dtype)], dim=-1)).squeeze(-1)
        if self.use_softplus:
            # softplus(x) = log1p(exp(x)); exp OVERFLOWS in fp16 for x > ~11 (=> inf). Compute in fp32 then cast
            # back so the field stays finite under autocast/fp16. No-op in pure fp32 (already fp32).
            density_pred = F.softplus(density_pred.float()).to(density_pred.dtype)  # non-negative field, ≥0 splats
        # A residue with NO valid context atom must predict NO occupancy. Zeroing vol_hidden above is
        # not enough -- the density MLP still emits query/bias-driven values from an all-zero latent, so
        # gate the predicted density by the SAME has_context mask ("no context -> exactly zero density").
        # Valid residues multiply by 1.0, so the normal path is byte-identical.
        density_pred = density_pred * has_context[:, :, None].to(density_pred.dtype)  # (B, L, Q)

        # SANDCLOCK: REROUTED (run10) -- the available-volume descriptor now conditions the DENSITY TRUNK
        # (`av_feat` added into `latent_b` above), NOT `vol_hidden`. The old `vol_hidden += available_volume_proj`
        # add here was DELETED so the projection is applied EXACTLY ONCE (a surviving copy would double-inject
        # it and confound the shared `available_volume_proj` gradient across the trunk and vol_hidden objectives).

        # OPTION (ii): add the pooled neighbour-occupancy field into the RETURNED `vol_hidden` so the pocket
        # context reaches GENERATION (deep-inject / stereo / existence consumers), not just vol_density_pred.
        # Placed AFTER the density field is read (density path byte-identical) and gated by the SAME has_context
        # mask ("no context -> no add"). Zero-init proj => +0 at init. Only when a field was actually built
        # (single-site on AND a context source supplied); None-source (sampling with no recycle x0) => no add.
        if vol_density_neighbor is not None:
            field_hidden = self.field_to_hidden_proj(vol_density_neighbor.to(vol_hidden.dtype))  # (B, L, hidden)
            vol_hidden = vol_hidden + field_hidden * has_context.unsqueeze(-1).to(vol_hidden.dtype)

        out: dict[str, torch.Tensor] = {"vol_hidden": vol_hidden, "vol_density_pred": density_pred}
        if vol_density_neighbor is not None:
            # Exposed for diagnostics / tests (the raw neighbour-occupancy conditioning field). Present only
            # when single-site conditioning is active this call.
            out["vol_density_neighbor"] = vol_density_neighbor

        if gt_sidechain_coords is not None and gt_sidechain_mask is not None:
            # GT side-chain atoms of residue i in i's local frame -> Gaussian splat at query points.
            # Uses the shared splat routine (weights = GT real-atom mask); 's self-consistency
            # reuses the SAME routine with soft P(real) weights on the identical lattice + sigma.
            out["vol_density_gt"] = self._splat_density(
                gt_sidechain_coords,
                gt_sidechain_mask,
                R,
                ca,
                element_ids=gt_sidechain_element,
                query_local_override=query_local_override,
            )

        return out
