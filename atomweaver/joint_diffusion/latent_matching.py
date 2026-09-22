"""Global latent-matching auxiliary (adapted from prior work, ``sidechain_redesign``).

At high flow-matching noise ``t`` the partially-built side chain is unidentifiable from
geometry alone, so we distil residue *identity* into a per-residue latent that conditions
generation. A **pocket-only, time-free** "clean context" stream (parallel to the main SE(3)
denoiser and never exposed to the generated side-chain atoms) is CLS-attention-pooled -- once
per designed residue over its local backbone/pocket neighbourhood -- into ``z_pred``. A
matching loss pulls ``z_pred`` toward a detached identity target ``z_gt`` (cosine), gated to
the high-noise regime. A confidence probe predicts ``cos(z_pred, z_gt)`` from detached
features, and the confidence-gated latent is FiLM'd back into the generated atoms.

Adaptation notes (vs. the ``model/global_latent_matching.py`` + ``clean_context_stream.py``)
------------------------------------------------------------------------------------------------
- **Per-residue, not global.** the model redesigns a single residue and pools one global
  latent per sample. AtomWeaver designs the *whole* peptide jointly, so the latent is
  per-designed-residue ``(B, L, embed_dim)`` and is matched per residue. Pooling is localised
  to each residue's spatial neighbourhood (self + residues within ``pool_radius`` on Cα).
- **z_gt is a LOOKUP, not an on-the-fly encode.** ``z_gt`` is a precomputed, rotamer-invariant
  QuPID identity embedding looked up by the GT residue-DB class id (``residue_indices``), not
  the output of a frozen atom-cloud encoder. See :func:`load_qupid_lookup`.
- **Clean stream seed.** Rather than a second atom-level SE(3) tower interwoven per layer, the
  clean stream refines the already-computed pocket-conditioned, time-free per-residue backbone
  features (detached, so no latent-loss gradient reaches the main denoiser) through a small
  stack of distance-biased self-attention layers. Conditioning back into the main stream is via
  the detached FiLM feedback (the ``condition_main_stream``).

Everything here is only constructed / run when ``use_global_latent_matching`` is on; with the
flag off the model is bit-exact identical to before.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from pathlib import Path


def load_qupid_lookup(path: str | Path) -> tuple[torch.Tensor, int, list[str] | None]:
    """Load + validate a precomputed QuPID identity-embedding lookup table.

    Expected schema -- a ``torch.save``'d object that is either

    - a ``dict`` with key ``"embeddings"`` mapping to a ``(R, embed_dim)`` float tensor
      (optionally ``"names"`` -- a length-``R`` list of residue-DB codes, ordered by disc index --
      and ``"embed_dim"`` for cross-checking), or
    - a bare ``(R, embed_dim)`` float tensor.

    Row ``r`` is the rotamer-invariant QuPID embedding of residue-DB class ``r`` and MUST align
    with the discretization DB ordering used to build ``batch["residue_indices"]`` (i.e. the
    ``name_to_idx`` mapping in the collate). Rows are L2-normalized on load; ``embed_dim`` is read
    off the table. Self-consistency is validated here (embeddings 2-D; ``embed_dim`` key, if present,
    matches the tensor width; ``names``, if present, has one entry per row). Cross-consistency against
    the disc-DB vocab (row count + name alignment) is checked by
    :func:`validate_qupid_lookup_alignment` once the DB is known.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.pt`` lookup artifact.

    Returns
    -------
    table : torch.Tensor
        ``(R, embed_dim)`` float tensor, each row L2-normalized.
    embed_dim : int
        The embedding width read from the table.
    names : list[str] or None
        Per-row residue-DB codes (disc-index ordered) if the file carries them, else ``None``.
    """
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    names: list[str] | None = None
    declared_dim: int | None = None
    if isinstance(obj, dict):
        if "embeddings" not in obj:
            raise KeyError(f"QuPID lookup dict at {path} has no 'embeddings' key (found {list(obj)[:8]})")
        table = obj["embeddings"]
        raw_names = obj.get("names")
        if raw_names is not None:
            names = [str(n) for n in raw_names]
        if obj.get("embed_dim") is not None:
            declared_dim = int(obj["embed_dim"])
    else:
        table = obj
    table = torch.as_tensor(table).float()
    if table.dim() != 2:
        raise ValueError(f"QuPID lookup must be (R, embed_dim); got shape {tuple(table.shape)}")
    embed_dim = int(table.shape[1])
    if declared_dim is not None and declared_dim != embed_dim:
        raise ValueError(
            f"QuPID lookup 'embed_dim'={declared_dim} disagrees with the embeddings width {embed_dim} at {path}."
        )
    if names is not None and len(names) != table.shape[0]:
        raise ValueError(
            f"QuPID lookup 'names' has {len(names)} entries but the table has {table.shape[0]} rows at {path}."
        )
    table = F.normalize(table, dim=-1)
    return table.contiguous(), embed_dim, names


def validate_qupid_lookup_alignment(
    num_rows: int,
    names: list[str] | None,
    disc_vocab_size: int,
    name_to_idx: dict[str, int],
    disc_names: list[str] | None = None,
) -> None:
    """Cross-check a QuPID lookup against the discretization-DB vocabulary.

    Guards against a stale / short / misaligned lookup silently mapping active residues to the wrong
    identity row (wrong ``z_gt`` is worse than no loss). Errors -- never clamps or truncates.

    Two alignment checks, strongest first:

    1. **Strong (primary):** if the lookup ``names`` and the disc-DB ``disc_names`` are both given,
       require an exact per-row equality (``names[i] == disc_names[i]`` for every row). This is a full
       proof that the lookup ordering matches the disc-index ordering -- it catches a scrambled order
       that the row-count check alone would miss.
    2. **Lenient (secondary, belt-and-suspenders):** for any lookup name that is a key of
       ``name_to_idx``, require it to map to its own row. Kept for the case where ``disc_names`` is
       unavailable, but note it is inert for full disc-DB names (e.g. ``"MK8"``) that are not
       ``name_to_idx`` keys -- the strong check is what actually proves alignment there.

    Parameters
    ----------
    num_rows : int
        Row count of the loaded lookup table.
    names : list[str] or None
        Per-row residue-DB codes (disc-index ordered) from the lookup, or ``None`` if absent.
    disc_vocab_size : int
        Number of discretization-DB classes (``len(db["metadata"])``); indices into this vocab are
        what ``batch["residue_indices"]`` carries.
    name_to_idx : dict[str, int]
        The collate's residue-name -> disc-index mapping.
    disc_names : list[str] or None, optional
        The disc-DB ``metadata[i]["name"]`` values in disc-index order. When provided together with
        the lookup ``names``, enables the strong per-row equality check.

    Raises
    ------
    ValueError
        If the row count does not match the disc-DB vocab size, if the lookup ``names`` and
        ``disc_names`` disagree on any row (strong check), or if any lookup name maps to a different
        disc index than its row position (lenient check).
    """
    if num_rows != disc_vocab_size:
        raise ValueError(
            f"QuPID lookup has {num_rows} rows but the discretization DB vocab has {disc_vocab_size} classes; "
            "the lookup must be disc-index-aligned (one row per disc class). Rebuild it against the current DB."
        )
    if names is not None and disc_names is not None:
        # Strong check: full per-row equality against the disc-DB metadata names.
        if len(names) != len(disc_names):
            raise ValueError(
                f"QuPID lookup has {len(names)} names but the disc DB has {len(disc_names)} metadata names; "
                "the lookup must have one name per disc class, in disc-index order. Rebuild it."
            )
        for row, (nm, disc_nm) in enumerate(zip(names, disc_names, strict=True)):
            if nm != disc_nm:
                raise ValueError(
                    f"QuPID lookup row {row} is {nm!r} but the disc DB metadata name at that index is "
                    f"{disc_nm!r}; the lookup ordering is misaligned with the discretization DB. Rebuild it."
                )
    if names is not None:
        # Lenient secondary check (kept as belt-and-suspenders; inert for non-name_to_idx keys).
        for row, nm in enumerate(names):
            mapped = name_to_idx.get(nm)
            if mapped is not None and mapped != row:
                raise ValueError(
                    f"QuPID lookup row {row} is {nm!r} but name_to_idx maps {nm!r} to disc index {mapped}; "
                    "the lookup ordering is misaligned with the discretization DB."
                )


class _DistanceBiasedSelfAttention(nn.Module):
    """One residue-level self-attention block with an additive Cα-distance bias.

    Invariant (scalar-feature) analogue of :class:`~atomweaver.joint_diffusion.se3_transformer`'s
    ``SE3AttentionLayer``: pairwise Cα distances are turned into per-head attention biases via a
    radial-basis MLP, but no coordinate update is produced -- the clean stream only needs an
    invariant per-residue representation for pooling.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        n_dist_features: int = 16,
        max_dist: float = 20.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.n_dist_features = n_dist_features

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_to_bias = nn.Sequential(
            nn.Linear(n_dist_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.register_buffer("rbf_centers", torch.linspace(0.0, max_dist, n_dist_features))
        self.rbf_width = max(max_dist / max(n_dist_features - 1, 1), 1e-3)

    def forward(self, h: torch.Tensor, ca_coords: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """Update per-residue features.

        Parameters
        ----------
        h : torch.Tensor
            ``(B, L, hidden_dim)`` per-residue features.
        ca_coords : torch.Tensor
            ``(B, L, 3)`` Cα coordinates.
        key_mask : torch.Tensor
            ``(B, L)`` bool; True = valid residue (attendable key).

        Returns
        -------
        torch.Tensor
            ``(B, L, hidden_dim)`` updated features.
        """
        b, n_res, _ = h.shape
        q = self.q_proj(h).view(b, n_res, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(b, n_res, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(b, n_res, self.num_heads, self.head_dim)

        scores = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale  # (B, H, L, L)

        dist = torch.cdist(ca_coords, ca_coords)  # (B, L, L)
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers) ** 2) / (2 * self.rbf_width**2))
        bias = self.dist_to_bias(rbf).permute(0, 3, 1, 2)  # (B, H, L, L)
        scores = scores + bias

        key_bias = torch.where(key_mask, 0.0, float("-inf")).view(b, 1, 1, n_res)
        scores = scores + key_bias
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # rows with no valid key -> zero
        attn = self.dropout(attn)

        pooled = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(b, n_res, -1)
        h = self.norm1(h + self.dropout(self.out_proj(pooled)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class CleanContextStream(nn.Module):
    """Pocket-only, time-free per-residue clean-context encoder.

    Refines the (detached) pocket-conditioned, time-free per-residue backbone features through a
    stack of Cα-distance-biased self-attention layers. Its output never depends on the generated
    side-chain atoms or on ``t``, so a single forward pass suffices at eval time.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int = 3,
        num_heads: int = 4,
        n_dist_features: int = 16,
        max_dist: float = 20.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DistanceBiasedSelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    n_dist_features=n_dist_features,
                    max_dist=max_dist,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, feats: torch.Tensor, ca_coords: torch.Tensor, seq_mask: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode ``(B, L, hidden)`` pocket features into clean per-residue context features.

        Parameters
        ----------
        feats : torch.Tensor
            ``(B, L, hidden)`` pocket-conditioned, time-free per-residue features. Should be
            passed detached -- no latent-matching gradient may reach the main denoiser.
        ca_coords : torch.Tensor
            ``(B, L, 3)`` Cα coordinates.
        seq_mask : torch.Tensor
            ``(B, L)`` bool; True = valid residue.

        Returns
        -------
        final : torch.Tensor
            ``(B, L, hidden)`` clean per-residue context features (last layer) -- read by the pool head.
        per_layer : list of torch.Tensor
            Length ``n_layers`` list of ``(B, L, hidden)`` per-layer outputs. Element ``i`` is the
            clean-stream layer-``i`` output, injected into main SE(3) layer ``i`` (per-layer interweave).
        """
        h = feats
        per_layer: list[torch.Tensor] = []
        for layer in self.layers:
            h = layer(h, ca_coords, seq_mask)
            per_layer.append(h)
        return h, per_layer


class LatentPoolHead(nn.Module):
    """Per-residue CLS-attention pool + confidence probe over the clean-context stream.

    For each designed residue, a shared CLS query attends over the residue's local Cα
    neighbourhood (self + residues within ``pool_radius``) to produce ``z_pred`` (L2-normalized).
    A second CLS query (own parameters) plus ``z_pred`` feeds a confidence MLP predicting
    ``cos(z_pred, z_gt)``; its input is detached, so the probe trains only itself.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        pool_hidden: int = 128,
        num_heads: int = 4,
        pool_radius: float = 10.0,
        confidence: bool = True,
        confidence_hidden: int = 64,
    ) -> None:
        super().__init__()
        if pool_hidden % num_heads != 0:
            raise ValueError(f"pool_hidden ({pool_hidden}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = int(embed_dim)
        self.num_heads = num_heads
        self.d_head = pool_hidden // num_heads
        self.pool_radius = float(pool_radius)

        self.k_proj = nn.Linear(hidden_dim, pool_hidden)
        self.v_proj = nn.Linear(hidden_dim, pool_hidden)
        self.cls_query = nn.Parameter(torch.randn(num_heads, self.d_head) / math.sqrt(self.d_head))
        self.out_proj = nn.Linear(pool_hidden, embed_dim)

        self.use_confidence = confidence
        if confidence:
            self.conf_query = nn.Parameter(torch.randn(num_heads, self.d_head) / math.sqrt(self.d_head))
            self.conf_mlp = nn.Sequential(
                nn.Linear(pool_hidden + embed_dim, confidence_hidden),
                nn.SiLU(),
                nn.Linear(confidence_hidden, 1),
            )

    def _neighbourhood_pool(
        self,
        query: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbour_bias: torch.Tensor,
    ) -> torch.Tensor:
        """CLS-attention pool of ``v`` over each residue's neighbourhood.

        Parameters
        ----------
        query : torch.Tensor
            ``(num_heads, d_head)`` shared CLS query.
        k, v : torch.Tensor
            ``(B, L, num_heads, d_head)`` projected keys / values.
        neighbour_bias : torch.Tensor
            ``(B, L_query, L_key)`` additive mask (0 in-neighbourhood, ``-inf`` outside).

        Returns
        -------
        torch.Tensor
            ``(B, L, num_heads * d_head)`` pooled features.
        """
        b, n_res = k.shape[0], k.shape[1]
        scores_j = (k * query).sum(-1) / math.sqrt(self.d_head)  # (B, L_key, H)
        scores = scores_j.permute(0, 2, 1).unsqueeze(2) + neighbour_bias.unsqueeze(1)  # (B, H, L_q, L_key)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.einsum("bhqj,bjhd->bqhd", attn, v)  # (B, L_q, H, d_head)
        return pooled.reshape(b, n_res, self.num_heads * self.d_head)

    def forward(
        self,
        clean_feats: torch.Tensor,
        ca_coords: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pool clean features into a per-residue latent and confidence.

        Parameters
        ----------
        clean_feats : torch.Tensor
            ``(B, L, hidden)`` clean-context features from :class:`CleanContextStream`.
        ca_coords : torch.Tensor
            ``(B, L, 3)`` Cα coordinates.
        seq_mask : torch.Tensor
            ``(B, L)`` bool; True = valid residue.

        Returns
        -------
        z_pred : torch.Tensor
            ``(B, L, embed_dim)`` L2-normalized per-residue latent.
        conf : torch.Tensor or None
            ``(B, L)`` predicted ``cos(z_pred, z_gt)`` in ``[-1, 1]`` (``None`` if disabled).
        """
        b, n_res, _ = clean_feats.shape
        k = self.k_proj(clean_feats).view(b, n_res, self.num_heads, self.d_head)
        v = self.v_proj(clean_feats).view(b, n_res, self.num_heads, self.d_head)

        dist = torch.cdist(ca_coords, ca_coords)  # (B, L, L)
        in_radius = dist <= self.pool_radius
        eye = torch.eye(n_res, dtype=torch.bool, device=clean_feats.device).unsqueeze(0)
        in_neigh = (in_radius | eye) & seq_mask.unsqueeze(1)  # (B, L_q, L_key)
        neighbour_bias = torch.where(in_neigh, 0.0, float("-inf"))

        pooled = self._neighbourhood_pool(self.cls_query, k, v, neighbour_bias)  # (B, L, pool_hidden)
        z_pred = F.normalize(self.out_proj(pooled), dim=-1)  # (B, L, embed_dim)

        conf = None
        if self.use_confidence:
            # Detach: the probe reads clean features + z_pred but trains only itself.
            conf_pooled = self._neighbourhood_pool(self.conf_query, k.detach(), v.detach(), neighbour_bias)
            conf_in = torch.cat([conf_pooled, z_pred.detach()], dim=-1)
            conf = torch.tanh(self.conf_mlp(conf_in).squeeze(-1))  # (B, L) in [-1, 1]
        return z_pred, conf


def clean_summary(z_pred: torch.Tensor, conf: torch.Tensor | None) -> torch.Tensor:
    """Build the detached FiLM feedback vector ``[conf, relu(conf) * z_pred]``.

    Negative/low confidence mutes the latent (``relu``) rather than flipping it. The result is
    detached so no gradient reaches the clean stream or pool head through the FiLM path.

    Parameters
    ----------
    z_pred : torch.Tensor
        ``(B, L, embed_dim)`` per-residue latent.
    conf : torch.Tensor or None
        ``(B, L)`` confidence in ``[-1, 1]``; ``None`` -> treated as ``1`` (no muting).

    Returns
    -------
    torch.Tensor
        ``(B, L, 1 + embed_dim)`` detached FiLM conditioning vector.
    """
    if conf is None:
        conf = torch.ones(z_pred.shape[:-1], device=z_pred.device, dtype=z_pred.dtype)
    gated = F.relu(conf).unsqueeze(-1) * z_pred
    return torch.cat([conf.unsqueeze(-1), gated], dim=-1).detach()


def global_latent_matching_loss(
    z_pred: torch.Tensor,
    z_gt: torch.Tensor,
    t_norm: torch.Tensor,
    mask: torch.Tensor,
    gate_t_min: float,
    gate_t_full: float,
    loss_type: Literal["cosine", "mse"] = "cosine",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """High-noise-gated per-residue latent-matching loss.

    Parameters
    ----------
    z_pred : torch.Tensor
        ``(B, L, D)`` L2-normalized predicted latent (carries gradient into the pool + clean stream).
    z_gt : torch.Tensor
        ``(B, L, D)`` L2-normalized target latent (detached by the caller).
    t_norm : torch.Tensor
        ``(B,)`` flow-matching time normalized to ``[0, 1]`` (1 = noisiest).
    mask : torch.Tensor
        ``(B, L)`` bool; True = designed position with a valid ``z_gt``.
    gate_t_min, gate_t_full : float
        Ramp bounds: gate is 0 below ``gate_t_min``, ramps to 1 at ``gate_t_full``. The loss is
        active only in the high-noise regime.
    loss_type : {"cosine", "mse"}
        Per-residue distance between ``z_pred`` and ``z_gt``.

    Returns
    -------
    loss : torch.Tensor
        Scalar masked, gate-weighted mean loss.
    cos : torch.Tensor
        ``(B, L)`` per-residue cosine similarity (for the confidence target).
    info : dict
        Diagnostics (``global_latent_cos``, ``global_latent_active_frac``).
    """
    if gate_t_full <= gate_t_min:
        raise ValueError(f"gate_t_full ({gate_t_full}) must exceed gate_t_min ({gate_t_min})")
    gate = ((t_norm - gate_t_min) / (gate_t_full - gate_t_min)).clamp(0.0, 1.0)  # (B,)
    cos = (z_pred * z_gt).sum(-1)  # (B, L)
    raw = ((z_pred - z_gt) ** 2).sum(-1) if loss_type == "mse" else (1.0 - cos)
    weight = gate.unsqueeze(-1) * mask.to(z_pred.dtype)  # (B, L)
    denom = weight.sum().clamp_min(1.0)
    loss = (raw * weight).sum() / denom
    with torch.no_grad():
        active = (weight > 0).to(z_pred.dtype)
        info = {
            "global_latent_cos": (cos * active).sum() / active.sum().clamp_min(1.0),
            "global_latent_active_frac": active.mean(),
        }
    return loss, cos, info


def global_latent_confidence_loss(
    conf: torch.Tensor,
    cos_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE of the confidence probe against the (detached) true ``cos(z_pred, z_gt)``.

    ``conf`` is produced from detached inputs, so this trains only the probe.

    Parameters
    ----------
    conf : torch.Tensor
        ``(B, L)`` predicted cosine.
    cos_target : torch.Tensor
        ``(B, L)`` true cosine; detached here.
    mask : torch.Tensor
        ``(B, L)`` bool; True = designed position.

    Returns
    -------
    torch.Tensor
        Scalar masked MSE.
    """
    err = (conf - cos_target.detach()) ** 2
    weight = mask.to(conf.dtype)
    return (err * weight).sum() / weight.sum().clamp_min(1.0)
