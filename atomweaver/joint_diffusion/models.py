"""
Denoising models for SE(3)-equivariant diffusion.

This module implements the neural network architecture for denoising side-chain
atom clouds conditioned on fixed backbone coordinates and diffusion timestep.
"""

from __future__ import annotations

import math
import os
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Sequence

from .diffusion import ELEMENT_PAD, NUM_ELEMENT_TYPES, TimestepEmbedding
from .egnn import (
    EDGE_TYPE_BINDER_TARGET,
    EdgeTypeEmbedding,
    build_multi_type_radius_graph,
)
from .latent_matching import CleanContextStream, LatentPoolHead, clean_summary
from .residue_frame_stream import (
    _STEREO_HEAD_LPRIOR_BIAS,
    ResidueFrameStream,
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
    default_sigma_element_scale,
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


def _count_pearson_intra_pooled(
    predicted_counts: torch.Tensor, target_counts: torch.Tensor, seq_mask_f: torch.Tensor
) -> torch.Tensor:
    """Per-sample (intra) atom-count Pearson r, with a batch-pooled fallback for single-site samples.

    For samples with >=2 valid positions, returns the per-sample (intra) Pearson r between predicted and
    target per-residue atom counts -- this forces learning of *within-peptide* big-vs-small allocation, the
    signal that transfers to val/test without the cross-sample length/backbone shortcut.

    For single-site samples (n_valid < 2) intra r is undefined (it degenerates to 0 -> loss 1), so those
    samples instead receive a single batch-pooled cross-sample r computed over all single-site samples'
    lone counts -- the legitimate "big vs small entity" signal for smallmol / single-site design, where
    there is no within-entity allocation to learn. A lone single-site sample (the only one in the batch)
    cannot be pooled and keeps r=0.

    NOTE (DDP locality): the single-site pool is over this rank's LOCAL microbatch only -- it is computed
    before DDP gradient sync, so world size / grad accumulation do NOT enlarge it. A rank that sees fewer
    than 2 single-site samples (per-rank bs=1, or one single-site mixed into a multi-site batch) falls to
    pooled_r=0 with no useful gradient. Benign for multi-site peptide training and smallmol at per-rank
    batch >= 2; for single-site-heavy phases at tiny per-rank batch, either guarantee >= 2 single-site
    samples per rank (dataloader) or replace this local pool with a differentiable distributed all-gather
    over the single-site counts. [reviewed 2026-06-14]

    Parameters
    ----------
    predicted_counts, target_counts : torch.Tensor
        Per-residue atom counts, shape (B, L).
    seq_mask_f : torch.Tensor
        Float validity mask, shape (B, L).

    Returns
    -------
    torch.Tensor
        Per-sample correlation of shape (B,): intra r for multi-site samples, the shared pooled r for
        single-site samples.
    """
    n_valid = seq_mask_f.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
    pred_mean = (predicted_counts * seq_mask_f).sum(dim=-1, keepdim=True) / n_valid
    gt_mean = (target_counts * seq_mask_f).sum(dim=-1, keepdim=True) / n_valid
    pred_c = (predicted_counts - pred_mean) * seq_mask_f
    gt_c = (target_counts - gt_mean) * seq_mask_f
    pred_n = pred_c.norm(dim=-1).clamp(min=1e-8)
    gt_n = gt_c.norm(dim=-1).clamp(min=1e-8)
    corr = (pred_c * gt_c).sum(dim=-1) / (pred_n * gt_n)  # (B,) intra; ~0 for single-site

    single_site = n_valid.squeeze(-1) < 2  # (B,)
    if single_site.any():
        lone_pred = (predicted_counts * seq_mask_f).sum(dim=-1)[single_site]  # the one count per sample
        lone_gt = (target_counts * seq_mask_f).sum(dim=-1)[single_site]
        if lone_pred.numel() >= 2:
            lp = lone_pred - lone_pred.mean()
            lg = lone_gt - lone_gt.mean()
            pooled_r = (lp * lg).sum() / (lp.norm().clamp(min=1e-8) * lg.norm().clamp(min=1e-8))
        else:
            pooled_r = corr.new_zeros(())  # lone single-site sample: nothing to pool
        corr = torch.where(single_site, pooled_r, corr)
    return corr


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


def revive_dead_ar_state_embed(model, std: float = 0.02, seed: int = 1234) -> bool:
    """Revive an all-zero ``ar_state_embed`` so the K-mask clean-context signal is live (BUG-A fix).

    A PRE-FIX checkpoint stored ``ar_state_embed`` as all-zeros (paired with a zero ``ar_state_proj`` ->
    dead branch: no output AND no gradient). ANY weights-load from such a checkpoint (``--resume-weights``,
    Ray/distributed resume, or a Lightning ``ckpt_path`` restore) re-kills the branch even though ``__init__``
    now small-random-inits it. Call this after every weights load (and in ``on_train_start`` to catch the
    post-restore case). Deterministic (seeded) so DDP ranks stay identical. ``ar_state_proj`` is left as
    loaded, so the residual is still a no-op initially but now has a live, gradient-carrying state signal.
    No-op (returns False) if the embed is already nonzero (post-fix checkpoint -> trained weights preserved).
    """
    # Accept either the inner InverseFoldingDiffusion (.denoiser) or a wrapper exposing .model (the Lightning
    # module passes self -> self.model.denoiser).
    denoiser = getattr(model, "denoiser", None) or getattr(getattr(model, "model", None), "denoiser", None)
    emb = getattr(denoiser, "ar_state_embed", None)
    if emb is None or int(torch.count_nonzero(emb.weight)) != 0:
        return False
    gen = torch.Generator(device=emb.weight.device).manual_seed(seed)
    with torch.no_grad():
        emb.weight.normal_(0.0, std, generator=gen)
    return True


# Graft init for the glycine PAD gate (see InverseFoldingDiffusion.__init__ for WHY it is 0.0 and not
# -4.0). Shared by the fresh graft and the resume-time reset so the two can never drift apart.
GLYCINE_PAD_GATE_INIT = 0.0


def reset_glycine_pad_gate(model, value: float = GLYCINE_PAD_GATE_INIT) -> float | None:
    """Force a LOADED ``glycine_pad_gate`` back to the current graft default (see ``GLYCINE_PAD_GATE_INIT``).

    The 0.0 gate init only helps a FRESH graft. A checkpoint that already contains ``glycine_pad_gate``
    restores whatever it stored, so a weights-resume from a run grafted under the old ``-4.0`` init
    reloads the gradient-STARVED value (measured: ``-4.00 -> -3.99`` over ~30 epochs of v34) and the
    init fix is a silent no-op. This overwrites just that one scalar so the successor run starts with a
    gate that can actually move.

    ONLY the gate scalar is touched: ``glycine_head`` (the dihedral->glycine classifier) keeps its
    learned weights, which are worth carrying forward (v34's output-weight RMS ~0.056, i.e. it DID
    learn to discriminate) -- it is the gate, not the classifier, that was starved.

    Unlike ``revive_dead_*`` this is NOT idempotent-safe (a trained gate would be clobbered), so call it
    ONLY at an explicit weights-resume site, never from a generic ``on_train_start`` hook that also
    fires after a mid-run Lightning ``ckpt_path`` restore.

    Parameters
    ----------
    model : nn.Module
        The ``InverseFoldingDiffusion`` itself, or a wrapper exposing it as ``.model`` (the Lightning
        module passes ``self`` -> ``self.model``).
    value : float
        Value to write into the gate. Defaults to the current graft init.

    Returns
    -------
    float or None
        The OLD gate value if the reset was applied (so the caller can log ``old -> new``), or ``None``
        if there is no glycine head / gate to reset (no-op).
    """
    core = model if hasattr(model, "glycine_pad_gate") else getattr(model, "model", None)
    gate = getattr(core, "glycine_pad_gate", None)
    if gate is None or not getattr(core, "use_glycine_head", False):
        return None  # head off (or absent): nothing to reset
    old = float(gate.detach().reshape(-1)[0].item())
    with torch.no_grad():
        gate.fill_(value)
    return old


def corrupt_evc_mask(
    gt_mask: torch.Tensor,
    seq_mask: "torch.Tensor | None",
    underfill_bias: float = 0.7,
    drop_lo: float = 0.05,
    drop_hi: float = 0.6,
    add_lo: float = 0.03,
    add_hi: float = 0.28,
) -> torch.Tensor:
    """Designed corruption of a binary GT sidechain mask for EVC scheduled sampling.

    Produces a spread of WRONG-count states: each selected sample either UNDER-fills (drops a
    per-sample-random fraction of present atoms) with probability ``underfill_bias``, or OVER-fills
    (adds spurious atoms on empty valid slots) otherwise. Corruption is per-atom Bernoulli with a
    per-sample rate, so the magnitude varies and states are partial (rarely all-empty / all-full).
    The point: the model learns to recover the correct count from wrong-count states in BOTH
    directions, so it never over-commits to ghost (all-PAD).

    Parameters
    ----------
    gt_mask : torch.Tensor
        Float {0, 1} mask of shape (B, L, max_sc).
    seq_mask : torch.Tensor or None
        Bool (B, L) valid-residue mask; corruption is restricted to valid residues.
    underfill_bias : float
        P(a corrupted sample is under-fill vs over-fill).
    drop_lo, drop_hi : float
        Per-sample under-fill drop-rate range (fraction of present atoms dropped).
    add_lo, add_hi : float
        Per-sample over-fill add-rate range (fraction of empty valid slots added).
    Used by the main-path underfill augmentation.

    """
    b, ll, s = gt_mask.shape
    device = gt_mask.device
    present = gt_mask > 0.5
    valid = seq_mask.to(torch.bool).unsqueeze(-1).expand(b, ll, s) if seq_mask is not None else torch.ones_like(present)
    do_underfill = torch.rand(b, device=device) < underfill_bias  # (b,)
    drop_rate = drop_lo + (drop_hi - drop_lo) * torch.rand(b, device=device)
    add_rate = add_lo + (add_hi - add_lo) * torch.rand(b, device=device)
    drop_draw = torch.rand(b, ll, s, device=device)
    add_draw = torch.rand(b, ll, s, device=device)
    drop = present & valid & do_underfill[:, None, None] & (drop_draw < drop_rate[:, None, None])
    add = (~present) & valid & (~do_underfill)[:, None, None] & (add_draw < add_rate[:, None, None])
    out = gt_mask.clone()
    out = out.masked_fill(drop, 0.0)
    return out.masked_fill(add, 1.0)


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

# Jackie biochemical features: 25-dim standardized descriptors for each amino acid.
# Encodes hydrophobicity, charge, aromaticity, size, H-bonding, etc.
# Ordering matches RESIDUE_TO_IDX: ALA=0, ARG=1, ..., VAL=19, UNK=20 (zeros).
# Source: pq_mlcore Jackie v3 feature set (Mordred descriptors, standardized).
JACKIE_DIM = 25
JACKIE_FEATURES = torch.tensor(
    [
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            0.3247,
            -0.1455,
            -0.8026,
            -0.3410,
            -0.8165,
            0.7273,
            0.0422,
            -1.1037,
            -0.5448,
            -0.3996,
            -0.9391,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            -1.2247,
            -0.5701,
        ],  # ALA
        [
            -0.3333,
            0.1302,
            -0.3333,
            4.1284,
            -0.4765,
            3.1690,
            -0.6727,
            0.0000,
            0.3247,
            -1.0954,
            -0.8026,
            0.6192,
            1.9052,
            -1.0909,
            0.5010,
            0.0880,
            0.0461,
            -0.3996,
            1.1619,
            -0.4201,
            -0.2294,
            0.0000,
            2.9666,
            0.8165,
            -0.5749,
        ],  # ARG
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            0.6835,
            0.8222,
            0.0000,
            -0.3068,
            -1.5823,
            0.9129,
            -0.4226,
            -0.8165,
            0.7273,
            0.7729,
            -0.5030,
            -0.2489,
            1.3578,
            0.3740,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            0.7839,
        ],  # ASN
        [
            -0.3333,
            0.1302,
            3.0000,
            -0.2949,
            -0.4765,
            -0.5592,
            2.3170,
            0.0000,
            -0.3068,
            -0.8299,
            0.9129,
            -0.4226,
            -0.8165,
            0.7273,
            0.7729,
            -0.5030,
            -0.2489,
            3.2010,
            0.3740,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            1.4692,
        ],  # ASP
        [
            3.0000,
            -2.4736,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            0.3247,
            -0.2586,
            -0.8026,
            -1.4644,
            -0.8165,
            0.7273,
            0.3791,
            -0.7465,
            -0.4458,
            -0.3996,
            -0.0199,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            1.7449,
        ],  # CYS
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            0.6835,
            0.8222,
            0.0000,
            0.0484,
            -1.0926,
            0.9129,
            0.6192,
            0.5443,
            -1.5455,
            0.6997,
            -0.2767,
            -0.1509,
            0.4791,
            0.9801,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            0.2370,
        ],  # GLN
        [
            -0.3333,
            0.1302,
            3.0000,
            -0.2949,
            -0.4765,
            -0.5592,
            2.3170,
            0.0000,
            0.0484,
            -0.3402,
            0.9129,
            0.6192,
            0.5443,
            -1.5455,
            0.6997,
            -0.2767,
            -0.1509,
            0.4883,
            0.9801,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            0.8293,
        ],  # GLU
        [
            -0.3333,
            -2.4736,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            -0.1173,
            -0.6332,
            -0.8026,
            -1.4644,
            -0.8165,
            0.7273,
            -0.5835,
            -1.0857,
            -0.6428,
            -1.2984,
            -0.6109,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            -1.2247,
            0.0509,
        ],  # GLY
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            1.2883,
            1.9262,
            -0.6727,
            0.0000,
            -1.1487,
            -0.2134,
            1.4948,
            -0.5405,
            0.5443,
            -1.5455,
            -1.5288,
            1.2994,
            0.0451,
            -0.3996,
            -0.4139,
            2.3805,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            0.8870,
        ],  # HIS
        [
            -0.3333,
            2.7340,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            1.0614,
            1.1429,
            -0.8026,
            1.8242,
            -0.8165,
            0.7273,
            0.9923,
            -0.5156,
            -0.2508,
            -0.3996,
            0.3740,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            -1.2247,
            -1.7118,
        ],  # ILE
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            1.0614,
            1.1429,
            -0.8026,
            1.8242,
            -0.8165,
            0.7273,
            0.7729,
            -0.5030,
            -0.2508,
            -0.3996,
            0.3740,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            -1.2247,
            -1.7118,
        ],  # LEU
        [
            -0.3333,
            0.1302,
            -0.3333,
            1.1795,
            -0.4765,
            0.6835,
            -0.6727,
            0.0000,
            1.0614,
            -0.0085,
            -0.8026,
            1.6610,
            0.5443,
            0.7273,
            0.5211,
            -0.0115,
            -0.1519,
            -0.3996,
            1.8892,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            -1.5213,
        ],  # LYS
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            0.8772,
            0.7751,
            -0.8026,
            -0.4226,
            0.5443,
            -1.5455,
            0.5554,
            -0.2162,
            -0.2499,
            -0.3996,
            1.3589,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            0.8165,
            0.4523,
        ],  # MET
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            1.6412,
            -0.5592,
            -0.6727,
            0.0000,
            -1.8854,
            1.3897,
            1.4948,
            -0.5617,
            0.5443,
            0.7273,
            -1.5084,
            1.5403,
            0.1411,
            -0.3996,
            -0.6109,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            -1.2247,
            0.1195,
        ],  # PHE
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            0.8772,
            0.4651,
            -0.8026,
            0.6192,
            -0.8165,
            0.7273,
            -1.3630,
            1.0623,
            -0.2508,
            -1.2984,
            -2.5806,
            2.3805,
            4.3589,
            0.0000,
            -2.0829,
            -1.2247,
            -0.5848,
        ],  # PRO
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            0.8222,
            0.0000,
            0.3247,
            -1.4356,
            -0.8026,
            -1.4644,
            -0.8165,
            0.7273,
            0.3791,
            -0.7465,
            -0.4458,
            0.4791,
            -0.0199,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            0.2900,
        ],  # SER
        [
            3.0000,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            0.8222,
            0.0000,
            0.6404,
            -0.9478,
            -0.8026,
            -0.3410,
            -0.8165,
            0.7273,
            0.7930,
            -0.7706,
            -0.3478,
            1.3871,
            -0.3423,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            -0.2630,
        ],  # THR
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            3.0530,
            0.6835,
            -0.6727,
            0.0000,
            -2.2011,
            1.9940,
            1.4948,
            -0.5617,
            1.9052,
            -1.0909,
            -2.1957,
            2.6202,
            4.2573,
            -0.3996,
            -1.2018,
            2.3805,
            -0.2294,
            0.0000,
            0.4418,
            -1.2247,
            0.7883,
        ],  # TRP
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            1.6412,
            -0.5592,
            0.8222,
            0.0000,
            -1.8854,
            1.0201,
            1.4948,
            -0.5617,
            1.9052,
            -1.0909,
            -1.4949,
            1.4190,
            0.2401,
            -0.3996,
            -0.7847,
            -0.4201,
            -0.2294,
            0.0000,
            0.4418,
            0.8165,
            0.6981,
        ],  # TYR
        [
            -0.3333,
            0.1302,
            -0.3333,
            -0.2949,
            -0.4765,
            -0.5592,
            -0.6727,
            0.0000,
            0.8772,
            0.6532,
            -0.8026,
            0.7824,
            -0.8165,
            0.7273,
            0.7930,
            -0.7706,
            -0.3488,
            -0.3996,
            -0.3423,
            -0.4201,
            -0.2294,
            0.0000,
            -0.8206,
            -1.2247,
            -1.4128,
        ],  # VAL
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
        ],  # UNK
    ],
    dtype=torch.float32,
)  # (21, 25)

# Standard amino acid mapping
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


def res_names_to_type_indices(res_names: list[list[str]]) -> torch.Tensor:
    """
    Convert residue names to type indices for embedding.

    Parameters
    ----------
    res_names : list[list[str]]
        Batch of residue name lists.

    Returns
    -------
    indices : torch.Tensor
        Tensor of shape (B, max_len) with residue type indices.
    """
    batch_size = len(res_names)
    max_len = max(len(names) for names in res_names)

    indices = torch.full((batch_size, max_len), RESIDUE_TO_IDX["UNK"], dtype=torch.long)
    for i, names in enumerate(res_names):
        for j, name in enumerate(names):
            indices[i, j] = RESIDUE_TO_IDX.get(name, RESIDUE_TO_IDX["UNK"])

    return indices


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


def rotate_about_axis_through_pivot(
    coords: torch.Tensor,
    pivot: torch.Tensor,
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Rigidly rotate points about a line through ``pivot`` along unit ``axis`` (Rodrigues).

    Used by the rotation-corruption augmentation: an azimuthal spin of a residue's
    side-chain atoms about its Cα->pseudo-Cβ (cone) axis. Because the rotation is an isometry
    that fixes every point on the axis (``pivot`` sits ON the axis), it preserves all pairwise
    distances AND every atom's distance from ``pivot`` -- i.e. it is a rigid re-orientation of the
    cloud, never a distortion. The backbone / pivot itself is unchanged.

    Parameters
    ----------
    coords : torch.Tensor
        Point positions, shape ``(..., 3)`` (e.g. ``(B, L, max_sc, 3)``).
    pivot : torch.Tensor
        A point on the rotation axis (the Cα), broadcastable to ``coords`` (e.g. ``(B, L, 1, 3)``).
    axis : torch.Tensor
        Rotation axis direction, broadcastable to ``coords``. Assumed unit-norm (the cone
        direction from :func:`compute_pseudo_cb_direction` already is); re-normalized defensively.
    angle : torch.Tensor
        Rotation angle in radians, broadcastable to ``coords[..., 0]`` (e.g. ``(B, L, 1)``).

    Returns
    -------
    torch.Tensor
        Rotated positions, same shape as ``coords``.
    """
    k = _safe_normalize(axis)
    v = coords - pivot  # offset from the on-axis pivot
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    kxv = torch.cross(k.expand_as(v), v, dim=-1)
    kdotv = (k * v).sum(dim=-1, keepdim=True)
    # Rodrigues: v_rot = v·cosθ + (k×v)·sinθ + k·(k·v)·(1-cosθ)
    v_rot = v * cos + kxv * sin + k * kdotv * (1.0 - cos)
    return pivot + v_rot


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

    def __init__(self, hidden_dim: int = 128, num_residue_types: int = NUM_RESIDUE_TYPES, use_jackie: bool = False):
        super().__init__()
        self.use_jackie = use_jackie

        # Residue type embedding: always use learned embedding
        self.residue_embed = nn.Embedding(num_residue_types, hidden_dim // 2)
        # Optionally also project Jackie biochemical features and add them
        if use_jackie:
            self.register_buffer("jackie_features", JACKIE_FEATURES)
            self.residue_proj = nn.Linear(JACKIE_DIM, hidden_dim // 2)

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

        # Encode residue types: learned embedding + optional Jackie biochemical features
        seq_features = self.residue_embed(residue_types)  # (B, L, hidden/2)
        if self.use_jackie:
            jackie_feat = self.jackie_features[residue_types.clamp(0, 20)]  # (B, L, 25)
            seq_features = seq_features + self.residue_proj(jackie_feat)  # additive

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


class IntraResidueSlotAttention(nn.Module):
    """
    Self-attention across the 14 sidechain slots within each residue.

    Lets slots coordinate ghost/real decisions and atom count by attending
    to each other's features. Operates on scalar features only (invariant).
    Slot position embeddings are added to Q/K so the model can learn
    ordered slot dependencies (e.g., slot 0 = Cb, almost always real).

    Output projection is zero-initialized so the block starts as identity,
    allowing warm-start from existing checkpoints.
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2, max_slots: int = 14, dropout: float = 0.1
    ):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for slot attention")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_slots = max_slots
        self.scale = self.head_dim**-0.5

        self.slot_pos_embed = nn.Embedding(max_slots, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(hidden_dim, hidden_dim),
                        "k_proj": nn.Linear(hidden_dim, hidden_dim),
                        "v_proj": nn.Linear(hidden_dim, hidden_dim),
                        "out_proj": nn.Linear(hidden_dim, hidden_dim),
                        "attn_dropout": nn.Dropout(dropout),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, 4 * hidden_dim),
                            nn.SiLU(),
                            nn.Linear(4 * hidden_dim, hidden_dim),
                            nn.Dropout(dropout),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        # Zero-init all output projections so block starts as identity
        for layer in self.layers:
            nn.init.zeros_(layer["out_proj"].weight)
            nn.init.zeros_(layer["out_proj"].bias)
            nn.init.zeros_(layer["ffn"][2].weight)
            nn.init.zeros_(layer["ffn"][2].bias)

    def forward(self, h: torch.Tensor, n_residues: int) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            Flat slot features of shape (N_sc, hidden_dim) where N_sc = n_residues * max_slots.
        n_residues : int
            Number of valid residues.

        Returns
        -------
        torch.Tensor
            Updated slot features, same shape as input.
        """
        if n_residues == 0:
            return h

        h = h.reshape(n_residues, self.max_slots, -1)  # (R, 14, D)
        slot_idx = torch.arange(self.max_slots, device=h.device)
        pos = self.slot_pos_embed(slot_idx)  # (14, D)

        for layer in self.layers:
            # Pre-norm self-attention with slot position on Q/K
            h_norm = layer["norm1"](h)
            q = layer["q_proj"](h_norm + pos)  # (R, 14, D)
            k = layer["k_proj"](h_norm + pos)  # (R, 14, D)
            v = layer["v_proj"](h_norm)  # (R, 14, D) -- no position on V

            # Reshape for multi-head attention: (R, heads, 14, head_dim)
            q = q.view(n_residues, self.max_slots, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(n_residues, self.max_slots, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(n_residues, self.max_slots, self.num_heads, self.head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (R, heads, 14, 14)
            attn = torch.softmax(attn, dim=-1)
            attn = layer["attn_dropout"](attn)

            out = torch.matmul(attn, v)  # (R, heads, 14, head_dim)
            out = out.transpose(1, 2).contiguous().view(n_residues, self.max_slots, -1)
            h = h + layer["out_proj"](out)

            # Pre-norm FFN
            h = h + layer["ffn"](layer["norm2"](h))

        return h.reshape(n_residues * self.max_slots, -1)  # (N_sc, D)


class DirectionalSlotAttention(nn.Module):
    """
    One-way attention from distal "scout" slots back to proximal receiver slots.

    Per residue, the n_scouts highest-indexed REAL slots (P(real) > 0.5) are
    designated as scouts. These are the most distal atoms -- they escape the CA
    cloud first during reverse sampling due to higher flow power and carry spatial
    information about the target pocket. All other real slots are receivers that
    attend to scouts via cross-attention.

    Scout assignment is dynamic per residue: a small amino acid like alanine with
    5 real atoms gets scouts from its own distal real slots (e.g. slots 2-4),
    not from ghost slots at indices 11-13 that are stuck at CA.

    Messages are gated by per-scout detached P(real) in the attention mask, so
    only confident scout signals propagate.

    Zero-init residual ensures warm-start compatibility.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        max_slots: int = 14,
        n_scouts: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for directional slot attention")
        if not (1 <= n_scouts <= max_slots - 1):
            raise ValueError(f"n_scouts must be in [1, {max_slots - 1}], got {n_scouts}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_slots = max_slots
        self.n_scouts = n_scouts
        self.scale = self.head_dim**-0.5

        # Slot position embeddings (shared for Q and K)
        self.slot_pos_embed = nn.Embedding(max_slots, hidden_dim)

        # Cross-attention: receivers (queries) attend to scouts (keys/values)
        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN for receiver slots
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Zero-init output projections for identity-start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(
        self,
        h: torch.Tensor,
        n_residues: int,
        occupancy_logits: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply dynamic scout-to-receiver cross-attention within each residue.

        Parameters
        ----------
        h : torch.Tensor
            Flat slot features (N_sc, hidden_dim) where N_sc = n_residues * max_slots.
        n_residues : int
            Number of valid residues.
        occupancy_logits : torch.Tensor or None
            Per-slot occupancy logits (N_sc, 1) from the occupancy head. Detached.
            Used for P(real) gating in attention and (when gt_mask is None) for
            determining which slots are real. If None, all slots treated as real.
        gt_mask : torch.Tensor or None
            Ground-truth sidechain mask (N_sc,) with 1=real, 0=ghost. Passed during
            training for stable scout assignment. During sampling this is None and
            scout assignment falls back to predicted P(real) > 0.5.

        Returns
        -------
        torch.Tensor
            Updated slot features, same shape. Only receiver (non-scout) slots are modified.
        """
        if n_residues == 0:
            return h

        n_slots = self.max_slots
        dim = self.hidden_dim
        h_3d = h.reshape(n_residues, n_slots, dim)  # (R, 14, D)
        device = h.device

        # P(real) per slot -- always needed for attention gating
        if occupancy_logits is not None:
            preal = torch.sigmoid(occupancy_logits.detach().reshape(n_residues, n_slots))
        else:
            preal = torch.ones(n_residues, n_slots, device=device)

        # Determine which slots are "real" for scout/receiver assignment.
        # Training: use GT mask for stable, correct assignment.
        # Sampling: fall back to predicted P(real) > 0.5.
        if gt_mask is not None:
            is_real = gt_mask.reshape(n_residues, n_slots).bool()
        else:
            is_real = preal > 0.5

        # For each residue, find the n_scouts highest-indexed REAL slots.
        # Scouts = the most distal real atoms; receivers = everything else.
        # Clamp: need at least 1 receiver, so effective scouts = min(n_scouts, n_real - 1).
        # Residues with 0 or 1 real atoms get 0 scouts (nothing useful to cross-attend).
        slot_indices = torch.arange(n_slots, device=device).expand(n_residues, n_slots)
        real_indices = torch.where(is_real, slot_indices, torch.full_like(slot_indices, -1))
        n_real = is_real.sum(dim=1)  # (n_residues,)
        effective_n = n_real.clamp(min=1) - 1  # reserve at least 1 receiver
        effective_n = effective_n.clamp(max=self.n_scouts)  # cap at n_scouts

        # Use topk to find the highest-indexed real slots per residue
        topk_vals, _ = real_indices.topk(self.n_scouts, dim=1)  # sorted descending; -1 for missing
        is_scout = torch.zeros(n_residues, n_slots, dtype=torch.bool, device=device)
        for k in range(self.n_scouts):
            scout_idx = topk_vals[:, k]
            # Only mark as scout if (a) it's a real slot and (b) k < effective_n for this residue
            valid = (scout_idx >= 0) & (k < effective_n)
            is_scout[torch.arange(n_residues, device=device)[valid], scout_idx[valid]] = True

        is_receiver = is_real & ~is_scout  # real non-scout slots

        # Pad to fixed sizes for batched attention
        max_recv = n_slots - self.n_scouts

        # Gather scout and receiver indices per residue, padded to fixed length
        scout_flat_idx = []
        recv_flat_idx = []
        scout_mask = torch.zeros(n_residues, self.n_scouts, dtype=torch.bool, device=device)
        recv_mask = torch.zeros(n_residues, max_recv, dtype=torch.bool, device=device)

        for r in range(n_residues):
            s_idx = is_scout[r].nonzero(as_tuple=False).squeeze(-1)
            r_idx = is_receiver[r].nonzero(as_tuple=False).squeeze(-1)

            n_s = s_idx.shape[0]
            n_r = r_idx.shape[0]
            s_padded = torch.zeros(self.n_scouts, dtype=torch.long, device=device)
            r_padded = torch.zeros(max_recv, dtype=torch.long, device=device)
            s_padded[:n_s] = s_idx
            r_padded[:n_r] = r_idx
            scout_flat_idx.append(s_padded)
            recv_flat_idx.append(r_padded)
            scout_mask[r, :n_s] = True
            recv_mask[r, :n_r] = True

        scout_idx_t = torch.stack(scout_flat_idx)  # (n_residues, n_scouts)
        recv_idx_t = torch.stack(recv_flat_idx)  # (n_residues, max_recv)

        # Gather features
        h_scouts = torch.gather(h_3d, 1, scout_idx_t.unsqueeze(-1).expand(-1, -1, dim))
        h_recv = torch.gather(h_3d, 1, recv_idx_t.unsqueeze(-1).expand(-1, -1, dim))

        # Gather position embeddings
        pos = self.slot_pos_embed.weight  # (n_slots, dim)
        pos_scouts = pos[scout_idx_t]
        pos_recv = pos[recv_idx_t]

        # Gather P(real) for scouts -- used to weight attention
        preal_scouts = torch.gather(preal, 1, scout_idx_t)

        # Pre-norm cross-attention: receivers attend to scouts
        h_recv_norm = self.norm(h_recv)
        h_scouts_norm = self.norm(h_scouts)

        q = self.q_proj(h_recv_norm + pos_recv)
        k = self.k_proj(h_scouts_norm + pos_scouts)
        v = self.v_proj(h_scouts_norm)  # no position on V

        # Multi-head reshape
        n_h = self.num_heads
        d_h = self.head_dim
        q = q.view(n_residues, max_recv, n_h, d_h).transpose(1, 2)
        k = k.view(n_residues, self.n_scouts, n_h, d_h).transpose(1, 2)
        v = v.view(n_residues, self.n_scouts, n_h, d_h).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (R, H, max_recv, n_scouts)

        # Mask out invalid scout padding positions
        scout_attn_mask = scout_mask.unsqueeze(1).unsqueeze(2)  # (R, 1, 1, n_scouts)
        attn = attn.masked_fill(~scout_attn_mask, float("-inf"))

        # Soft P(real) bias: log P(real) as additive logit so lower-confidence scouts contribute less
        preal_bias = preal_scouts.unsqueeze(1).unsqueeze(2)  # (R, 1, 1, n_scouts)
        attn = attn + preal_bias.log().clamp(min=-10)

        attn = torch.softmax(attn, dim=-1)
        # NaN guard: residues with no valid scouts get all-NaN rows -> zero them
        attn = attn.nan_to_num(0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (R, H, max_recv, d_h)
        out = out.transpose(1, 2).contiguous().view(n_residues, max_recv, dim)
        out = self.out_proj(out)

        # Zero out padded receiver positions
        recv_mask_3d = recv_mask.unsqueeze(-1)  # (R, max_recv, 1)
        out = out * recv_mask_3d

        # Residual + FFN on receiver slots (only valid positions)
        h_recv_updated = h_recv + out
        h_recv_updated = h_recv_updated + self.ffn(self.norm_ffn(h_recv_updated))

        # For padding rows, restore original gathered features so scatter is harmless.
        # Without this, FFN transforms padding features and overwrites slot 0.
        h_recv_updated = torch.where(recv_mask_3d, h_recv_updated, h_recv)

        # Scatter updated receiver features back into full tensor
        # Padded indices (0) still scatter, but with original h_recv values -> no corruption
        h_out = h_3d.clone()
        h_out.scatter_(1, recv_idx_t.unsqueeze(-1).expand(-1, -1, dim), h_recv_updated)

        return h_out.reshape(n_residues * n_slots, dim)


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


def corrupt_neighbor_x0(
    neighbor_coords: torch.Tensor,
    neighbor_valid: torch.Tensor,
    ca_coords: torch.Tensor,
    add_prob: float,
    drop_prob: float,
    noise_prob: float,
    noise_std: float,
    disconnect_prob: float = 0.0,
    radius: float = 8.0,
    generator: torch.Generator | None = None,
    protect_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TRAINING-side corruption of the neighbour-x0 packing signal ("weaken the neighbour signal").

    Makes the neighbour clean-endpoint context deliberately unreliable during training so the model
    learns to DISCOUNT it rather than over-trust it. Three coordinate-space corruptions, all keyed on
    the Cα=ghost / away-from-Cα=real convention (existence is encoded by POSITION, not just the mask):

    1. **REMOVE** (per REAL slot, chance ``drop_prob``): move the atom's coordinate TO the residue's Cα
       and set ``valid=False`` -- it becomes a ghost, encoded by sitting at Cα.
    2. **ADD** (per GHOST slot, chance ``add_prob``): move the ghost atom (at Cα) to a plausible REAL
       position IN THE DIRECTION of that residue's existing real-atom cloud -- along the ray from Cα
       toward the centroid of the residue's real atoms, at a random radius within the cloud's spread
       (``>1.5`` Å from Cα) plus a small jitter; set ``valid=True``. Residues with NO real atoms have
       no cloud to point toward, so the add is SKIPPED for them (never a random direction).
    3. **NOISE** (whole-call Bernoulli, chance ``noise_prob``): if it fires, add ``N(0, noise_std²)``
       to every currently-valid coordinate (ghost slots, sitting at Cα, are left alone). Applied AFTER
       remove/add so ADD-ed atoms are noised too and blend in with the real ones.

    **Track DESYNC** (chance ``disconnect_prob``, per add/remove op). Normally an add/remove moves BOTH
    the coordinate AND the existence (``valid``) together, so the slot stays self-consistent
    (real = away-from-Cα + ``valid=True``; ghost = at-Cα + ``valid=False``). With chance
    ``disconnect_prob`` the op is applied to ONLY ONE track (coord-only vs existence-only at 50/50),
    leaving the slot in a MISMATCHED state -- teaching the model the two tracks can disagree. E.g. a
    coord-only add moves the coord to the cloud position but keeps ``valid=False``; an existence-only
    remove flips ``valid=False`` but leaves the coord away from Cα. ``0.0`` => always consistent.

    **Byte-identical guard.** If ``add_prob == 0`` and ``drop_prob == 0`` and ``noise_prob == 0`` the
    inputs are returned UNCHANGED and NO RNG is touched (early return before any ``torch.rand``/``randn``).
    ``disconnect_prob`` alone drives no ops, so it is not part of the guard; with ``disconnect_prob == 0``
    the add/remove RNG stream is bit-identical to the no-desync path (the desync draws are skipped).

    Parameters
    ----------
    neighbor_coords : torch.Tensor
        Per-residue neighbour atom positions (x0 estimates), shape (L, S, 3).
    neighbor_valid : torch.Tensor
        Bool mask of which neighbour slots hold a real atom, shape (L, S).
    ca_coords : torch.Tensor
        Per-residue Cα positions, shape (L, 3). ADD ray origin / REMOVE target.
    add_prob, drop_prob : float
        Per-slot add / remove probabilities.
    noise_prob : float
        Whole-call probability that the coordinate noise is applied at all.
    noise_std : float
        Gaussian coordinate-noise std (Å), used only when the noise fires.
    disconnect_prob : float
        Per-op chance to desync the coord and existence tracks (apply the op to only one track).
    radius : float
        Fallback outer radius (Å) if a residue's real-atom spread is degenerate.
    generator : torch.Generator, optional
        Seedable RNG source for ALL randomness (reproducible / resumable).
    protect_mask : torch.Tensor, optional
        Per-residue (L,) bool. Residues where this is True are left BYTE-UNTOUCHED on both the coord
        and existence tracks -- no REMOVE, no ADD, and the NOISE step skips them. Used to shield the
        clean/pinned GT-context residues (``trust == 1.0``) so corruption only ever hits the designed /
        co-generated neighbours the model is meant to learn to discount. ``None`` protects nothing
        (historical behaviour).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(coords, valid)`` -- corrupted coordinates (L, S, 3) and updated bool validity mask (L, S).
    """
    # BYTE-IDENTICAL GUARD: nothing enabled -> return inputs untouched, consume no RNG.
    if add_prob == 0.0 and drop_prob == 0.0 and noise_prob == 0.0:
        return neighbor_coords, neighbor_valid

    # FIX D1: cheap scalar range guard on the forward/env-resolved path. The launch validator covers the
    # constructed hparams, but ANY caller reaching here (e.g. a stray env override) must not pass a
    # nonsensical probability/std. Placed AFTER the byte-identical early-return so the all-zero no-op path
    # stays free of any check.
    for _name, _p in (
        ("add_prob", add_prob),
        ("drop_prob", drop_prob),
        ("noise_prob", noise_prob),
        ("disconnect_prob", disconnect_prob),
    ):
        if not math.isfinite(_p) or _p < 0.0 or _p > 1.0:
            raise ValueError(f"corrupt_neighbor_x0: {_name}={_p} must be finite and in [0, 1].")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError(f"corrupt_neighbor_x0: noise_std={noise_std} must be finite and >= 0.")

    device = neighbor_coords.device
    dtype = neighbor_coords.dtype
    n_res, n_slots, _ = neighbor_coords.shape

    coords = neighbor_coords.clone()
    valid = neighbor_valid.bool().clone()
    ca = ca_coords.to(dtype)  # (L, 3)

    if n_res == 0 or n_slots == 0:
        return coords, valid

    # FIX C: per-residue protection mask. Residues flagged True are clean/pinned GT context (trust==1.0)
    # and must be left byte-untouched -- corruption only ever hits the designed / co-generated neighbours.
    # None => protect nothing (historical behaviour).
    if protect_mask is not None:
        protect_row = protect_mask.to(device=device, dtype=torch.bool).reshape(n_res)  # (L,)
    else:
        protect_row = torch.zeros(n_res, dtype=torch.bool, device=device)  # (L,)

    def _rand(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, device=device, dtype=dtype, generator=generator)

    def _randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    # Cloud geometry from the ORIGINAL real atoms (computed before any mutation so REMOVE-to-Cα does
    # not pollute the direction/spread the ADD points along).
    orig_valid = neighbor_valid.bool()  # (L, S)
    orig_valid_f = orig_valid.to(dtype)  # (L, S)
    n_real = orig_valid_f.sum(dim=1)  # (L,)
    has_real = n_real > 0  # (L,)

    centroid = (coords * orig_valid_f.unsqueeze(-1)).sum(dim=1) / n_real.clamp(min=1.0).unsqueeze(-1)  # (L, 3)
    cloud_dir = centroid - ca  # (L, 3)
    cloud_dir = cloud_dir / cloud_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # unit ray Cα -> cloud
    # Radial spread: farthest real atom from Cα per residue, floored so ADD lands > 1.5 Å from Cα.
    real_dist = (coords - ca.unsqueeze(1)).norm(dim=-1)  # (L, S)
    real_dist = torch.where(orig_valid, real_dist, torch.zeros_like(real_dist))
    r_max = real_dist.max(dim=1).values  # (L,)
    r_max = torch.clamp(r_max, min=1.5 + 1e-3, max=float(2.0 * radius))

    ca_row = ca.unsqueeze(1).expand(n_res, n_slots, 3)  # (L, S, 3)

    def _split_desync(op_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split an op mask into (coord-track apply, existence-track apply).

        With ``disconnect_prob`` an op desyncs: applied to ONLY one track (coord-only vs existence-only
        at 50/50). Otherwise both tracks move together. Draws are skipped entirely when
        ``disconnect_prob == 0`` so the RNG stream stays bit-identical to the consistent-only path.
        """
        if disconnect_prob <= 0.0:
            return op_mask, op_mask
        desync = op_mask & (_rand((n_res, n_slots)) < disconnect_prob)  # ops that split
        side_coord = desync & (_rand((n_res, n_slots)) < 0.5)  # coord-only among desynced
        side_exist = desync & ~side_coord  # existence-only among desynced
        coord_apply = op_mask & ~side_exist  # move coord unless this op is existence-only
        exist_apply = op_mask & ~side_coord  # flip existence unless this op is coord-only
        return coord_apply, exist_apply

    # --- REMOVE: real -> Cα, ghost it. ---
    if drop_prob > 0.0:
        drop_mask = valid & (_rand((n_res, n_slots)) < drop_prob)  # (L, S)
        drop_mask = drop_mask & ~protect_row.unsqueeze(1)  # FIX C: never touch protected residues
        drop_coord, drop_exist = _split_desync(drop_mask)
        coords = torch.where(drop_coord.unsqueeze(-1), ca_row, coords)
        valid = valid & ~drop_exist

    # --- ADD: ghost (at Cα) -> plausible real position along the residue's cloud ray. ---
    if add_prob > 0.0:
        ghost = ~valid  # after REMOVE
        add_mask = ghost & (_rand((n_res, n_slots)) < add_prob) & has_real.unsqueeze(1)  # skip no-cloud residues
        add_mask = add_mask & ~protect_row.unsqueeze(1)  # FIX C: never touch protected residues
        _u = _rand((n_res, n_slots))  # per-slot radius fraction
        radius_slot = 1.5 + (r_max.unsqueeze(1) - 1.5).clamp(min=0.0) * _u  # (L, S) in [1.5, r_max]
        jitter = _randn((n_res, n_slots, 3)) * float(_NX0_ADD_JITTER_STD)
        added = ca.unsqueeze(1) + cloud_dir.unsqueeze(1) * radius_slot.unsqueeze(-1) + jitter  # (L, S, 3)
        add_coord, add_exist = _split_desync(add_mask)
        coords = torch.where(add_coord.unsqueeze(-1), added, coords)
        valid = valid | add_exist

    # --- NOISE: probabilistic whole-call Gaussian on the FINAL valid set (added atoms included). ---
    if noise_prob > 0.0:
        fire = _rand(()) < noise_prob  # single Bernoulli for the whole call
        if bool(fire) and noise_std > 0.0:
            noise = _randn((n_res, n_slots, 3)) * float(noise_std)
            # FIX C: noise only unprotected, currently-valid slots (protected residues stay byte-exact).
            noise_slots = valid & ~protect_row.unsqueeze(1)  # (L, S)
            coords = torch.where(noise_slots.unsqueeze(-1), coords + noise, coords)

    return coords, valid


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


def bond_geom_descriptor(
    coords: torch.Tensor,
    valid: torch.Tensor,
    n_bins: int = BOND_GEOM_HIST_BINS,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Per-residue, distance-weighted internal-geometry descriptor ("bond-inject") of a predicted-x0 cloud.

    ``coords`` (L, S, 3) is a PREDICTED CLEAN side-chain x0 (freshest recycle / same-step pass); ``valid``
    (L, S) marks REAL atom slots (graph-valid AND predicted-existent -- ghost/Cα-collapsed slots are
    excluded by the caller). Because residue identity (hence bond topology) is unknown at generation,
    neighbours are selected by GEOMETRY, not a bond mask: pairs/triples are weighted by ``exp(-d/2σ)`` so
    near (bond-length-scale) atoms dominate, approximating the bonded angles/lengths the losses supervise.
    Identity-free and computed identically at train and inference (no train/eval divergence).

    Returns (L, ``BOND_GEOM_DESC_DIM``) = two blocks concatenated:
      * ANGLES ``[w-mean cos, w-std cos, soft-hist over n_bins in [-1,1]]`` (SAME angle def as
        :func:`training.bond_angle_loss`); zeroed for residues with <3 valid atoms.
      * LENGTHS ``[w-mean d, w-std d, soft-hist over n_bins in [MIN,MAX] Å]`` over unordered valid pairs;
        zeroed for residues with <2 valid atoms.
    Both blocks are functions of edge directions/magnitudes only (translation- and rotation-invariant) ->
    the downstream inject stays SE(3)-invariant.
    """
    L, S, _ = coords.shape
    dtype = coords.dtype
    e = coords.unsqueeze(1) - coords.unsqueeze(2)  # (L, center j, other i, 3): x_i - x_j
    d = e.norm(dim=-1)  # (L, S, S)
    u = e / d.clamp(min=1e-6).unsqueeze(-1)  # unit edges
    cos = torch.einsum("ljid,ljkd->ljik", u, u).clamp(-1.0, 1.0)  # (L, j, i, k)
    vv = valid.to(torch.bool)
    idx = torch.arange(S, device=coords.device)
    # --- ANGLE block: unordered pairs (i<k) of OTHER valid atoms around a valid center j ---
    tri = (
        vv.view(L, S, 1, 1)
        & vv.view(L, 1, S, 1)
        & vv.view(L, 1, 1, S)
        & (idx.view(1, S, 1, 1) != idx.view(1, 1, S, 1))  # i != j
        & (idx.view(1, S, 1, 1) != idx.view(1, 1, 1, S))  # k != j
        & (idx.view(1, 1, S, 1) < idx.view(1, 1, 1, S))  # i < k (unordered)
    )
    w = torch.exp(-(d.view(L, S, S, 1) + d.view(L, S, 1, S)) / (2.0 * sigma)) * tri.to(dtype)
    inv = 1.0 / w.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    a_mean = (w * cos).sum(dim=(1, 2, 3)) * inv  # (L,)
    a_std = ((w * (cos - a_mean.view(L, 1, 1, 1)) ** 2).sum(dim=(1, 2, 3)) * inv).clamp(min=0.0).sqrt()
    a_centers = torch.linspace(-1.0, 1.0, n_bins, device=coords.device, dtype=dtype)
    a_bw = 2.0 / n_bins
    a_hb = torch.exp(-((cos.unsqueeze(-1) - a_centers) ** 2) / (2.0 * a_bw * a_bw))  # (L, j, i, k, n_bins)
    a_hist = (w.unsqueeze(-1) * a_hb).sum(dim=(1, 2, 3)) * inv.view(L, 1)  # (L, n_bins)
    a_desc = torch.cat([a_mean.view(L, 1), a_std.view(L, 1), a_hist], dim=-1)  # (L, 2 + n_bins)
    a_desc = torch.where((vv.sum(dim=-1) >= 3).view(L, 1), a_desc, torch.zeros_like(a_desc))
    # --- LENGTH block: unordered valid pairs (i<k), distances weighted toward bonded scale ---
    pair = vv.view(L, S, 1) & vv.view(L, 1, S) & (idx.view(1, S, 1) < idx.view(1, 1, S))  # (L, S, S)
    wl = torch.exp(-d / (2.0 * sigma)) * pair.to(dtype)  # (L, S, S)
    invl = 1.0 / wl.sum(dim=(1, 2)).clamp(min=1e-6)
    l_mean = (wl * d).sum(dim=(1, 2)) * invl  # (L,)
    l_std = ((wl * (d - l_mean.view(L, 1, 1)) ** 2).sum(dim=(1, 2)) * invl).clamp(min=0.0).sqrt()
    l_centers = torch.linspace(BOND_LENGTH_HIST_MIN, BOND_LENGTH_HIST_MAX, n_bins, device=coords.device, dtype=dtype)
    l_bw = (BOND_LENGTH_HIST_MAX - BOND_LENGTH_HIST_MIN) / n_bins
    l_hb = torch.exp(-((d.unsqueeze(-1) - l_centers) ** 2) / (2.0 * l_bw * l_bw))  # (L, S, S, n_bins)
    l_hist = (wl.unsqueeze(-1) * l_hb).sum(dim=(1, 2)) * invl.view(L, 1)  # (L, n_bins)
    l_desc = torch.cat([l_mean.view(L, 1), l_std.view(L, 1), l_hist], dim=-1)  # (L, 2 + n_bins)
    l_desc = torch.where((vv.sum(dim=-1) >= 2).view(L, 1), l_desc, torch.zeros_like(l_desc))
    return torch.cat([a_desc, l_desc], dim=-1)  # (L, BOND_GEOM_DESC_DIM)


class SidechainDenoiser(nn.Module):
    """
    EGNN-based denoiser for side-chain atom clouds.

    Takes noised side-chain coordinates and predicts the noise (or denoised coords),
    conditioned on:
    - Fixed backbone coordinates (included in graph as fixed nodes)
    - Diffusion timestep
    - Target chain atoms (included in graph as fixed nodes for pocket awareness)

    The EGNN graph includes three types of edges:
    - Intra-residue: within the same binder residue
    - Inter-residue: across different binder residues
    - Binder-target: between binder and target atoms

    Target conditioning uses:
    - Multiple cross-attention layers (not just one)
    - FiLM modulation for stronger conditioning signal
    - Per-atom target features in graph edges

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for features.
    num_layers : int
        Number of EGNN layers.
    time_embed_dim : int
        Dimension for timestep embeddings.
    max_sidechain_atoms : int
        Maximum number of side-chain atoms per residue.
    use_target_conditioning : bool
        Whether to use target chain conditioning.
    num_cross_attn_layers : int
        Number of cross-attention layers for target conditioning. Default 3.
    num_cross_attn_heads : int
        Number of attention heads per cross-attention layer. Default 8.
    use_film : bool
        Whether to use FiLM modulation for target conditioning.
    use_bidirectional_target_conditioning : bool
        Whether to update target residue features with peptide backbone context
        before conditioning backbone features on target.
    use_sidechain_target_residue_attention : bool
        Whether to directly inject target residue context into sidechain particle
        features before the SE(3) graph update.
    edge_embed_dim : int
        Dimension for edge type embeddings.
    intra_residue_cutoff : float
        Distance cutoff for intra-residue edges.
    inter_residue_cutoff : float
        Distance cutoff for inter-residue edges.
    sidechain_target_cutoff : float
        Distance cutoff for SC->target edges.
    backbone_target_cutoff : float
        Distance cutoff for CA->target edges (split cutoff mode).
    ca_ca_prefilter : float
        CA-CA distance prefilter for binder-target edges.
    dropout : float
        Dropout rate for SE(3) transformer attention and FFN layers.
    """

    #: Process-wide latch so the neighbour-x0 training-corruption prints its "active" marker only once.
    _nx0_corrupt_announced: bool = False

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_embed_dim: int = 128,
        max_sidechain_atoms: int = 14,
        count_embed_mode: str = "linear",  # "linear" (Linear(1,d) ramp) | "ordinal" (Embedding per count)
        use_target_conditioning: bool = True,
        num_cross_attn_layers: int = 3,  # Deep target conditioning
        num_cross_attn_heads: int = 8,  # Multi-head target attention
        use_film: bool = True,  # FiLM: multiplicative target gating
        use_bidirectional_target_conditioning: bool = True,
        use_sidechain_target_residue_attention: bool = True,
        target_condition_scale: float = 1.0,
        cluster_target_condition_scale: float = 1.0,
        edge_embed_dim: int = 16,
        intra_residue_cutoff: float = 8.0,
        inter_residue_cutoff: float = 8.0,
        sidechain_target_cutoff: float = 12.0,
        backbone_target_cutoff: float = 25.0,
        # SE(3) RBF span selector. False (default) => legacy 0-10 A RBF span (rbf_max_dist=None below);
        # True => widen the span to backbone_target_cutoff. See the IFD-level docstring above.
        rbf_span_to_cutoff: bool = False,
        ca_ca_prefilter: float = 20.0,
        dropout: float = 0.0,
        use_jackie: bool = False,
        preal_gate_target: str = "none",
        use_slot_attention: bool = False,
        num_slot_attn_layers: int = 2,
        num_slot_attn_heads: int = 4,
        use_directional_slot_attention: bool = False,
        n_scouts: int = 3,
        num_element_classes: int = NUM_ELEMENT_TYPES,
        use_element_velocity_coupling: bool = False,
        num_timesteps: int = 250,  # T for tau = t/T normalization used by non-EVC timestep ramps
        use_ca_dist_element_feature: bool = False,
        use_count_velocity_coupling: bool = False,
        use_bond_attention: bool = False,
        zero_init_bond_attention: bool = False,  # bond-attn identity-start (safe retrofit onto pre-bond-attn ckpts)
        use_valence_demand: bool = False,
        use_cross_residue_packing: bool = False,
        cross_residue_packing_spatial: bool = False,  # pair residues by SPATIAL proximity instead of i±1
        cross_residue_packing_k: int = 4,  # k nearest residues per query (spatial mode)
        cross_residue_packing_radius: float = 10.0,  # pseudo-Cbeta cutoff, <=0 disables the radius gate
        cross_residue_packing_min_seq_sep: int = 2,  # minimum |i-j|; 2 excludes i±1 (matches eval Buried)
        use_neighbor_x0_packing: bool = False,  # condition packing on NEIGHBOURS' predicted x0 (anti-self)
        neighbor_x0_packing_radius: float = 8.0,  # outer shell radius for the neighbour-x0 context features
        neighbor_x0_highnoise_weight: float = 1.0,  # amplify the neighbour-x0 residual at high noise; 1.0 = exact no-op
        # TRAINING-side corruption of the neighbour-x0 packing signal (weaken-the-neighbour-signal ablation).
        # All 0 => byte-identical (no corruption, no RNG). See corrupt_neighbor_x0. Applied ONLY when training.
        neighbor_x0_corrupt_add_prob: float = 0.0,  # per-GHOST-slot chance to ADD a phantom atom along the cloud ray
        neighbor_x0_corrupt_drop_prob: float = 0.0,  # per-REAL-slot chance to REMOVE (ghost to Cα) an atom
        neighbor_x0_corrupt_noise_prob: float = 0.0,  # whole-call chance the coord noise is applied at all
        neighbor_x0_corrupt_coord_noise: float = 0.0,  # Gaussian coord-noise std (Angstrom) when noise fires
        neighbor_x0_corrupt_disconnect_prob: float = 0.0,  # per-op chance to DESYNC coord vs existence tracks
        use_shape_prior: bool = False,
        shape_prior_n_anchors: int = 4,
        use_bond_co_diffusion: bool = False,
        use_coord_self_conditioning: bool = False,
        use_plan_latent: bool = False,
        use_interaction_intent: bool = False,
        interaction_intent_num_classes: int = 4,
        interaction_intent_velocity_bias: bool = True,
        use_chirality: bool = False,
        num_edge_types: int = 3,
        graph_num_edge_types: int | None = None,
        use_backbone_dihedral: bool = False,  # backbone dihedral graft (zero-init add)
        use_burial_feature: bool = False,  # Cbeta-burial graft (zero-init add)
        use_residue_frame_stream: bool = False,  # DRAFT: residue-level frame-aware stream (zero-init inject)
        residue_frame_layers: int = 2,  # #layers in the residue frame-attention stack
        use_residue_frame_deep_inject: bool = False,  # DRAFT: also inject the residue latent into EVERY SE3 layer
        # volumetric deep-inject. When on, the (t-independent) VolumetricOccupancyHead's per-residue
        # latent `vol_hidden` (computed once at the InverseFoldingDiffusion level) is deep-injected into EVERY
        # SE(3) layer, mirroring residue_frame_deep_proj. The head lives one level up, so the latent is passed
        # DOWN into forward(vol_hidden=...); this flag only builds the per-layer projections + applies them.
        # The IFD ANDs this with use_volumetric_head before it reaches here, so it is already effective-gated.
        use_volumetric_deep_inject: bool = False,
        # TARGET deep-inject (run10): reinforce sc_target_attn (sidechain->target cross-attention) at EVERY
        # SE(3) layer, not just layer 0. Local to this module (sc_target_attn is computed in forward), so no
        # IFD-side latent threading is needed; applied on sc nodes only. Zero-init => byte-identical off.
        use_target_deep_inject: bool = False,
        use_backbone_deep_inject: bool = False,
        frame_v2_deep_inject_detach: bool = False,
        # x0 BOND-INJECT deep-inject (angle + length): featurize the PREDICTED CLEAN x0 side-chain cloud's
        # internal bond angles AND bonded-pair lengths (bond_geom_descriptor) -> per-residue latent
        # (bond_angle_inject_encoder) -> ZERO-INIT per-SE(3)-layer projection (bond_angle_deep_proj), mirroring
        # residue_frame_v2_deep_proj. Source = bond_prev_x0 (detached predicted x0); absent => inject skipped =>
        # inert. Byte-identical off. Orthogonal to bond_angle_loss + bond_length_loss (which supervise the
        # angles/lengths); this feeds predicted internal geometry BACK.
        use_bond_angle_deep_inject: bool = False,
        # volumetric -> existence (element-track) coupling. When on, the head's per-residue `vol_hidden`
        # latent (computed once at the IFD level, threaded DOWN into forward(vol_hidden=...)) modulates the
        # element track's per-slot PAD/GHOST (existence) logit via a single ZERO-INIT projection. The IFD ANDs
        # this with use_volumetric_head before it reaches here, so it is already effective-gated. Applied on
        # BOTH the training forward AND the sampling reverse loop (the element head runs at both).
        use_volumetric_existence_coupling: bool = False,
        # v2 residue-frame graph (binder+target residues, orientation-only heads). Independent
        # of (and mutually exclusive with) the v1 stream above. Off => not built, byte-identical.
        use_residue_frame_stream_v2: bool = False,
        residue_frame_v2_layers: int = 2,  # #layers in the v2 cross-attention stack
        frame_v2_clean_input: bool = True,  # True: clean geometry-only binder embed; False: pocket-conditioned backbone_features
        # NEW (v2 frame deep-inject): per-SE(3)-layer LIVE-gradient deep-injection of the v2 residue-frame
        # stream's per-residue latent into the atom SE(3) transformer. Mirror of residue_frame_deep_proj but
        # gated on the v2 stream and -- critically -- LIVE (source NOT detached; see the module def below). The
        # IFD ANDs this with use_residue_frame_stream_v2 before it reaches here, so it is already effective-gated.
        use_residue_frame_v2_deep_inject: bool = False,
        # supervised stereochemistry (e3-sign) head on the v2 frame graph. Only meaningful with
        # use_residue_frame_stream_v2 (the IFD ANDs the two before it reaches here). Off => head not built.
        use_stereochem_head: bool = False,
        # t-resolution head. A t-DEPENDENT stereochem prediction that reads the CURRENT (noised)
        # side-chain atom cloud's in-frame out-of-plane (e3) configuration and blends it with the STATIC
        # Chunk-4a prior: P(D)_t = (1 - w(t))·P(D)_prior + w(t)·sigmoid(atom_logit). The IFD ANDs this with
        # use_stereochem_head (which itself ANDs the v2 stream) before it reaches here, so it is only ever
        # True when the static stereo head is on. Off => head not built, byte-identical.
        use_stereochem_t_resolution: bool = False,
        # t-resolution COORD-FLOW FEEDBACK. The resolved P(D)_t () steers the coordinate flow
        # to pull atoms onto the chosen L/D (e3) face. Two ZERO-INIT channels under one flag: (1) a learned
        # per-SE(3)-layer deep-inject (`stereo_feedback_deep_proj`, exact mirror of volumetric_deep_proj) that
        # scatters a face_pref-derived per-residue latent into each layer's invariant node features, and (2) an
        # explicit, equivariant e3-directional velocity bias (face_pref·e3, gated by a zero-init learnable scale)
        # that GUARANTEES atoms move toward the resolved face. The IFD ANDs this with use_stereochem_t_resolution
        # (which itself needs the static stereo head + v2 stream) before it reaches here, so it is only ever True
        # when P(D)_t exists to steer with. Off => no modules built, no RNG advance, byte-identical.
        use_stereochem_t_resolution_feedback: bool = False,
        activation_checkpointing: bool = False,
        activation_checkpoint_stride: float = 1.0,  # >1 = selective: checkpoint every Nth SE3 layer (float-OK, e.g. 1.5)
        use_global_latent_matching: bool = False,  # clean-context stream + per-residue latent pool (ported)
        global_latent_embed_dim: int = 64,  # z_pred / z_gt width (read off the QuPID lookup when wired)
        global_latent_pool_hidden: int = 128,  # CLS-pool width (divisible by n_heads)
        global_latent_pool_radius: float = 10.0,  # Cα neighbourhood radius for per-residue pooling
        global_latent_confidence: bool = True,  # confidence probe + gated FiLM feedback
        global_latent_confidence_hidden: int = 64,  # confidence MLP hidden width
        global_latent_condition_main_stream: bool = True,  # FiLM the gated latent into generated atoms
        global_latent_film_layers: str = "last",  # where to inject the FiLM feedback (this port: "last")
        graft_init_std: float = 0.0,  # >0: init the zero-init graft INJECTION modules with N(0, std) instead of zeros
    ):
        super().__init__()

        # Category-3 injection-graft init. Default 0.0 => zeros (bit-exact current behaviour). When > 0,
        # the zero-init INJECTION projections (neighbor_x0_proj, t_cond_delta_proj, recycle_index_proj,
        # clean_inject_proj, latent_film, and -- when residue-frame deep-inject is on --
        # residue_frame_deep_proj) start from N(0, graft_init_std) instead -- testing whether a non-zero
        # init helps the injection escape the zero-basin. Only these injection modules are
        # touched; encoders (clean_context_stream / latent_pool_head) and every other module keep their
        # normal init. On a --resume-weights run the checkpoint overwrites any module it contains, so this
        # only affects FRESH modules (e.g. a v35 resume: the latent injection is fresh -> gets the non-zero
        # init; the packing projections are in the ckpt -> overwritten by their trained values, as expected).

        # RNG ISOLATION: the graft draws come from a DEDICATED, fixed-seeded torch.Generator, never the
        # global stream. So enabling graft_init_std does NOT advance the global RNG and leaves every
        # NON-injection module (encoders, trunk, heads -- constructed before AND after these grafts)
        # byte-identical to the zeros build under the same seed. The draw is still deterministic (fixed
        # sub-seed) and independent across the five modules.
        _graft_std = float(graft_init_std)
        _graft_gen = torch.Generator()
        _graft_gen.manual_seed(0x6A17)  # fixed constant -> deterministic, isolated from the global RNG

        def _init_graft_weight(w: torch.Tensor) -> None:
            if _graft_std > 0.0:
                with torch.no_grad():
                    w.normal_(0.0, _graft_std, generator=_graft_gen)
            else:
                nn.init.zeros_(w)

        self.use_backbone_dihedral = use_backbone_dihedral
        self.use_burial_feature = use_burial_feature
        self.hidden_dim = hidden_dim
        # Graph-side edge-type count may differ from the model's embedding row
        # count. Used during monomer fine-tune from a smallmol->3-edge-duplicated
        # checkpoint: model has 3 embedding rows but we want the graph to emit
        # only intra (0) and inter/extra (1), preserving row 2 (binder-target)
        # for the eventual peptide:protein phase.
        self._graph_num_edge_types: int | None = graph_num_edge_types
        self.use_coord_self_conditioning = use_coord_self_conditioning
        self.use_plan_latent = use_plan_latent
        self.use_interaction_intent = use_interaction_intent
        self.use_bond_attention = use_bond_attention
        self.use_valence_demand = use_valence_demand
        self.use_cross_residue_packing = use_cross_residue_packing
        self.cross_residue_packing_spatial = cross_residue_packing_spatial
        self.use_neighbor_x0_packing = use_neighbor_x0_packing
        self.neighbor_x0_packing_radius = neighbor_x0_packing_radius
        self.neighbor_x0_highnoise_weight = float(neighbor_x0_highnoise_weight)
        self.neighbor_x0_corrupt_add_prob = float(neighbor_x0_corrupt_add_prob)
        self.neighbor_x0_corrupt_drop_prob = float(neighbor_x0_corrupt_drop_prob)
        self.neighbor_x0_corrupt_noise_prob = float(neighbor_x0_corrupt_noise_prob)
        self.neighbor_x0_corrupt_coord_noise = float(neighbor_x0_corrupt_coord_noise)
        self.neighbor_x0_corrupt_disconnect_prob = float(neighbor_x0_corrupt_disconnect_prob)
        self.num_element_classes = num_element_classes
        self.use_jackie = use_jackie
        self.preal_gate_target = preal_gate_target
        self.use_slot_attention = use_slot_attention
        self.use_directional_slot_attention = use_directional_slot_attention
        self.use_element_velocity_coupling = use_element_velocity_coupling
        self.num_timesteps = num_timesteps
        self.use_ca_dist_element_feature = use_ca_dist_element_feature
        self.use_count_velocity_coupling = use_count_velocity_coupling
        self.max_sidechain_atoms = max_sidechain_atoms
        self.use_target_conditioning = use_target_conditioning
        self.num_cross_attn_layers = num_cross_attn_layers
        self.num_cross_attn_heads = num_cross_attn_heads
        self.use_film = use_film
        self.use_bidirectional_target_conditioning = use_bidirectional_target_conditioning
        self.use_sidechain_target_residue_attention = use_sidechain_target_residue_attention
        self.target_condition_scale = target_condition_scale
        self.cluster_target_condition_scale = cluster_target_condition_scale
        self.intra_residue_cutoff = intra_residue_cutoff
        self.inter_residue_cutoff = inter_residue_cutoff
        self.sidechain_target_cutoff = sidechain_target_cutoff
        self.backbone_target_cutoff = backbone_target_cutoff
        self.ca_ca_prefilter = ca_ca_prefilter

        # Backbone encoder (for cross-attention context)
        self.backbone_encoder = BackboneEncoder(
            hidden_dim, use_backbone_dihedral=use_backbone_dihedral, use_burial_feature=use_burial_feature
        )

        # DRAFT: residue-level frame-aware stream. Built only when opted in. Reads the pocket-
        # conditioned per-residue features, reasons over residues in their local backbone frames,
        # and (a) injects a ZERO-INIT additive residual back into backbone_features (byte-identical
        # at graft, resume-safe) and (b) predicts coarse side-chain latents for the supervision +
        # consistency losses assembled in InverseFoldingDiffusion.forward. See residue_frame_stream.py.
        self.use_residue_frame_stream = use_residue_frame_stream
        # DRAFT deep injection: only meaningful when the stream itself is on. The effective flag AND's
        # the two so downstream logic is a single guard, and no deep-inject modules are built (nor any
        # tensors added to the state_dict) unless BOTH are requested.
        self.use_residue_frame_deep_inject = bool(use_residue_frame_stream and use_residue_frame_deep_inject)
        if use_residue_frame_stream:
            self.residue_frame_stream = ResidueFrameStream(
                hidden_dim=hidden_dim,
                n_layers=residue_frame_layers,
            )

        # v2 residue-frame graph (binder+target residues; orientation-only heads: in-frame
        # centroid + χ1 (cos,sin); no count/radial). SE(3)-INVARIANT exactly like v1 -- geometry
        # enters only as neighbour Cα in the query's local frame, now with the key set extended to
        # include target residues (their Cα + a residue-type embedding). Built ONLY when opted in, so
        # the model is byte-identical (no new params, no RNG advance) when off, and resume-safe (the
        # injection is zero-init). v1 and v2 are mutually exclusive: setting BOTH is a launch error --
        # they contend for the same injection slot, so we refuse rather than silently pick one.
        self.use_residue_frame_stream_v2 = bool(use_residue_frame_stream_v2)
        # supervised stereochemistry head lives INSIDE the v2 stream; the IFD has already AND'd
        # this flag with use_residue_frame_stream_v2, so it is only ever True when the v2 stream is on.
        self.use_stereochem_head = bool(use_stereochem_head)
        if use_residue_frame_stream and use_residue_frame_stream_v2:
            raise ValueError(
                "use_residue_frame_stream and use_residue_frame_stream_v2 are mutually exclusive "
                "(both inject into backbone_features). Enable exactly one."
            )
        if self.use_residue_frame_stream_v2:
            self.residue_frame_stream_v2 = ResidueFrameStreamV2(
                hidden_dim=hidden_dim,
                n_layers=residue_frame_v2_layers,
                clean_input=frame_v2_clean_input,
                num_residue_types=NUM_RESIDUE_TYPES,
                use_stereochem_head=self.use_stereochem_head,
            )
        # t-resolution head. Lives on the DENOISER (not inside the v2 module) because it is
        # t-DEPENDENT and reads the current side-chain atom cloud + `t` -- both available only here, not in
        # the static frame stream. A small MLP maps the 3 in-frame e3 (out-of-plane) atom-cloud moments to a
        # scalar per-residue `atom_logit`, blended with the static P(D)_prior by the per-timestep schedule
        # w(t). The IFD has already AND'd this flag with use_stereochem_head, so it is only ever True when
        # the static head exists (to supply P(D)_prior). Built ONLY when opted in => no params / no RNG
        # advance when off (byte-identical). The final-layer OUTPUT bias mirrors the static head's L-prior
        # (_STEREO_HEAD_LPRIOR_BIAS) so an untrained atom head gives sigmoid(atom_logit)≈0 (L-leaning),
        # keeping P(D)_t≈P(D)_prior on a fresh graft rather than pulling it toward 0.5.
        self.use_stereochem_t_resolution = bool(use_stereochem_t_resolution)
        if self.use_stereochem_t_resolution:
            self.stereo_t_atom_head = nn.Sequential(
                nn.Linear(_STEREO_T_ATOM_FEAT_DIM, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.constant_(self.stereo_t_atom_head[-1].bias, _STEREO_HEAD_LPRIOR_BIAS)
        # coord-flow feedback modules. Built ONLY when opted in (the IFD has AND'd the flag with
        # use_stereochem_t_resolution) => no params / no RNG advance when off (byte-identical). Every OUTPUT of
        # the feedback is ZERO at graft/resume, so the whole coupling is EXACTLY 0 at init (byte-identical, and
        # bit-identical even ON-at-init once shared weights are loaded): `stereo_feedback_deep_proj` is a
        # per-SE(3)-layer BIAS-FREE, ZERO-INIT projection (exact mirror of volumetric_deep_proj; respects
        # graft_init_std) and `stereo_feedback_face_scale` is a ZERO-INIT learnable scalar gating the explicit
        # e3 velocity bias. The face->latent `stereo_feedback_embed` is a normal Linear because its output only
        # ever feeds the zero-init deep-inject, so the injected residual is still 0 at init.
        self.use_stereochem_t_resolution_feedback = bool(use_stereochem_t_resolution_feedback)
        if self.use_stereochem_t_resolution_feedback:
            self.stereo_feedback_embed = nn.Linear(1, hidden_dim)  # face_pref (scalar) -> per-residue latent
            self.stereo_feedback_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.stereo_feedback_deep_proj:
                _init_graft_weight(_proj.weight)
            # LEARNED sign + magnitude of the explicit e3 push (zero-init => 0 at graft). The model discovers how
            # hard, and toward which sign of e3, to steer for the resolved face.
            self.stereo_feedback_face_scale = nn.Parameter(torch.zeros(1))
        if self.use_residue_frame_deep_inject:
            # One BIAS-FREE, ZERO-INIT projection per SE(3) layer (num_layers == len(transformer.layers)).
            # Each maps the per-residue frame latent -> a per-node residual added to the layer's INVARIANT
            # (type-0) node features only (see _forward_single). Zero-init => exact no-op at graft/resume;
            # gradient still reaches these projections (grad_W = grad_out^T @ in, with in = the residue latent),
            # so the deep path is live-but-identity at init (NOT a zero-init-both-ends deadlock). Same
            # category-3 graft-init framework as clean_inject_proj (respects graft_init_std; zeros by default).
            self.residue_frame_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.residue_frame_deep_proj:
                _init_graft_weight(_proj.weight)

        # NEW: v2 residue-frame deep per-layer injection. One BIAS-FREE, ZERO-INIT projection per SE(3) layer,
        # structurally IDENTICAL to residue_frame_deep_proj / volumetric_deep_proj: each maps the v2 stream's
        # per-residue latent `rf_hidden` -> a per-node residual added to that layer's INVARIANT (type-0) node
        # features (residue->atom scatter in _forward_single). Zero-init => exact no-op at graft/resume =>
        # byte-identical + resume-safe; the gradient still reaches these projections (grad_W = grad_out^T @
        # rf_hidden), so the deep path is live-but-identity at init. Respects graft_init_std (zeros by default).
        # Built ONLY when the effective flag is on (IFD has AND'd it with use_residue_frame_stream_v2).

        # DOCTRINE -- THIS CHANNEL IS *LIVE*, NOT DETACHED (deliberate, load-bearing). The v1 frame deep-inject
        # (residue_frame_hidden) AND the two volumetric-derived channels (vol_hidden for volumetric_deep_proj,
        # face_pref for stereo_feedback_deep_proj) all pass their source `.detach()`ed: those streams are
        # trained ONE-WAY by their own supervision losses and consumed by the atom flow as literal, immutable
        # read-outs. This v2 channel is the OPPOSITE: its source `rf_hidden` is fed to the deep-inject WITHOUT
        # detaching, so the atom-flow (coord/velocity) loss backprops THROUGH the deep path into the v2 frame
        # stream's own parameters. The doctrine is that the frame stream is a SOFT guideline (rotamer/centroid
        # orientation) that SHOULD be co-trained BIDIRECTIONALLY by the atom flow, whereas volume/stereo are
        # CONFIDENT/immutable read-outs that stay one-way. This restores what v1's frame stream does (and what
        # run VM1 empirically benefits from) inside the v2 architecture. Byte-identical when off.
        self.use_residue_frame_v2_deep_inject = bool(use_residue_frame_stream_v2 and use_residue_frame_v2_deep_inject)
        self.frame_v2_deep_inject_detach = bool(frame_v2_deep_inject_detach)
        if self.use_residue_frame_v2_deep_inject:
            self.residue_frame_v2_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.residue_frame_v2_deep_proj:
                _init_graft_weight(_proj.weight)

        # volumetric deep per-layer injection. One BIAS-FREE, ZERO-INIT projection per SE(3) layer,
        # mirroring residue_frame_deep_proj EXACTLY. Each maps the (detached) per-residue `vol_hidden` latent
        # -> a per-node residual added to that layer's INVARIANT (type-0) node features (residue->atom scatter
        # in _forward_single). Zero-init => exact no-op at graft/resume => byte-identical + resume-safe; the
        # gradient still reaches these projections (grad_W = grad_out^T @ vol_hidden), so the deep path is
        # live-but-identity at init. Respects graft_init_std (zeros by default). Built ONLY when the effective
        # flag is on (IFD has already AND'd it with use_volumetric_head), so nothing is added to the
        # state_dict when off. `vol_hidden` has width hidden_dim (the head is constructed with hidden_dim).
        self.use_volumetric_deep_inject = bool(use_volumetric_deep_inject)
        if self.use_volumetric_deep_inject:
            self.volumetric_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.volumetric_deep_proj:
                _init_graft_weight(_proj.weight)

        # TARGET deep per-layer injection (run10). One BIAS-FREE ZERO-INIT projection per SE(3) layer, mirroring
        # residue_frame_deep_proj / volumetric_deep_proj EXACTLY. Maps the (detached) per-sc-node target-context
        # latent `sc_target_attn` (sidechain->target cross-attention) -> a per-node residual on the SIDECHAIN
        # nodes only (bb + target nodes get 0). Reinforces the dense target signal -- otherwise applied ONLY at
        # layer 0 (`sidechain_target_film`) -- at every SE(3) layer. Zero-init => exact no-op at init/resume =>
        # byte-identical off; gradient still reaches the projections. Gated at forward on sc_target_attn is not None.
        self.use_target_deep_inject = bool(use_target_deep_inject)
        if self.use_target_deep_inject:
            self.target_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.target_deep_proj:
                _init_graft_weight(_proj.weight)

        # BACKBONE deep-inject (run10-next): reinforce the BackboneEncoder per-residue latent (backbone geometry
        # + target cross-attention + FiLM) -- otherwise applied only at SE(3) layer 0 (concatenated into the initial
        # sc/bb node features) -- at EVERY layer. Mirrors residue_frame_deep_proj (sc + bb scatter). DETACHED at apply
        # (the backbone encoder is upstream + heavily shared; do not reshape it via a new per-layer path). Zero-init
        # => byte-identical off. `backbone_features` (L, hidden) is already a _forward_single arg, so no new threading.
        self.use_backbone_deep_inject = bool(use_backbone_deep_inject)
        if self.use_backbone_deep_inject:
            self.backbone_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.backbone_deep_proj:
                _init_graft_weight(_proj.weight)

        # x0 BOND-ANGLE deep-inject. `bond_angle_inject_encoder` maps the (L, BOND_ANGLE_DESC_DIM) per-residue
        # angle descriptor of the predicted-x0 cloud -> a per-residue latent (L, hidden). `bond_angle_deep_proj`
        # is one BIAS-FREE ZERO-INIT Linear per SE(3) layer, IDENTICAL in structure to residue_frame_v2_deep_proj:
        # each maps that latent -> a per-node residual on the INVARIANT (type-0) node features (residue->atom
        # scatter in _forward_single). Zero-init => exact no-op at init/resume => byte-identical + resume-safe;
        # gradient still reaches proj AND (through it) the encoder, so the path is live-but-identity at init. The
        # encoder is NOT zero-init (it is a fresh reader of the descriptor), but its output is gated by the
        # zero-init proj, so nothing perturbs the flow until training moves proj off zero.
        self.use_bond_angle_deep_inject = bool(use_bond_angle_deep_inject)
        if self.use_bond_angle_deep_inject:
            self.bond_angle_inject_encoder = nn.Sequential(
                nn.Linear(BOND_GEOM_DESC_DIM, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.bond_angle_deep_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.bond_angle_deep_proj:
                _init_graft_weight(_proj.weight)

        # volumetric -> existence coupling. ONE ZERO-INIT projection mapping the (detached) per-residue
        # `vol_hidden` latent -> a PER-SLOT existence-logit bias (L, max_sc). The bias is SUBTRACTED from the
        # element track's PAD/GHOST logit (class ELEMENT_PAD=0) in _forward_single: a positive bias LOWERS the
        # PAD logit => LOWERS P(PAD) => RAISES existence, so "more predicted volume => more real (non-PAD)
        # atoms" is directly expressible. Both weight AND bias are zero-init (respecting graft_init_std; zeros
        # by default) => the bias is EXACTLY 0 at graft/resume => byte-identical + resume-safe, and it cannot
        # perturb P(PAD) at init (a 0 additive shift is softmax-inert). The sign is LEARNED (zero-init), so the
        # model discovers whether volume should raise or lower existence; the gradient still reaches the
        # projection (grad_W = grad_out^T @ vol_hidden). Built ONLY when the effective flag is on (IFD has
        # already AND'd it with use_volumetric_head), so nothing is added to the state_dict when off. Per-SLOT
        # (not per-residue) for expressiveness; since it is zero-init and only a single additive DOF per slot,
        # it stays a gentle learned lever that cannot destabilise the PAD-sink at init.
        self.use_volumetric_existence_coupling = bool(use_volumetric_existence_coupling)
        if self.use_volumetric_existence_coupling:
            self.volumetric_existence_proj = nn.Linear(hidden_dim, max_sidechain_atoms)
            _init_graft_weight(self.volumetric_existence_proj.weight)
            _init_graft_weight(self.volumetric_existence_proj.bias)

        # Target encoder and cross-attention (for sequence-level context)
        if use_target_conditioning:
            self.target_encoder = TargetEncoder(hidden_dim, use_jackie=use_jackie)
            pair_geom_dim = 3 + 3 + 9 + 16
            self.residue_pair_bias_proj = nn.Sequential(
                nn.Linear(pair_geom_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_cross_attn_heads),
            )
            self.sidechain_pair_bias_proj = nn.Sequential(
                nn.Linear(pair_geom_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_cross_attn_heads),
            )
            if use_bidirectional_target_conditioning:
                self.target_to_backbone_attention_layers = nn.ModuleList(
                    [
                        CrossAttention(hidden_dim, num_heads=num_cross_attn_heads, dropout=dropout)
                        for _ in range(num_cross_attn_layers)
                    ]
                )
                self.target_to_backbone_norms = nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(num_cross_attn_layers)]
                )
            # Stack of cross-attention layers (configurable depth)
            self.cross_attention_layers = nn.ModuleList(
                [
                    CrossAttention(hidden_dim, num_heads=num_cross_attn_heads, dropout=dropout)
                    for _ in range(num_cross_attn_layers)
                ]
            )
            self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cross_attn_layers)])
            if use_sidechain_target_residue_attention:
                self.sidechain_target_attention = CrossAttention(
                    hidden_dim, num_heads=num_cross_attn_heads, dropout=dropout
                )
                self.sidechain_target_norm = nn.LayerNorm(hidden_dim)
                self.sidechain_target_film = FiLMLayer(hidden_dim, hidden_dim)
                self.cluster_target_context_proj = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                # Backward-compatible no-op init for checkpoints created before this branch existed.
                nn.init.zeros_(self.cluster_target_context_proj[0].weight)
                nn.init.zeros_(self.cluster_target_context_proj[0].bias)
                nn.init.zeros_(self.cluster_target_context_proj[2].weight)
                nn.init.zeros_(self.cluster_target_context_proj[2].bias)
            # Optional FiLM layer: target global features modulate backbone features
            if use_film:
                self.target_film = FiLMLayer(hidden_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_embed_dim, hidden_dim)

        # Node embeddings for different atom types in the graph
        # Binder sidechain: atom position embedding (which atom slot 0-13)
        self.sc_atom_embed = nn.Embedding(max_sidechain_atoms, hidden_dim // 4)
        # Binder sidechain: element type embedding (PAD, C, N, O, S, and optionally MASK)
        self.element_type_embed = nn.Embedding(num_element_classes, hidden_dim // 4)
        # Binder backbone: atom type embedding (N, CA, C, O)
        self.bb_atom_embed = nn.Embedding(4, hidden_dim // 4)
        # Target atoms: per-atom feature embeddings for proper local conditioning
        # These replace the old mean-pooled approach with proper per-atom features
        self.target_atom_type_embed = nn.Embedding(18, hidden_dim // 4)  # 4 backbone + 14 sidechain slots
        # PAD-aware (v1): rows 0=PAD,1=C,2=N,3=O,4=X (vocab5-X); X keeps its own row.
        self.target_element_type_embed = nn.Embedding(NUM_ELEMENT_TYPES, hidden_dim // 4)
        # Residue type embedding for target atoms: always use learned embedding
        self.target_residue_type_embed = nn.Embedding(21, hidden_dim // 4)  # 20 amino acids + unknown
        # Optionally also project Jackie biochemical features and add them
        if use_jackie:
            self.register_buffer("jackie_features_graph", JACKIE_FEATURES)
            self.target_residue_type_proj = nn.Linear(JACKIE_DIM, hidden_dim // 4)
        self.target_is_backbone_embed = nn.Embedding(2, hidden_dim // 8)  # 0=sidechain, 1=backbone
        # Project concatenated target features to hidden_dim
        # 3 * (hidden//4) + hidden//8 = 3*hidden/4 + hidden/8 = 7*hidden/8
        target_proj_in = 3 * (hidden_dim // 4) + (hidden_dim // 8)
        self.target_atom_proj = nn.Linear(target_proj_in, hidden_dim)

        # Noised count embedding: tells the model roughly how many atoms are "on"
        # at this noise level. Derived from (noised_element_types != PAD).sum().

        # "linear" (default, byte-identical to history): Linear(1, d) of the raw scalar count. A
        # Linear(1, d) can only emit a monotone ramp -- every count maps to a scalar multiple of one
        # direction, so nearby counts (e.g. 7 vs 8) are near-collinear and hard to resolve.
        # "ordinal": Embedding(max_sidechain_atoms + 1, d) indexed by the integer count clamped to
        # [0, max_sidechain_atoms]. Each count gets its own freely-learned code -> distinct,
        # resolvable embeddings. Param-shape change, so it only activates behind the flag; a base
        # checkpoint (trained "linear") loads unchanged when the flag is left at "linear".
        self.count_embed_mode = count_embed_mode
        if count_embed_mode == "ordinal":
            self.noised_count_embed = nn.Embedding(max_sidechain_atoms + 1, hidden_dim // 8)
        elif count_embed_mode == "linear":
            self.noised_count_embed = nn.Linear(1, hidden_dim // 8)
        else:
            raise ValueError(f"count_embed_mode must be 'linear' or 'ordinal', got {count_embed_mode!r}")

        # Self-conditioning: previous prediction embeddings
        # During training, we sometimes run the model twice - first to get predictions,
        # then again with those predictions as extra inputs. This makes the model robust to
        # its own errors during sampling (reduces exposure bias).
        # Element self-cond: softmax probabilities from previous pass.
        self.self_cond_element_proj = nn.Linear(num_element_classes, hidden_dim // 8)
        self.cluster_id_embed = nn.Embedding(max_sidechain_atoms, hidden_dim // 8)
        self.self_cond_cluster_proj = nn.Linear(max_sidechain_atoms, hidden_dim // 8)

        # Residue index embedding: tells atoms which residue they belong to
        # This helps group intra-sidechain atoms together during diffusion
        # Use sinusoidal encoding to generalize to longer sequences
        self.max_residue_idx = 256  # Max sequence length we'll support
        self.residue_idx_embed = nn.Embedding(self.max_residue_idx, hidden_dim // 8)

        # Autoregressive / K-mask state embedding: 0=hidden, 1=focus (being denoised), 2=revealed (clean context)
        # Added as a residual after sc_node_proj so sc_node_proj keeps its original size
        # (backward compatible with pre-AR checkpoints).
        # BUG-A fix (2026-06-25): the residual is ar_state_proj(ar_state_embed(state)). If BOTH matrices are
        # zero-init the branch is DEAD -- zero output AND zero gradient into either matrix, so wiring
        # design_mask -> ar_state would do nothing. Safe pattern (per review agent): small-RANDOM ar_state_embed
        # (distinct per-state signal) + ZERO ar_state_proj (residual is initially a no-op but the projection can
        # learn). NOTE: a weights-resume from a PRE-FIX checkpoint reloads a zero embed and re-kills the branch --
        # the trainer re-inits ar_state_embed post-load (on the training resume path).
        self.ar_state_embed = nn.Embedding(3, hidden_dim // 8)
        nn.init.normal_(self.ar_state_embed.weight, std=0.02)
        self.ar_state_proj = nn.Linear(hidden_dim // 8, hidden_dim, bias=False)
        nn.init.zeros_(self.ar_state_proj.weight)

        # Projection for binder sidechain nodes:
        # atom_embed + element_embed + noised_count_embed
        # + self_cond_mask + self_cond_element
        # + residue_idx_embed + backbone_context + time
        sc_proj_in = (
            hidden_dim // 4  # atom_embed
            + hidden_dim // 4  # element_embed
            + hidden_dim // 8  # noised_count_embed
            + hidden_dim // 8  # self_cond_element
            + hidden_dim // 8  # cluster_id_embed
            + hidden_dim // 8  # self_cond_cluster
            + hidden_dim // 8  # residue_idx_embed
            + hidden_dim  # backbone_context
            + hidden_dim  # time
        )
        if use_coord_self_conditioning:
            sc_proj_in += hidden_dim // 8  # coord_self_cond
        self.sc_node_proj = nn.Linear(sc_proj_in, hidden_dim)
        # Projection for binder backbone nodes: atom_embed + backbone_context + time
        self.bb_node_proj = nn.Linear(hidden_dim // 4 + hidden_dim + hidden_dim, hidden_dim)

        # Edge type embedding (num_edge_types=3 -> intra/inter/binder-target;
        # num_edge_types=2 -> intra/extra-residue, forward-compatible with monomer/small-molecule)
        self.num_edge_types = num_edge_types
        self.edge_embed = EdgeTypeEmbedding(num_types=num_edge_types, embed_dim=edge_embed_dim)

        # SE(3) Transformer (replaces EGNN for better all-to-all attention).
        # SE(3)-core heads follow num_cross_attn_heads (the "attention heads" knob): both were 4
        # historically, so this is a no-op for every existing run (cross=4 -> SE(3)=4) but lets a
        # run opt into 8 SE(3) heads by setting --num-cross-attn-heads 8. Auto-persists via hparams
        # (num_cross_attn_heads is already saved), so resumes rebuild the SE(3) core at the right width.
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
            # Thread the largest active edge cutoff (CA->target context edges) as the RBF max radial
            # extent so the attention-bias net stays distance-sensitive out to the target. Buffer-only
            # change (center COUNT unchanged) -> loads cleanly from a base checkpoint. FLAG-GATED (audit
            # follow-up): only widen when rbf_span_to_cutoff is opted in; otherwise pass None so the
            # transformer keeps its LEGACY 0-10 A span (what every pre-ce7ed5d70 checkpoint was trained with).
            rbf_max_dist=(backbone_target_cutoff if rbf_span_to_cutoff else None),
        )

        # Intra-residue slot attention: structured all-to-all communication within each residue
        if use_slot_attention:
            self.slot_attention = IntraResidueSlotAttention(
                hidden_dim=hidden_dim,
                num_heads=num_slot_attn_heads,
                num_layers=num_slot_attn_layers,
                max_slots=max_sidechain_atoms,
                dropout=dropout,
            )

        # Directional slot attention: distal scouts -> proximal sticky slots (one-way)
        if use_directional_slot_attention:
            self.directional_slot_attention = DirectionalSlotAttention(
                hidden_dim=hidden_dim,
                num_heads=num_slot_attn_heads,
                max_slots=max_sidechain_atoms,
                n_scouts=n_scouts,
                dropout=dropout,
            )

        # Element-velocity coupling: FiLM modulation of sc_out_features by PAD/non-PAD state
        if use_element_velocity_coupling:
            self.evc_film = FiLMLayer(hidden_dim, hidden_dim)
            self.evc_state_proj = nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
            # Nonzero state_proj init so EVC learns from step 0. This is a from-scratch lineage (no
            # pre-EVC checkpoint to preserve), so we do NOT want the old zero-init identity start:
            # a zero state_proj + zero FiLM weights are mutually gradient-starved (the EVC deadlock).
            # The FiLM bias stays zero so the FiLM contribution itself starts identity-ish, while the
            # nonzero state_proj weight is what breaks the deadlock and makes EVC live from the start.
            nn.init.normal_(self.evc_state_proj[-1].weight, std=0.02)
            nn.init.zeros_(self.evc_state_proj[-1].bias)

        # Count-velocity coupling: FiLM modulation of sc_out_features by per-residue count.
        # Uses count head predictions (backbone-only, stable) -> per-slot P(real) via sigmoid.
        # Unlike EVC (noisy at sampling start), count predictions are available and stable from step 0.
        if use_count_velocity_coupling:
            self.cvc_film = FiLMLayer(hidden_dim, hidden_dim)
            self.cvc_proj = nn.Sequential(
                nn.Linear(1, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
            nn.init.zeros_(self.cvc_proj[-1].weight)
            nn.init.zeros_(self.cvc_proj[-1].bias)

        # Output projection for noise prediction
        self.output_proj = nn.Linear(hidden_dim, 3)

        # Element type prediction head (predicts clean element types from noised)
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

        # Occupancy head: per-slot binary classifier for ghost vs real (P(real | features))
        self.occupancy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Pocket contact prediction: per-slot binary "contacts target?" head
        self.use_pocket_contact_prediction = getattr(self, "use_pocket_contact_prediction", False)
        self.pocket_contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Intra-residue bond attention: refines per-slot features using predicted bond graph
        if use_bond_attention:
            self.bond_attention = IntraResidueBondAttention(
                hidden_dim=hidden_dim,
                max_sc=max_sidechain_atoms,
                num_heads=4,
                dropout=dropout,
                zero_init_out=zero_init_bond_attention,
            )

        # Valence demand track: per-slot prediction of remaining heavy-atom valence (0-4).
        # Supervised from GT bond counts. Feeds predicted valence as FiLM conditioning to
        # element and coord heads -- an oxygen with 2 bonds satisfied behaves differently
        # from one with 1 remaining.
        if use_valence_demand:
            self.valence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 5),  # 5 classes: 0,1,2,3,4 bonds
            )
            # FiLM conditioning: predicted valence modulates features before output heads
            self.valence_film = FiLMLayer(hidden_dim, hidden_dim)
            self.valence_embed = nn.Sequential(
                nn.Linear(5, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
            )
            # Zero-init so valence conditioning is identity at start
            nn.init.zeros_(self.valence_embed[-1].weight)
            nn.init.zeros_(self.valence_embed[-1].bias)

        # Learned sidechain shape prior: predict K anchor points per residue from backbone+target
        # features. During coord flow, bias velocities toward nearest predicted anchor.
        # Supervised by matching anchors to GT atom positions.
        self.use_shape_prior = use_shape_prior
        if use_shape_prior:
            self.shape_prior_n_anchors = shape_prior_n_anchors
            self.shape_prior_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, self.shape_prior_n_anchors * 3),  # K anchors × 3D
            )
            # Zero-init so anchors start at CA (no bias initially)
            nn.init.zeros_(self.shape_prior_head[-1].weight)
            nn.init.zeros_(self.shape_prior_head[-1].bias)
            # Learnable velocity bias strength (starts small)
            self.shape_prior_bias_scale = nn.Parameter(torch.tensor(0.1))

        # Cross-residue packing attention: inter-residue contact prediction + attention
        if use_cross_residue_packing:
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

        # === Feature: neighbour-x0 packing context ===
        # Zero-init projection of the anti-self neighbour-x0 context vector, added as a residual
        # to the sidechain node features. Zero weight => exact no-op at graft (safe retrofit onto
        # any existing checkpoint) but the gradient w.r.t. the weight is the (non-zero) feature
        # vector, so the branch is NOT dead.
        if use_neighbor_x0_packing:
            self.neighbor_x0_proj = nn.Linear(NEIGHBOR_X0_FEAT_DIM, hidden_dim, bias=False)
            _init_graft_weight(self.neighbor_x0_proj.weight)
            # Two-timestep encoding. The model already knows t_original (the residue's own noise
            # level); this branch adds the DELTA to t_conditioning (the nominal noise level of the
            # coordinates it is being shown). The input is the difference of the two time embeddings
            # plus the normalised scalar gap, so it is EXACTLY zero whenever
            # t_conditioning == t_original -- i.e. today's single-t behaviour is an exact special
            # case for ANY value of the weights, not just at init (bias=False is load-bearing).
            # That is what makes the new regime a smooth extension an existing checkpoint can be
            # fine-tuned into rather than a distribution it has never seen.
            self.t_cond_delta_proj = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
            _init_graft_weight(self.t_cond_delta_proj.weight)
            # Recycle-index encoding. Tells the model WHICH refinement pass it is on, so the recycle
            # count no longer has to match between training and sampling. Built on the ABSOLUTE pass
            # index j (1-based), NEVER the fraction j/N: pass 2-of-2 and pass 2-of-5 receive literally
            # identical conditioning (both are conditioned on exactly one prior refinement), and future
            # passes cannot influence the current input, so conditioning quality is a function of j
            # alone and is INDEPENDENT of N. Encoding j/N would hand the model different codes (1.0 vs
            # 0.4) for identical inputs -- noise by construction. Same bias-free difference trick as
            # t_cond_delta_proj: the input is (embed(j) - embed(1), 1 - 1/j), identically zero at j=1,
            # so the unconditioned first pass is bit-exact for ANY weights. See _forward_single.
            self.recycle_index_proj = nn.Linear(hidden_dim + 1, hidden_dim, bias=False)
            _init_graft_weight(self.recycle_index_proj.weight)

        # === Feature: Coordinate self-conditioning ===
        # Project previous step's predicted x₀ coords into per-slot features.
        # Zero-init for backward compatibility with existing checkpoints.
        if use_coord_self_conditioning:
            self.coord_self_cond_proj = nn.Linear(3, hidden_dim // 8)
            nn.init.zeros_(self.coord_self_cond_proj.weight)
            nn.init.zeros_(self.coord_self_cond_proj.bias)

        # === Feature: Residue-level plan latent ===
        # Predict per-residue latent (dim=16) from backbone+target features.
        # Supervised by GT residue type through information bottleneck.
        # Modulates per-slot features via additive projection.
        if use_plan_latent:
            self.plan_latent_dim = 16
            self.plan_latent_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, self.plan_latent_dim),
            )
            # Frozen GT embedding: 20 AA types -> 16-dim latent (information bottleneck)
            self.plan_latent_gt_embed = nn.Embedding(21, self.plan_latent_dim)  # 20 AA + unknown
            self.plan_latent_gt_embed.weight.requires_grad = False
            # Initialize with random orthogonal-ish vectors for good separation
            nn.init.normal_(self.plan_latent_gt_embed.weight, std=0.5)
            # Project plan latent to hidden_dim for additive modulation of per-slot features
            self.plan_latent_to_slot = nn.Linear(self.plan_latent_dim, hidden_dim, bias=False)
            nn.init.zeros_(self.plan_latent_to_slot.weight)

        # === Feature: Target-interaction intent ===
        # Per-slot prediction of interaction mode with target. Can be 2-class (binary
        # contact) or 4-class (none, hydrophobic, H-bond, aromatic). Velocity bias
        # toward nearest target atom is optional and orthogonal to the classification.
        self.interaction_intent_num_classes = interaction_intent_num_classes
        self.interaction_intent_velocity_bias = interaction_intent_velocity_bias
        if use_interaction_intent:
            self.interaction_intent_head = nn.Linear(hidden_dim, interaction_intent_num_classes)
            nn.init.zeros_(self.interaction_intent_head.weight)
            nn.init.zeros_(self.interaction_intent_head.bias)
            # Learnable velocity bias scale for interacting atoms (only used if velocity_bias=True)
            self.interaction_bias_scale = nn.Parameter(torch.tensor(0.05))

        # === Feature: Chirality attention ===
        # Signed volume from first 3 atoms per residue -> per-slot scalar feature.
        # Breaks SE(3) equivariance to distinguish L vs D configurations.
        self.use_chirality = use_chirality
        if use_chirality:
            self.chirality_proj = nn.Linear(1, hidden_dim)
            nn.init.zeros_(self.chirality_proj.weight)
            nn.init.zeros_(self.chirality_proj.bias)

        # Stage 2 per-residue mixture heads: predict shared sidechain cloud parameters.
        # All real atoms within a residue share one centroid and one variance,
        # rather than each slot having independent Gaussian parameters.
        # Pool per-slot features to per-residue, then predict centroid offset + cloud logvar.
        self.residue_mixture_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residue_centroid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.residue_cloud_logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Residue-level count head: predicts per-residue atom count from backbone+target features.
        # Operates BEFORE sidechain processing, giving a direct gradient path for count prediction
        # independent of element type predictions.
        self.residue_count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Init: both layers use default (Kaiming) init for non-trivial initial predictions.
        # Scale output layer to small values so initial predictions stay near the bias of 4.0.
        nn.init.normal_(self.residue_count_head[2].weight, std=0.01)
        nn.init.constant_(self.residue_count_head[2].bias, 4.0)  # ~4 atoms/residue mean

        # Per-residue LRT head: predicts a delta to the global mixture_lr_threshold
        # from backbone+target features. Tight pockets -> positive delta (stricter, fewer atoms),
        # spacious pockets -> negative delta (permissive, more atoms).
        # Initialized to zero output so initial behavior matches global LRT.
        self.residue_lrt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.residue_lrt_head[2].weight)
        nn.init.zeros_(self.residue_lrt_head[2].bias)

        # === Feature: global latent matching (clean-context stream + per-residue latent pool) ===
        # Pocket-only, time-free stream refines the (detached) per-residue backbone features into a
        # clean context; a per-residue CLS-attention pool reads off z_pred; a confidence probe gates
        # a detached FiLM feedback into the generated atoms. Built ONLY when the flag is on, so with
        # the flag off there are no extra tensors and the forward path is bit-exact unchanged.
        self.use_global_latent_matching = use_global_latent_matching
        self.global_latent_embed_dim = int(global_latent_embed_dim)
        self.global_latent_condition_main_stream = global_latent_condition_main_stream
        self.global_latent_film_layers = global_latent_film_layers
        if use_global_latent_matching:
            # Clean-stream depth MATCHES the main SE(3) depth so each main layer has a corresponding
            # clean-stream layer output for the per-layer interweave (clean_stream_n_layers=null).
            self.clean_context_stream = CleanContextStream(
                hidden_dim=hidden_dim,
                n_layers=num_layers,
                num_heads=num_cross_attn_heads,
            )
            self.latent_pool_head = LatentPoolHead(
                hidden_dim=hidden_dim,
                embed_dim=global_latent_embed_dim,
                pool_hidden=global_latent_pool_hidden,
                num_heads=num_cross_attn_heads,
                pool_radius=global_latent_pool_radius,
                confidence=global_latent_confidence,
                confidence_hidden=global_latent_confidence_hidden,
            )
            # Per-layer interweave (the clean_inject_proj): one BIAS-FREE, ZERO-INIT projection per
            # main SE(3) layer. The clean-stream layer-i output (mapped to graph nodes) is projected and
            # added into the main node features before main layer i. Zero-init => EXACT no-op at init,
            # ramps as it learns (same safe-graft pattern as neighbor_x0_proj / t_cond_delta_proj). Unlike
            # the confidence FiLM below, this path is NOT detached: the main-task loss trains the clean
            # stream + these projections to produce useful per-layer pocket->main conditioning.
            self.clean_inject_proj = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)]
            )
            for _proj in self.clean_inject_proj:
                _init_graft_weight(_proj.weight)
            if global_latent_condition_main_stream:
                # cond_dim = 1 (confidence) + embed_dim (relu(conf) * z_pred); FiLMLayer is zero-init identity.
                self.latent_film = FiLMLayer(hidden_dim, 1 + global_latent_embed_dim)
                # Category-3 graft init: re-init the FiLM identity params (weights + biases) off zero when
                # requested, through the SAME isolated-RNG helper (done here, not in FiLMLayer, so other
                # FiLM layers keep their zero-init). At graft_std=0 this re-zeros already-zero params (no-op).
                _init_graft_weight(self.latent_film.scale_proj.weight)
                _init_graft_weight(self.latent_film.scale_proj.bias)
                _init_graft_weight(self.latent_film.shift_proj.weight)
                _init_graft_weight(self.latent_film.shift_proj.bias)

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
        t_norm = (t.float() / max(self.num_timesteps - 1, 1)).clamp(0.0, 1.0)  # (B,)
        return ((0.5 - t_norm) / 0.5).clamp(0.0, 1.0)  # (B,)

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
        eps = 1e-8
        R, ca = build_local_frames(backbone_coords, backbone_mask)  # (B,L,3,3), (B,L,3)
        offset = sidechain_coords - ca.unsqueeze(2)  # (B, L, K, 3): atom - CA (global)
        # e3 (out-of-plane) component in the local frame: local = Rᵀ(atom - CA), and its 3rd entry is
        # eq3 · offset (eq3 = R[..., :, 2], the frame's third column / backbone-plane normal).
        eq3 = R[..., :, 2]  # (B, L, 3)
        e3 = (offset * eq3.unsqueeze(2)).sum(dim=-1)  # (B, L, K) per-atom out-of-plane signed component
        w = real_weight.clamp(min=0.0)  # (B, L, K)
        wsum = w.sum(dim=-1).clamp(min=eps)  # (B, L)
        mean_e3 = (e3 * w).sum(dim=-1) / wsum  # (B, L) signed mean (which face)
        mean_abs_e3 = (e3.abs() * w).sum(dim=-1) / wsum  # (B, L) mean magnitude (how far off-plane)
        mean_sq_e3 = ((e3**2) * w).sum(dim=-1) / wsum  # (B, L) second moment (spread)
        feats = torch.stack([mean_e3, mean_abs_e3, mean_sq_e3], dim=-1)  # (B, L, 3)

        atom_logit = self.stereo_t_atom_head(feats).squeeze(-1)  # (B, L)
        pd_prior = torch.sigmoid(stereo_prior_logit)  # (B, L)
        pd_atom = torch.sigmoid(atom_logit)  # (B, L)
        wt = self._t_resolution_weight(t).view(-1, 1)  # (B, 1) broadcast over residues
        return (1.0 - wt) * pd_prior + wt * pd_atom  # (B, L) P(D)_t in [0,1]

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
        bond_prev_x0: torch.Tensor
        | None = None,  # (B, L, max_sc, 3) detached predicted-x0 for the bond-inject descriptor
        bond_prev_mask: torch.Tensor
        | None = None,  # (B, L, max_sc) real-atom (predicted-existence) mask for the descriptor
        neighbor_x0_coords: torch.Tensor | None = None,
        neighbor_x0_mask: torch.Tensor | None = None,
        neighbor_x0_trust: torch.Tensor | None = None,
        neighbor_x0_apply: torch.Tensor | None = None,
        t_original_res: torch.Tensor | None = None,
        t_conditioning_res: torch.Tensor | None = None,
        recycle_index: int | None = None,
        vol_hidden: torch.Tensor | None = None,  # (B, L, hidden) volumetric latent for deep-inject
        vol_density_inject: torch.Tensor
        | None = None,  # (B, L, hidden) zero-init proj of vol_density_pred; when supplied, deep-injects THIS instead of vol_hidden
        stereo_feedback_ramp: float = 1.0,  # head-first ramp scale (1.0 = inference/full; 0 = off)
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

        # Compute per-residue occupancy gate for target conditioning.
        # Uses the hard noised PAD/non-PAD state from element diffusion (not the model's
        # occupancy belief, which isn't available until after the transformer).
        # At low noise this closely tracks true ghost/real; at high noise most slots
        # are PAD so the floor (0.3) dominates and all residues get near-equal signal.
        if self.preal_gate_target in ("cross_attention", "film"):
            if noised_element_types is not None:
                occ_gate = (noised_element_types != 0).float()  # (B, L, max_sc)
                occ_gate = torch.clamp(occ_gate, min=0.3)
                occ_gate_residue = occ_gate.mean(dim=-1, keepdim=True)  # (B, L, 1)
            else:
                occ_gate_residue = torch.ones(batch_size, seq_len, 1, device=sidechain_coords.device)

        # Encode backbone for cross-attention context
        backbone_features = self.backbone_encoder(backbone_coords, backbone_mask)  # (B, L, hidden)

        # Short-circuit: skip residue-level target conditioning in graph-only mode.
        # Must also check use_film since FiLM needs target_features from TargetEncoder.
        skip_residue_target = (
            self.num_cross_attn_layers == 0 and not self.use_film and not self.use_sidechain_target_residue_attention
        )

        # Encode target and apply MULTIPLE cross-attention layers + FiLM for stronger conditioning
        if self.use_target_conditioning and target_backbone_coords is not None and not skip_residue_target:
            target_features = self.target_encoder(
                target_backbone_coords,
                target_backbone_mask,
                target_residue_types,
                target_seq_mask,
            )  # (B, L_t, hidden)
            residue_pair_geometry = compute_residue_pair_geometry(backbone_coords, target_backbone_coords)
            residue_pair_bias = self.residue_pair_bias_proj(residue_pair_geometry).permute(0, 3, 1, 2)

            if self.use_bidirectional_target_conditioning:
                for target_cross_attn, target_norm, backbone_cross_attn, backbone_norm in zip(
                    self.target_to_backbone_attention_layers,
                    self.target_to_backbone_norms,
                    self.cross_attention_layers,
                    self.cross_attn_norms,
                    strict=False,
                ):
                    # Let fixed target residues read the current peptide state first, then
                    # feed that peptide-aware target context back into the peptide.
                    target_attn_out = target_cross_attn(
                        query=target_features,
                        key=backbone_features,
                        value=backbone_features,
                        key_mask=seq_mask,
                        attn_bias=residue_pair_bias.transpose(-2, -1),
                    )
                    target_features = target_norm(target_features + target_attn_out)
                    backbone_attn_out = backbone_cross_attn(
                        query=backbone_features,
                        key=target_features,
                        value=target_features,
                        key_mask=target_seq_mask,
                        attn_bias=residue_pair_bias,
                    )
                    if self.preal_gate_target == "cross_attention":
                        backbone_attn_out = backbone_attn_out * occ_gate_residue
                    backbone_features = backbone_norm(backbone_features + backbone_attn_out)
            else:
                # Legacy one-way peptide<-target conditioning.
                for cross_attn, norm in zip(self.cross_attention_layers, self.cross_attn_norms, strict=False):
                    cross_attn_out = cross_attn(
                        query=backbone_features,
                        key=target_features,
                        value=target_features,
                        key_mask=target_seq_mask,
                        attn_bias=residue_pair_bias,
                    )
                    if self.preal_gate_target == "cross_attention":
                        cross_attn_out = cross_attn_out * occ_gate_residue
                    backbone_features = norm(backbone_features + cross_attn_out)
        else:
            target_features = None
            residue_pair_geometry = None

        # Optional FiLM modulation using global target context
        if (
            self.use_target_conditioning
            and self.use_film
            and target_backbone_coords is not None
            and target_features is not None
        ):
            # Pool target features to get global conditioning signal
            if target_seq_mask is not None:
                # Masked mean pooling
                target_mask_expanded = target_seq_mask.unsqueeze(-1).float()  # (B, L_t, 1)
                target_global = (target_features * target_mask_expanded).sum(dim=1) / (
                    target_mask_expanded.sum(dim=1) + 1e-8
                )  # (B, hidden)
            else:
                target_global = target_features.mean(dim=1)  # (B, hidden)

            # FiLM: modulate backbone features based on global target context
            target_global_expanded = target_global.unsqueeze(1).expand_as(backbone_features)  # (B, L, hidden)
            film_out = self.target_film(backbone_features, target_global_expanded)
            if self.preal_gate_target == "film":
                # Blend: gate=0.3 -> mostly keep original, gate=1.0 -> full FiLM
                backbone_features = backbone_features + (film_out - backbone_features) * occ_gate_residue
            else:
                backbone_features = film_out

        # DRAFT residue-frame stream: reason over residues in their local frames, inject a ZERO-INIT
        # additive residual into the pocket-conditioned per-residue features (byte-identical at graft,
        # so every downstream head + _forward_single is unchanged when off / at init), and stash the
        # coarse-latent predictions for the losses in InverseFoldingDiffusion.forward.
        residue_frame_outputs = None
        if self.use_residue_frame_stream:
            # Graft-safety: feed a DETACHED copy of backbone_features to the stream, so the stream's
            # supervision/consistency losses train the stream's own params but never backprop into the
            # base encoder (the stream reads the base as read-only conditioning). The injection below is
            # still added live (zero-init -> resume-safe), so its downstream gradient reaches the stream.
            residue_frame_outputs = self.residue_frame_stream(
                backbone_features.detach(), backbone_coords, backbone_mask, seq_mask
            )
            backbone_features = backbone_features + residue_frame_outputs["rf_injection"]

        # v2 residue-frame graph. Same graft convention as v1 (detached read of the pocket-
        # conditioned features so the stream's losses never backprop into the base encoder; zero-init
        # injection added live so its downstream gradient reaches the stream). v2 ALSO reads the target
        # residues (Cα + type) as extra attention keys -- still SE(3)-invariant. Mutually exclusive with
        # v1 (enforced at construction). Skipped entirely (byte-identical) unless the v2 flag is on.
        residue_frame_v2_outputs = None
        if self.use_residue_frame_stream_v2:
            residue_frame_v2_outputs = self.residue_frame_stream_v2(
                backbone_features.detach(),
                backbone_coords,
                backbone_mask,
                seq_mask,
                target_backbone_coords=target_backbone_coords,
                target_backbone_mask=target_backbone_mask,
                target_residue_types=target_residue_types,
                target_seq_mask=target_seq_mask,
                # hand the volumetric latent to the stereochem head (DETACHED -- graft-safety,
                # so the stereo BCE trains only its own head/vol-proj, never the volumetric head). None
                # when volumetric is off => the head falls back to frame features alone.
                vol_hidden=(vol_hidden.detach() if vol_hidden is not None else None),
            )
            backbone_features = backbone_features + residue_frame_v2_outputs["rf_injection"]

        # t-resolution head. Compute the EVOLVING P(D)_t here (in forward, once per batch), where
        # the current side-chain atoms + `t` + the static P(D)_prior (rfv2_stereo_logit) are all in scope.
        # The real-atom weight comes from the hard noised element state (PAD/GHOST=0 => weight 0); if
        # noised_element_types is absent we weight all slots equally (the head still reads the mean e3). The
        # None guards are belt-and-braces -- use_stereochem_t_resolution is AND-gated (IFD) with the stereo
        # head, so the v2 stream + rfv2_stereo_logit are always present when the flag is on. NOT consumed by
        # coords/element in this piece (that is ); exposed for 2b + sampling. Byte-identical when off.
        rfv2_stereo_pd_t = None
        if (
            self.use_stereochem_t_resolution
            and residue_frame_v2_outputs is not None
            and residue_frame_v2_outputs.get("rfv2_stereo_logit") is not None
        ):
            if noised_element_types is not None:
                stereo_t_real_w = (noised_element_types != 0).float()  # (B, L, max_sc) real atoms
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

        # coord-flow feedback sources. Turn the resolved P(D)_t into a per-residue e3-face-preference
        # `face_pref = 2·P(D)_t - 1` (+1 => D face, -1 => L face, 0 => undecided), DETACHED for graft-safety (the
        # feedback trains ONLY its own embed/deep-proj/scale, never the stereo/atom heads -- those are trained by
        # their own BCE, exactly as the volumetric deep-inject detaches `vol_hidden`). Two per-residue signals go
        # DOWN to _forward_single: (a) a learned latent (`stereo_feedback_embed`) for the zero-init per-layer
        # deep-inject, and (b) the SIGNED e3 axis (face_pref · frame-normal) for the explicit velocity bias. Both
        # None (=> feedback skipped, byte-identical) unless the flag is on AND P(D)_t was produced this pass.
        stereo_feedback_latent = None
        stereo_feedback_e3 = None
        if self.use_stereochem_t_resolution_feedback and rfv2_stereo_pd_t is not None:
            face_pref = 2.0 * rfv2_stereo_pd_t.detach() - 1.0  # (B, L) in [-1, 1]
            stereo_feedback_latent = self.stereo_feedback_embed(face_pref.unsqueeze(-1))  # (B, L, hidden)
            stereo_R_fb, _ = build_local_frames(backbone_coords, backbone_mask)  # (B, L, 3, 3)
            stereo_e3_axis = stereo_R_fb[..., :, 2]  # (B, L, 3): the frame's out-of-plane (L/D) axis
            stereo_feedback_e3 = face_pref.unsqueeze(-1) * stereo_e3_axis  # (B, L, 3) toward the resolved face

        # DRAFT deep per-layer injection: the per-residue frame latent (B, L, hidden), DETACHED, is handed
        # to _forward_single so each SE(3) layer can add a zero-init-projected, residue->atom-scattered
        # residual to its invariant node features. Detaching mirrors the neck's graft-safety: the deep path
        # trains ONLY its own per-layer projections and never backprops into the stream trunk (which is
        # trained by its supervision/consistency losses + the live neck injection above). None when off.
        residue_frame_hidden_deep = None
        if self.use_residue_frame_deep_inject and residue_frame_outputs is not None:
            residue_frame_hidden_deep = residue_frame_outputs["rf_hidden"].detach()

        # NEW: v2 residue-frame deep per-layer injection. The v2 stream ran a few lines up (inside THIS forward,
        # so it is recomputed at every sampling reverse step too -> no inference-silence risk), exposing its
        # post-trunk per-residue latent `rf_hidden`. Hand it down to _forward_single for the per-layer,
        # residue->atom-scattered zero-init deep-inject. CRITICAL: NOT detached (see the module-def doctrine) --
        # the atom-flow loss co-trains the v2 frame stream through this path. Contrast with the v1 line above,
        # which detaches. None when off / when the v2 stream is off (=> byte-identical).
        residue_frame_v2_hidden_deep = None
        if self.use_residue_frame_v2_deep_inject and residue_frame_v2_outputs is not None:
            residue_frame_v2_hidden_deep = residue_frame_v2_outputs["rf_hidden"]

        # volumetric deep per-layer injection. The volumetric head runs one level up (in
        # InverseFoldingDiffusion), which computes `vol_hidden` ONCE (t-independent) and hands it down. Detach
        # it here -- same graft-safety as the frame deep-inject: the deep path trains ONLY its own per-layer
        # projections and never backprops into the head (which is trained by its own self-occupancy loss).
        # None (=> injection skipped) unless the flag is on AND the IFD supplied a latent.
        vol_hidden_deep = None
        if self.use_volumetric_deep_inject and vol_hidden is not None:
            vol_hidden_deep = vol_hidden.detach()

        # (run10): density-field projection latent for the deep-inject -- replaces the degenerate pooled
        # vol_hidden as the source. NOT detached here (unlike vol_hidden): the volumetric HEAD is already
        # protected upstream -- the IFD builds this from `vol_density_pred.detach()`, so no gradient reaches the
        # head -- but `volumetric_density_inject_proj` is a fresh projection that MUST receive the flow's
        # coord/element gradient to train (detaching here would leave it zero-init forever). Preferred over
        # vol_hidden_deep in ; existence () + stereo below keep the original vol_hidden, so this
        # changes ONLY the deep-inject. None (=> fall back to vol_hidden) unless the flag is on AND supplied.
        vol_density_inject_deep = None
        if self.use_volumetric_deep_inject and vol_density_inject is not None:
            vol_density_inject_deep = vol_density_inject

        # volumetric -> existence coupling. Same t-independent `vol_hidden` handed down from the IFD,
        # DETACHED here for the SAME graft-safety reason as the deep-inject above: the existence bias trains
        # ONLY its own projection and never backprops into the volumetric head. Threaded into _forward_single
        # (below) where it biases the element track's PAD/GHOST logit. None (=> no bias applied) unless the flag
        # is on AND the IFD supplied a latent. Independent of deep-inject (either can be on without the other).
        vol_hidden_exist = None
        if self.use_volumetric_existence_coupling and vol_hidden is not None:
            vol_hidden_exist = vol_hidden.detach()

        # Global latent matching: pocket-only, time-free clean-context stream + per-residue latent pool.
        # backbone_features is the pocket-conditioned, time-free per-residue representation and never
        # sees the generated side-chain atoms, so it IS our f_pocket. Detach it so the latent-matching
        # gradient reaches only the clean stream + pool head, never the main denoiser.
        global_latent_z_pred = None
        global_latent_conf = None
        latent_clean_summary = None
        clean_per_layer = None
        if self.use_global_latent_matching:
            seq_mask_bool = (
                seq_mask.bool()
                if seq_mask is not None
                else torch.ones(batch_size, seq_len, dtype=torch.bool, device=sidechain_coords.device)
            )
            ca_coords_gl = backbone_coords[:, :, 1, :]  # (B, L, 3) -- CA is index 1
            # Detach the input: the clean stream + its downstream projections train from the latent loss
            # AND the main-task loss (via the per-layer injection), but neither perturbs the shared
            # backbone encoder through this path. `clean_per_layer` (list length num_layers) feeds the
            # per-layer interweave; `clean_feats` (final layer) feeds the pool head.
            clean_feats, clean_per_layer = self.clean_context_stream(
                backbone_features.detach(), ca_coords_gl, seq_mask_bool
            )
            global_latent_z_pred, global_latent_conf = self.latent_pool_head(clean_feats, ca_coords_gl, seq_mask_bool)
            if self.global_latent_condition_main_stream:
                # Detached [conf, relu(conf) * z_pred] -> FiLM'd into generated atoms in _forward_single.
                latent_clean_summary = clean_summary(global_latent_z_pred, global_latent_conf)  # (B, L, 1+D)

        # Residue-level count prediction from backbone+target features (before sidechain processing)
        residue_count_pred = self.residue_count_head(backbone_features).squeeze(-1)  # (B, L)

        # Shape prior: predict K anchor points per residue from backbone+target features
        shape_prior_anchors = None
        if self.use_shape_prior:
            K = self.shape_prior_n_anchors
            anchor_offsets = self.shape_prior_head(backbone_features)  # (B, L, K*3)
            anchor_offsets = anchor_offsets.view(batch_size, seq_len, K, 3)  # (B, L, K, 3)
            # Anchors are offsets from CA
            ca_coords = backbone_coords[:, :, 1, :]  # (B, L, 3) -- CA is index 1
            shape_prior_anchors = ca_coords.unsqueeze(2) + anchor_offsets  # (B, L, K, 3)

        # Plan latent: predict per-residue latent from backbone+target features
        plan_latent = None
        if self.use_plan_latent:
            plan_latent = self.plan_latent_head(backbone_features)  # (B, L, plan_latent_dim)

        # Per-residue LRT delta: predict how much to adjust the global LRT per residue
        lrt_input = backbone_features.detach() if getattr(self, "_dlrt_detach", False) else backbone_features
        residue_lrt_delta = self.residue_lrt_head(lrt_input).squeeze(-1)  # (B, L)
        # At sampling time, use EMA shadow weights for more stable delta if enabled
        ema_decay = getattr(self, "_dlrt_ema_decay", 0.0)
        if not self.training and ema_decay > 0 and hasattr(self, "_lrt_ema_shadow"):
            with torch.no_grad():
                residue_lrt_delta = self._lrt_ema_shadow(lrt_input.detach()).squeeze(-1)

        # Encode timestep
        time_features = self.time_embed(t.float())  # (B, hidden)

        # High-noise-weighted packing (FEATURE 2). Amplify the neighbour-x0 residual only where it is
        # non-redundant -- at high noise (large t) a DENOISED neighbour estimate carries information the
        # near-clean graph does not. Per-sample multiplier ``1 + (w - 1) * ramp`` with a smooth ramp
        # ``clamp((t_norm - 0.5) / 0.5, 0, 1)`` (0 for t_norm<=0.5, linear to 1 at t_norm=1.0),
        # t_norm = t / T. Default weight 1.0 -> factor is exactly 1.0 everywhere -> bit-exact no-op, so
        # the multiplier tensor is left as None (never touches the residual) unless the feature is on.
        neighbor_x0_highnoise_factor_all = None
        if self.use_neighbor_x0_packing and self.neighbor_x0_highnoise_weight != 1.0:
            _t_norm = (t.float() / max(self.num_timesteps, 1)).clamp(0.0, 1.0)  # (B,)
            _hn_ramp = ((_t_norm - 0.5) / 0.5).clamp(0.0, 1.0)  # (B,) 0 below t_norm=0.5, ->1 at t_norm=1
            neighbor_x0_highnoise_factor_all = 1.0 + (self.neighbor_x0_highnoise_weight - 1.0) * _hn_ramp  # (B,)

        # Build combined mask for backbone (still uses backbone_mask for valid backbone atoms)
        bb_combined_mask = backbone_mask * seq_mask.unsqueeze(-1) if seq_mask is not None else backbone_mask

        # Default noised_count to max_sc if not provided
        if noised_count is None:
            noised_count = torch.full(
                (batch_size, seq_len), max_sc, dtype=torch.float32, device=sidechain_coords.device
            )

        # Default prev_element_pred to zeros if not provided (no self-conditioning)
        if prev_element_pred is None:
            prev_element_pred = torch.zeros(
                batch_size, seq_len, max_sc, self.num_element_classes, device=sidechain_coords.device
            )

        # Process each sample in the batch separately (different graph sizes)
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
                neighbor_x0_highnoise_factor=(
                    neighbor_x0_highnoise_factor_all[b] if neighbor_x0_highnoise_factor_all is not None else None
                ),
                noised_element_types=noised_element_types[b] if noised_element_types is not None else None,
                noised_cluster_ids=noised_cluster_ids[b] if noised_cluster_ids is not None else None,
                noised_count=noised_count[b],
                prev_element_pred=prev_element_pred[b],
                prev_cluster_pred=prev_cluster_pred[b] if prev_cluster_pred is not None else None,
                cluster_feature_scale=cluster_feature_scale,
                target_coords=target_coords[b] if target_coords is not None else None,
                target_mask=target_mask[b] if target_mask is not None else None,
                target_features=(
                    target_features[b] if self.use_target_conditioning and target_features is not None else None
                ),
                target_seq_mask=target_seq_mask[b] if target_seq_mask is not None else None,
                target_residue_pair_geometry=(
                    residue_pair_geometry[b]
                    if self.use_target_conditioning and residue_pair_geometry is not None
                    else None
                ),
                target_atom_type=target_atom_type[b] if target_atom_type is not None else None,
                target_atom_element_type=target_atom_element_type[b] if target_atom_element_type is not None else None,
                target_atom_residue_type=(
                    target_atom_residue_type[b] if target_atom_residue_type is not None else None
                ),
                target_atom_is_backbone=(target_atom_is_backbone[b] if target_atom_is_backbone is not None else None),
                gt_sidechain_mask=gt_sidechain_mask[b] if gt_sidechain_mask is not None else None,
                ar_state=ar_state[b] if ar_state is not None else None,
                element_velocity_conditioning=(
                    element_velocity_conditioning[b] if element_velocity_conditioning is not None else None
                ),
                count_velocity_conditioning=(
                    count_velocity_conditioning[b] if count_velocity_conditioning is not None else None
                ),
                soft_element_probs=soft_element_probs[b] if soft_element_probs is not None else None,
                noised_bond_probs=noised_bond_probs[b] if noised_bond_probs is not None else None,
                shape_prior_anchors=shape_prior_anchors[b] if shape_prior_anchors is not None else None,
                prev_coord_pred=None,  # coord self-conditioning: still not wired from the outer path (dormant)
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
                latent_clean_per_layer=([cl[b] for cl in clean_per_layer] if clean_per_layer is not None else None),
                residue_frame_hidden=(residue_frame_hidden_deep[b] if residue_frame_hidden_deep is not None else None),
                residue_frame_v2_hidden=(
                    residue_frame_v2_hidden_deep[b] if residue_frame_v2_hidden_deep is not None else None
                ),
                volumetric_hidden=(vol_hidden_deep[b] if vol_hidden_deep is not None else None),
                volumetric_density_inject=(vol_density_inject_deep[b] if vol_density_inject_deep is not None else None),
                volumetric_hidden_exist=(vol_hidden_exist[b] if vol_hidden_exist is not None else None),
                stereo_feedback_hidden=(stereo_feedback_latent[b] if stereo_feedback_latent is not None else None),
                stereo_feedback_e3=(stereo_feedback_e3[b] if stereo_feedback_e3 is not None else None),
                stereo_feedback_ramp=stereo_feedback_ramp,
            )
            noise_pred[b] = outputs_b["noise_pred"]
            element_logits[b] = outputs_b["element_logits"]
            cluster_logits[b] = outputs_b["cluster_logits"]
            occupancy_logits[b] = outputs_b["occupancy_logits"]
            residue_centroid[b] = outputs_b["residue_centroid"]
            residue_cloud_logvar[b] = outputs_b["residue_cloud_logvar"]
            # The packing aux outputs are collected for EVERY batch element (ragged per-sample
            # shapes, so a list rather than a stacked tensor); the loss accumulates over all of
            # them. The `packing_contact_logits` key keeps its historical b==0 value for any
            # external reader. The remaining aux outputs stay b==0-only (pre-existing behaviour,
            # untouched here).
            packing_logits_per_sample.append(outputs_b.get("packing_contact_logits"))
            packing_nb_idx_per_sample.append(outputs_b.get("packing_neighbor_idx"))
            packing_nb_valid_per_sample.append(outputs_b.get("packing_neighbor_valid"))
            if b == 0:
                bond_logits_all = outputs_b.get("bond_logits")  # (n_res, S, S) or None
                cl = outputs_b.get("contact_logits")
                contact_logits_all = cl.unsqueeze(0) if cl is not None else None  # (1, L, S) -- loss expects (B, L, S)
                packing_logits_all = outputs_b.get("packing_contact_logits")  # (n_pairs, S, S) or None
                packing_nb_idx_all = outputs_b.get("packing_neighbor_idx")  # (n_res, k) or None (spatial only)
                packing_nb_valid_all = outputs_b.get("packing_neighbor_valid")  # (n_res, k) or None (spatial only)
                valence_logits_all = outputs_b.get("valence_logits")  # (L, S, 5) or None
                interaction_intent_all = outputs_b.get("interaction_intent_logits")  # (L, S, 4) or None

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
            # DRAFT residue-frame stream coarse latents (None unless use_residue_frame_stream).
            "rf_centroid_pred": residue_frame_outputs["rf_centroid_pred"]
            if residue_frame_outputs is not None
            else None,
            "rf_radial_pred": residue_frame_outputs["rf_radial_pred"] if residue_frame_outputs is not None else None,
            "rf_count_pred": residue_frame_outputs["rf_count_pred"] if residue_frame_outputs is not None else None,
            # v2 orientation-only latents (None unless use_residue_frame_stream_v2).
            "rfv2_centroid_pred": (
                residue_frame_v2_outputs["rfv2_centroid_pred"] if residue_frame_v2_outputs is not None else None
            ),
            "rfv2_chi1_pred": (
                residue_frame_v2_outputs["rfv2_chi1_pred"] if residue_frame_v2_outputs is not None else None
            ),
            # supervised stereochemistry (e3-sign) logit (None unless use_stereochem_head).
            "rfv2_stereo_logit": (
                residue_frame_v2_outputs["rfv2_stereo_logit"] if residue_frame_v2_outputs is not None else None
            ),
            # evolving atom-informed P(D)_t (None unless use_stereochem_t_resolution). Consumed by
            # its own BCE supervision now; by coords at sampling in .
            "rfv2_stereo_pd_t": rfv2_stereo_pd_t,
            "rfv2_stereo_w_t": (self._t_resolution_weight(t) if rfv2_stereo_pd_t is not None else None),
            "bond_logits": bond_logits_all if batch_size > 0 else None,
            "contact_logits": contact_logits_all if batch_size > 0 else None,
            "packing_contact_logits": packing_logits_all if batch_size > 0 else None,
            "packing_neighbor_idx": packing_nb_idx_all if batch_size > 0 else None,
            "packing_neighbor_valid": packing_nb_valid_all if batch_size > 0 else None,
            # Per-sample lists (length B) -- the packing contact loss sums over the whole batch.
            "packing_contact_logits_per_sample": packing_logits_per_sample,
            "packing_neighbor_idx_per_sample": packing_nb_idx_per_sample,
            "packing_neighbor_valid_per_sample": packing_nb_valid_per_sample,
            "valence_logits": valence_logits_all if batch_size > 0 else None,
            "shape_prior_anchors": shape_prior_anchors,  # (B, L, K, 3) from forward-level computation
            "interaction_intent_logits": interaction_intent_all if batch_size > 0 else None,
            "plan_latent": plan_latent,  # (B, L, plan_latent_dim) or None
            "global_latent_z_pred": global_latent_z_pred,  # (B, L, embed_dim) or None
            "global_latent_conf": global_latent_conf,  # (B, L) or None
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
            return coords_valid, node_features, cluster_condition, residue_idx_valid, identity

        cluster_keys = residue_idx_valid * self.max_sidechain_atoms + cluster_ids_valid
        unique_keys, inverse = torch.unique(cluster_keys, sorted=False, return_inverse=True)
        n_clusters = unique_keys.numel()

        pooled_counts = torch.bincount(inverse, minlength=n_clusters).to(coords_valid.dtype).unsqueeze(-1)
        pooled_coords = torch.zeros(
            n_clusters,
            coords_valid.shape[-1],
            device=coords_valid.device,
            dtype=coords_valid.dtype,
        )
        pooled_features = torch.zeros(
            n_clusters,
            node_features.shape[-1],
            device=node_features.device,
            dtype=node_features.dtype,
        )
        pooled_cluster_condition = torch.zeros(
            n_clusters,
            cluster_condition.shape[-1],
            device=cluster_condition.device,
            dtype=cluster_condition.dtype,
        )
        pooled_coords.index_add_(0, inverse, coords_valid)
        pooled_features.index_add_(0, inverse, node_features)
        pooled_cluster_condition.index_add_(0, inverse, cluster_condition)
        pooled_coords = pooled_coords / pooled_counts.clamp_min(1.0)
        pooled_features = pooled_features / pooled_counts.clamp_min(1.0)
        pooled_cluster_condition = pooled_cluster_condition / pooled_counts.clamp_min(1.0)

        pooled_residue_idx = torch.div(unique_keys, self.max_sidechain_atoms, rounding_mode="floor")
        return pooled_coords, pooled_features, pooled_cluster_condition, pooled_residue_idx, inverse

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
        bond_prev_x0: torch.Tensor | None = None,  # (L, max_sc, 3) detached predicted-x0 for the bond-inject descriptor
        bond_prev_mask: torch.Tensor
        | None = None,  # (L, max_sc) real-atom (predicted-existence) mask for the descriptor
        plan_latent: torch.Tensor | None = None,
        target_coords_for_intent: torch.Tensor | None = None,
        target_mask_for_intent: torch.Tensor | None = None,
        neighbor_x0_highnoise_factor: torch.Tensor | None = None,  # () per-sample high-noise packing multiplier
        neighbor_x0_coords: torch.Tensor | None = None,  # (L, max_sc, 3) neighbour clean-endpoint estimates
        neighbor_x0_mask: torch.Tensor | None = None,  # (L, max_sc) which neighbour slots hold a real atom
        neighbor_x0_trust: torch.Tensor | None = None,  # (L,) per-residue trust in [0, 1]
        neighbor_x0_apply: torch.Tensor | None = None,  # () scalar gate in [0, 1] for this sample
        t_original_res: torch.Tensor | None = None,  # (L,) per-residue original noise level
        t_conditioning_res: torch.Tensor | None = None,  # (L,) per-residue conditioning noise level
        recycle_index: int | None = None,  # 1-based ABSOLUTE recycle pass index (1 = unconditioned first pass)
        latent_clean_summary: torch.Tensor | None = None,  # (L, 1+embed_dim) detached global-latent FiLM feedback
        latent_clean_per_layer: list[torch.Tensor] | None = None,  # per-main-layer (L, hidden) clean features to inject
        residue_frame_hidden: torch.Tensor | None = None,  # (L, hidden) DETACHED residue-frame latent for deep inject
        residue_frame_v2_hidden: torch.Tensor | None = None,  # (L, hidden) LIVE v2-frame latent for the v2 deep inject
        volumetric_hidden: torch.Tensor | None = None,  # (L, hidden) DETACHED volumetric latent for deep inject
        volumetric_density_inject: torch.Tensor
        | None = None,  # (L, hidden) DETACHED density-field proj; preferred over volumetric_hidden for
        volumetric_hidden_exist: torch.Tensor | None = None,  # (L, hidden) DETACHED vol latent, existence bias
        stereo_feedback_hidden: torch.Tensor | None = None,  # (L, hidden) face_pref latent, deep-inject
        stereo_feedback_e3: torch.Tensor | None = None,  # (L, 3) signed e3 axis (face_pref * frame-normal)
        stereo_feedback_ramp: float = 1.0,  # head-first ramp scale (1.0 = inference/full; 0 = off)
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

        # --- Extract ALL sidechain atoms for valid residues ---
        # Build mask: all max_sc atoms for residues where seq_mask=True
        residue_mask = seq_mask.unsqueeze(-1).expand(-1, max_sc)  # (L, max_sc)
        sc_valid_idx = torch.where(residue_mask.reshape(-1))[0]  # reshape for non-contiguous

        if len(sc_valid_idx) == 0:
            return {
                "noise_pred": torch.zeros_like(sidechain_coords),
                "element_logits": torch.zeros(seq_len, max_sc, self.num_element_classes, device=device),
                "cluster_logits": torch.zeros(seq_len, max_sc, max_sc, device=device),
            }
        sc_cluster_condition = torch.zeros(len(sc_valid_idx), self.hidden_dim, device=device)

        sc_coords_flat = sidechain_coords.view(-1, 3)
        sc_coords_valid = sc_coords_flat[sc_valid_idx]  # (N_sc, 3)

        # Residue index for each sidechain atom
        sc_residue_idx_flat = torch.arange(seq_len, device=device).unsqueeze(-1).expand(-1, max_sc).reshape(-1)
        sc_residue_idx_valid = sc_residue_idx_flat[sc_valid_idx]

        # Atom position index for each sidechain atom
        sc_atom_idx_flat = torch.arange(max_sc, device=device).unsqueeze(0).expand(seq_len, -1).reshape(-1)
        sc_atom_idx_valid = sc_atom_idx_flat[sc_valid_idx]

        # Get noised_count values for valid atoms (used as input feature)
        # noised_count is shape (L,). Broadcast to (N_sc,) using residue indices.
        if self.count_embed_mode == "ordinal":
            # Ordinal: index an Embedding table by the integer count clamped to [0, max_sidechain_atoms].
            if noised_count is not None:
                count_idx = noised_count[sc_residue_idx_valid]  # (N_sc,)
            else:
                count_idx = torch.full((len(sc_valid_idx),), float(max_sc), device=device)
            count_idx = count_idx.round().clamp(0, self.max_sidechain_atoms).long()  # (N_sc,)
            sc_noised_count_features = self.noised_count_embed(count_idx)  # (N_sc, hidden//8)
        else:
            # Linear: monotone ramp of the raw scalar count (historical default).
            if noised_count is not None:
                noised_count_valid = noised_count[sc_residue_idx_valid].unsqueeze(-1)  # (N_sc, 1)
            else:
                noised_count_valid = torch.full((len(sc_valid_idx), 1), float(max_sc), device=device)
            sc_noised_count_features = self.noised_count_embed(noised_count_valid)  # (N_sc, hidden//8)

        # Get residue index for each sidechain atom (clamped to max supported)
        sc_residue_idx_clamped = sc_residue_idx_valid.clamp(0, self.max_residue_idx - 1)
        sc_residue_idx_features = self.residue_idx_embed(sc_residue_idx_clamped)  # (N_sc, hidden//8)

        # Get self-conditioning features (previous element prediction)
        if prev_element_pred is not None:
            prev_element_pred_flat = prev_element_pred.reshape(-1, self.num_element_classes)
            prev_element_pred_valid = prev_element_pred_flat[sc_valid_idx]
        else:
            prev_element_pred_valid = torch.zeros(len(sc_valid_idx), self.num_element_classes, device=device)
        sc_self_cond_element_features = self.self_cond_element_proj(prev_element_pred_valid)  # (N_sc, hidden//8)

        if noised_cluster_ids is not None:
            cluster_ids_flat = noised_cluster_ids.reshape(-1)
            cluster_ids_valid = cluster_ids_flat[sc_valid_idx].clamp(0, max_sc - 1)
        else:
            cluster_ids_valid = sc_atom_idx_valid
        sc_cluster_features = self.cluster_id_embed(cluster_ids_valid)  # (N_sc, hidden//8)

        if prev_cluster_pred is not None:
            prev_cluster_pred_flat = prev_cluster_pred.reshape(-1, max_sc)
            prev_cluster_pred_valid = prev_cluster_pred_flat[sc_valid_idx]
        else:
            prev_cluster_pred_valid = F.one_hot(sc_atom_idx_valid, num_classes=max_sc).float()
        sc_self_cond_cluster_features = self.self_cond_cluster_proj(prev_cluster_pred_valid.float())
        sc_cluster_features = sc_cluster_features * cluster_feature_scale
        sc_self_cond_cluster_features = sc_self_cond_cluster_features * cluster_feature_scale

        # Get element types for valid sidechain atoms
        if soft_element_probs is not None:
            # Element flow matching: soft probability vector matmul with embedding weights
            probs_flat = soft_element_probs.reshape(-1, soft_element_probs.shape[-1])  # (L*max_sc, C)
            probs_valid = probs_flat[sc_valid_idx]  # (N_sc, C)
            sc_element_features = probs_valid @ self.element_type_embed.weight  # (N_sc, hidden//4)
        elif noised_element_types is not None:
            element_types_flat = noised_element_types.view(-1)
            element_types_valid = element_types_flat[sc_valid_idx]  # (N_sc,)
            # Clamp to valid range [0, num_element_classes-1] (handles -1 padding that slipped through)
            element_types_valid = element_types_valid.clamp(0, self.num_element_classes - 1)
            sc_element_features = self.element_type_embed(element_types_valid)  # (N_sc, hidden//4)
        else:
            # If no element types provided, use PAD embedding for all
            sc_element_features = self.element_type_embed(
                torch.zeros(len(sc_valid_idx), device=device, dtype=torch.long)
            )  # (N_sc, hidden//4)

        # Build sidechain node features:
        # atom_embed + element_embed + noised_count_embed + self_cond_element
        # + cluster + self_cond_cluster + residue_idx + backbone_context + time
        sc_atom_features = self.sc_atom_embed(sc_atom_idx_valid)  # (N_sc, hidden//4)
        sc_backbone_context = backbone_features[sc_residue_idx_valid]  # (N_sc, hidden)
        sc_time_features = time_features.unsqueeze(0).expand(len(sc_valid_idx), -1)  # (N_sc, hidden)
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
        # Coordinate self-conditioning: project previous step's predicted x₀ into features
        if self.use_coord_self_conditioning:
            if prev_coord_pred is not None:
                prev_coords_flat = prev_coord_pred.reshape(-1, 3)  # (L*max_sc, 3)
                prev_coords_valid = prev_coords_flat[sc_valid_idx]  # (N_sc, 3)
            else:
                prev_coords_valid = torch.zeros(len(sc_valid_idx), 3, device=device)
            sc_coord_self_cond = self.coord_self_cond_proj(prev_coords_valid)  # (N_sc, hidden//8)
            cat_features.append(sc_coord_self_cond)
        sc_node_features = self.sc_node_proj(torch.cat(cat_features, dim=-1))  # (N_sc, hidden)

        # Autoregressive / K-mask state features: added as a residual after sc_node_proj (zero-init proj keeps
        # it a no-op on pre-AR checkpoints).
        if ar_state is not None:
            ar_state_flat = ar_state.reshape(-1)  # (L*max_sc,)
            ar_state_valid = ar_state_flat[sc_valid_idx]  # (N_sc,)
            # State 0 = inert sentinel: full-design-sample residues (K-mask) and padding get NO residual, so a
            # full-design sample is identical alone or in a mixed batch and matches leg-A sampling (ar_state=None).
            # Masking the residual (rather than zeroing embed row 0) keeps it stable under training -- no gradient
            # leaks into embed[0]. (Also restores pre-BUG-A behavior for AR-path "hidden" residues = no residual.)
            ar_state_residual = self.ar_state_proj(self.ar_state_embed(ar_state_valid))
            ar_state_apply = (ar_state_valid > 0).unsqueeze(-1).to(ar_state_residual.dtype)  # (N_sc, 1)
            sc_node_features = sc_node_features + ar_state_residual * ar_state_apply

        # Neighbour-x0 packing context: describe each slot by the OTHER residues' predicted clean
        # endpoints (or, for clean-context/inpainting positions, their given GT side chains). Added as
        # a zero-init residual so this is an exact no-op at graft. See compute_neighbor_x0_features for
        # the anti-self rule -- a residue never receives its own x0 estimate back.
        if self.use_neighbor_x0_packing:
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
                # TRAINING-ONLY corruption of the neighbour-x0 packing signal (weaken-the-neighbour ablation):
                # make the neighbour context unreliable so the model learns to DISCOUNT it. Off (byte-identical,
                # no RNG) unless a knob is >0. Values come purely from constructed hparams (env is resolved at the
                # LAUNCH layer into these hparams -- no forward-time env read). NEVER at inference (guarded on
                # self.training). See corrupt_neighbor_x0.
                _c_add = self.neighbor_x0_corrupt_add_prob
                _c_drop = self.neighbor_x0_corrupt_drop_prob
                _c_noise_prob = self.neighbor_x0_corrupt_noise_prob
                _c_noise_std = self.neighbor_x0_corrupt_coord_noise
                _c_disc = self.neighbor_x0_corrupt_disconnect_prob
                # FIX B: only run corruption when at least one sample in the batch actually applies the
                # packing residual (gate != 0). Otherwise the residual is zeroed downstream anyway, so
                # corrupting here would only burn RNG and drift an apply=0 sample.
                _gate_on = (neighbor_x0_apply is None) or bool((neighbor_x0_apply != 0).any())
                if self.training and _gate_on and (_c_add > 0.0 or _c_drop > 0.0 or _c_noise_prob > 0.0):
                    if not SidechainDenoiser._nx0_corrupt_announced:
                        print(
                            f"[nx0-corrupt] add={_c_add} drop={_c_drop} noise_prob={_c_noise_prob} "
                            f"noise={_c_noise_std} disconnect={_c_disc} active",
                            file=sys.stderr,
                            flush=True,
                        )
                        SidechainDenoiser._nx0_corrupt_announced = True
                    _ca_res = backbone_coords[:, 1, :].to(sc_coords_valid.dtype)  # (L, 3) per-residue Cα
                    # FIX C: protect clean/pinned context (trust==1.0 EXACTLY) -- corruption only ever hits
                    # the designed / co-generated neighbours (trust = 1 - t/T, strictly < 1). Gap >= 1/T so
                    # a >= 1 - 1e-4 threshold reliably separates the two.
                    _protect = nb_trust >= (1.0 - 1e-4)  # (L,) bool
                    # FIX B: isolate corruption RNG on a dedicated device generator so the L,S-dependent draw
                    # COUNT never perturbs the device (CUDA) global stream that dropout/SE3 consume. Seed it
                    # from ONE draw off the CPU global RNG (torch.randint's default generator is the CPU one)
                    # -> advances the CPU global by exactly 1, leaves the CUDA global untouched, stays
                    # deterministic/resumable.
                    _cgen = torch.Generator(device=_nx0_coords.device)
                    _cgen.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
                    _nx0_coords, _nb_valid = corrupt_neighbor_x0(
                        _nx0_coords,
                        _nb_valid,
                        _ca_res,
                        add_prob=_c_add,
                        drop_prob=_c_drop,
                        noise_prob=_c_noise_prob,
                        noise_std=_c_noise_std,
                        disconnect_prob=_c_disc,
                        radius=self.neighbor_x0_packing_radius,
                        generator=_cgen,
                        protect_mask=_protect,
                    )
                nb_feats = compute_neighbor_x0_features(
                    query_coords=sc_coords_valid,
                    query_residue_idx=sc_residue_idx_valid,
                    neighbor_coords=_nx0_coords,
                    neighbor_valid=_nb_valid,
                    neighbor_trust=nb_trust,
                    radius=self.neighbor_x0_packing_radius,
                )  # (N_sc, NEIGHBOR_X0_FEAT_DIM)
                nb_residual = self.neighbor_x0_proj(nb_feats.to(sc_node_features.dtype))
                # High-noise-weighted packing (FEATURE 2): scale the packing residual by the per-sample
                # multiplier (1.0 => unchanged => bit-exact). None on the default no-op path, so the
                # residual is untouched. Broadcasts over (N_sc, hidden). Applied in BOTH training and
                # sampling because both regimes reach this single denoiser path.
                if neighbor_x0_highnoise_factor is not None:
                    nb_residual = nb_residual * neighbor_x0_highnoise_factor.to(nb_residual.dtype)
                sc_node_features = sc_node_features + nb_residual * gate

            # Two-timestep delta: (embed(t_conditioning) - embed(t_original), (t_cond - t_orig)/T).
            # Identically zero when the two agree -> today's models are the exact special case.
            if t_original_res is not None and t_conditioning_res is not None:
                t_o = t_original_res.to(sc_node_features.dtype).reshape(-1)  # (L,)
                t_c = t_conditioning_res.to(sc_node_features.dtype).reshape(-1)  # (L,)
                delta_embed = self.time_embed(t_c) - self.time_embed(t_o)  # (L, hidden)
                # num_timesteps is wired from the model's `timesteps` (T) at construction.
                delta_scalar = ((t_c - t_o) / max(self.num_timesteps, 1)).unsqueeze(-1)  # (L, 1)
                t_delta_res = self.t_cond_delta_proj(torch.cat([delta_embed, delta_scalar], dim=-1))  # (L, hidden)
                sc_node_features = sc_node_features + t_delta_res[sc_residue_idx_valid] * gate

            # Recycle index: which refinement pass is this? Input is
            # (embed(j) - embed(1), 1 - 1/j) through a BIAS-FREE linear, so pass 1 contributes
            # EXACTLY zero for any weights -- the unconditioned first pass stays bit-exact.

            # ABSOLUTE j, never j/N. Pass 2-of-2 and pass 2-of-5 consume literally the same
            # conditioning (one prior refinement each) and future passes cannot reach backwards
            # into the current input, so conditioning quality depends on j alone. Encoding j/N
            # would give identical inputs different codes (1.0 vs 0.4) -- noise by construction.
            # It also means the recycle count no longer has to match between train and sample.

            # SATURATING scalar 1 - 1/j (0.00 / 0.50 / 0.67 / 0.75 / 0.80 at j = 1..5): information
            # gain from pass 1 -> 2 is large, 4 -> 5 nearly nil (AF2 recycling is largely saturated
            # by ~3), and it stays bounded for j never seen in training. The sinusoidal timestep
            # embedding is reused (rather than a fresh table) for the same reason -- it is defined
            # and bounded for every integer j, so extrapolation is well-posed.
            if recycle_index is not None:
                j = float(max(1, int(recycle_index)))
                j_t = torch.tensor([j], device=device, dtype=sc_node_features.dtype)  # (1,)
                j_embed = self.time_embed(j_t) - self.time_embed(torch.ones_like(j_t))  # (1, hidden)
                j_scalar = (1.0 - 1.0 / j_t).unsqueeze(-1)  # (1, 1)
                rc_res = self.recycle_index_proj(torch.cat([j_embed, j_scalar], dim=-1))  # (1, hidden)
                sc_node_features = sc_node_features + rc_res * gate

        # Chirality: signed volume from first 3 atoms per residue -> per-slot feature
        if self.use_chirality:
            # Compute signed volume per residue from slots 0,1,2
            max_sc = sidechain_coords.shape[1]
            if max_sc >= 3:
                a0 = sidechain_coords[:, 0, :]  # (L, 3)
                a1 = sidechain_coords[:, 1, :]  # (L, 3)
                a2 = sidechain_coords[:, 2, :]  # (L, 3)
                v01 = a1 - a0
                v02 = a2 - a0
                signed_vol = (v01 * torch.cross(v02, v01, dim=-1)).sum(dim=-1)  # (L,)
                # Normalize to ~[-1, 1] range (typical volumes ~10-50 Å³)
                signed_vol = torch.tanh(signed_vol / 20.0)
            else:
                signed_vol = torch.zeros(sidechain_coords.shape[0], device=device)
            # Expand to per-slot and project
            chirality_per_slot = signed_vol[sc_residue_idx_valid].unsqueeze(-1)  # (N_sc, 1)
            sc_node_features = sc_node_features + self.chirality_proj(chirality_per_slot)

        # Plan latent modulation: add per-residue plan latent to per-slot features
        if self.use_plan_latent and plan_latent is not None:
            # plan_latent: (L, plan_latent_dim) -> expand to per-slot
            plan_per_slot = self.plan_latent_to_slot(plan_latent[sc_residue_idx_valid])  # (N_sc, hidden)
            sc_node_features = sc_node_features + plan_per_slot

        # run10 target deep-inject source; stays None unless the sidechain->target cross-attention below runs.
        sc_target_attn = None
        if self.use_target_conditioning and self.use_sidechain_target_residue_attention and target_features is not None:
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
        (
            sc_graph_coords,
            sc_graph_features,
            sc_graph_cluster_condition,
            sc_graph_residue_idx,
            sc_cluster_inverse,
        ) = self._pool_cluster_nodes(
            sc_coords_valid,
            sc_node_features,
            sc_cluster_condition,
            sc_residue_idx_valid,
            cluster_ids_valid if noised_cluster_ids is not None else None,
        )
        n_binder_sc = len(sc_graph_coords)

        # --- Extract valid binder backbone atoms ---
        bb_valid_idx = torch.where(backbone_mask.view(-1))[0]
        bb_coords_flat = backbone_coords.view(-1, 3)
        bb_coords_valid = bb_coords_flat[bb_valid_idx]  # (N_bb, 3)

        # Residue and atom indices for backbone
        bb_residue_idx_flat = torch.arange(seq_len, device=device).unsqueeze(-1).expand(-1, 4).reshape(-1)
        bb_residue_idx_valid = bb_residue_idx_flat[bb_valid_idx]
        bb_atom_idx_flat = torch.arange(4, device=device).unsqueeze(0).expand(seq_len, -1).reshape(-1)
        bb_atom_idx_valid = bb_atom_idx_flat[bb_valid_idx]

        # Build backbone node features
        bb_atom_features = self.bb_atom_embed(bb_atom_idx_valid)  # (N_bb, hidden//4)
        bb_backbone_context = backbone_features[bb_residue_idx_valid]  # (N_bb, hidden)
        bb_time_features = time_features.unsqueeze(0).expand(len(bb_valid_idx), -1)  # (N_bb, hidden)
        bb_node_features = self.bb_node_proj(
            torch.cat([bb_atom_features, bb_backbone_context, bb_time_features], dim=-1)
        )  # (N_bb, hidden)

        n_binder_bb = len(bb_valid_idx)

        # --- Extract valid target atoms with proper per-atom features ---
        target_coords_valid = None
        target_node_features = None
        n_target = 0

        if target_coords is not None and target_mask is not None:
            target_valid_idx = torch.where(target_mask)[0]
            if len(target_valid_idx) > 0:
                target_coords_valid = target_coords[target_valid_idx]  # (N_target, 3)
                n_target = len(target_valid_idx)

                # Build per-atom target features from atom type, element type, residue type, and is_backbone
                # This provides local structural information for each target atom
                if (
                    target_atom_type is not None
                    and target_atom_element_type is not None
                    and target_atom_residue_type is not None
                    and target_atom_is_backbone is not None
                ):
                    # Extract features for valid atoms
                    atom_type_valid = target_atom_type[target_valid_idx]  # (N_target,)
                    element_type_valid = target_atom_element_type[target_valid_idx]  # (N_target,)
                    residue_type_valid = target_atom_residue_type[target_valid_idx]  # (N_target,)
                    is_backbone_valid = target_atom_is_backbone[target_valid_idx].long()  # (N_target,)

                    # Clamp indices to valid ranges to handle any edge cases
                    atom_type_valid = atom_type_valid.clamp(0, 17)  # 0-17 for 4 bb + 14 sc
                    # PAD-aware (v1): clamp 0..NUM-1 so X (id 4) keeps its row, not clamped onto O.
                    element_type_valid = element_type_valid.clamp(0, NUM_ELEMENT_TYPES - 1)
                    residue_type_valid = residue_type_valid.clamp(0, 20)  # 0-20 for amino acids + unknown

                    # Embed each feature type
                    atom_type_feat = self.target_atom_type_embed(atom_type_valid)  # (N_target, hidden//4)
                    element_type_feat = self.target_element_type_embed(element_type_valid)  # (N_target, hidden//4)
                    residue_type_feat = self.target_residue_type_embed(residue_type_valid)  # (N_target, hidden//4)
                    if self.use_jackie:
                        jackie_feat = self.jackie_features_graph[residue_type_valid.clamp(0, 20)]  # (N_target, 25)
                        residue_type_feat = residue_type_feat + self.target_residue_type_proj(jackie_feat)  # additive
                    is_backbone_feat = self.target_is_backbone_embed(is_backbone_valid)  # (N_target, hidden//8)

                    # Concatenate and project to hidden dim
                    target_cat_features = torch.cat(
                        [atom_type_feat, element_type_feat, residue_type_feat, is_backbone_feat], dim=-1
                    )  # (N_target, hidden + hidden//8)
                    target_node_features = self.target_atom_proj(target_cat_features)  # (N_target, hidden)
                else:
                    # Fallback: zeros if per-atom features not provided
                    target_node_features = torch.zeros(n_target, self.hidden_dim, device=device)

        # --- Build combined graph ---
        # Node order: [binder_sc, binder_bb, target]
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

        # Get CA coordinates for prefiltering
        # CA is index 1 in backbone (N, CA, C, O)
        binder_ca_coords = backbone_coords[:, 1, :]  # (L, 3)

        # Extract CA atoms from valid backbone block for split cutoff mode
        ca_mask = bb_atom_idx_valid == 1
        binder_ca_only_coords = bb_coords_valid[ca_mask]  # (N_ca, 3)
        ca_bb_indices = torch.where(ca_mask)[0]  # positions within backbone block

        # Build multi-type radius graph
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
            num_edge_types=(
                self._graph_num_edge_types if self._graph_num_edge_types is not None else self.num_edge_types
            ),
        )

        # Embed edge types
        edge_attr = self.edge_embed(edge_type)  # (E, edge_embed_dim)

        # Occupancy-gated graph edges: attenuate binder-target edge features by binder node's
        # noised occupancy (non-PAD fraction). Same signal as residue-level gating above.
        if self.preal_gate_target == "graph_edges" and noised_element_types is not None:
            # Compute occupancy gate per slot from noised element types
            element_types_flat = noised_element_types.view(-1)
            occ_flat = (element_types_flat != 0).float().clamp(min=0.3)
            occ_valid = occ_flat[sc_valid_idx]  # (N_sc,)

            # Pool through cluster pooling (same mean-pool as features)
            occ_graph = torch.zeros(n_binder_sc, device=device)
            occ_graph.index_add_(0, sc_cluster_inverse, occ_valid)
            pool_counts = torch.bincount(sc_cluster_inverse, minlength=n_binder_sc).float().clamp(min=1.0)
            occ_graph = occ_graph / pool_counts  # (n_binder_sc,)

            # Gate binder-target edges by the binder sidechain node's occupancy
            bt_mask = edge_type == EDGE_TYPE_BINDER_TARGET
            if bt_mask.any():
                src = edge_index[0, bt_mask]
                dst = edge_index[1, bt_mask]
                # For each binder-target edge, one end is binder sc (< n_binder_sc)
                src_is_sc = src < n_binder_sc
                dst_is_sc = dst < n_binder_sc
                binder_sc_idx = torch.where(src_is_sc, src, dst)
                has_sc = src_is_sc | dst_is_sc
                gate = torch.ones(bt_mask.sum(), 1, device=device)
                if has_sc.any():
                    gate[has_sc] = occ_graph[binder_sc_idx[has_sc].clamp(max=n_binder_sc - 1)].unsqueeze(-1)
                edge_attr = edge_attr.clone()
                edge_attr[bt_mask] = edge_attr[bt_mask] * gate

        # Per-layer clean-context interweave (the clean_inject_proj): map each clean-stream layer's
        # per-residue features onto the graph nodes (binder side-chain + backbone; target nodes get zero)
        # via the corresponding ZERO-INIT, bias-free projection, so the injection is an exact no-op at
        # init and grows as it learns. Passed to the SE(3) stack; each main layer i adds injection i.
        layer_conditioning = None
        if self.use_global_latent_matching and latent_clean_per_layer is not None:
            n_total = all_features.shape[0]
            layer_conditioning = []
            for li, clean_li in enumerate(latent_clean_per_layer):
                proj = self.clean_inject_proj[li]
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(clean_li[sc_graph_residue_idx].to(all_features.dtype))
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(
                        clean_li[bb_residue_idx_valid].to(all_features.dtype)
                    )
                layer_conditioning.append(node_cond)

        # DRAFT deep per-layer residue-frame injection. Same residue->atom scatter + ZERO-INIT projection
        # pattern as the clean-context interweave above, but the source is the residue-frame stream's
        # (detached) per-residue latent. Each SE(3) layer i adds residue_frame_deep_proj[i](latent[res(node)])
        # to that layer's node features. CRITICAL (equivariance): node_cond is added to the SE(3) node
        # feature stream, which is TYPE-0 / invariant (coordinates are a pure pass-through; the layers never
        # touch x). The latent itself is a function of local-frame invariants only, so the injected signal is
        # SE(3)-invariant and adding it cannot break the transformer's equivariance. Target nodes get 0. When
        # both this and the clean-context interweave are active, the two per-layer residuals SUM (independent
        # zero-init grafts). Zero-init => exact no-op at init => byte-identical + resume-safe.
        if self.use_residue_frame_deep_inject and residue_frame_hidden is not None:
            n_total = all_features.shape[0]
            rf_latent = residue_frame_hidden.to(all_features.dtype)  # (L, hidden), already detached
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.residue_frame_deep_proj)
            for li, proj in enumerate(self.residue_frame_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(rf_latent[sc_graph_residue_idx])
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(rf_latent[bb_residue_idx_valid])
                # Sum with the clean-context injection when present (both are zero-init grafts).
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )

        # BACKBONE deep-inject (run10-next): SAME sc+bb residue->node scatter as the frame deep-inject above,
        # sourced from the (DETACHED) BackboneEncoder per-residue latent `backbone_features` (L, hidden) -- the
        # backbone geometry + target cross-attn + FiLM encoding that is otherwise only concatenated into the
        # layer-0 node features. Zero-init proj => +0 at init => byte-identical off; detached => the atom-flow
        # loss cannot reshape the shared backbone encoder through this per-layer path.
        if self.use_backbone_deep_inject and backbone_features is not None:
            n_total = all_features.shape[0]
            bb_latent = backbone_features.detach().to(all_features.dtype)  # (L, hidden)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.backbone_deep_proj)
            for li, proj in enumerate(self.backbone_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(bb_latent[sc_graph_residue_idx])
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(bb_latent[bb_residue_idx_valid])
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )

        # NEW: v2 residue-frame deep per-layer injection. IDENTICAL residue->atom scatter + ZERO-INIT projection
        # pattern as the v1 frame deep-inject above, but the source is the v2 stream's per-residue `rf_hidden`
        # latent and -- critically -- it is NOT detached (the caller handed a LIVE tensor; see the module-def
        # doctrine). So the coord/velocity loss backprops through proj[li] into the v2 frame stream's own params
        # (bidirectional co-training), whereas the v1/volumetric/stereo channels are one-way (detached source).
        # Equivariance is unchanged from the v1 argument: the injection goes into the SE(3) node feature stream,
        # which is TYPE-0 / invariant (coordinates pass through untouched), and `rf_hidden` is a function of
        # local-frame invariants only, so the added signal is SE(3)-invariant. Target nodes get 0. When any other
        # deep-inject graft is active, all zero-init residuals SUM. Zero-init => exact no-op at init =>
        # byte-identical + resume-safe.
        if self.use_residue_frame_v2_deep_inject and residue_frame_v2_hidden is not None:
            n_total = all_features.shape[0]
            # frame_v2_deep_inject_detach (run10-next): default LIVE (flow co-trains the v2 stream); True =>
            # detach so noisy atom-flow losses cannot repurpose the interpretable geometry/stereo prior.
            rfv2_latent = (
                residue_frame_v2_hidden.detach() if self.frame_v2_deep_inject_detach else residue_frame_v2_hidden
            ).to(all_features.dtype)  # (L, hidden)
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

        # x0 BOND-INJECT deep per-layer injection (angle + length). Featurize the PREDICTED CLEAN x0 side-chain
        # cloud (bond_prev_x0) into a per-residue internal-geometry descriptor (angles + lengths), encode to a
        # latent, and inject it into EVERY SE(3) layer via ZERO-INIT projections -- IDENTICAL residue->atom
        # scatter pattern as the frame-v2 deep-inject above. bond_prev_x0 is the DETACHED predicted x0, so this channel is a
        # one-way read-out of the model's own predicted geometry (not a gradient path back into the flow that
        # produced it); the coord-flow loss still trains bond_angle_deep_proj + bond_angle_inject_encoder. The
        # descriptor is a function of edge DIRECTIONS + bonded-pair LENGTH magnitudes (translation/rotation-invariant) -> the added type-0
        # signal is SE(3)-invariant (same argument as the frame inject). Target nodes get 0. Skipped when
        # bond_prev_x0 is None (no recycle pass ran) => inert. Zero-init => byte-identical off.
        # NOTE (review): the descriptor uses `residue_mask` (graph-valid slots); ghost slots sit at Cα and are
        # EXCLUDED via bond_prev_mask (predicted-real, P(real)>0.5) -- not merely down-weighted.
        if self.use_bond_angle_deep_inject and bond_prev_x0 is not None:
            n_total = all_features.shape[0]
            _ba_coords = bond_prev_x0.reshape(seq_len, max_sc, 3).detach().to(all_features.dtype)
            # Descriptor over REAL predicted atoms only: AND the graph-valid residue_mask with the recycle's
            # hard predicted-existence mask (P(real)>0.5). Ghost / Cα-collapsed slots are EXCLUDED, not merely
            # distance-down-weighted (review: leaving them in shifts the descriptor by up to ~0.7 max-abs).
            _ba_valid = residue_mask.to(torch.bool)
            if bond_prev_mask is not None:
                _ba_valid = _ba_valid & bond_prev_mask.reshape(seq_len, max_sc).to(torch.bool)
            ba_latent = self.bond_angle_inject_encoder(bond_geom_descriptor(_ba_coords, _ba_valid))  # (L, hidden)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.bond_angle_deep_proj)
            for li, proj in enumerate(self.bond_angle_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                if n_binder_sc > 0:
                    node_cond[:n_binder_sc] = proj(ba_latent[sc_graph_residue_idx])
                if n_binder_bb > 0:
                    node_cond[n_binder_sc : n_binder_sc + n_binder_bb] = proj(ba_latent[bb_residue_idx_valid])
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )

        # volumetric deep per-layer injection. IDENTICAL residue->atom scatter + ZERO-INIT projection
        # pattern as the residue-frame deep-inject above, but the source is the volumetric head's (detached)
        # per-residue `vol_hidden` latent. Each SE(3) layer i adds volumetric_deep_proj[i](vol_latent[res(node)])
        # to that layer's node features. Equivariance (same argument as the frame deep-inject): the injection
        # goes into the SE(3) node feature stream, which is TYPE-0 / invariant (coordinates pass through
        # untouched), and `vol_hidden` is a function of local-frame invariants only, so the added signal is
        # SE(3)-invariant. Target nodes get 0. When the clean-context and/or frame deep-inject residuals are
        # also active, all zero-init grafts SUM. Zero-init => exact no-op at init => byte-identical + resume-safe.
        if self.use_volumetric_deep_inject and (volumetric_density_inject is not None or volumetric_hidden is not None):
            n_total = all_features.shape[0]
            # (run10): prefer the zero-init density-field projection (informative) over the degenerate pooled
            # vol_hidden. When the density-inject latent is supplied it fully replaces vol_hidden AS THE DEEP-INJECT
            # SOURCE (existence/stereo above still use vol_hidden). Falls back to vol_hidden when not supplied.
            _vol_src = volumetric_density_inject if volumetric_density_inject is not None else volumetric_hidden
            vol_latent = _vol_src.to(all_features.dtype)  # (L, hidden), already detached
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

        # TARGET deep per-layer injection (run10): reinforce sc_target_attn at every SE(3) layer (SIDECHAIN nodes
        # only). Zero-init proj => +0 at init => byte-identical off. Detached: the deep path trains ONLY its own
        # per-layer projections; the cross-attention itself is trained one-way by the layer-0 sidechain_target_film.
        # Gated on sc_target_attn (built above only when target-conditioning + sidechain-target attention are on).
        if self.use_target_deep_inject and sc_target_attn is not None and n_binder_sc > 0:
            n_total = all_features.shape[0]
            tgt_latent = sc_target_attn.detach().to(all_features.dtype)  # (N_sc == n_binder_sc, hidden)
            if layer_conditioning is None:
                layer_conditioning = [None] * len(self.target_deep_proj)
            for li, proj in enumerate(self.target_deep_proj):
                node_cond = torch.zeros(n_total, self.hidden_dim, device=device, dtype=all_features.dtype)
                node_cond[:n_binder_sc] = proj(tgt_latent)
                layer_conditioning[li] = (
                    node_cond if layer_conditioning[li] is None else layer_conditioning[li] + node_cond
                )

        # t-resolution coord-flow feedback -- LEARNED channel. IDENTICAL residue->atom scatter +
        # ZERO-INIT projection pattern as the volumetric deep-inject above, but the source is the (detached)
        # face_pref-derived per-residue latent and each layer's residual is additionally scaled by the head-first
        # ramp, so an UNTRAINED P(D)_t head never steers coords early. Equivariance: same argument as the
        # volumetric deep-inject (injection into TYPE-0/invariant node features; face_pref is a frame invariant).
        # Skipped at ramp==0 (=> byte-identical at training_progress=0) and zero-init => no-op at graft/resume.
        # Target nodes get 0; sums with any other active grafts.
        if (
            self.use_stereochem_t_resolution_feedback
            and stereo_feedback_hidden is not None
            and stereo_feedback_ramp > 0.0
        ):
            n_total = all_features.shape[0]
            st_latent = stereo_feedback_hidden.to(all_features.dtype)  # (L, hidden), already detached
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

        # Run SE(3) Transformer
        out_features, out_coords = self.transformer(
            all_features, all_coords, edge_index, edge_attr, layer_conditioning=layer_conditioning
        )

        # Extract outputs for sidechain atoms only
        # The coordinate stream is now an identity pass-through: the transformer returns the input
        # coords unchanged, and we predict velocity/noise from the invariant feature stream only.
        _ = out_coords[:n_binder_sc]  # (N_sc, 3) - coords unused; identity pass-through
        sc_out_features = out_features[:n_binder_sc]
        sc_out_features = sc_out_features[sc_cluster_inverse]

        # Intra-residue slot attention: structured all-to-all within each residue
        if self.use_slot_attention:
            n_valid_residues = seq_mask.sum().item()
            sc_out_features = self.slot_attention(sc_out_features, n_valid_residues)

        # Directional slot attention: distal scouts -> proximal sticky slots
        # Compute occupancy logits first for P(real) gating, then recompute after attention
        if self.use_directional_slot_attention:
            pre_attn_occupancy = self.occupancy_head(sc_out_features)  # (N_sc, 1) -- for gating only
            n_valid_residues = seq_mask.sum().item()
            # During training: use GT mask for stable scout assignment
            # During sampling: gt_sidechain_mask is None, falls back to predicted P(real)
            gt_mask_flat = None
            if gt_sidechain_mask is not None:
                gt_mask_flat = gt_sidechain_mask[seq_mask].reshape(n_valid_residues * self.max_sidechain_atoms)
            sc_out_features = self.directional_slot_attention(
                sc_out_features,
                n_valid_residues,
                occupancy_logits=pre_attn_occupancy,
                gt_mask=gt_mask_flat,
            )

        # Element-velocity coupling: FiLM modulation of sc_out_features by PAD/non-PAD state.
        # Intentionally placed BEFORE all heads (velocity, element, occupancy, mixture) so that
        # the ghost/real state modulates the shared representation -- this is a broad feedback
        # mechanism, not velocity-only, creating coupling across all output predictions.
        if self.use_element_velocity_coupling and element_velocity_conditioning is not None:
            evc_flat = element_velocity_conditioning[:seq_len].reshape(-1)  # (L*max_sc,)
            evc_valid = evc_flat[sc_valid_idx].unsqueeze(-1)  # (N_sc, 1)
            evc_embed = self.evc_state_proj(evc_valid)  # (N_sc, hidden_dim)
            sc_out_features = self.evc_film(sc_out_features, evc_embed)

        # Count-velocity coupling: per-slot P(real) from per-residue count via sigmoid step.
        # count_velocity_conditioning is (L,) per-residue count. Convert to per-slot via
        # sigmoid(4*(count - slot_idx - 0.5)): slots below count -> ~1.0, above -> ~0.0.
        if self.use_count_velocity_coupling and count_velocity_conditioning is not None:
            slot_idx = torch.arange(max_sc, device=device).float()  # (max_sc,)
            count_per_res = count_velocity_conditioning[:seq_len]  # (L,)
            p_real_per_slot = torch.sigmoid(4.0 * (count_per_res.unsqueeze(1) - slot_idx - 0.5))  # (L, max_sc)
            cvc_flat = p_real_per_slot.reshape(-1)  # (L*max_sc,)
            cvc_valid = cvc_flat[sc_valid_idx].unsqueeze(-1)  # (N_sc, 1)
            cvc_embed = self.cvc_proj(cvc_valid)  # (N_sc, hidden_dim)
            sc_out_features = self.cvc_film(sc_out_features, cvc_embed)

        # Intra-residue bond attention: refine per-slot features using predicted bond graph
        bond_logits_out = None
        if self.use_bond_attention:
            # Reshape flat (N_sc,) features to (n_residues, max_sc, D) for intra-residue attention
            n_valid_res_ba = int(sc_residue_idx_valid.max().item()) + 1
            h_3d = torch.zeros(n_valid_res_ba, max_sc, sc_out_features.shape[-1], device=device)
            coords_3d = torch.zeros(n_valid_res_ba, max_sc, 3, device=device)
            mask_3d = torch.zeros(n_valid_res_ba, max_sc, dtype=torch.bool, device=device)
            slot_within_res = sc_valid_idx % max_sc
            h_3d[sc_residue_idx_valid, slot_within_res] = sc_out_features
            coords_3d[sc_residue_idx_valid, slot_within_res] = sc_coords_valid
            mask_3d[sc_residue_idx_valid, slot_within_res] = True

            # For co-diffusion: reshape noised bond probs to match bond attention's (n_res, S, S) shape
            noised_bp_3d = None
            if noised_bond_probs is not None:
                # noised_bond_probs is (L, max_sc, max_sc) -- per-sample from _forward_single
                noised_bp_3d = noised_bond_probs[:n_valid_res_ba]  # (n_res, max_sc, max_sc)

            h_3d, bond_logits_3d = self.bond_attention(h_3d, coords_3d, mask_3d, noised_bond_probs=noised_bp_3d)
            bond_logits_out = bond_logits_3d  # (n_res, max_sc, max_sc) -- for loss

            # Write back to flat features
            sc_out_features = h_3d[sc_residue_idx_valid, slot_within_res]

        # Cross-residue packing attention: attend between atoms of paired residues
        # (sequence-consecutive by default; spatially-nearest when cross_residue_packing_spatial).
        packing_contact_logits_out = None
        packing_neighbor_idx_out = None
        packing_neighbor_valid_out = None
        if self.use_cross_residue_packing:
            n_valid_res_crp = int(sc_residue_idx_valid.max().item()) + 1
            h_3d_crp = torch.zeros(n_valid_res_crp, max_sc, sc_out_features.shape[-1], device=device)
            coords_3d_crp = torch.zeros(n_valid_res_crp, max_sc, 3, device=device)
            mask_3d_crp = torch.zeros(n_valid_res_crp, max_sc, dtype=torch.bool, device=device)
            slot_within_res_crp = sc_valid_idx % max_sc
            h_3d_crp[sc_residue_idx_valid, slot_within_res_crp] = sc_out_features
            coords_3d_crp[sc_residue_idx_valid, slot_within_res_crp] = sc_coords_valid
            mask_3d_crp[sc_residue_idx_valid, slot_within_res_crp] = True

            # Spatial mode needs a per-residue anchor. Reuse the backbone-only pseudo-Cbeta
            # (an INPUT, never GT side chains) so neighbour selection stays leak-free.
            crp_residue_pos = None
            if self.cross_residue_packing_spatial:
                cb_dir = compute_pseudo_cb_direction(backbone_coords)  # (L, 3)
                crp_residue_pos = backbone_coords[:, 1, :] + 1.53 * cb_dir  # (L, 3) virtual CB

            (
                h_3d_crp,
                packing_contact_logits_out,
                packing_neighbor_idx_out,
                packing_neighbor_valid_out,
            ) = self.cross_residue_packing(h_3d_crp, coords_3d_crp, mask_3d_crp, residue_pos=crp_residue_pos)
            sc_out_features = h_3d_crp[sc_residue_idx_valid, slot_within_res_crp]

        # Valence demand: predict per-slot valence, use as FiLM conditioning before output heads
        valence_logits_valid = None
        if self.use_valence_demand:
            valence_logits_valid = self.valence_head(sc_out_features)  # (N_sc, 5)
            valence_probs = torch.softmax(valence_logits_valid, dim=-1)  # (N_sc, 5)
            valence_embed = self.valence_embed(valence_probs)  # (N_sc, hidden_dim)
            sc_out_features = self.valence_film(sc_out_features, valence_embed)

        # Global-latent FiLM feedback: the confidence-gated per-residue latent (detached) modulates
        # the generated side-chain features just before the output heads (film_layers="last"). The
        # summary is already detached, so this conditions the main stream without any gradient
        # reaching the clean stream or the pool head.
        if (
            self.use_global_latent_matching
            and self.global_latent_condition_main_stream
            and latent_clean_summary is not None
        ):
            latent_summary_per_slot = latent_clean_summary[sc_residue_idx_valid]  # (N_sc, 1+embed_dim)
            sc_out_features = self.latent_film(sc_out_features, latent_summary_per_slot)

        # Predict occupancy and mixture from features -- optionally with info bottleneck dropout
        # to prevent the mixture/occupancy heads from reading GT count info from teacher-forced features
        if self.training and hasattr(self, "_mixture_head_dropout") and self._mixture_head_dropout > 0:
            mixture_features = F.dropout(sc_out_features, p=self._mixture_head_dropout, training=True)
        else:
            mixture_features = sc_out_features
        occupancy_logits_valid = self.occupancy_head(mixture_features)  # (N_sc, 1)

        # Predict pocket contacts: per-slot binary "contacts target within 4Å?"
        contact_logits_valid = self.pocket_contact_head(sc_out_features)  # (N_sc, 1)

        # Predict noise for coordinates
        noise_valid = self.output_proj(sc_out_features)  # (N_sc, 3)

        # t-resolution coord-flow feedback -- EXPLICIT DIRECTIONAL channel. Add a velocity bias along the
        # SIGNED e3 axis (face_pref · frame-normal) so atoms are pulled toward the RESOLVED L/D face. This is the
        # part that GUARANTEES the steer direction; the invariant deep-inject above is expressive but not
        # directionally constrained. Equivariant: e3 rotates with the frame, face_pref is a frame invariant, so
        # face_pref·e3 is a proper type-1 vector added to the (global-frame) velocity. Gated by a ZERO-INIT
        # learnable scale (=> bias EXACTLY 0 at graft/resume => byte-identical) and the head-first ramp; skipped
        # at ramp==0 (byte-identical at training_progress=0). Applied in BOTH training and sampling (this head
        # runs at both), so the mid-trajectory decision actually moves the atoms at inference.
        if self.use_stereochem_t_resolution_feedback and stereo_feedback_e3 is not None and stereo_feedback_ramp > 0.0:
            e3_per_slot = stereo_feedback_e3[sc_residue_idx_valid].to(noise_valid.dtype)  # (N_sc, 3)
            noise_valid = noise_valid + stereo_feedback_ramp * self.stereo_feedback_face_scale * e3_per_slot

        # Shape prior velocity bias: pull atoms toward nearest predicted anchor
        if self.use_shape_prior and shape_prior_anchors is not None:
            # shape_prior_anchors is (L, K, 3) -- per-sample from _forward_single
            K = self.shape_prior_n_anchors
            # Get anchors for each valid slot's residue
            slot_anchors = shape_prior_anchors[sc_residue_idx_valid]  # (N_sc, K, 3)
            # Distance from each slot's current coords to each anchor
            slot_coords = sc_coords_valid.unsqueeze(1)  # (N_sc, 1, 3)
            anchor_dists = (slot_coords - slot_anchors).norm(dim=-1)  # (N_sc, K)
            # Find nearest anchor per slot
            nearest_idx = anchor_dists.argmin(dim=-1)  # (N_sc,)
            nearest_anchor = slot_anchors[torch.arange(len(nearest_idx), device=device), nearest_idx]  # (N_sc, 3)
            # Velocity bias: direction toward nearest anchor, scaled by learnable strength
            direction = nearest_anchor - sc_coords_valid  # (N_sc, 3)
            # Normalize direction and scale by learnable bias (avoid /0 for atoms at anchor)
            dist_to_anchor = direction.norm(dim=-1, keepdim=True).clamp(min=0.1)
            shape_bias = direction / dist_to_anchor * self.shape_prior_bias_scale
            noise_valid = noise_valid + shape_bias

        # Target-interaction intent: predict per-slot interaction type, optionally bias velocity toward target
        interaction_intent_logits_valid = None
        if self.use_interaction_intent:
            interaction_intent_logits_valid = self.interaction_intent_head(sc_out_features)  # (N_sc, num_classes)
            # Velocity bias is optional and orthogonal to the classification loss
            if (
                self.interaction_intent_velocity_bias
                and target_coords_for_intent is not None
                and target_mask_for_intent is not None
            ):
                # P(interacting) = 1 - P(class 0 = no interaction) for both 2-class and 4-class
                p_interact = 1.0 - torch.softmax(interaction_intent_logits_valid, dim=-1)[:, 0]  # (N_sc,)
                # Find nearest target atom for each sidechain atom
                tgt_valid = target_coords_for_intent[target_mask_for_intent]  # (N_tgt, 3)
                if tgt_valid.shape[0] > 0:
                    dists = torch.cdist(sc_coords_valid, tgt_valid)  # (N_sc, N_tgt)
                    nearest_tgt = tgt_valid[dists.argmin(dim=-1)]  # (N_sc, 3)
                    intent_direction = nearest_tgt - sc_coords_valid  # (N_sc, 3)
                    intent_dist = intent_direction.norm(dim=-1, keepdim=True).clamp(min=0.1)
                    intent_bias = (
                        intent_direction / intent_dist * p_interact.unsqueeze(-1) * self.interaction_bias_scale
                    )
                    noise_valid = noise_valid + intent_bias

        # Predict element types -- optionally with distance-from-CA as explicit feature
        if self.use_ca_dist_element_feature:
            # Compute per-atom distance from CA (normalized by typical sidechain radius ~5Å)
            ca_for_atoms = backbone_coords[sc_residue_idx_valid, 1, :]  # (N_sc, 3) -- CA coords per atom
            dist_from_ca = (sc_coords_valid - ca_for_atoms).norm(dim=-1, keepdim=True)  # (N_sc, 1)
            dist_from_ca_norm = dist_from_ca / 5.0  # normalize to ~1.0 for typical sidechain atoms
            element_input = torch.cat([sc_out_features, dist_from_ca_norm], dim=-1)
        else:
            element_input = sc_out_features
        element_logits_valid = self.element_type_head(element_input)  # (N_sc, 5)

        # Predict per-residue mixture centroid + cloud logvar (Stage 2).
        # Pool per-slot features to per-residue via mean, then predict shared sidechain cloud.
        n_valid_res = seq_mask.sum().item() if seq_mask is not None else seq_len
        residue_pooled = torch.zeros(n_valid_res, mixture_features.shape[-1], device=device)
        # sc_residue_idx_valid maps each valid slot to its residue index (0..n_valid_res-1 after masking)
        # We need to map to contiguous 0..n_valid_res-1
        unique_res, slot_to_res = torch.unique(sc_residue_idx_valid, return_inverse=True)
        residue_pooled.index_add_(0, slot_to_res, mixture_features)
        res_counts = torch.zeros(n_valid_res, device=device)
        res_counts.index_add_(0, slot_to_res, torch.ones(len(slot_to_res), device=device))
        residue_pooled = residue_pooled / res_counts.unsqueeze(-1).clamp(min=1)
        residue_pooled = self.residue_mixture_pool(residue_pooled)  # (n_valid_res, hidden)
        residue_centroid_valid = self.residue_centroid_head(residue_pooled)  # (n_valid_res, 3)
        residue_cloud_logvar_valid = self.residue_cloud_logvar_head(residue_pooled).squeeze(-1)  # (n_valid_res,)

        cluster_logits_valid = self.cluster_assignment_head(
            sc_out_features + sc_graph_cluster_condition[sc_cluster_inverse]
        )  # (N_sc, max_sc)

        # Scatter noise back to full tensor (match source dtype so bf16-mixed autocast
        # outputs can index-put into the destination -- mainline parity).
        noise_flat = torch.zeros(seq_len * max_sc, 3, device=device, dtype=noise_valid.dtype)
        noise_flat[sc_valid_idx] = noise_valid
        noise_pred = noise_flat.view(seq_len, max_sc, 3)

        # Scatter element logits back to full tensor
        element_logits_flat = torch.zeros(
            seq_len * max_sc, self.num_element_classes, device=device, dtype=element_logits_valid.dtype
        )
        element_logits_flat[sc_valid_idx] = element_logits_valid
        element_logits = element_logits_flat.view(seq_len, max_sc, self.num_element_classes)

        # volumetric -> existence coupling. Project the (detached) per-residue `vol_hidden` latent to a
        # PER-SLOT bias (L, max_sc) and SUBTRACT it from the PAD/GHOST logit (class ELEMENT_PAD=0). Positive bias
        # => lower PAD logit => lower P(PAD) => higher existence, so predicted VOLUME becomes a direct lever on
        # how many atoms should EXIST. Zero-init projection => bias is EXACTLY 0 at graft/resume => this whole
        # block is a no-op (byte-identical) until the projection learns; the additive form (build a zero tensor,
        # write only the PAD column, add) keeps every non-PAD logit untouched and avoids any in-place mutation of
        # the autograd graph. This runs in the ELEMENT HEAD, which executes at BOTH training AND sampling, so the
        # coupling is live at inference exactly as in training (the sample path threads `vol_hidden` in the same
        # way -- see sample()/sample_autoregressive()).
        if self.use_volumetric_existence_coupling and volumetric_hidden_exist is not None:
            exist_bias = self.volumetric_existence_proj(volumetric_hidden_exist.to(element_logits.dtype))  # (L, max_sc)
            pad_bias = torch.zeros_like(element_logits)
            pad_bias[..., ELEMENT_PAD] = -exist_bias
            element_logits = element_logits + pad_bias

        cluster_logits_flat = torch.zeros(seq_len * max_sc, max_sc, device=device, dtype=cluster_logits_valid.dtype)
        cluster_logits_flat[sc_valid_idx] = cluster_logits_valid
        cluster_logits = cluster_logits_flat.view(seq_len, max_sc, max_sc)

        # Scatter occupancy logits back to full tensor
        occupancy_logits_flat = torch.zeros(seq_len * max_sc, 1, device=device, dtype=occupancy_logits_valid.dtype)
        occupancy_logits_flat[sc_valid_idx] = occupancy_logits_valid
        occupancy_logits = occupancy_logits_flat.view(seq_len, max_sc)

        # Scatter contact logits back to full tensor
        contact_logits_flat = torch.zeros(seq_len * max_sc, 1, device=device, dtype=contact_logits_valid.dtype)
        contact_logits_flat[sc_valid_idx] = contact_logits_valid
        contact_logits = contact_logits_flat.view(seq_len, max_sc)

        # Scatter per-residue mixture params back to full tensors
        residue_centroid = torch.zeros(seq_len, 3, device=device, dtype=residue_centroid_valid.dtype)
        residue_centroid[unique_res] = residue_centroid_valid
        residue_cloud_logvar = torch.zeros(seq_len, device=device, dtype=residue_cloud_logvar_valid.dtype)
        residue_cloud_logvar[unique_res] = residue_cloud_logvar_valid

        # Note: Outputs are already zero for invalid residue positions (not in sc_valid_idx)
        # because the scatter operation only fills valid positions.
        # seq_mask already gated which residues to process.

        # Scatter valence logits back to full tensor
        valence_logits = None
        if valence_logits_valid is not None:
            valence_logits_flat = torch.zeros(seq_len * max_sc, 5, device=device, dtype=valence_logits_valid.dtype)
            valence_logits_flat[sc_valid_idx] = valence_logits_valid
            valence_logits = valence_logits_flat.view(seq_len, max_sc, 5)

        # Scatter interaction intent logits to full (L, max_sc, 4)
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
    """
    Complete inverse folding diffusion model with triple diffusion.

    Combines the denoiser with two parallel diffusion processes:
    1. Continuous Gaussian diffusion for 3D coordinates (v-prediction)
    2. Class-weighted discrete diffusion for element types (PAD=0, C=1, N=2, O=3, S=4)

    Atom existence is determined by element_type != PAD. There is no separate mask
    diffusion -- the mask is derived from element type predictions. A hard prefix
    constraint is enforced during sampling: if slot i is PAD, all j>i must be PAD.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension.
    num_layers : int
        Number of SE(3) transformer layers.
    timesteps : int
        Number of diffusion timesteps.
    schedule : str
        Noise schedule type.
    max_sidechain_atoms : int
        Maximum number of sidechain atoms per residue.
    use_target_conditioning : bool
        Whether to condition on target chain.
    timestep_sampling : str
        Strategy for sampling timesteps during training:
        - 'uniform': Standard uniform sampling over [0, timesteps)
        - 'low_noise_bias': Quadratic bias toward low noise (t^2 distribution)
        - 'sqrt': Square root bias toward high noise
        - 'stratified': Stratified sampling for coverage
    element_loss_weight : float
        Weight for element type prediction loss.
    atom_count_loss_weight : float
        Weight for atom count loss.
    count_correlation_loss_weight : float
        Weight for per-residue atom count correlation loss.
    count_pearson_loss_weight : float
        Weight for the independent (1 - Pearson r) count-differentiation loss. Runs alongside
        count_correlation_loss (the MSE mean-calibrator) so both can be active at once. 0=disabled.
    element_type_drop_prob : float
        Probability of replacing ALL element types with PAD (global dropout).
    element_type_token_drop_prob : float
        Probability of replacing each individual element type with PAD (token dropout).
    element_type_drop_schedule : bool
        If True, scale dropout probability with timestep (more dropout at high t).
    self_conditioning_prob : float
        Probability of using self-conditioning during training.
    ghost_weight : float
        Weight for ghost atom (PAD) coordinate loss. When 0.0 (default), coordinate
        loss is hard-gated to non-PAD atoms only. When > 0, PAD slots contribute
        to coord loss with this weight, providing gradient flow for ghost atoms.
    min_snr_gamma : float
        If > 0, apply Min-SNR-gamma loss weighting to the coordinate loss.
    num_cross_attn_layers : int
        Number of cross-attention layers for target conditioning. Default 5.
    num_cross_attn_heads : int
        Number of attention heads per cross-attention layer. Default 8.
    use_film : bool
        Whether to use FiLM modulation for target conditioning. Default True.
    use_bidirectional_target_conditioning : bool
        Whether target residue features attend to the current peptide backbone. Default True.
    use_sidechain_target_residue_attention : bool
        Whether sidechain particles attend to residue-level target features. Default True.
    sidechain_target_cutoff : float
        Distance cutoff (Å) for SC->target edges. Default 15.0.
    backbone_target_cutoff : float
        Distance cutoff (Å) for CA->target edges. Default 25.0.
    dropout : float
        Dropout rate for SE(3) transformer layers. Default 0.1.
    coord_noise_std : float
        Std of Gaussian noise added to input coordinates during training. Default 0.1.
    disable_element_types : bool
        If True, disable element type diffusion. All types set to PAD, element loss
        zeroed. Default True.
    """

    #: One-time announce guard for the bond-inject SOURCE corruption (nx0-OFF path). Class-level so the
    #: message prints once per process, mirroring ``SidechainDenoiser._nx0_corrupt_announced``.
    _ba_corrupt_announced: bool = False

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        timesteps: int = 250,
        schedule: str = "cosine",
        max_sidechain_atoms: int = 14,
        count_embed_mode: str = "linear",  # "linear" (default, byte-identical) | "ordinal" (Embedding)
        use_target_conditioning: bool = True,
        timestep_sampling: str = "uniform",
        coord_loss_weight: float = 1.25,
        element_loss_weight: float = 0.5,
        atom_count_loss_weight: float = 0.5,
        coord_scale: float = 100.0,
        prediction_type: str = "v",
        coord_process_type: str = "ddpm",  # "ddpm" or "flow_matching"
        flow_ghost_power: float = 2.0,
        flow_real_proximal_power: float = 2.0,
        flow_real_distal_power: float = 1.0,
        flow_use_conditional_groupwise: bool = False,
        flow_noise_scale: float = 1.0,  # Source distribution std (Å) for coordinate flow. Smaller = tighter ghost cloud.
        element_type_drop_prob: float = 0.05,
        element_type_token_drop_prob: float = 0.02,
        element_type_drop_schedule: bool = True,
        self_conditioning_prob: float = 0.75,
        use_cluster_particle_diffusion: bool = False,
        oracle_atom_counts: bool = False,
        cluster_assignment_loss_weight: float = 0.0,
        cluster_cohesion_loss_weight: float = 0.0,
        cluster_structure_loss_weight: float = 0.0,
        cluster_contrastive_loss_weight: float = 0.0,
        target_condition_scale: float = 1.0,
        cluster_target_condition_scale: float = 1.0,
        cluster_split_ramp_power: float = 1.0,
        cluster_label_permutation_prob: float = 0.0,
        cluster_feature_warmup_fraction: float = 0.0,
        min_snr_gamma: float = 0.0,
        count_correlation_loss_weight: float = 5.0,
        count_loss_type: str = "mse",  # "correlation" or "mse" -- per-residue count loss formulation
        num_cross_attn_layers: int = 5,
        num_cross_attn_heads: int = 8,
        use_film: bool = True,
        use_bidirectional_target_conditioning: bool = True,
        use_sidechain_target_residue_attention: bool = True,
        sidechain_target_cutoff: float = 15.0,
        backbone_target_cutoff: float = 25.0,
        # SE(3) RBF span selector (audit follow-up, ce7ed5d70). False (default / absent from old hparams) =>
        # the SE(3) attention-bias RBF keeps its LEGACY 0-10 A span (rbf_max_dist=None in the transformer);
        # True => the run OPTS IN to widening the span to backbone_target_cutoff so the distance bias stays
        # sensitive on the long CA->target context edges. Flag-gated (NOT default-on) so an OLD checkpoint --
        # which carries backbone_target_cutoff in hparams but was TRAINED with the 0-10 span -- rebuilds with
        # the legacy 0-10 features it actually saw. Persisted via save_hyperparameters so
        # eval rebuilds the exact span. Mirrors the --count-embed-mode plumbing.
        rbf_span_to_cutoff: bool = False,
        dropout: float = 0.1,
        coord_noise_std: float = 0.1,
        disable_element_types: bool = False,
        ghost_weight: float = 0.5,  # Weight for ghost atom coord loss (0=hard gating, 1.0=equal weight)
        pad_sampling_init: str | None = "bare",  # "bare"=all-PAD init with time-varying prior, None=fixed prior
        element_fn_weight: float = 1.0,  # Extra multiplier on element loss for non-PAD GT positions (false neg penalty)
        element_loss_non_pad_only: bool = False,  # Only compute element CE on GT non-PAD positions
        atom_mask_loss_weight: float = 0.0,  # Binary CE on P(non-PAD) vs GT mask (PAD existence loss)
        mask_bce_pos_weight_cap: float = 0.0,  # Cap pos_weight in mask BCE (0=no cap, e.g. 10=clip at 10)
        soft_count_loss_weight: float = 0.0,  # MSE on sum(P(non-PAD)) per residue vs GT count
        use_multi_count_discretization: bool = False,  # Try {n-1, n, n+1} atoms in discretization
        multi_count_max_plus: int = 1,  # Max positive delta for multi-count disc (1={-1,0,+1}, 2={-1,0,+1,+2})
        decoupled_count: bool = False,  # Decouple atom count from element identity noise
        count_ramp_threshold: float = 0.9,  # t_norm threshold for delayed-eq count schedule
        count_overdispersion: float = 1.5,  # NegBin overdispersion φ (Var = μ·φ)
        use_jackie: bool = False,  # Use Jackie biochemical features instead of learned residue embeddings
        count_perturb_prob: float = 0.0,  # Prob of perturbing per-residue atom counts during training
        count_corr_tau: float = 0.5,  # Tau for count correlation/MSE loss gating (1.0=no gating)
        count_tau_floor: float = 0.0,  # Floor on the tau-gated count-loss weight so count differentiation fires at ALL t (0.0=current low-t-only behavior; e.g. 0.4 => high-t weight 0.4, low-t 1.0)
        occupancy_loss_weight: float = 0.0,  # Weight for per-slot ghost-vs-real BCE (0=disabled)
        occupancy_gate_elements: bool = False,  # Use occupancy logits to gate element predictions
        occupancy_gate_strength: float = 3.0,  # Strength of the occupancy gate on element logits
        coord_dropout: float = 0.0,  # Prob of replacing sidechain coords with CA (forces context-based element pred)
        mixture_gate_weight: float = 0.0,  # Blend weight for mixture posterior in element gating (0=occupancy only)
        ghost_var_floor: float = 0.3,  # Min ghost variance (Å²) to avoid degenerate posterior at low noise
        mixture_loss_weight: float = 0.0,  # Weight for Stage 2 learned mixture param supervision (0=Stage 1 only)
        mixture_gate_max_noise: float = 1.0,  # Max noise fraction (t/T) for mixture gating (1.0=no limit)
        mixture_override_pad: bool = False,  # Hard-override PAD state from mixture posterior instead of element diffusion
        disc_detach_mask: bool = False,  # DETACH the disc soft mask -> disc trains COORDS only, no gradient into P(PAD) (severs the "drop the atom you can't place" NDM count-hack without swapping the 2-track existence mechanism, unlike mixture_override_pad)
        all_carbon_sampling: bool = False,  # Feed all-Carbon elements to denoiser during sampling; ghost/real from mixture posterior only
        position_based_element_powers: bool = True,  # Use all-real mask for element schedule powers in sampling (Fix A)
        preal_gate_target: str = "none",  # Gate target conditioning by P(real): "none", "cross_attention", "film", "graph_edges"
        residue_count_loss_weight: float = 0.0,  # Weight for residue-level count head supervision (0=disabled)
        count_ranking_loss_weight: float = 0.0,  # Weight for pairwise count-ranking loss (0=disabled)
        count_pearson_loss_weight: float = 0.0,  # Weight for independent (1 - Pearson r) count correlation loss (0=disabled)
        noise_dependent_ghost_weight: bool = False,  # Ramp ghost_weight from 0.5 (high noise) to 0.1 (low noise)
        element_pad_prior: float
        | None = None,  # Override PAD weight in element prior (default=0.71). Lower -> more non-PAD during noise/sampling.
        mixture_lr_threshold: float = 0.0,  # Log-LR threshold for mixture posterior (0=standard Bayes, >0=require stronger evidence for real)
        mixture_real_var_floor: float = 0.0,  # Extra variance floor (Å²) for real component, scaled by s². Broad at t=T -> easier escape from ghost.
        use_existence_flow: bool = False,  # Replace occupancy BCE with flow-matched existence velocity loss
        existence_loss_weight: float = 1.0,  # Weight for existence velocity MSE (replaces occupancy_loss_weight when active)
        non_pad_element_sampling: bool = False,  # 4-class element diffusion {C,N,O,S} with no PAD; ghost/real from mixture at t=0
        late_element_resolution: bool = False,  # Resolve element types in last 25% of trajectory; all-Carbon before that
        late_element_cutoff: float = 0.25,  # t_norm cutoff: elements are all-Carbon above this, resolve below
        contact_weight_min: float = 1.0,  # Min loss weight for distant residues (1.0=disabled, <1.0=contact upweighting)
        contact_weight_scale: float = 5.0,  # Distance scale (Å) for contact weight exponential decay
        contact_weight_boost: float = 0.0,  # Upweight-only mode: w = 1 + boost*exp(-d/scale). 0=disabled.
        contact_weight_coord: bool = True,  # Apply contact weighting to coord loss (False=element loss only)
        contact_weight_count: bool = False,  # Apply contact weighting to per-residue count losses (atom_count + count-corr MSE)
        fill_corrected_coord_weight: float = 0.0,  # Weight for the fill-corrected coord loss (0=off): subtracts fill-attributable coord error so count can reach GT without geometry fighting it
        fcc_slope: float = 0.097,  # Å-per-atom slope subtracted as fill-attributable coord error (empirical rmsd-vs-fill, interface value)
        fcc_tau: float = 0.5,  # Tau-gate: fill-corrected loss applied only for t/T < fcc_tau (low noise)
        fcc_per_env: bool = False,  # Per-env FCC slopes: gather 0.097/0.067/0.082 by the residue's B/E/I env (needs bei_env in forward); off => scalar fcc_slope
        fcc_slope_buried: float = 0.067,  # Å-per-atom FCC slope for BURIED residues (env code 1), used only when fcc_per_env and bei_env are supplied
        fcc_slope_exposed: float = 0.082,  # Å-per-atom FCC slope for EXPOSED residues (env code 2), used only when fcc_per_env and bei_env are supplied
        occupancy_match_loss_weight: float = 0.0,  # Weight for the occupancy-matching loss (0=off): differentiable Gaussian atom-density agreement between predicted and GT sidechain clouds (permutation-invariant, self-occupancy-esque)
        occupancy_match_sigma: float = 1.0,  # Gaussian sigma (Å) for the atom-presence density
        occupancy_match_tau: float = 0.5,  # Tau-gate: occupancy-matching loss applied only for t/T < occupancy_match_tau (low noise)
        occupancy_match_timestep_floor: float = 0.0,  # Soft floor on the occupancy-match tau-gated weight so it fires across the WHOLE trajectory (0.0=byte-identical hard cosine-tau gate; e.g. 0.5 => high-t weight floored at 0.5, t=0 weight 1.0). Mirrors count_tau_floor.
        # === Volumetric self-occupancy head (self-occupancy-style graft; head + loss only, NO flow injection). ===
        # Additive: not built + byte-identical when off. See volumetric_head.py.
        use_volumetric_head: bool = False,  # build + run the volumetric occupancy head (default off = byte-identical)
        # SANDCLOCK available-volume INPUT to the volumetric head: a GT-free 2-cone descriptor of how much
        # room each e3 (out-of-plane / L-D) face has, fed as an extra per-residue input the head projects
        # (zero-init) into `vol_hidden`. AND-gated with use_volumetric_head; byte-identical when off. See
        # volumetric_head.available_volume_cones. Meaningful only in full-arch FT (inert in the head-only pretrain).
        use_available_volume: bool = False,
        # SINGLE-SITE POCKET CONTEXT: condition the volumetric head on a per-query NEIGHBOR-OCCUPANCY FIELD --
        # the OTHER residues' side chains (+ binder backbone + target) splatted onto each site's own query
        # lattice, with the site's own side chain self-excluded -- fed via a zero-init projection so the head
        # predicts its own occupancy knowing which query points the neighbour pocket blocks. Raw neighbour atoms
        # never enter the head's feature set. Pretrain source = GT side chains (documented predicted-x0 seam for
        # FT). AND-gated with use_volumetric_head; byte-identical when off. See volumetric_head.py.
        use_single_site_context: bool = False,
        # SINGLE-SITE SCHEDULED-SAMPLING context mix (FT-only; default OFF = teacher forcing). Only the
        # INTEGRATED (non-pretrain-only) forward is affected, and only when use_single_site_context is on:
        # the volumetric head's neighbour-occupancy CONTEXT source is mixed PER BINDER RESIDUE between the
        # GT clean x0 (teacher forcing = what the head saw in pretraining) and the flow's PREDICTED x0
        # (reused from the neighbour-x0 recycle machinery; GT fallback when recycling did not run). The
        # per-site predicted fraction ramps linearly 0 -> p_max over [start_epoch, max_epochs]. p_max=0.0
        # (default) => all-GT => byte-identical to a plain teacher-forced single-site run, so a run that sets
        # use_single_site_context but not these still behaves sanely. Guarded below (>0 requires single-site
        # AND non-pretrain-only). See _single_site_context_source / get_ss_context_pred_prob.
        volumetric_ss_context_p_max: float = 0.0,
        volumetric_ss_context_start_epoch: int = 40,
        volumetric_loss_weight: float = 0.0,  # weight of the self-occupancy MSE loss (0=off; head still runs when use_volumetric_head)
        # VOLUMETRIC DECOY CROSS-ENTROPY (FT-only plumbing; default 0.0 = OFF = byte-identical). A discriminative
        # "on-top" term that pressures the head's predicted own-sidechain density field to score its GT residue-
        # type reference density above a set of DECOY-type reference densities -- REUSING the SAME decoys the
        # atom-disc discretization loss draws for that position (threaded out of DiscretizationLoss; never
        # re-sampled). The CE itself is added in DiffusionLightningModule.training_step alongside the integrated
        # volumetric loss (NOT in --volumetric-pretrain-only, which returns before it). Requires use_volumetric_head
        # (guarded below) AND the stratified discretization/residue-DB machinery it borrows decoys from (guarded in
        # validate_training_flag_coherence). Weight 0 => no decoy reuse, no reference-density library build, no CE.
        volumetric_decoy_ce_weight: float = 0.0,
        volumetric_decoy_ce_temperature: float = 1.0,  # softmax temperature on the density-similarity logits (>0)
        # INTEGRATED volumetric-loss TARGET selector. Default False = the historical self-SIDECHAIN-ONLY GT
        # (vol_density_gt, the head's simplified anti-leak MSE) => byte-identical when off. True switches the
        # integrated volumetric loss to the SAME own-only region-bucketed self-occupancy objective (via
        # build_full_occupancy_target) that --volumetric-pretrain-only supervises against -- so a Phase-2 FT
        # of a self-occupancy-pretrained head does not train it AWAY from the objective it was pretrained on. See the
        # coherence guard in validate_training_flag_coherence (pretrained + thawed + weight>0 + self-only
        # target is rejected as off-objective).
        volumetric_loss_supervision_target: bool = False,
        # A/B toggle for the self-occupancy occupancy TARGET composition (both the --volumetric-pretrain-only and
        # --volumetric-loss-self-occupancy-target paths flow through _supervision_full_field_occupancy_loss). Default False =
        # OWN-ONLY (faithful): the head regresses the masked residue's own side chain; context is INPUT +
        # CONTEXT-region label, not part of the target. True = legacy own+context field (comparison only).
        volumetric_target_include_context: bool = False,
        volumetric_context_radius: float = 10.0,  # Å radius of the target-aware context sphere (around each CA)
        volumetric_n_query: int = 128,  # #query points in the fixed local-frame occupancy lattice
        volumetric_sigma: float = 1.0,  # Gaussian sigma (Å) for the GT self-occupancy splat
        # PER-ELEMENT vdW-derived splat sigma: replace the single volumetric_sigma with a per-atom Bondi-radius
        # sigma (small O vs bulky S/X) on the GT-target / self-occupancy-target / self-consistency splats (and the eval
        # references). Off (default) => byte-identical uniform sigma. Scale None => mean-over-C/N/O/X normalized
        # to 1.0. AND'd with use_volumetric_head; stored on the head so the eval gate auto-detects it.
        volumetric_per_element_sigma: bool = False,
        volumetric_sigma_element_scale: float | None = None,
        volumetric_empty_weight: float = 1.75,  # up-weight for ~empty (anti-leak) query points (self-occupancy default)
        # FOURIER query lift + trunk dropout (faithful field expressivity). The query point is lifted to
        # 2*fourier_frequencies random Fourier features before the density trunk (0 disables the lift). The final
        # density is softplus'd (non-negative field). These CHANGE the head's arch/state_dict (a Fourier-widened
        # density_mlp) => a checkpoint from a differently-configured head will not strict-load. Passed to the head.
        volumetric_fourier_frequencies: int = 64,
        volumetric_fourier_scale: float = 10.0,
        volumetric_dropout: float = 0.0,  # module default 0.0 = byte-identical; the CLI drives the self-occupancy 0.1
        # Softplus (non-negative) density field at the head output (6f7648f04). True (production default) =>
        # softplus'd field aligned to the >=0 GT/reference splats. False => the LEGACY signed raw output. This
        # does NOT change the head's state_dict (softplus is an activation), but it DOES change the scored
        # density, so an OLD checkpoint trained WITHOUT softplus (absent from its hparams) must rebuild with
        # False for faithful density scoring; the eval-rebuild defaults it legacy-when-absent. Passed to the head.
        volumetric_use_softplus: bool = True,
        # PER-QUERY kNN CONTEXT (self-occupancy ContextEncoder). Off (default) => the density trunk reads the shared
        # attention-pooled `vol_hidden` broadcast to every query point (byte-identical; the head builds NO
        # ContextEncoder submodule). On => EACH query point gets its OWN kNN neighborhood over the single-site
        # context atoms (backbone + target + OTHER residues' side chains, self-excluded), the per-point
        # discrimination signal the pooled broadcast cannot carry. use_per_query_context BUILDS a new submodule
        # (context_encoder.*) => it changes the head state_dict (a strict load catches a mismatch); context_k /
        # context_heads tune the kNN + attention. Persisted via save_hyperparameters for eval/Ray rebuild.
        volumetric_per_query_context: bool = False,
        volumetric_context_k: int = 128,  # self-occupancy k_neighbors
        volumetric_context_heads: int = 4,  # self-occupancy NeighborhoodAttention num_heads (must divide hidden_dim)
        # Residue-axis chunk for the per-query ContextEncoder (OOM guard). The (G, Q, k, hidden) neighbor
        # tensor OOMs at G=B*L residues in parallel; each residue is INDEPENDENT so chunking G is NUMERICALLY
        # EXACT (peak set by the chunk, not B*L). Only active with per_query on; compute-only (does NOT change
        # the state_dict or numerics), so it is deliberately NOT a checkpoint-compat check.
        volumetric_context_chunk: int = 32,
        # VOLUMETRIC-FAITHFUL ATOM-ANCHORED QUERIES. Off (default) => the fixed CA-centered Fibonacci lattice
        # (byte-identical). On => the pretrain-only forward SAMPLES per-residue queries anchored on each GT
        # own-atom (center + around cluster + context negatives + banded empties; sample_atom_anchored_queries)
        # so the positive buckets are populated, and supervises the head's density at those SAME queries. The
        # head builds NO extra parameters for this (stateless sampler + per-call query override), so the head
        # state_dict is unchanged whether it is on or off => resume-safe / checkpoint-transferable. Persisted
        # via save_hyperparameters for eval/Ray rebuild.
        volumetric_atom_anchored_queries: bool = False,
        # SCALE-ANCHOR loss: pin the predicted density's per-residue total mass to the GT splat's mass. The
        # z-scored decoy-CE is scale-INVARIANT, so it leaves absolute magnitude unconstrained and the mass
        # drifts (neg_mse recovery collapses while cosine rises). This term lets the decoy-CE sharpen SHAPE
        # while magnitude stays calibrated. 0.0 (default) => not added (byte-identical); the decoy-CE is
        # untouched. Added to the volumetric loss (both the pretrain-only and integrated paths).
        volumetric_scale_anchor_weight: float = 0.0,
        # self-consistency loss = MSE between the FLOW's predicted-x0 density and the volumetric
        # HEAD's predicted density (the head's density is DETACHED => the flow chases the head one-way, not
        # the head drifting toward the collapsing flow). The flow-x0 cloud is splatted on the head's OWN
        # query lattice + sigma, weighted by the model's OWN soft P(real) (NOT the GT mask, unlike
        # occupancy_match) so the signal is exactly what the model reads at inference (the head exists at
        # inference; GT does not). This is the FREE-SAMPLING anchor, complementary to occupancy_match (the
        # low-t GT teacher-forcing guard). NOT tau-gated -- ramped IN by epoch (head-first curriculum: the
        # head warms on its GT loss before the flow is forced to it). Additive + byte-identical when off;
        # meaningful only WITH use_volumetric_head (AND'd below).
        use_volumetric_self_consistency: bool = False,  # add the self-consistency loss (default off = byte-identical)
        self_consistency_weight: float = 1.0,  # weight of the self-consistency MSE (0=off; tuned HIGH at training)
        self_consistency_ramp_start: float = 0.2,  # epoch frac where the ramp begins (0 weight before this)
        self_consistency_ramp_end: float = 0.6,  # epoch frac where the ramp reaches full weight
        # deep-inject the head's per-residue latent `vol_hidden` into EVERY SE(3) layer (zero-init,
        # byte-identical off, resume-safe). Only meaningful WITH use_volumetric_head (AND'd below). Applied on
        # BOTH the training forward AND the sampling reverse loop (the head is computed once per call/step-loop).
        use_volumetric_deep_inject: bool = False,
        # (run10): repoint the deep-inject from the degenerate pooled `vol_hidden` to a zero-init
        # projection of `vol_density_pred` (the informative per-query density field). Only meaningful WITH
        # use_volumetric_head AND use_volumetric_deep_inject (AND'd below). Zero-init => byte-identical when off.
        use_volumetric_density_inject: bool = False,
        # TARGET deep-inject (run10): reinforce the sidechain->target cross-attention at every SE(3) layer.
        # Threaded straight through to the SidechainDenoiser (which owns sc_target_attn). Zero-init => off = identical.
        use_target_deep_inject: bool = False,
        use_backbone_deep_inject: bool = False,
        frame_v2_deep_inject_detach: bool = False,
        use_bond_angle_deep_inject: bool = False,  # x0 bond-angle repr deep-inject (see SidechainDenoiser flag)
        # volumetric -> existence (element-track) coupling. The head's per-residue `vol_hidden` latent
        # modulates the element track's per-slot PAD/GHOST logit via a zero-init projection (more predicted
        # volume => shift probability toward real, non-PAD atoms -- the direct lever on undercount / size
        # collapse). Additive + byte-identical when off; meaningful only WITH use_volumetric_head (AND'd below).
        # Applied on BOTH the training forward AND the sampling reverse loop (the element head runs at both;
        # `vol_hidden` is computed once per call/step-loop and threaded in).
        use_volumetric_existence_coupling: bool = False,
        # PHASE 2: load a SEPARATELY-pretrained volumetric occupancy head (produced by
        # scripts/joint_diffusion/train_volumetric_head.py, i.e. a standalone-trained
        # atomweaver.joint_diffusion.volumetric_head.VolumetricOccupancyHead) into self.volumetric_head so the
        # atom flow is guided by an already-good head instead of one competing for gradient from scratch. The
        # path may be a raw head state_dict OR a Lightning/standalone-trainer checkpoint whose keys are
        # prefixed (e.g. "state_dict"/"model" wrapper + "...volumetric_head.*"); any common prefix is stripped and
        # the load is strict (a silent partial load would read as random guidance). Only meaningful WITH
        # use_volumetric_head (validated below). None/"" => no load (byte-identical). If freeze is on, the
        # head's params are frozen (requires_grad=False) so it acts as a fixed teacher; the optimizer
        # the training-time param-group logic skips requires_grad=False params, so frozen params land in NO group.
        volumetric_head_pretrained: str | None = None,
        freeze_volumetric_head: bool = False,
        # VOLUMETRIC PRETRAIN-ONLY: build + run ONLY the VolumetricOccupancyHead (skip the SE(3) atom
        # transformer / flow), so the head can be pretrained through the standard harness (DDP, data
        # pipeline, holdout flags, val, checkpointing) at big batch. The SidechainDenoiser is NOT
        # constructed when this is on (that is what frees the VRAM); forward() computes the head sub-graph
        # (byte-identical head input to the full net) and the self-occupancy occupancy loss, and STOPS there -- no
        # coord/element/count/any other loss. Requires use_volumetric_head. Off (default) => byte-identical:
        # the denoiser is built and forward()/loss run exactly as today. The head's state_dict keys+shapes
        # are unchanged either way, so a pretrain checkpoint loads into the full net via PHASE 2
        # (--volumetric-head-pretrained) with the same strict-load + config-validate.
        volumetric_pretrain_only: bool = False,
        use_slot_attention: bool = False,  # Intra-residue slot attention for ghost/real coordination
        num_slot_attn_layers: int = 2,  # Number of slot attention layers
        num_slot_attn_heads: int = 4,  # Number of attention heads in slot attention
        use_directional_slot_attention: bool = False,  # Distal scout -> proximal one-way attention
        n_scouts: int = 3,  # Number of scout slots: the N most-distal REAL slots per residue send one-way messages
        autoregressive_training: bool = False,  # Per-residue AR training: predict one focus residue conditioned on revealed GT neighbors
        ar_isolated_residues: bool = False,  # Single-residue mode: no inter-residue context (all non-focus hidden)
        count_extreme_alpha: float = 0.0,  # Reweight count losses by |gt_count - mean|^alpha + 1. 0=disabled.
        asymmetric_perturb: bool = False,  # Bias count perturbation toward mean: high-count residues lose atoms, low-count gain
        mixture_head_dropout: float = 0.0,  # Dropout on features feeding occupancy/mixture heads (info bottleneck, 0=disabled)
        rc_head_tau: float = -1.0,  # Separate tau for residue count head loss (-1=use count_corr_tau, >0=independent)
        dynamic_lrt: bool = False,  # Per-residue LRT: predict LRT offset from backbone+target features
        dynamic_lrt_weight: float = 2.0,  # Loss weight for dynamic LRT loss
        dynamic_lrt_clamp: float = 0.0,  # Clamp LRT delta to [-val, +val]. 0=no clamp
        dynamic_lrt_loss_type: str = "l1",  # l1, mse, or huber for dlrt count-matching loss
        dynamic_lrt_rank_weight: float = 0.0,  # Pairwise ranking loss on lrt_delta vs GT count (0=disabled)
        dynamic_lrt_reg_weight: float = 0.0,  # L2 regularizer on lrt_delta magnitude (0=disabled)
        dlrt_analytical_scale: float = 0.0,  # >0: use count head prediction to set threshold analytically (no learning)
        dlrt_detach: bool = False,  # Detach backbone features before LRT head (prevents gradient interference with backbone encoder)
        dlrt_sample_scale: float = 1.0,  # Scale LRT delta at sampling time (0.5 = half the learned delta). 1.0=no change.
        dlrt_ema_decay: float = 0.0,  # EMA decay for LRT head weights. 0=disabled. 0.99=slow EMA. Use EMA weights at sampling time.
        use_split_velocity: bool = False,  # Split velocity: learned v_real + analytical v_ghost. Ghost slots not trained.
        split_velocity_sampling_only: bool = False,  # Sampling-only split: standard training, blend v_real+v_ghost at sampling time only
        use_prior_cloud: bool = False,  # Backbone-conditioned prior for centroid/logvar (stable signal before sidechain denoising)
        prior_cloud_loss_weight: float = 0.5,  # Loss weight for prior cloud centroid/logvar supervision
        prior_blend_power: float = 2.0,  # Blending schedule: w(t) = (1-tau)^power (1=linear, 2=quadratic)
        sharpen_temperature_min: float = 1.0,  # Min temperature for mixture posterior at t=0 (1.0=off, 0.1=very sharp)
        sharpen_temperature_power: float = 1.0,  # Annealing speed for temperature schedule
        sidechain_corrupt_prob: float = 0.0,  # Prob of corrupting sidechain geometry per residue (0=off). Exposure bias fix.
        sidechain_corrupt_noise_gate: float = 0.5,  # Only corrupt when t_norm < this (mid/low noise)
        sidechain_corrupt_distal_only: bool = False,  # Only flip distal slots (slot-index-biased), not whole-residue
        sidechain_compress_prob: float = 0.0,  # Radial compression prob per real slot (0=off). r~U(0.3,0.8).
        rotation_corruption_prob: float = 0.0,  # per-residue prob of azimuthally spinning the NOISED sidechain INPUT about the Cα->pseudo-Cβ cone axis (0=off, byte-identical). x0 target stays true -> model learns to de-rotate via the frame. Training-only.
        rotation_corruption_max_angle: float = 90.0,  # max |spin| in DEGREES; angle ~ U(-cap, +cap). Cap mitigates near-symmetric sidechains (Phe/Tyr ring, Asp/Glu carboxylate, Arg guanidinium) whose true x0 would otherwise be a near-degenerate target under a large spin.
        rotation_corruption_min_tau: float = 0.05,  # skip rotation-corruption for samples with tau=t/(T-1) below this. The FM velocity-target recompute inverts x_t=(1-s)x0+s*x1 via 1/s (s=tau^power -> 0 as t->0), so it is singular near t=0; gating there avoids the singularity (and the near-clean regime is where the augmentation is least useful). Corrupted samples get their target REBUILT so the x0-inverse still reconstructs the true x0.
        # (stereochem): two low-rate coord-input corruptions that teach the t-resolution head to break the L/D symmetry. Both reflect/flatten the NOISED sidechain about the residue BACKBONE PLANE (the N-CA-C plane whose unit normal is e3 of build_local_frames), backbone untouched, self.training-only, and -- exactly like corrupt the model INPUT (x_t) so the FM velocity target is REBUILT from the implied corrupted source (recompute_target_for_corrupted_xt). Both reuse rotation_corruption_min_tau as the t≈0 floor (same 1/s singularity) and the FM-only construction guard. Byte-identical when both == 0.0 (block skipped, no RNG).
        mirror_corruption_prob: float = 0.0,  # per-residue prob of MIRROR-FLIPping the NOISED sidechain across the backbone plane (x' = x - 2·((x-CA)·e3)·e3) -> atoms on the WRONG L/D face; the t-resolution head must recover the correct face (the rarer wrong-face rescue). 0=off, byte-identical.
        inplane_corruption_prob: float = 0.0,  # per-residue prob of IN-PLANE-FLATTENing the NOISED sidechain onto the backbone plane (x' = x - ((x-CA)·e3)·e3, zeroing the out-of-plane component) -> the ambiguous "which side?" mid-state; the head must resolve to the correct side. This is the PRIMARY symmetry-break (more important of the two). 0=off, byte-identical.
        # Existence/count exposure-bias fix: corrupt CONDITIONING toward under-fill; loss targets stay GT.
        main_path_underfill_prob: float = 0.0,  # Frac of samples with corrupted existence/count cond (0=off).
        main_path_underfill_bias: float = 0.85,  # P(under-fill vs over-fill) for a corrupted sample.
        use_donut_source: bool = False,  # Per-slot radial shell source (donut) instead of isotropic Gaussian
        donut_thickness_ratio: float = 0.2,  # Shell thickness as fraction of shell radius
        use_empirical_shell_thickness: bool = False,  # data-driven radial std sqrt(Var(||x-CA||)) instead of tau*r jitter
        shell_target_var_scale: float = 1.0,  # scale empirical per-slot radial variance (target shells) DOWN; 1.0 = OFF (byte-identical)
        donut_element_init: str = "pad",  # Element init at t=T for donut mode: "pad"=all-PAD, "carbon"=all-Carbon, "mask"=MASK
        absorbing_mask: bool = True,  # MASK is absorbing: once resolved to {PAD,C,N,O,S}, can't return to MASK
        use_element_velocity_coupling: bool = False,  # FiLM modulation of velocity by PAD/non-PAD state (mixDDPM-style)
        evc_scheduled_sampling_prob: float = 0.0,  # Prob of non-GT existence mask for EVC in training. REQUIRES a delivery mode (self_conditioning_prob>0 OR evc_ss_corruption=True); was historically 0.3 but INERT without one, so default is now 0.0 (honest)
        evc_soft_conditioning: bool = False,  # Use soft P(non-PAD) instead of hard binary for EVC at sampling
        evc_velocity_blend: bool = False,  # At sampling: blend learned velocity with ghost velocity using EVC state
        evc_ss_corruption: bool = False,  # SS: feed a designed under/over-fill corruption of the GT mask
        ss_underfill_bias: float = 0.7,  # fraction of SS-corrupted samples that under-fill vs over-fill
        evc_ss_corrupt_selfcond: bool = False,  # SS STACK: corrupt the SELF-CONDITIONED predicted mask (not GT)
        evc_ss_noised_element_prob: float = 0.0,  # SS: feed EVC the noised-element existence (MASK->0.5) at t; also switches 2-track SAMPLING evc read to MASK->0.5
        use_ca_dist_element_feature: bool = False,  # Concat distance-from-CA to element head input
        use_count_velocity_coupling: bool = False,  # FiLM modulation by per-residue count -> per-slot P(real)
        use_bond_attention: bool = False,  # Intra-residue bond graph attention for coupling the 3 tracks
        zero_init_bond_attention: bool = False,  # bond-attn identity-start (retrofit onto pre-bond-attn ckpts)
        bond_loss_weight: float = 1.0,  # Weight for bond prediction auxiliary loss
        pocket_contact_loss_weight: float = 1.0,  # Weight for pocket contact prediction loss
        valence_loss_weight: float = 0.0,  # Weight for valence demand prediction loss (CE on bond count 0-4). 0=off.
        use_cross_residue_packing: bool = False,  # Inter-residue packing attention: contact prediction + cross-residue attention
        packing_contact_loss_weight: float = 1.0,  # Weight for inter-residue contact prediction auxiliary loss
        cross_residue_packing_spatial: bool = False,  # Pair residues by SPATIAL proximity (pseudo-CB kNN) instead of i±1. The eval's Buried metric is |q-p|>1 tip-packing, which the i±1-only pairing structurally cannot serve.
        cross_residue_packing_k: int = 4,  # k nearest residues per query in spatial mode (memory ~ k x consecutive mode)
        cross_residue_packing_radius: float = 10.0,  # Pseudo-CB distance cutoff in Angstrom; <=0 disables the radius gate
        cross_residue_packing_min_seq_sep: int = 2,  # Minimum |i-j| in spatial mode; 2 excludes i±1 (matches the eval Buried definition)
        use_neighbor_x0_packing: bool = False,  # Condition each residue's packing on its NEIGHBOURS' predicted x0 (anti-self); leg-A's approximation of leg-B's clean neighbour context
        neighbor_x0_packing_recycles: int = 2,  # TOTAL forward passes (AF2-style recycling). 1 = unconditioned single pass, 2 = the original two-pass scheme. Wall-clock scales ~linearly.
        neighbor_x0_packing_radius: float = 8.0,  # Outer shell radius (Angstrom) for the neighbour-x0 context features
        neighbor_self_dropout_prob: float = 0.0,  # FEATURE 1: per-residue P of ghosting a DESIGNED residue's own noised side chain to CA, forcing it to pack from neighbour-x0 context. 0.0 = bit-exact off.
        neighbor_x0_highnoise_weight: float = 1.0,  # FEATURE 2: amplify the neighbour-x0 packing residual at high noise (t_norm>0.5). 1.0 = exact no-op everywhere.
        # TRAINING-side corruption of the neighbour-x0 packing signal (weaken-the-neighbour ablation). All 0 =>
        # byte-identical (no corruption, no RNG). Applied ONLY during training; env-overridable at the call site.
        neighbor_x0_corrupt_add_prob: float = 0.0,  # per-GHOST-slot chance to ADD a phantom atom along the cloud ray
        neighbor_x0_corrupt_drop_prob: float = 0.0,  # per-REAL-slot chance to REMOVE (ghost to Cα) an atom
        neighbor_x0_corrupt_noise_prob: float = 0.0,  # whole-call chance the coord noise is applied at all
        neighbor_x0_corrupt_coord_noise: float = 0.0,  # Gaussian coord-noise std (Angstrom) when noise fires
        neighbor_x0_corrupt_disconnect_prob: float = 0.0,  # per-op chance to DESYNC coord vs existence tracks
        neighbor_x0_packing_prob_start: float = 0.0,  # Per-SAMPLE firing probability at epoch 0 (0.0 = start bit-identical to the parent checkpoint)
        neighbor_x0_packing_prob_end: float = 0.8,  # Firing probability after the ramp
        neighbor_x0_packing_ramp_epochs: int = 50,  # Epochs over which the firing probability ramps start -> end
        # Randomised recycle count. PAIRS WITH the recycle-index encoding: encoding j makes a
        # train/sample recycle mismatch HONEST, but only training over a spread of N makes it WORK
        # (train at N=2 and the model has never seen j=3,4,5). This is exactly why AF2 trains with a
        # random recycle count. Turning one on without the other is half the feature.
        neighbor_x0_packing_random_recycles: bool = False,  # draw N per BATCH from the weighted categorical below
        neighbor_x0_packing_max_recycles: int = 5,  # largest N with non-zero weight (must match the weight length)
        neighbor_x0_packing_recycle_weights: str
        | Sequence[float]
        | None = None,  # relative P(N=2), P(N=3), ...; None = NEIGHBOR_X0_RECYCLE_WEIGHTS_DEFAULT (E[N]=2.5)
        use_transition_weighted_loss: bool = False,  # Reweight the FINAL-pass per-slot loss by HOW THE PREDICTION CHANGED vs the earlier (detached) recycle pass -- targets STICKY FAILURES. See _transition_weights().
        transition_weight_sticky: float = 2.5,  # Max multiplier: wrong -> SAME wrong (repeated its own error; the behaviour to destroy)
        transition_weight_revised: float = 1.25,  # Max multiplier: wrong -> DIFFERENTLY wrong (revised, still wrong -- it at least tried)
        transition_weight_regression: float = 2.0,  # Max multiplier: right -> wrong (two-sided, so we do not train flightiness)
        transition_weight_cap: float = 3.0,  # Hard ceiling on any per-slot weight, so one position cannot dominate the batch
        transition_coord_tol: float = 1.0,  # Angstrom; per-slot x0 error above this counts as "wrong" on the COORD channel
        transition_coord_same_tol: float = 0.25,  # Angstrom; final x0 moved less than this from the earlier pass -> "SAME wrong" (sticky)
        transition_coord_mag_scale: float = 2.0,  # Angstrom of excess error / degradation that saturates the coord magnitude ramp
        occupancy_weighted_source: bool = False,  # Scale donut shell by per-slot P(real) so ghost-heavy slots start near CA
        leak_gt_count: bool = False,  # Positive control: leak GT atom mask into source (real->shell, ghost->CA)
        leak_gt_direction: bool = False,  # Positive control: leak GT sidechain direction into donut source
        pseudo_cb_direction: bool = False,  # Use virtual CB direction from backbone geometry (N, CA, C) instead of GT
        d_aa_l_source_prob: float = 0.0,  # Per-D-residue probability of forcing the SOURCE cone to L (+1) in TRAINING (D residues start on L hemisphere; flow must cross the mirror). >=1.0 = always L (old always_L_source_prior=True); <=0.0 = GT chirality everywhere (old False, new default); 0<p<1 = per-residue Bernoulli. Flow target + chirality-head supervision stay GT. Sampling always respects caller chirality.
        element_flow_matching: bool = False,  # Use flow matching on probability simplex for element types instead of absorbing diffusion
        split_element_existence: bool = False,  # Separate existence flow (PAD/non-PAD) from element absorbing diffusion ({C,N,O,S,MASK})
        split_absorbing_element: bool = True,  # If True, split element diffusion is absorbing (MASK->{C,N,O,S} only). If False, non-absorbing (allows re-masking mid-trajectory).
        split_element_flow: bool = False,  # Use flow matching on {C,N,O,S,MASK} simplex instead of discrete diffusion for split elements
        split_element_flow_temp: float = 0.0,  # Temperature for element flow denoiser input: 0=hard argmax, >0=sharpened soft probs
        split_element_uniform_power: bool = False,  # If True, element track uses uniform power (1.0) while existence/coords use slot schedule
        existence_source_value: float = 0.5,  # Source value for existence flow (0.5=max entropy start)
        existence_power: float = 1.0,  # Power schedule for existence flow (1=linear, 2=quadratic)
        existence_threshold: float = 0.5,  # Threshold for ghost/real classification (>= threshold = real)
        existence_absorbing: bool = False,  # Use 3-class absorbing diffusion {ghost,real,MASK} instead of continuous existence flow
        existence_velocity_scale: float = 1.0,  # Sampling-time velocity multiplier for existence flow
        use_shape_prior: bool = False,  # Learned sidechain shape prior: predict anchor points, bias coord flow
        shape_prior_n_anchors: int = 4,  # Number of anchor points per residue (K). 0 disables shape prior entirely.
        shape_prior_loss_weight: float = 1.0,  # Weight for shape prior anchor matching loss
        use_bond_co_diffusion: bool = False,  # Latent bond graph co-diffusion: noise GT bonds, predict bond velocity
        bond_co_diffusion_weight: float = 1.0,  # Weight for bond co-diffusion flow matching loss
        use_coord_self_conditioning: bool = False,  # Feed previous step's predicted x₀ coords as input
        use_plan_latent: bool = False,  # Per-residue plan latent from backbone+target
        plan_latent_loss_weight: float = 1.0,  # Weight for plan latent supervision loss
        use_interaction_intent: bool = False,  # Per-slot target interaction type prediction
        interaction_intent_loss_weight: float = 1.0,  # Weight for interaction intent CE loss
        interaction_intent_num_classes: int = 4,  # Number of intent classes (2=binary, 4=multiclass)
        interaction_intent_velocity_bias: bool = True,  # Whether to apply velocity bias toward target for interactors
        use_chirality: bool = False,  # Signed volume feature breaks SE(3) to distinguish L/D
        pairwise_distance_loss_weight: float = 0.0,  # Intra-residue pairwise distance MSE (0=off)
        bonded_geometry_loss_weight: float = 0.0,  # Bonded + nonbonded split geometry loss (0=off)
        rotamer_rmsd_loss_weight: float = 0.0,  # Low-noise rotamer RMSD loss (0=off)
        bond_window_loss_weight: float = 0.0,  # Bond-window classification loss (0=off)
        bond_angle_loss_weight: float = 0.0,  # Bond angle loss on bonded triples (0=off)
        bond_length_loss_weight: float = 0.0,  # Bond length loss on bonded pairs (bond-inject length half; 0=off)
        scale_loss_weight: float = 0.0,  # Per-residue scale loss: penalize compressed pairwise distances (0=off)
        dist_from_ca_loss_weight: float = 0.0,  # Per-slot distance-from-CA MSE vs GT (0=off)
        num_edge_types: int = 3,  # 3=intra/inter/binder-target (legacy), 2=intra/extra-residue (forward-compat)
        graph_num_edge_types: int
        | None = None,  # If set, graph emits this many edge types (decoupled from embedding rows)
        use_backbone_dihedral: bool = False,  # backbone dihedral graft (zero-init add)
        use_burial_feature: bool = False,  # Cbeta-burial graft (zero-init add)
        # === DRAFT: residue-level frame-aware stream + coarse-latent consistency (see residue_frame_stream.py) ===
        # Off by default; byte-identical + resume-safe when off (zero-init injection). FIRST CUT -- needs review.
        use_residue_frame_stream: bool = False,  # build + run the residue frame stream
        residue_frame_layers: int = 2,  # #layers in the residue frame-attention stack
        use_residue_frame_deep_inject: bool = False,  # DRAFT: ALSO inject the residue latent per-SE3-layer (needs stream on)
        residue_frame_supervision_weight: float = 1.0,  # weight of the GT-supervision latent loss (keeps stream live)
        residue_frame_consistency_weight: float = 1.0,  # weight of the atom-cloud consistency loss (RAMPED, tau-gated). Was 5.0 -- dropped 2026-08-05: at 5.0 it dominated gradient (~5x coord loss) + destabilized the coord track from-scratch.
        residue_frame_consistency_ramp_frac: float = 0.3,  # ramp consistency 0->full over this fraction of max_epochs
        # === v2 residue-frame graph (binder+target residues; orientation-only). Off by ===
        # default; byte-identical + resume-safe when off. Mutually exclusive with the v1 stream above.
        use_residue_frame_stream_v2: bool = False,  # build + run the v2 residue-frame graph
        residue_frame_v2_layers: int = 2,  # #layers in the v2 cross-attention stack
        frame_v2_clean_input: bool = True,  # True: clean geometry-only binder embed; False: pocket-conditioned fallback (v1-style)
        # NEW: per-SE(3)-layer LIVE deep-inject of the v2 frame stream's latent into the atom transformer.
        # Only meaningful WITH use_residue_frame_stream_v2 (AND'd below). Off => byte-identical + resume-safe.
        use_residue_frame_v2_deep_inject: bool = False,
        frame_v2_centroid_weight: float = 1.0,  # weight of the v2 in-frame-centroid supervision loss
        frame_v2_chi1_weight: float = 1.0,  # weight of the v2 χ1 (cos,sin) supervision loss
        # supervised stereochemistry (e3-sign) head on the v2 frame graph, reading vol_hidden as
        # an auxiliary input. Off by default; only meaningful WITH use_residue_frame_stream_v2 (AND'd below,
        # + a coherence guard in validate_training_flag_coherence). Byte-identical when off.
        use_stereochem_head: bool = False,  # build + supervise the e3-sign (L vs D) head
        frame_v2_stereo_weight: float = 1.0,  # weight of the stereochem BCE (only active when use_stereochem_head)
        # t-resolution head. Off by default; only meaningful WITH use_stereochem_head (which itself
        # needs use_residue_frame_stream_v2). AND'd below + a coherence guard in
        # validate_training_flag_coherence. Byte-identical when off. Supervised-only (its own BCE); it does
        # NOT feed back into coords/element/existence in this piece (that is ).
        use_stereochem_t_resolution: bool = False,
        stereo_t_resolution_weight: float = 1.0,  # weight of the P(D)_t BCE (only active with the flag)
        # t-resolution COORD-FLOW FEEDBACK. Off by default; only meaningful WITH
        # use_stereochem_t_resolution (which supplies the resolved P(D)_t to steer with, and itself needs the
        # static stereo head + v2 stream). AND'd below + a coherence guard in validate_training_flag_coherence.
        # Byte-identical when off (no modules built). HEAD-FIRST RAMPED (knobs below) so an UNTRAINED P(D)_t head
        # -- whose resolved face is meaningless early -- never steers coords; =1 at inference.
        use_stereochem_t_resolution_feedback: bool = False,
        t_resolution_feedback_ramp_start: float = 0.1,  # epoch frac where the feedback ramp begins (off before)
        t_resolution_feedback_ramp_end: float = 0.5,  # epoch frac where the feedback ramp reaches full (s=1)
        # -1: sigmoid-gated cone init. Steer the pseudo-Cβ donut cone's OUT-OF-PLANE (e3) sign by the
        # stereochem head's P(D) so D-amino acids initialize on the correct face at the SOURCE distribution.
        # Effective only WITH use_residue_frame_stream_v2 AND use_stereochem_head AND pseudo_cb_direction
        # (AND'd below + a coherence guard in validate_training_flag_coherence). Off => the source dist uses the
        # ungated analytic cone unchanged => BYTE-IDENTICAL (no new params -- reuses the frame-v2 + volumetric heads).
        use_stereochem_gated_init: bool = False,
        # -1 head-first ramp (graft safety): blend the source-dist e3 from the ungated analytic L
        # cone (s=0) to the P(D)-gated value (s=1) over training, mirroring the self-consistency ramp. At
        # s=0 the gated init is byte-identical to the L cone, so an UNTRAINED stereo head (P(D)≈0.5 on a
        # fresh graft) does NOT flatten every cone in-plane. Inference (no training_progress) => s=1 (full).
        gated_init_ramp_start: float = 0.1,  # epoch frac where the ramp begins (s=0 == L cone before this)
        gated_init_ramp_end: float = 0.5,  # epoch frac where the ramp reaches s=1 (full P(D)-gating)
        # v2 REUSES residue_frame_consistency_weight / _ramp_frac / _tau for the centroid consistency term.
        use_polarity_head: bool = False,  # burial/backbone -> element-composition expert (product-of-experts on element logits; see below). Most meaningful with use_burial_feature=True.
        polarity_loss_weight: float = 0.1,  # weight of the supervised polarity composition CE (only active when use_polarity_head)
        use_glycine_head: bool = False,  # backbone-DIHEDRAL(phi,psi)-only -> glycine expert: a bounded, zero-init, GT-glycine-supervised push on P(PAD) (glycine==0 real atoms). Narrow input+supervision confine it to the axis where all-PAD is CORRECT; NOT a general count lever. WARNING: modulates the fragile existence axis -- keep OFF until undercount is stable.
        glycine_loss_weight: float = 0.1,  # weight of the supervised glycine BCE (only active when use_glycine_head)
        glycine_pad_cap: float = 4.0,  # cap (logits) on the additive P(PAD) bias the glycine head may apply, gated by a zero-init scalar
        activation_checkpointing: bool = False,  # Recompute SE3 layer activations during backward to save memory
        activation_checkpoint_stride: float = 1.0,  # >1 = selective: checkpoint only every Nth SE3 layer (float-OK, e.g. 1.5)
        use_global_latent_matching: bool = False,  # clean-context stream + per-residue latent matching (ported; default OFF = no-op)
        global_latent_weight: float = 1.0,  # weight of the latent-matching (cosine) loss
        global_latent_gate_t_min: float = 0.5,  # t_norm below this -> no latent loss (ramp start; high-noise gate)
        global_latent_gate_t_full: float = 1.0,  # t_norm at/above this -> full latent-loss weight
        global_latent_embed_dim: int = 64,  # z_pred/z_gt width; OVERRIDDEN by the lookup table width when wired
        global_latent_qupid_lookup: str = "",  # path to the QuPID identity-embedding lookup .pt (launch-time input)
        global_latent_confidence: bool = True,  # confidence probe + gated FiLM feedback
        global_latent_confidence_weight: float = 0.1,  # weight of the confidence-probe MSE loss
        global_latent_condition_main_stream: bool = True,  # FiLM the gated latent into generated atoms
        global_latent_film_layers: str = "last",  # where to inject the FiLM feedback (this port: "last")
        global_latent_loss_type: str = "cosine",  # "cosine" | "mse"
        latent_self_dropout_prob: float = 0.0,  # FEATURE 3: per-residue P of blinding a DESIGNED residue's own element identity to MASK, forcing it to read identity from the injected z_pred latent. 0.0 = bit-exact off.
        distal_threshold_cap: float = 0.0,  # FEATURE 4: LOWER the MASK-slot existence read-threshold to min(0.5*r, cap) for mid/distal slots (0.5*r>cap), recovering under-read atoms; proximal unchanged. 0.0 = OFF (bit-exact)
        distal_jitter_cap: float = 0.0,  # FEATURE 5: cap the source jitter (tau*r) at this Angstrom for distal slots; 0.0 = OFF (bit-exact). Threaded to the coord flow.
        distal_shell_ramp_epochs: int = 0,  # shared distal-shell warmup window for features 4 & 5; 0 = instant/no-ramp
        graft_init_std: float = 0.0,  # >0: init the zero-init graft INJECTION modules (incl. residue_frame_deep_proj when deep-inject is on) with N(0, std) instead of zeros. 0.0 = zeros (bit-exact).
    ):
        super().__init__()

        from .diffusion import (
            EXISTENCE_MASK,
            NUM_EXISTENCE_CLASSES,
            NUM_SPLIT_ELEMENT_TYPES,
            SPLIT_ELEMENT_DATA_PRIOR,
            SPLIT_ELEMENT_MASK,
            ClassWeightedDiscreteDiffusion,
            ElementFlowMatching,
            ExistenceFlowMatching,
            GaussianDiffusion,
            GhostRealFlowMatching,
            cosine_beta_schedule,
        )

        if coord_process_type not in {"ddpm", "flow_matching"}:
            raise ValueError(f"Unknown coord_process_type={coord_process_type}")

        # Determine number of element classes: 5 (PAD,C,N,O,S), 6 (+MASK), or 5 (split: C,N,O,S,MASK)
        from .diffusion import ELEMENT_MASK

        self.split_element_existence = split_element_existence
        self.existence_source_value = existence_source_value
        self.existence_power = existence_power
        self.existence_threshold = existence_threshold
        self.existence_velocity_scale = existence_velocity_scale
        if split_element_existence:
            # Split mode: denoiser operates on {C=0,N=1,O=2,S=3,MASK=4} -- no PAD class
            self.num_element_classes = NUM_SPLIT_ELEMENT_TYPES
        else:
            self.num_element_classes = NUM_ELEMENT_TYPES + (1 if donut_element_init == "mask" else 0)

        # === Global latent matching (ported) -- default OFF = bit-exact no-op ===
        # z_gt is a LOOKUP of precomputed rotamer-invariant QuPID identity embeddings, keyed by the GT
        # residue-DB class id (batch["residue_indices"], NCAA-aware via name_to_idx). The lookup width
        # sets embed_dim (the CLI value is only a fallback when no table is wired). The loss itself is
        # assembled during training; here we build the buffer + thread the flags into the denoiser.
        self.use_global_latent_matching = use_global_latent_matching
        self.global_latent_weight = float(global_latent_weight)
        self.global_latent_gate_t_min = float(global_latent_gate_t_min)
        self.global_latent_gate_t_full = float(global_latent_gate_t_full)
        self.global_latent_confidence = bool(global_latent_confidence)
        self.global_latent_confidence_weight = float(global_latent_confidence_weight)
        self.global_latent_loss_type = global_latent_loss_type
        self.global_latent_embed_dim = int(global_latent_embed_dim)
        self.global_latent_names = None
        if use_global_latent_matching:
            # FIX 3 (pocket-only invariant): the clean stream reads backbone_features, which
            # preal_gate_target would gate by the noised element state -> z_pred would depend on the
            # generated atoms, breaking the pocket-only/time-free invariant. Reject the combo rather
            # than silently degrade it (launch defaults leave preal_gate_target='none').
            if preal_gate_target not in (None, "", "none"):
                raise ValueError(
                    "use_global_latent_matching=True is incompatible with preal_gate_target="
                    f"{preal_gate_target!r}: gating backbone_features by the noised element state makes the "
                    "clean-context latent depend on the generated atoms, breaking the pocket-only/time-free "
                    "invariant. Use preal_gate_target='none' (the default) with global latent matching."
                )
            # FIX 1 (fail loud, no silent-skip): the clean stream / injection / FiLM would otherwise
            # still train via the main loss while the load-bearing matching objective is silently off.
            if not global_latent_qupid_lookup:
                raise ValueError(
                    "use_global_latent_matching=True requires --global-latent-qupid-lookup (path to the "
                    "QuPID identity-embedding lookup .pt). Refusing to run with the matching loss silently off."
                )
            from pathlib import Path as _Path

            if not _Path(global_latent_qupid_lookup).is_file():
                raise FileNotFoundError(
                    f"global_latent_qupid_lookup not found: {global_latent_qupid_lookup!r}. "
                    "Provide the disc-index-aligned QuPID lookup .pt."
                )
            from .latent_matching import load_qupid_lookup

            table, self.global_latent_embed_dim, self.global_latent_names = load_qupid_lookup(
                global_latent_qupid_lookup
            )
            self.register_buffer("global_latent_lookup", table, persistent=False)
        else:
            self.global_latent_lookup = None

        # DRAFT residue-frame stream config (stored so forward can assemble the two losses + ramp).
        self.use_residue_frame_stream = use_residue_frame_stream
        # DRAFT deep per-layer injection (effective only WITH the stream; the denoiser AND's the two).
        self.use_residue_frame_deep_inject = bool(use_residue_frame_stream and use_residue_frame_deep_inject)
        self.residue_frame_supervision_weight = float(residue_frame_supervision_weight)
        self.residue_frame_consistency_weight = float(residue_frame_consistency_weight)
        self.residue_frame_consistency_ramp_frac = float(residue_frame_consistency_ramp_frac)
        # Tau-gate the consistency loss to low noise (mirrors FCC's schedule); fixed for the first cut.
        self.residue_frame_consistency_tau = 0.5

        # v2 residue-frame graph (orientation-only). Loss weights stored here; the module is
        # built inside the denoiser. Off => weights unused + nothing runs (byte-identical).
        self.use_residue_frame_stream_v2 = bool(use_residue_frame_stream_v2)
        # NEW: v2 frame deep-inject is effective only WITH the v2 stream (AND). Stored for hparams/resume; the
        # denoiser stores the same AND'd value and owns the per-layer projections.
        self.use_residue_frame_v2_deep_inject = bool(use_residue_frame_stream_v2 and use_residue_frame_v2_deep_inject)
        self.frame_v2_deep_inject_detach = bool(frame_v2_deep_inject_detach)
        self.use_bond_angle_deep_inject = bool(use_bond_angle_deep_inject)
        self.frame_v2_centroid_weight = float(frame_v2_centroid_weight)
        self.frame_v2_chi1_weight = float(frame_v2_chi1_weight)
        # effective-gate the stereochem head with the v2 stream (the head lives inside the v2
        # module; enabling it without the stream would do nothing). The denoiser stores the AND'd value.
        self.use_stereochem_head = bool(use_residue_frame_stream_v2 and use_stereochem_head)
        self.frame_v2_stereo_weight = float(frame_v2_stereo_weight)
        # effective-gate the t-resolution head with the static stereo head (which supplies
        # P(D)_prior) -- and, transitively, with the v2 stream (use_stereochem_head already ANDs it). Enabling
        # it without the static head would have no prior to blend against. The denoiser stores the AND'd value.
        self.use_stereochem_t_resolution = bool(
            use_residue_frame_stream_v2 and use_stereochem_head and use_stereochem_t_resolution
        )
        self.stereo_t_resolution_weight = float(stereo_t_resolution_weight)
        # effective-gate the coord-flow feedback with the t-resolution head (which produces the P(D)_t
        # it steers with) -- and, transitively, with the static stereo head + v2 stream. The denoiser stores the
        # AND'd value. Ramp knobs live here (training_progress is available at the IFD level; see
        # _t_resolution_feedback_ramp, threaded DOWN to the denoiser as a per-forward scalar).
        self.use_stereochem_t_resolution_feedback = bool(
            use_residue_frame_stream_v2
            and use_stereochem_head
            and use_stereochem_t_resolution
            and use_stereochem_t_resolution_feedback
        )
        self.t_resolution_feedback_ramp_start = float(t_resolution_feedback_ramp_start)
        self.t_resolution_feedback_ramp_end = float(t_resolution_feedback_ramp_end)
        # -1: sigmoid-gated cone init. Effective ONLY with the v2 stream + stereochem head (which
        # produce P(D)) AND pseudo_cb_direction (the analytic cone `d` is the thing we steer -- without it the
        # donut uses a RANDOM direction with no frame-consistent e3 to gate) AND use_donut_source==True (the
        # donut is the ONLY source that consumes `direction_override`; an isotropic-Gaussian source ignores
        # it) AND coord_process_type=="flow_matching" (the whole recompute/steer is FM-specific -- DDPM never
        # takes the direction_override path). When any is off, the source dist is byte-identical to today (the
        # gating sits inside `if self.pseudo_cb_direction` and this AND'd flag).
        self.use_stereochem_gated_init = bool(
            use_residue_frame_stream_v2
            and use_stereochem_head
            and pseudo_cb_direction
            and use_donut_source
            and coord_process_type == "flow_matching"
            and use_stereochem_gated_init
        )
        # Head-first ramp knobs (graft safety); see _gated_init_ramp / _stereochem_gate_cone_direction.
        self.gated_init_ramp_start = float(gated_init_ramp_start)
        self.gated_init_ramp_end = float(gated_init_ramp_end)

        # VOLUMETRIC PRETRAIN-ONLY: a head-only pretrain build. Requires the head (there is nothing to
        # pretrain otherwise) and skips the SE(3) atom transformer / flow denoiser entirely so a much
        # bigger batch fits. Set BEFORE the denoiser construction so the build can be gated on it.
        self.volumetric_pretrain_only = bool(volumetric_pretrain_only)
        if self.volumetric_pretrain_only and not use_volumetric_head:
            raise ValueError(
                "volumetric_pretrain_only=True requires use_volumetric_head=True: the pretrain-only build "
                "trains ONLY the VolumetricOccupancyHead (the SE(3) atom transformer is not constructed), so "
                "with the head off there is nothing to train. Enable use_volumetric_head, or set "
                "volumetric_pretrain_only=False."
            )

        # Skip the (VRAM-dominant) SE(3) atom denoiser in pretrain-only mode. forward() branches out at the
        # volumetric-head compute and never touches self.denoiser, so leaving it None is safe there.
        if self.volumetric_pretrain_only:
            self.denoiser = None
        else:
            self.denoiser = SidechainDenoiser(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                max_sidechain_atoms=max_sidechain_atoms,
                count_embed_mode=count_embed_mode,
                use_target_conditioning=use_target_conditioning,
                num_cross_attn_layers=num_cross_attn_layers,
                num_cross_attn_heads=num_cross_attn_heads,
                use_film=use_film,
                use_bidirectional_target_conditioning=use_bidirectional_target_conditioning,
                use_sidechain_target_residue_attention=use_sidechain_target_residue_attention,
                target_condition_scale=target_condition_scale,
                cluster_target_condition_scale=cluster_target_condition_scale,
                sidechain_target_cutoff=sidechain_target_cutoff,
                backbone_target_cutoff=backbone_target_cutoff,
                rbf_span_to_cutoff=rbf_span_to_cutoff,
                dropout=dropout,
                use_jackie=use_jackie,
                preal_gate_target=preal_gate_target,
                use_slot_attention=use_slot_attention,
                num_slot_attn_layers=num_slot_attn_layers,
                num_slot_attn_heads=num_slot_attn_heads,
                use_directional_slot_attention=use_directional_slot_attention,
                n_scouts=n_scouts,
                num_element_classes=self.num_element_classes,
                use_element_velocity_coupling=use_element_velocity_coupling,
                num_timesteps=timesteps,
                use_ca_dist_element_feature=use_ca_dist_element_feature,
                use_count_velocity_coupling=use_count_velocity_coupling,
                use_bond_attention=use_bond_attention,
                zero_init_bond_attention=zero_init_bond_attention,
                use_valence_demand=valence_loss_weight > 0,
                use_cross_residue_packing=use_cross_residue_packing,
                cross_residue_packing_spatial=cross_residue_packing_spatial,
                cross_residue_packing_k=cross_residue_packing_k,
                cross_residue_packing_radius=cross_residue_packing_radius,
                cross_residue_packing_min_seq_sep=cross_residue_packing_min_seq_sep,
                use_neighbor_x0_packing=use_neighbor_x0_packing,
                neighbor_x0_packing_radius=neighbor_x0_packing_radius,
                neighbor_x0_highnoise_weight=neighbor_x0_highnoise_weight,
                neighbor_x0_corrupt_add_prob=neighbor_x0_corrupt_add_prob,
                neighbor_x0_corrupt_drop_prob=neighbor_x0_corrupt_drop_prob,
                neighbor_x0_corrupt_noise_prob=neighbor_x0_corrupt_noise_prob,
                neighbor_x0_corrupt_coord_noise=neighbor_x0_corrupt_coord_noise,
                neighbor_x0_corrupt_disconnect_prob=neighbor_x0_corrupt_disconnect_prob,
                use_shape_prior=use_shape_prior and shape_prior_n_anchors > 0,
                shape_prior_n_anchors=shape_prior_n_anchors if shape_prior_n_anchors > 0 else 4,
                use_bond_co_diffusion=use_bond_co_diffusion,
                use_coord_self_conditioning=use_coord_self_conditioning,
                use_plan_latent=use_plan_latent,
                use_interaction_intent=use_interaction_intent,
                interaction_intent_num_classes=interaction_intent_num_classes,
                interaction_intent_velocity_bias=interaction_intent_velocity_bias,
                use_chirality=use_chirality,
                num_edge_types=num_edge_types,
                graph_num_edge_types=graph_num_edge_types,
                use_backbone_dihedral=use_backbone_dihedral,
                use_burial_feature=use_burial_feature,
                use_residue_frame_stream=use_residue_frame_stream,
                residue_frame_layers=residue_frame_layers,
                use_residue_frame_deep_inject=use_residue_frame_deep_inject,
                # effective-gate the volumetric deep-inject with use_volumetric_head here (the head lives
                # in this module, so an inject with no head would build dead projections). The denoiser stores the
                # AND'd value directly.
                use_volumetric_deep_inject=bool(use_volumetric_head and use_volumetric_deep_inject),
                use_target_deep_inject=use_target_deep_inject,
                use_backbone_deep_inject=use_backbone_deep_inject,
                frame_v2_deep_inject_detach=frame_v2_deep_inject_detach,
                use_bond_angle_deep_inject=use_bond_angle_deep_inject,
                # effective-gate the existence coupling with use_volumetric_head here (the head lives in
                # this module, so a coupling with no head would build a dead projection). The denoiser stores the
                # AND'd value directly.
                use_volumetric_existence_coupling=bool(use_volumetric_head and use_volumetric_existence_coupling),
                use_residue_frame_stream_v2=use_residue_frame_stream_v2,
                residue_frame_v2_layers=residue_frame_v2_layers,
                frame_v2_clean_input=frame_v2_clean_input,
                # NEW: AND with the v2 stream here (the deep-inject projections live in the denoiser, so a deep-inject
                # with no v2 stream would build dead projections + have no source latent). The denoiser stores the AND'd value.
                use_residue_frame_v2_deep_inject=bool(use_residue_frame_stream_v2 and use_residue_frame_v2_deep_inject),
                # AND with the v2 stream here (the head lives inside the v2 module, so a head with no
                # stream would build dead params). The denoiser stores the AND'd value directly.
                use_stereochem_head=bool(use_residue_frame_stream_v2 and use_stereochem_head),
                # AND with the v2 stream + static stereo head here (the t-resolution head needs the
                # static P(D)_prior). The denoiser stores the AND'd value directly.
                use_stereochem_t_resolution=bool(
                    use_residue_frame_stream_v2 and use_stereochem_head and use_stereochem_t_resolution
                ),
                # AND with the t-resolution head (+ static stereo head + v2 stream). The denoiser stores
                # the AND'd value directly.
                use_stereochem_t_resolution_feedback=bool(
                    use_residue_frame_stream_v2
                    and use_stereochem_head
                    and use_stereochem_t_resolution
                    and use_stereochem_t_resolution_feedback
                ),
                activation_checkpointing=activation_checkpointing,
                activation_checkpoint_stride=activation_checkpoint_stride,
                use_global_latent_matching=use_global_latent_matching,
                global_latent_embed_dim=self.global_latent_embed_dim,
                global_latent_confidence=global_latent_confidence,
                global_latent_condition_main_stream=global_latent_condition_main_stream,
                global_latent_film_layers=global_latent_film_layers,
                graft_init_std=graft_init_std,
            )
        # === Polarity head (burial/backbone -> element composition) ===================
        # A supervised bottleneck expert: from the per-residue backbone encoding (which carries the
        # Cbeta-burial contribution when use_burial_feature is on), predict the sidechain's element
        # COMPOSITION as a distribution over the real element classes {C, N, O, X} (everything except
        # PAD). Its log-distribution is added as a product-of-experts bias to the real element logits
        # and RENORMALISED within the real block so P(PAD) is preserved EXACTLY (existence stays with
        # EVC / the element head; polarity only reweights identity-given-existence). See forward().
        # Zero-init the last layer so at graft time polarity_logits==0 -> uniform -> the renormalised
        # bias is a no-op, i.e. identity when resuming a checkpoint; the supervised loss grows it in.
        self.use_polarity_head = use_polarity_head
        self.polarity_loss_weight = polarity_loss_weight
        if use_polarity_head:
            # The polarity PoE assumes class 0 is PAD (it holds class 0 fixed and reweights classes 1..K-1
            # to keep P(PAD) invariant). Split-existence mode has NO PAD class (class 0 = Carbon), so this
            # would preserve Carbon and reweight the wrong block. Reject loudly; a split-aware polarity path
            # over all real element classes would be needed instead (not implemented -- 2-track is prod).
            if split_element_existence:
                raise ValueError(
                    "use_polarity_head=True is incompatible with split_element_existence=True: the polarity "
                    "PoE holds class 0 fixed as PAD, but in split mode class 0 is Carbon (there is no PAD "
                    "class), so it would preserve Carbon and reweight the wrong block. Disable one of the two."
                )
            n_real_classes = self.num_element_classes - 1  # exclude PAD (index 0)
            self.polarity_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, n_real_classes),
            )
            nn.init.zeros_(self.polarity_head[-1].weight)
            nn.init.zeros_(self.polarity_head[-1].bias)
        self.use_glycine_head = use_glycine_head
        self.glycine_loss_weight = glycine_loss_weight
        self.glycine_pad_cap = glycine_pad_cap
        if use_glycine_head:
            # Split-existence mode has NO PAD class (class 0 = Carbon), so a "raise P(PAD)" push would
            # boost carbon instead of the existence axis. Reject loudly rather than silently corrupting
            # the element head; glycine must instead go through the split existence head in that mode.
            if split_element_existence:
                raise ValueError(
                    "use_glycine_head=True is incompatible with split_element_existence=True: in split "
                    "mode class 0 is Carbon (there is no PAD class), so the glycine PAD push would boost "
                    "carbon. Implement glycine via the split existence head, or disable one of the two."
                )
            # Input = the RAW 8-dim backbone dihedral features ONLY (phi,psi,omega,tau x [sin,cos]) via
            # backbone_encoder._backbone_dihedral_features -> a scalar glycine logit per residue.
            self.glycine_head = nn.Sequential(
                nn.Linear(8, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            nn.init.zeros_(self.glycine_head[-1].weight)
            # LOW classifier-bias init: sigmoid(-4)~=0.018 for EVERY residue at graft. So even if the
            # (nonnegative) gate grows before the BCE has separated glycine from non-glycine, the push
            # stays ~0 for all residues -> NO broad PAD bias. The BCE lifts the logit only where GT-glycine.
            nn.init.constant_(self.glycine_head[-1].bias, -4.0)
            # Gate parameter is passed through tanh(softplus(.)) at use -> strictly NONNEGATIVE (the push
            # can only ADD to P(PAD), never subtract) and bounded by `glycine_pad_cap`.

            # INIT 0.0 -- DO NOT set this back to -4. The push is a PRODUCT of two dampers,
            # push = cap * tanh(softplus(gate)) * sigmoid(classifier_logit),
            # so the gradient into `gate` is proportional to sigmoid(classifier_logit) and vice versa.
            # With BOTH damped at -4 the gate's gradient is ~0.0013 (vs ~0.64 with both open): a ~495x
            # attenuation that gradient-STARVES the gate. It cannot bootstrap -- it needs the classifier
            # confident to receive gradient, and the classifier's push needs the gate open to matter, so
            # each holds the other shut. Measured: the gate moved -4.00 -> -3.99 over ~30 epochs of v34.
            # FIX: safety comes from ONE damper (the classifier bias, still -4), not two multiplied.
            # * at graft (gate=0, classifier=-4): push = 4.0 * 0.600 * 0.0180 = 0.043 logits -- still
            # negligible (~1% shift in P(PAD)), so near-identity-at-graft is preserved;
            # * gate gradient at graft: ~0.023, i.e. ~18x larger -> the gate can actually MOVE if the
            # loss wants it to;
            # * add-only + capped is unchanged (tanh(softplus(.)) is still nonnegative and bounded).
            # NOTE: this only helps a FRESH graft -- a checkpoint containing `glycine_pad_gate` reloads
            # its stored (starved) value. Use `reset_glycine_gate` on a weights-resume to force it back
            # to this default; see reset_glycine_pad_gate().
            self.glycine_pad_gate = nn.Parameter(torch.full((1,), GLYCINE_PAD_GATE_INIT))
        self.use_coord_self_conditioning = use_coord_self_conditioning
        self.use_plan_latent = use_plan_latent
        self.use_interaction_intent = use_interaction_intent
        self.use_chirality = use_chirality
        self.pairwise_distance_loss_weight = pairwise_distance_loss_weight
        self.bonded_geometry_loss_weight = bonded_geometry_loss_weight
        self.rotamer_rmsd_loss_weight = rotamer_rmsd_loss_weight
        self.bond_window_loss_weight = bond_window_loss_weight
        self.bond_angle_loss_weight = bond_angle_loss_weight
        self.bond_length_loss_weight = bond_length_loss_weight
        self.scale_loss_weight = scale_loss_weight
        self.dist_from_ca_loss_weight = dist_from_ca_loss_weight
        self.use_element_velocity_coupling = use_element_velocity_coupling
        self.use_count_velocity_coupling = use_count_velocity_coupling
        self.use_bond_attention = use_bond_attention
        self.bond_loss_weight = bond_loss_weight
        self.pocket_contact_loss_weight = pocket_contact_loss_weight
        self.valence_loss_weight = valence_loss_weight
        self.use_cross_residue_packing = use_cross_residue_packing
        self.packing_contact_loss_weight = packing_contact_loss_weight
        self.cross_residue_packing_spatial = cross_residue_packing_spatial
        self.cross_residue_packing_k = cross_residue_packing_k
        self.cross_residue_packing_radius = cross_residue_packing_radius
        self.cross_residue_packing_min_seq_sep = cross_residue_packing_min_seq_sep
        self.use_neighbor_x0_packing = use_neighbor_x0_packing
        self.neighbor_x0_packing_recycles = max(1, int(neighbor_x0_packing_recycles))
        self.neighbor_x0_packing_radius = neighbor_x0_packing_radius
        self.neighbor_self_dropout_prob = float(neighbor_self_dropout_prob)
        self.neighbor_x0_highnoise_weight = float(neighbor_x0_highnoise_weight)
        self.latent_self_dropout_prob = float(latent_self_dropout_prob)
        self.distal_threshold_cap = float(distal_threshold_cap)
        self.distal_shell_ramp_epochs = int(distal_shell_ramp_epochs)
        # MODE guard (v36 FEATURE 3): latent self-dropout writes ELEMENT_MASK, which is a distinct
        # 'unknown' embedding row ONLY in 2-track (split_element_existence=False) + donut_element_init
        # =='mask'. In split mode / non-mask init the denoiser clamps the unknown id to a real element
        # row (or it collides with carbon at index 0), so the blinding is silently WRONG rather than
        # merely inert. Fail loud at construction (the flag is new, so no historical checkpoint trips
        # this). Also enforced in validate_training_flag_coherence for a friendlier launch message.
        if self.latent_self_dropout_prob > 0.0 and (split_element_existence or donut_element_init != "mask"):
            raise ValueError(
                f"latent_self_dropout_prob={self.latent_self_dropout_prob} requires split_element_existence="
                f"False (got {split_element_existence}) AND donut_element_init=='mask' (got "
                f"{donut_element_init!r}): ELEMENT_MASK is only a distinct identity-blind token in the "
                f"2-track mask-init regime; any other mode blinds to the wrong element. Use 2-track + "
                f"donut_element_init=mask, or set latent_self_dropout_prob=0."
            )
        self.neighbor_x0_packing_prob_start = neighbor_x0_packing_prob_start
        self.neighbor_x0_packing_prob_end = neighbor_x0_packing_prob_end
        self.neighbor_x0_packing_ramp_epochs = neighbor_x0_packing_ramp_epochs
        self.neighbor_x0_packing_random_recycles = bool(neighbor_x0_packing_random_recycles)
        self.neighbor_x0_packing_max_recycles = int(neighbor_x0_packing_max_recycles)
        # Parsed (and validated) only when the draw can actually happen, so an eval/resume loader
        # rebuilding a historical checkpoint never trips on a config it will not use.
        self.neighbor_x0_packing_recycle_weights = (
            parse_neighbor_x0_recycle_weights(
                neighbor_x0_packing_recycle_weights, self.neighbor_x0_packing_max_recycles
            )
            if (use_neighbor_x0_packing and self.neighbor_x0_packing_random_recycles)
            else None
        )
        self.use_transition_weighted_loss = use_transition_weighted_loss
        self.transition_weight_sticky = float(transition_weight_sticky)
        self.transition_weight_revised = float(transition_weight_revised)
        self.transition_weight_regression = float(transition_weight_regression)
        self.transition_weight_cap = float(transition_weight_cap)
        self.transition_coord_tol = float(transition_coord_tol)
        self.transition_coord_same_tol = float(transition_coord_same_tol)
        self.transition_coord_mag_scale = float(transition_coord_mag_scale)
        self.use_shape_prior = use_shape_prior
        self.shape_prior_loss_weight = shape_prior_loss_weight
        self.use_bond_co_diffusion = use_bond_co_diffusion
        self.bond_co_diffusion_weight = bond_co_diffusion_weight
        self.plan_latent_loss_weight = plan_latent_loss_weight
        self.interaction_intent_loss_weight = interaction_intent_loss_weight
        self.leak_gt_count = leak_gt_count
        self.leak_gt_direction = leak_gt_direction
        self.pseudo_cb_direction = pseudo_cb_direction
        self.d_aa_l_source_prob = float(d_aa_l_source_prob)
        if pseudo_cb_direction and leak_gt_direction:
            raise ValueError("--pseudo-cb-direction and --leak-gt-direction are mutually exclusive")
        self.evc_scheduled_sampling_prob = evc_scheduled_sampling_prob
        self.evc_soft_conditioning = evc_soft_conditioning
        self.evc_velocity_blend = evc_velocity_blend
        self.evc_ss_corruption = evc_ss_corruption
        self.ss_underfill_bias = ss_underfill_bias
        self.evc_ss_noised_element_prob = evc_ss_noised_element_prob
        self.evc_ss_corrupt_selfcond = evc_ss_corrupt_selfcond
        # PRETRAIN-ONLY: the denoiser is None, so its post-construction attribute wiring is skipped (nothing
        # downstream in the pretrain path reads these). Off => byte-identical (the block runs exactly as today).
        if self.denoiser is not None:
            self.denoiser._mixture_head_dropout = mixture_head_dropout
            self.denoiser._dlrt_detach = dlrt_detach
            self.denoiser._dlrt_ema_decay = dlrt_ema_decay
            if dlrt_ema_decay > 0:
                # Create EMA shadow copy of LRT head (not a registered submodule -- shouldn't affect optimizer)
                import copy

                self.denoiser._lrt_ema_shadow = copy.deepcopy(self.denoiser.residue_lrt_head)
                for p in self.denoiser._lrt_ema_shadow.parameters():
                    p.requires_grad = False

        # Continuous diffusion for coordinates with v-prediction for stability
        self.diffusion = GaussianDiffusion(
            timesteps=timesteps,
            schedule=schedule,
            noise_scale=4.0,  # Scale noise to 4Å std
            prediction_type=prediction_type,
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

        # Discrete diffusion for element types (PAD=0, C=1, N=2, O=3, S=4)
        # PAD encodes "no atom" -- replaces separate mask diffusion track entirely.
        # With prior_schedule="bare", the prior interpolates from data distribution
        # toward all-PAD as t->T: at t=T everything is PAD (bare backbone).
        # Build element prior -- default is empirical [0.71, 0.22, 0.028, 0.039, 0.002]
        # If element_pad_prior is set, override PAD weight and redistribute non-PAD proportionally
        custom_prior = None
        if element_pad_prior is not None:
            from .diffusion import DEFAULT_ELEMENT_PRIOR

            base = DEFAULT_ELEMENT_PRIOR[:NUM_ELEMENT_TYPES].clone()
            non_pad = base[1:]  # C, N, O, S
            non_pad = non_pad / non_pad.sum()  # Normalize to relative proportions
            custom_prior = torch.zeros(NUM_ELEMENT_TYPES)
            custom_prior[0] = element_pad_prior
            custom_prior[1:] = (1.0 - element_pad_prior) * non_pad

        # Determine bare target for time-varying prior schedule.
        # Default (PAD): at t=T all elements -> PAD (bare backbone).
        # Carbon: at t=T all elements -> Carbon (matches donut source where all atoms look real).
        # Mask: at t=T all elements -> MASK ("unknown" -- 6th class, never in GT).
        from .diffusion import ELEMENT_C

        self.donut_element_init = donut_element_init
        if donut_element_init == "carbon":
            bare_target = ELEMENT_C
        elif donut_element_init == "mask":
            bare_target = ELEMENT_MASK
        else:
            bare_target = ELEMENT_PAD

        # Extend prior to 6 classes if using MASK mode
        self.absorbing_mask = absorbing_mask and (donut_element_init == "mask")
        elem_prior = custom_prior
        elem_prior_schedule = pad_sampling_init  # "bare" or None

        if donut_element_init == "mask" and not split_element_existence:
            from .diffusion import DEFAULT_ELEMENT_PRIOR

            data_prior_5 = (
                custom_prior if custom_prior is not None else DEFAULT_ELEMENT_PRIOR[:NUM_ELEMENT_TYPES].clone()
            )

            if self.absorbing_mask:
                # Absorbing MASK: fixed all-MASK prior. The existing q_sample/q_posterior
                # math automatically gives absorbing behavior: elements either stay at x_0
                # or go to MASK (the only prior sample), and can never leave MASK.
                elem_prior = torch.zeros(self.num_element_classes)
                elem_prior[ELEMENT_MASK] = 1.0
                elem_prior_schedule = None  # Fixed prior, no time-varying schedule
            else:
                # Non-absorbing MASK: time-varying schedule drives toward MASK at t=T
                # but allows re-masking/un-masking at intermediate timesteps.
                elem_prior = torch.cat([data_prior_5, torch.zeros(1)])  # MASK=0 in data prior

        # In split mode, the standard element_diffusion is a simple 5-class placeholder;
        # actual element diffusion uses self.split_element_diffusion (created below).
        if split_element_existence:
            # Minimal 5-class diffusion (not used in forward/sample, but must exist for module structure)
            elem_prior = None
            elem_prior_schedule = None
            bare_target = ELEMENT_PAD

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

        # For absorbing MASK: override class weights with data-distribution weights.
        # The all-MASK prior gives equal weights to all non-MASK classes (all clamped at 20),
        # losing the rebalancing between PAD(71%), C(22%), etc.
        if self.absorbing_mask and not split_element_existence:
            data_prior_6 = torch.cat([data_prior_5, torch.ones(1)])  # MASK gets weight 1 (never in GT)
            data_weights = (1.0 / data_prior_6.clamp(min=0.01)).clamp(max=20.0)
            # Normalizer: ANCHOR to the canonical vocab5 reference {PAD,C,N,O,S,MASK} (default) so that
            # extending the element vocab (vocab12) does NOT rescale the canonical weights via the normalizer.
            # The old unanchored mean-over-ALL-classes normalizer is exactly what diluted PAD ~0.13->0.09 at
            # vocab12 (the 7 rare classes pin at the clamp ceiling of 20 and inflate the mean), mushing the
            # P(PAD) signal that gates coordinate routing. With anchoring, PAD is vocab-invariant (same prior
            # 0.71 -> 0.126 exactly); CNOS are near-invariant (any residual is the small vocab5-vs-vocab12
            # default-prior delta, e.g. C 0.220 vs 0.218, NOT normalizer dilution). For vocab5 anchored ==
            # unanchored (the reference IS the full set).
            _anchor = True
            if _anchor:
                from .diffusion import DEFAULT_ELEMENT_PRIOR

                _canon = torch.cat([DEFAULT_ELEMENT_PRIOR[:5].clone(), torch.ones(1)])  # PAD,C,N,O,S,MASK
                _norm = (1.0 / _canon.clamp(min=0.01)).clamp(max=20.0).mean()
            else:
                _norm = data_weights.mean()
            data_weights = data_weights / _norm
            # Element-loss class weights; overall element-loss scale is governed by element_loss_weight.
            _pad_scale = 1.0
            os.environ.get("ATOMWEAVER_VERBOSE") and print(
                f"[element-loss] vocab={data_prior_6.numel() - 1} anchor={_anchor} pad_scale={_pad_scale} "
                f"-> PAD class weight {float(data_weights[ELEMENT_PAD]):.4f}"
            )
            self.element_diffusion.class_weights.copy_(data_weights)

        # Element flow matching: continuous flow on probability simplex as alternative to absorbing diffusion
        self.element_flow_matching = element_flow_matching
        if element_flow_matching:
            self.element_flow = ElementFlowMatching(
                timesteps=timesteps,
                num_classes=self.num_element_classes,
                prior=self.element_diffusion.prior.clone(),
            )

        # Late element resolution: standard PAD-aware element diffusion throughout, but
        # non-PAD slots are stochastically collapsed to Carbon based on a cosine schedule
        # compressed into [0, cutoff]. Chemistry {C,N,O,S} resolves smoothly in the final
        # portion of the trajectory. PAD/non-PAD signal is preserved at all noise levels.
        self.non_pad_element_sampling = non_pad_element_sampling
        self.late_element_resolution = late_element_resolution
        self.late_element_cutoff = late_element_cutoff
        self.contact_weight_min = contact_weight_min
        self.contact_weight_scale = contact_weight_scale
        self.contact_weight_boost = contact_weight_boost
        self.contact_weight_coord = contact_weight_coord
        self.contact_weight_count = contact_weight_count
        self.fill_corrected_coord_weight = fill_corrected_coord_weight
        self.fcc_slope = fcc_slope
        self.fcc_tau = fcc_tau
        self.fcc_per_env = fcc_per_env
        self.fcc_slope_buried = fcc_slope_buried
        self.fcc_slope_exposed = fcc_slope_exposed
        self.occupancy_match_loss_weight = occupancy_match_loss_weight
        self.occupancy_match_sigma = occupancy_match_sigma
        self.occupancy_match_tau = occupancy_match_tau
        # Validated fail-loud (finite, in [0.0, 1.0]) -- this floor is relaunch-critical: >1 reverses the
        # floor + (1-floor)*base interpolation, nan poisons the loss, <0 silently disables. Checked inline
        # (not via a training-side validator) because the training module imports models.py -> importing it back here
        # would be a circular import.
        _occ_floor = float(occupancy_match_timestep_floor)
        if not (math.isfinite(_occ_floor) and 0.0 <= _occ_floor <= 1.0):
            raise ValueError(
                f"occupancy_match_timestep_floor must be finite in [0.0, 1.0] (got {_occ_floor!r}); "
                ">1 reverses the interpolation, nan poisons the loss, <0 silently disables"
            )
        self.occupancy_match_timestep_floor = _occ_floor

        # Volumetric self-occupancy head (). Built ONLY when opted in, so the model is
        # byte-identical (no new params / no RNG advance) when off, and resume-safe. The head is
        # t-independent and reads only the fixed backbone + fixed target (no side chains in its
        # context), so nothing downstream changes yet -- this chunk only adds a supervision loss.
        # A later chunk will deep-inject the exposed `vol_hidden` into the flow.
        self.use_volumetric_head = bool(use_volumetric_head)
        # Slot budget, mirrored at the top level so it is reachable even in pretrain-only builds where the
        # denoiser (which also stores it) is never constructed. Consumed by the pretrain-only volumetric decoy
        # CE to filter references whose sidechain exceeds the model's budget (representability), matching the
        # atom-disc path's max_sc filter.
        self.max_sidechain_atoms = int(max_sidechain_atoms)
        self.volumetric_loss_weight = float(volumetric_loss_weight)
        # SCALE-ANCHOR weight (per-residue total-mass match). 0.0 => not added (byte-identical). Meaningful
        # only WITH use_volumetric_head (guarded at the loss add-sites below).
        self.volumetric_scale_anchor_weight = float(volumetric_scale_anchor_weight)
        # VOLUMETRIC DECOY CE (FT-only). Plain scalars consumed by DiffusionLightningModule.training_step (which
        # owns the residue-DB decoys + the reference-density library). Stored raw (NOT AND'd with the head) so the
        # __init__ guard below can fail loud on an incoherent request; the training_step CE path is head-gated.
        self.volumetric_decoy_ce_weight = float(volumetric_decoy_ce_weight)
        self.volumetric_decoy_ce_temperature = float(volumetric_decoy_ce_temperature)
        # INTEGRATED volumetric loss target: self-sidechain-only GT (False, historical) vs own-only
        # region-bucketed self-occupancy objective (True). Effective only WITH the head (the loss term is head-gated),
        # so store the AND -- False when the head is off keeps every off path byte-identical.
        self.volumetric_loss_supervision_target = bool(use_volumetric_head and volumetric_loss_supervision_target)
        # A/B toggle for the self-occupancy occupancy target composition. Default False = own-only (faithful);
        # True = legacy own+context. Threaded into _supervision_full_field_occupancy_loss (both the pretrain-only and
        # integrated self-occupancy-target paths).
        self.volumetric_target_include_context = bool(volumetric_target_include_context)
        self.volumetric_empty_weight = float(volumetric_empty_weight)
        # VOLUMETRIC-FAITHFUL ATOM-ANCHORED QUERIES. Effective only WITH the head (AND); stored so the pretrain-only
        # forward knows to sample per-residue atom-anchored queries + supervise the head at them. False when the
        # head is off keeps every off path byte-identical.
        self.volumetric_atom_anchored_queries = bool(use_volumetric_head and volumetric_atom_anchored_queries)
        # volumetric deep-inject is effective only WITH the head (AND). Stored so forward/sample can
        # decide to compute `vol_hidden` early (before the denoiser) and thread it into every SE(3) layer.
        self.use_volumetric_deep_inject = bool(use_volumetric_head and use_volumetric_deep_inject)
        # (run10): the deep-inject reads a zero-init Linear(n_query -> hidden) of the DETACHED density field
        # instead of the degenerate pooled vol_hidden. Built here (IFD level, where vol_density_pred is produced);
        # the (B,L,hidden) latent threads down to exactly like vol_hidden. Zero-init weight => +0 at init
        # => byte-identical for run9-lineage resumes until it learns. Bias-free (mirrors the vol_hidden path).
        self.use_volumetric_density_inject = bool(
            use_volumetric_head and use_volumetric_deep_inject and use_volumetric_density_inject
        )
        if self.use_volumetric_density_inject:
            self.volumetric_density_inject_proj = nn.Linear(volumetric_n_query, hidden_dim, bias=False)
            nn.init.zeros_(self.volumetric_density_inject_proj.weight)
        # run10 graft coherence -- fail-loud rather than silently inert (the "flag set but no-op" launch trap):
        if use_volumetric_density_inject and not (use_volumetric_head and use_volumetric_deep_inject):
            raise ValueError(
                "use_volumetric_density_inject=True requires use_volumetric_head AND use_volumetric_deep_inject: "
                "the density projection is the deep-inject source, AND-gated off (never built) otherwise."
            )
        if use_target_deep_inject and not (use_sidechain_target_residue_attention and use_target_conditioning):
            raise ValueError(
                "use_target_deep_inject=True requires BOTH use_sidechain_target_residue_attention=True AND "
                "use_target_conditioning=True: sc_target_attn is only computed under both flags (and with target "
                "inputs present); otherwise target_deep_proj is built but receives zero gradient and trains nothing."
            )
        if use_target_deep_inject and use_cluster_particle_diffusion:
            raise ValueError(
                "use_target_deep_inject is not supported with use_cluster_particle_diffusion: sc_target_attn is "
                "per-slot (N_sc) while cluster-particle pooling collapses to n_binder_sc < N_sc nodes, so the "
                "per-layer node_cond[:n_binder_sc] = target_deep_proj(sc_target_attn) shape-mismatches."
            )
        # volumetric -> existence coupling is effective only WITH the head (AND). Stored so forward AND
        # both sample paths know to compute `vol_hidden` early (before the denoiser) and thread it in, so the
        # element head's PAD-logit bias is live at training AND inference (the coupling itself lives in the
        # denoiser, effective-gated with the head at construction below).
        # The coupling SUBTRACTS its bias from element_logits[..., ELEMENT_PAD] (class 0). Split-existence mode
        # has NO PAD class in element_logits (class 0 = Carbon; existence lives on the separate occupancy track),
        # so the bias would silently corrupt the Carbon logit. Reject loudly -- mirrors the polarity/glycine
        # heads above, which raise for the same "class 0 is not PAD in split mode" reason. A split-aware coupling
        # would have to bias the split existence head instead (not implemented -- 2-track is prod).
        if use_volumetric_existence_coupling and split_element_existence:
            raise ValueError(
                "use_volumetric_existence_coupling=True is incompatible with split_element_existence=True: the "
                "coupling subtracts its bias from element_logits[..., ELEMENT_PAD] (class 0), but in split mode "
                "class 0 is Carbon (there is no PAD class in element_logits -- existence is a separate track), so "
                "it would corrupt the Carbon logit. Implement the coupling via the split existence head, or "
                "disable one of the two."
            )
        self.use_volumetric_existence_coupling = bool(use_volumetric_head and use_volumetric_existence_coupling)
        # SANDCLOCK available-volume input. AND-gated with the head: the descriptor is only projected+added
        # inside the head, so with the head off there is nothing to feed it into. Refuse the incoherent combo
        # (flag set but inert) at EVERY construction path -- mirrors the deep-inject / existence-coupling guards.
        if use_available_volume and not use_volumetric_head:
            raise ValueError(
                "use_available_volume=True requires use_volumetric_head=True: the available-volume descriptor "
                "is a GT-free INPUT projected into the volumetric head's vol_hidden latent, so with the head off "
                "there is nothing to feed it into and the flag does NOTHING (the model AND-gates the two, building "
                "no projection). Enable use_volumetric_head, or set use_available_volume=False."
            )
        self.use_available_volume = bool(use_volumetric_head and use_available_volume)
        # SINGLE-SITE POCKET CONTEXT. AND-gated with the head: the neighbour-occupancy field + its zero-init
        # conditioning projection live inside the volumetric head, so with the head off there is nothing to
        # condition and the flag does NOTHING. Refuse the incoherent combo (flag set but inert) at EVERY
        # construction path -- mirrors the available-volume / deep-inject / existence-coupling guards.
        if use_single_site_context and not use_volumetric_head:
            raise ValueError(
                "use_single_site_context=True requires use_volumetric_head=True: single-site pocket context "
                "conditions the volumetric head on a neighbour-occupancy field (built + projected inside the "
                "head), so with the head off there is nothing to condition and the flag does NOTHING (the model "
                "AND-gates the two). Enable use_volumetric_head, or set use_single_site_context=False."
            )
        self.use_single_site_context = bool(use_volumetric_head and use_single_site_context)
        # SINGLE-SITE SCHEDULED-SAMPLING knobs. p_max in [0, 1]; ramp start >= 0. A positive p_max is only
        # meaningful when the single-site context is actually built (use_single_site_context, which is itself
        # AND-gated with the head above) AND in the INTEGRATED forward (pretrain-only computes the head on GT
        # neighbours and returns before any flow, so there is no predicted x0 to mix in). Refuse both incoherent
        # combos loudly on EVERY construction path -- mirrors the decoy-CE / available-volume guards.
        self.volumetric_ss_context_p_max = float(volumetric_ss_context_p_max)
        self.volumetric_ss_context_start_epoch = int(volumetric_ss_context_start_epoch)
        if not (0.0 <= self.volumetric_ss_context_p_max <= 1.0):
            raise ValueError(
                f"volumetric_ss_context_p_max={self.volumetric_ss_context_p_max} must be in [0, 1] (it is a "
                "per-site Bernoulli probability of using the predicted x0 instead of the GT clean x0)."
            )
        if self.volumetric_ss_context_start_epoch < 0:
            raise ValueError(
                f"volumetric_ss_context_start_epoch={self.volumetric_ss_context_start_epoch} must be >= 0."
            )
        if self.volumetric_ss_context_p_max > 0.0:
            if not self.use_single_site_context:
                raise ValueError(
                    f"volumetric_ss_context_p_max={self.volumetric_ss_context_p_max} (>0) requires "
                    "use_single_site_context=True: the scheduled-sampling mix swaps the SINGLE-SITE pocket "
                    "context source between GT and predicted x0, so with single-site context off there is no "
                    "context to mix and the ramp does NOTHING. Enable use_single_site_context (and "
                    "use_volumetric_head), or set volumetric_ss_context_p_max=0."
                )
            if self.volumetric_pretrain_only:
                raise ValueError(
                    f"volumetric_ss_context_p_max={self.volumetric_ss_context_p_max} (>0) is incompatible with "
                    "volumetric_pretrain_only=True: the pretrain-only forward computes the head on GT neighbour "
                    "side chains and returns BEFORE any flow denoiser runs, so there is no predicted x0 to mix "
                    "in -- the ramp would silently do nothing (and the 247/249 pretrains depend on the byte-"
                    "identical GT context). Drop volumetric_pretrain_only for the FT run, or set "
                    "volumetric_ss_context_p_max=0."
                )
            if not self.use_neighbor_x0_packing or self._neighbor_x0_effective_recycles() <= 1:
                raise ValueError(
                    f"volumetric_ss_context_p_max={self.volumetric_ss_context_p_max} (>0) requires "
                    f"use_neighbor_x0_packing=True (currently {self.use_neighbor_x0_packing}) AND effective "
                    f"recycles>1 (currently {self._neighbor_x0_effective_recycles()}): the predicted-x0 pocket "
                    "context is stashed ONLY inside the neighbour-x0 recycle loop (2-pass), so without it "
                    "ss_pred_coords is never populated, the option-(ii) recompute never fires, and the "
                    "scheduled-sampling ramp silently no-ops to pure GT teacher forcing for the ENTIRE run "
                    "(vol_hidden keeps the GT context). Enable use_neighbor_x0_packing with "
                    "neighbor_x0_packing_recycles>=2, or set volumetric_ss_context_p_max=0."
                )
        # PHASE 2 coherence: loading / freezing a pretrained volumetric head is meaningless without the head
        # (there is nothing to load into / nothing to freeze). Refuse loudly rather than silently ignore --
        # mirrors the deep-inject / existence-coupling "flag set but inert" guards in
        # validate_training_flag_coherence. Kept here so EVERY construction path (Ray/eval rebuild, tests,
        # direct instantiation) is protected, not just the CLI.
        if volumetric_head_pretrained and not use_volumetric_head:
            raise ValueError(
                "volumetric_head_pretrained is set but use_volumetric_head=False: there is no volumetric head "
                "to load the pretrained weights into, so the path does NOTHING. Enable use_volumetric_head, or "
                "clear volumetric_head_pretrained."
            )
        if freeze_volumetric_head and not use_volumetric_head:
            raise ValueError(
                "freeze_volumetric_head=True but use_volumetric_head=False: there is no volumetric head to "
                "freeze, so the flag does NOTHING. Enable use_volumetric_head, or set freeze_volumetric_head=False."
            )
        self.freeze_volumetric_head = bool(freeze_volumetric_head)
        # a self-occupancy-PRETRAINED head that is LEFT THAWED (not frozen) and trained with a
        # positive integrated volumetric_loss_weight against the SELF-SIDECHAIN-ONLY target (the head's simplified
        # anti-leak MSE) would be pulled OFF the own-only region-bucketed self-occupancy objective it was just pretrained
        # on. Refuse the combo -- the launcher must either switch to the self-occupancy-objective target
        # (volumetric_loss_supervision_target=True), freeze the head, or set weight 0.
        # Only meaningful with the head on (all four inputs are head-gated downstream). Mirrored in
        # validate_training_flag_coherence for a friendlier launch message; kept here so EVERY rebuild path
        # (Ray/eval, tests, direct instantiation) is protected.
        if (
            volumetric_head_pretrained
            and use_volumetric_head
            and not freeze_volumetric_head
            and volumetric_loss_weight > 0.0
            and not volumetric_loss_supervision_target
        ):
            raise ValueError(
                "volumetric_head_pretrained is set with the head THAWED (freeze_volumetric_head=False) and "
                f"volumetric_loss_weight={volumetric_loss_weight} (>0), but volumetric_loss_supervision_target=False: "
                "the integrated volumetric loss would supervise the head with the SELF-SIDECHAIN-ONLY objective "
                "(the head's simplified anti-leak MSE) instead of the region-bucketed self-occupancy objective it was "
                "pretrained on. Set volumetric_loss_supervision_target=True (supervise against the same field + "
                "bucketed loss the pretrain used), OR "
                "freeze_volumetric_head=True (fixed teacher), OR volumetric_loss_weight=0 (no head gradient)."
            )
        # VOLUMETRIC DECOY CE (coherence): the CE scores the head's predicted density field against reference
        # densities, so it structurally requires the volumetric head. A positive weight without the head would be
        # silently inert (the training_step CE path is head-gated). Refuse it. The disc-DB / stratified-decoy
        # prerequisites (which the model does not know about) are checked in validate_training_flag_coherence.
        # Kept here so EVERY rebuild path (Ray/eval, tests, direct instantiation) is protected.
        if self.volumetric_decoy_ce_weight > 0.0 and not use_volumetric_head:
            raise ValueError(
                f"volumetric_decoy_ce_weight={volumetric_decoy_ce_weight} (>0) requires use_volumetric_head=True: "
                "the volumetric decoy cross-entropy scores the volumetric head's predicted own-sidechain density "
                "field against per-type reference densities, so with the head off it would do NOTHING. Enable "
                "--use-volumetric-head, or set volumetric_decoy_ce_weight=0."
            )
        if self.volumetric_decoy_ce_weight > 0.0 and self.volumetric_decoy_ce_temperature <= 0.0:
            raise ValueError(
                f"volumetric_decoy_ce_temperature={volumetric_decoy_ce_temperature} must be > 0 when "
                "volumetric_decoy_ce_weight>0 (it is the softmax temperature on the density-similarity logits)."
            )
        # PRETRAIN-ONLY decoy CE (2026-08-07): the decoy CE now ALSO runs in the pretrain-only path. The head-only
        # build has no atom-disc decoys to reuse, so the LightningModule gives the CE its OWN decoy source straight
        # from the residue DB (a StratifiedDecoySampler + per-type reference densities built during training) and
        # the pretrain-only training_step/validation_step add ``volumetric_decoy_ce_weight * ce`` to the occupancy
        # loss. The disc-DB / stratified-decoy / cluster-file preconditions (which the model does not know about)
        # are checked in validate_training_flag_coherence. The former "pretrain-only forbids decoy CE" guard is
        # therefore retired; only the head-off / temperature guards above remain structural to the model.
        # pretrain-only builds the head as the ONLY trainable module (denoiser=None), so
        # freezing it means NOTHING trains at all. Refuse the contradictory combo. Mirrored in
        # validate_training_flag_coherence.
        if volumetric_pretrain_only and freeze_volumetric_head:
            raise ValueError(
                "volumetric_pretrain_only=True with freeze_volumetric_head=True is incoherent: pretrain-only "
                "builds ONLY the VolumetricOccupancyHead (the SE(3) atom transformer / flow is not constructed), "
                "so freezing the head leaves NO trainable parameters and the run does nothing. Set "
                "freeze_volumetric_head=False to pretrain the head, or drop volumetric_pretrain_only."
            )
        if self.use_volumetric_head:
            self.volumetric_head = VolumetricOccupancyHead(
                hidden_dim=hidden_dim,
                num_element_types=NUM_ELEMENT_TYPES,
                context_radius=volumetric_context_radius,
                n_query=volumetric_n_query,
                sigma=volumetric_sigma,
                empty_weight=volumetric_empty_weight,
                use_available_volume=self.use_available_volume,
                per_element_sigma=bool(volumetric_per_element_sigma),
                sigma_element_scale=volumetric_sigma_element_scale,
                use_single_site_context=self.use_single_site_context,
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
            # PHASE 2: optionally warm-start the head from a separately-pretrained checkpoint, then optionally
            # freeze it into a fixed teacher. Order matters -- load BEFORE freeze so the loaded weights are the
            # ones frozen. Both are no-ops (byte-identical) when the new params are at their None/False defaults.
            # Default (no pretrained load -> fresh grafts): EVERY optional key is fresh zero-init, so all stay
            # trainable under freeze (historical behavior). A pretrained load NARROWS this to only the optional
            # keys actually ABSENT from the checkpoint; optional keys that WERE loaded are trained teacher
            # tensors and must freeze with the rest.
            missing_optional_keys = {
                name for name, _ in self.volumetric_head.named_parameters() if _is_optional_volumetric_head_key(name)
            }
            if volumetric_head_pretrained:
                missing_optional_keys = self._load_pretrained_volumetric_head(volumetric_head_pretrained)
            if self.freeze_volumetric_head:
                n_frozen = 0
                n_exempt = 0
                for name, p in self.volumetric_head.named_parameters():
                    # EXEMPT a fresh (zero-init) optional graft absent from the checkpoint -- e.g. the SANDCLOCK
                    # available_volume_proj.* or the single-site field_proj.*: freezing the pretrained head into a
                    # fixed teacher must NOT lock a fresh graft at zero; it has to stay trainable so it can LEARN
                    # to modulate the frozen head. A LOADED optional graft (present in the checkpoint) is a trained
                    # teacher tensor and freezes like every other head param.
                    if _is_optional_volumetric_head_key(name) and name in missing_optional_keys:
                        n_exempt += 1
                        continue
                    p.requires_grad = False
                    n_frozen += 1
                os.environ.get("ATOMWEAVER_VERBOSE") and print(
                    f"[volumetric-head] FROZE {n_frozen} volumetric_head tensors "
                    f"(requires_grad=False); they are excluded from all optimizer param groups."
                    + (
                        f" EXEMPTED {n_exempt} trainable fresh-graft tensors absent from the checkpoint."
                        if n_exempt
                        else ""
                    )
                )
        # self-consistency (flow-x0 density <-> head density). Effective only WITH the head, so
        # the AND makes the flag inert (byte-identical) when the head is off. Ramp knobs are epoch fracs.
        self.use_volumetric_self_consistency = bool(use_volumetric_head and use_volumetric_self_consistency)
        self.self_consistency_weight = float(self_consistency_weight)
        self.self_consistency_ramp_start = float(self_consistency_ramp_start)
        self.self_consistency_ramp_end = float(self_consistency_ramp_end)
        # Positive-value guards: bad launch values would divide-by-zero in the tau gate / density kernel.
        if not self.fcc_tau > 0:
            raise ValueError(f"fcc_tau must be > 0 (got {self.fcc_tau})")
        if not self.occupancy_match_tau > 0:
            raise ValueError(f"occupancy_match_tau must be > 0 (got {self.occupancy_match_tau})")
        if not self.occupancy_match_sigma > 0:
            raise ValueError(f"occupancy_match_sigma must be > 0 (got {self.occupancy_match_sigma})")
        if late_element_resolution or non_pad_element_sampling:
            # Pre-compute chemistry retention schedule: alpha_bar for chemistry collapse.
            # Uses the same cosine schedule shape as the element diffusion, compressed
            # into [0, cutoff] of the global timeline.
            chem_betas = cosine_beta_schedule(timesteps)
            chem_alphas = 1.0 - chem_betas
            chem_alphas_cumprod = torch.cumprod(chem_alphas, dim=0)
            self.register_buffer("chem_retention", chem_alphas_cumprod)

        # Existence flow: continuous flow-matched existence variable (0=ghost, 1=real)
        self.mixture_lr_threshold = mixture_lr_threshold
        self.mixture_real_var_floor = mixture_real_var_floor
        self.use_existence_flow = use_existence_flow
        self.existence_loss_weight = existence_loss_weight
        self.existence_absorbing = existence_absorbing
        if use_existence_flow and not existence_absorbing:
            self.existence_flow = ExistenceFlowMatching(
                timesteps=timesteps, source_value=existence_source_value, power=existence_power
            )

        # Split existence/element: separate existence flow from element diffusion.
        # Existence flow handles PAD/non-PAD (atom count), element diffusion handles {C,N,O,S,MASK}.
        self.split_absorbing_element = split_absorbing_element
        if split_element_existence:
            if existence_absorbing:
                # 3-class discrete diffusion for existence: {GHOST=0, REAL=1, MASK=2}
                # At t=T all slots converge to MASK; they resolve to GHOST or REAL during reverse.
                # Remaining MASK at t=0 -> GHOST (conservative: uncertain = ghost).
                self.use_existence_flow = True  # Flag reuse for downstream code paths
                EXISTENCE_DATA_PRIOR = torch.tensor([0.71, 0.29, 0.0])  # GHOST, REAL, MASK
                if split_absorbing_element:
                    # Absorbing: fixed all-MASK prior. Slots can only leave MASK, never return.
                    exist_prior = torch.zeros(NUM_EXISTENCE_CLASSES)
                    exist_prior[EXISTENCE_MASK] = 1.0
                    exist_prior_schedule = None
                else:
                    # Non-absorbing: time-varying prior drives toward MASK at t=T,
                    # but allows re-MASKing at intermediate timesteps (mirroring element pattern).
                    exist_prior = EXISTENCE_DATA_PRIOR.clone()  # MASK=0 in data
                    exist_prior_schedule = "bare"
                self.existence_diffusion = ClassWeightedDiscreteDiffusion(
                    timesteps=timesteps,
                    schedule=schedule,
                    num_classes=NUM_EXISTENCE_CLASSES,
                    prior=exist_prior,
                    prior_schedule=exist_prior_schedule,
                    bare_target=EXISTENCE_MASK,
                )
                # Weight classes by data distribution: ~71% ghost, ~29% real
                exist_data_weights = torch.cat([EXISTENCE_DATA_PRIOR[:2], torch.ones(1)])  # MASK=1
                exist_weights = (1.0 / exist_data_weights.clamp(min=0.01)).clamp(max=20.0)
                exist_weights = exist_weights / exist_weights.mean()
                self.existence_diffusion.class_weights.copy_(exist_weights)
            elif not use_existence_flow:
                # Auto-enable continuous existence flow for split mode
                self.use_existence_flow = True
                self.existence_flow = ExistenceFlowMatching(
                    timesteps=timesteps, source_value=existence_source_value, power=existence_power
                )
            # 5-class diffusion: {C=0, N=1, O=2, S=3, MASK=4}
            if split_absorbing_element:
                # Absorbing: fixed all-MASK prior. Elements can only leave MASK, never return.
                split_prior = torch.zeros(NUM_SPLIT_ELEMENT_TYPES)
                split_prior[SPLIT_ELEMENT_MASK] = 1.0
                split_prior_schedule = None
            else:
                # Non-absorbing: time-varying prior drives toward MASK at t=T,
                # but allows re-masking at intermediate timesteps.
                split_prior = torch.cat([SPLIT_ELEMENT_DATA_PRIOR, torch.zeros(1)])  # MASK=0 in data
                split_prior_schedule = "bare"
            self.split_element_diffusion = ClassWeightedDiscreteDiffusion(
                timesteps=timesteps,
                schedule=schedule,
                num_classes=NUM_SPLIT_ELEMENT_TYPES,
                prior=split_prior,
                prior_schedule=split_prior_schedule,
                bare_target=SPLIT_ELEMENT_MASK,
            )
            # Override class weights with non-PAD data distribution
            data_prior_5 = torch.cat([SPLIT_ELEMENT_DATA_PRIOR, torch.ones(1)])  # MASK gets weight 1
            data_weights = (1.0 / data_prior_5.clamp(min=0.01)).clamp(max=20.0)
            data_weights = data_weights / data_weights.mean()
            self.split_element_diffusion.class_weights.copy_(data_weights)

            # Joint class weights over the 2-track encoding {PAD,C,N,O,S,...} for the chain-rule-coupled
            # existence+element loss (see the element-loss block in forward). Same recipe as 2-track
            # (1/prior, clamp 20) but the mean-normalizer is ANCHORED to the canonical vocab5 reference
            # {PAD,C,N,O,S,MASK} -- so for vocab5 this reproduces the baseline's element class weights exactly, AND
            # extending the element vocab can NEVER rescale/shrink the PAD & CNOS weights (that
            # rescaling is the 2-track vocab12 PAD-washout this whole vocab extension exists to avoid). Indexed per slot
            # by the joint target (PAD for ghost, element for real).
            from .diffusion import DEFAULT_ELEMENT_PRIOR

            _joint_prior = torch.cat([DEFAULT_ELEMENT_PRIOR[:NUM_ELEMENT_TYPES].clone(), torch.ones(1)])
            _joint_w = (1.0 / _joint_prior.clamp(min=0.01)).clamp(max=20.0)
            _canon_prior = torch.cat([DEFAULT_ELEMENT_PRIOR[:5].clone(), torch.ones(1)])  # PAD,C,N,O,S,MASK
            _ref_mean = (1.0 / _canon_prior.clamp(min=0.01)).clamp(max=20.0).mean()  # fixed vocab5 normalizer
            _joint_w = _joint_w / _ref_mean
            self.register_buffer("coupled_joint_class_weights", _joint_w[:NUM_ELEMENT_TYPES].clone())

        # Split element flow matching: continuous flow on {C,N,O,S,MASK} simplex
        # Replaces discrete diffusion for element types in split mode.
        self.split_element_flow = split_element_flow
        self.split_element_flow_temp = split_element_flow_temp
        self.split_element_uniform_power = split_element_uniform_power
        if split_element_flow and split_element_existence:
            # Prior: MASK-heavy for absorbing-like behavior, or data prior for non-absorbing
            if split_absorbing_element:
                # Absorbing-like: prior is all-MASK (flow drives to MASK at t=T)
                flow_prior = torch.zeros(NUM_SPLIT_ELEMENT_TYPES)
                flow_prior[SPLIT_ELEMENT_MASK] = 1.0
            else:
                # Non-absorbing: data prior with small MASK weight (allows re-masking)
                flow_prior = torch.cat([SPLIT_ELEMENT_DATA_PRIOR, torch.tensor([0.1])])
            self.split_element_flow_proc = ElementFlowMatching(
                timesteps=timesteps,
                num_classes=NUM_SPLIT_ELEMENT_TYPES,
                prior=flow_prior,
            )

        # Prior cloud: backbone-conditioned centroid/logvar heads.
        # Predicts from backbone_features (stable, available before sidechain denoising)
        # to provide a meaningful mixture signal at high noise when sidechain atoms are garbage.
        if use_prior_cloud:
            self.prior_centroid_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 3),
            )
            self.prior_cloud_logvar_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.prediction_type = prediction_type
        self.coord_process_type = coord_process_type
        self.flow_ghost_power = flow_ghost_power
        self.flow_real_proximal_power = flow_real_proximal_power
        self.flow_real_distal_power = flow_real_distal_power
        self.flow_use_conditional_groupwise = flow_use_conditional_groupwise
        self.ghost_weight = ghost_weight
        self.pad_sampling_init = pad_sampling_init
        self.element_fn_weight = element_fn_weight
        self.element_loss_non_pad_only = element_loss_non_pad_only
        self.atom_mask_loss_weight = atom_mask_loss_weight
        self.mask_bce_pos_weight_cap = mask_bce_pos_weight_cap
        self.soft_count_loss_weight = soft_count_loss_weight
        self.use_multi_count_discretization = use_multi_count_discretization
        self.multi_count_max_plus = multi_count_max_plus
        self.decoupled_count = decoupled_count
        self.count_perturb_prob = count_perturb_prob
        self.count_corr_tau = count_corr_tau
        self.count_tau_floor = count_tau_floor
        self.occupancy_loss_weight = occupancy_loss_weight
        self.occupancy_gate_elements = occupancy_gate_elements
        self.occupancy_gate_strength = occupancy_gate_strength
        self.coord_dropout = coord_dropout
        self.mixture_gate_weight = mixture_gate_weight
        self.ghost_var_floor = ghost_var_floor
        self.mixture_loss_weight = mixture_loss_weight
        self.mixture_gate_max_noise = mixture_gate_max_noise
        self.mixture_override_pad = mixture_override_pad
        self.disc_detach_mask = disc_detach_mask
        self.all_carbon_sampling = all_carbon_sampling
        self.position_based_element_powers = position_based_element_powers
        self.preal_gate_target = preal_gate_target
        self.residue_count_loss_weight = residue_count_loss_weight
        self.count_ranking_loss_weight = count_ranking_loss_weight
        self.count_pearson_loss_weight = count_pearson_loss_weight
        self.noise_dependent_ghost_weight = noise_dependent_ghost_weight
        # Whether to use per-slot element noise schedules matched to coordinate flow powers.
        # When True, distal real atoms stay non-PAD longer in forward (matching their coord schedule),
        # and ghost atoms transition to PAD faster. Controlled by flow_use_conditional_groupwise.
        self.groupwise_element_schedule = flow_use_conditional_groupwise and coord_process_type == "flow_matching"
        self.count_extreme_alpha = count_extreme_alpha
        self.asymmetric_perturb = asymmetric_perturb
        self.mixture_head_dropout = mixture_head_dropout
        self.rc_head_tau = rc_head_tau
        self.dynamic_lrt = dynamic_lrt
        self.dynamic_lrt_weight = dynamic_lrt_weight
        self.dynamic_lrt_clamp = dynamic_lrt_clamp
        self.dynamic_lrt_loss_type = dynamic_lrt_loss_type
        self.dynamic_lrt_rank_weight = dynamic_lrt_rank_weight
        self.dynamic_lrt_reg_weight = dynamic_lrt_reg_weight
        self.dlrt_analytical_scale = dlrt_analytical_scale
        self.dlrt_detach = dlrt_detach
        self.dlrt_sample_scale = dlrt_sample_scale
        self.dlrt_ema_decay = dlrt_ema_decay
        self.use_split_velocity = use_split_velocity
        self.split_velocity_sampling_only = split_velocity_sampling_only
        self.use_prior_cloud = use_prior_cloud
        self.prior_cloud_loss_weight = prior_cloud_loss_weight
        self.prior_blend_power = prior_blend_power
        self.sharpen_temperature_min = sharpen_temperature_min
        self.sharpen_temperature_power = sharpen_temperature_power
        self.sidechain_corrupt_prob = sidechain_corrupt_prob
        self.sidechain_corrupt_noise_gate = sidechain_corrupt_noise_gate
        self.sidechain_corrupt_distal_only = sidechain_corrupt_distal_only
        self.sidechain_compress_prob = sidechain_compress_prob
        self.rotation_corruption_prob = rotation_corruption_prob
        self.rotation_corruption_max_angle = rotation_corruption_max_angle
        self.rotation_corruption_min_tau = float(rotation_corruption_min_tau)
        # (stereochem): mirror-flip + in-plane-flatten coord-input corruption probabilities. Both
        # reuse rotation_corruption_min_tau as the t≈0 floor (identical 1/s singularity in the recompute).
        self.mirror_corruption_prob = float(mirror_corruption_prob)
        self.inplane_corruption_prob = float(inplane_corruption_prob)
        # FM-ONLY guard: rotation-corruption edits the noised INPUT (x_t) and REQUIRES rebuilding the
        # flow-matching velocity target from the implied corrupted source (recompute_target_for_corrupted_xt).
        # That recompute is specific to the linear-interpolant velocity target; under DDPM/v/epsilon there is
        # no such re-derivation, so the corruption would silently train the model on a stale target. Fail loud.
        if rotation_corruption_prob > 0.0 and coord_process_type != "flow_matching":
            raise ValueError(
                f"rotation_corruption_prob={rotation_corruption_prob} requires "
                f'coord_process_type="flow_matching" (got "{coord_process_type}"): the velocity-target '
                "recompute after corrupting x_t is flow-matching-specific. Disable rotation-corruption or "
                "switch to flow_matching."
            )
        # FM-ONLY guard: the mirror/in-plane corruptions edit x_t identically and share the SAME
        # velocity-target recompute, so they carry the SAME flow-matching-only requirement. Fail loud.
        if (mirror_corruption_prob > 0.0 or inplane_corruption_prob > 0.0) and coord_process_type != "flow_matching":
            raise ValueError(
                f"mirror_corruption_prob={mirror_corruption_prob} / inplane_corruption_prob="
                f'{inplane_corruption_prob} require coord_process_type="flow_matching" (got '
                f'"{coord_process_type}"): the velocity-target recompute after corrupting x_t is '
                "flow-matching-specific. Disable the stereo corruptions or switch to flow_matching."
            )

        # Range-validate the graft ramp / gate knobs, but ONLY when the owning (effective, AND-computed)
        # feature flag is ON, so a default/unused knob never blocks an eval/resume/launch. Bad ranges here
        # silently misbehave (a start>=end gives an always-0 or always-full ramp; an out-of-[0,1] epoch
        # fraction never fires the crossover), so fail loud at construction instead. Mirrors the fcc_tau /
        # occupancy_match_tau > 0 guards above.
        if self.use_stereochem_gated_init and not (0.0 <= self.gated_init_ramp_start < self.gated_init_ramp_end <= 1.0):
            raise ValueError(
                "gated_init_ramp_start/gated_init_ramp_end must satisfy "
                "0 <= start < end <= 1 (epoch fractions) when use_stereochem_gated_init is on, got "
                f"start={self.gated_init_ramp_start}, end={self.gated_init_ramp_end}."
            )
        if self.use_volumetric_self_consistency and not (
            0.0 <= self.self_consistency_ramp_start < self.self_consistency_ramp_end <= 1.0
        ):
            raise ValueError(
                "self_consistency_ramp_start/self_consistency_ramp_end must satisfy "
                "0 <= start < end <= 1 (epoch fractions) when use_volumetric_self_consistency is on, got "
                f"start={self.self_consistency_ramp_start}, end={self.self_consistency_ramp_end}."
            )
        if self.use_stereochem_t_resolution_feedback and not (
            0.0 <= self.t_resolution_feedback_ramp_start < self.t_resolution_feedback_ramp_end <= 1.0
        ):
            raise ValueError(
                "t_resolution_feedback_ramp_start/t_resolution_feedback_ramp_end must satisfy "
                "0 <= start < end <= 1 (epoch fractions) when use_stereochem_t_resolution_feedback is on, got "
                f"start={self.t_resolution_feedback_ramp_start}, end={self.t_resolution_feedback_ramp_end}."
            )
        # rotation_corruption_min_tau is the shared t≈0 floor for BOTH the rotation corruption and the
        # stereo (mirror / in-plane) corruptions; it is a tau=t/(T-1) fraction, so 0 < min_tau <= 1
        # (0 would re-admit the 1/s singularity it exists to gate). Enforce when ANY of those corruptions is on.
        if (
            self.rotation_corruption_prob > 0.0
            or self.mirror_corruption_prob > 0.0
            or self.inplane_corruption_prob > 0.0
        ) and not (0.0 < self.rotation_corruption_min_tau <= 1.0):
            raise ValueError(
                "rotation_corruption_min_tau must satisfy 0 < min_tau <= 1 (a tau=t/(T-1) floor) when "
                "rotation / mirror / in-plane corruption is on, got "
                f"{self.rotation_corruption_min_tau}."
            )

        self.main_path_underfill_prob = main_path_underfill_prob
        self.main_path_underfill_bias = main_path_underfill_bias
        self.autoregressive_training = autoregressive_training
        self.ar_isolated_residues = ar_isolated_residues
        if ar_isolated_residues and not autoregressive_training:
            raise ValueError("ar_isolated_residues requires autoregressive_training=True")

        if occupancy_gate_elements and occupancy_loss_weight <= 0:
            raise ValueError(
                "occupancy_gate_elements requires occupancy_loss_weight > 0, "
                "otherwise the occupancy head is untrained and gates with random noise"
            )
        sampling_mode_count = sum(
            int(flag) for flag in (all_carbon_sampling, late_element_resolution, non_pad_element_sampling)
        )
        if sampling_mode_count > 1:
            raise ValueError(
                "all_carbon_sampling, late_element_resolution, and non_pad_element_sampling are mutually exclusive"
            )
        if mixture_override_pad and sampling_mode_count > 0:
            raise ValueError(
                "mixture_override_pad is incompatible with all_carbon_sampling, late_element_resolution, "
                "and non_pad_element_sampling"
            )
        if (late_element_resolution or non_pad_element_sampling) and not (0.0 < late_element_cutoff <= 1.0):
            raise ValueError(
                "late_element_cutoff must be in (0, 1] when late_element_resolution or non_pad_element_sampling is enabled"
            )
        if split_element_existence and (element_flow_matching or mixture_override_pad or sampling_mode_count > 0):
            raise ValueError(
                "split_element_existence is incompatible with element_flow_matching, mixture_override_pad, "
                "all_carbon_sampling, late_element_resolution, and non_pad_element_sampling"
            )

        # Donut source validation: reject incompatible legacy settings.
        # Donut mode uses normal PAD element diffusion + direct mask BCE for ghost/real.
        # Mixture-based occupancy paths are not compatible.
        if use_donut_source:
            if non_pad_element_sampling:
                raise ValueError(
                    "use_donut_source is incompatible with non_pad_element_sampling. "
                    "Donut mode uses normal PAD element diffusion for ghost/real, not mixture posterior at t=0."
                )
            if all_carbon_sampling:
                raise ValueError(
                    "use_donut_source is incompatible with all_carbon_sampling. "
                    "Use --donut-element-init carbon instead."
                )
            if late_element_resolution:
                raise ValueError(
                    "use_donut_source is incompatible with late_element_resolution. "
                    "Use --donut-element-init carbon or mask instead."
                )
            if atom_mask_loss_weight <= 0:
                raise ValueError(
                    "use_donut_source requires atom_mask_loss_weight > 0 for direct ghost/real supervision."
                )
            if mixture_loss_weight > 0:
                raise ValueError(
                    "use_donut_source is incompatible with mixture_loss_weight > 0. "
                    "Donut mode does not use mixture modeling for occupancy."
                )
            if mixture_gate_weight > 0:
                raise ValueError(
                    "use_donut_source is incompatible with mixture_gate_weight > 0. "
                    "Donut mode does not use mixture gating for element logits."
                )
            if occupancy_gate_elements:
                raise ValueError(
                    "use_donut_source is incompatible with occupancy_gate_elements. "
                    "Donut mode uses direct mask BCE, not occupancy head gating."
                )

        # Per-slot fill rate buffer (set externally from dataset before training).
        # Used by atom_mask_loss for stable class balancing.
        # Falls back to per-batch computation if not set.
        self.register_buffer("_slot_fill_rate", torch.zeros(max_sidechain_atoms))
        # Per-slot coordinate statistics for mixture model (set externally from dataset).
        # _slot_coord_mean[j] = mean 3D offset from CA for real atoms in slot j.
        # _slot_coord_var[j] = scalar isotropic variance of (x0 - CA) for real atoms in slot j.
        self.register_buffer("_slot_coord_mean", torch.zeros(max_sidechain_atoms, 3))
        self.register_buffer("_slot_coord_var", torch.ones(max_sidechain_atoms))

        self.timesteps = timesteps
        self.use_target_conditioning = use_target_conditioning
        self.timestep_sampling = timestep_sampling
        self.coord_loss_weight = coord_loss_weight
        self.element_loss_weight = element_loss_weight
        self.atom_count_loss_weight = atom_count_loss_weight
        self.count_correlation_loss_weight = count_correlation_loss_weight
        self.count_loss_type = count_loss_type
        # Coordinate normalization: training coords have std ~100 Angstroms
        # Divide by this to get unit variance (~1.0) for proper diffusion
        # Only scale, don't shift mean (SE(3) equivariance requires translation invariance)
        self.coord_scale = coord_scale

        # Element type dropout: prevents model from over-relying on element types
        self.element_type_drop_prob = element_type_drop_prob
        self.element_type_token_drop_prob = element_type_token_drop_prob
        self.element_type_drop_schedule = element_type_drop_schedule
        self.self_conditioning_prob = self_conditioning_prob
        self.use_cluster_particle_diffusion = use_cluster_particle_diffusion
        self.oracle_atom_counts = oracle_atom_counts
        self.cluster_assignment_loss_weight = cluster_assignment_loss_weight
        self.cluster_cohesion_loss_weight = cluster_cohesion_loss_weight
        self.cluster_structure_loss_weight = cluster_structure_loss_weight
        self.cluster_contrastive_loss_weight = cluster_contrastive_loss_weight
        self.target_condition_scale = target_condition_scale
        self.cluster_target_condition_scale = cluster_target_condition_scale
        self.cluster_split_ramp_power = cluster_split_ramp_power
        self.cluster_label_permutation_prob = cluster_label_permutation_prob
        self.cluster_feature_warmup_fraction = cluster_feature_warmup_fraction
        # Self-conditioning: reduces exposure bias by training model on its own predictions
        self.self_conditioning_prob = self_conditioning_prob
        self.cluster_sampling_keep_start = 8.0  # Strongly prefer keeping current cluster at high noise
        self.cluster_sampling_keep_end = 0.2  # Keep a little late inertia to avoid collapsing far below target counts
        self.cluster_sampling_local_top_k = 4  # Allow only a few nearby merge targets per step
        self.cluster_sampling_soft_occupancy_floor = 0.85  # Do not merge far below soft expected occupancy
        self.cluster_sampling_soft_keep_strength = 4.0  # Preserve labels the model still believes are occupied
        # Preserve more late occupancy to prevent undercount collapse.
        self.cluster_sampling_soft_occupancy_floor_end = 0.7
        self.cluster_sampling_temperature_start = 0.85  # Slightly sharper updates at high noise
        self.cluster_sampling_temperature_end = 1.15  # Keep some late flexibility without over-merging stochastically
        self.cluster_sampling_resplit_start = 0.0  # Disabled by default; can be enabled for sampler-only repair tests
        self.cluster_sampling_resplit_end = 0.0
        self.cluster_sampling_resplit_min_occupancy = 0.25

        # Min-SNR-gamma: boost coordinate loss at low noise where v-prediction
        # provides weak x₀ reconstruction gradients (SNR is high -> x₀ ≈ xₜ)
        self.min_snr_gamma = min_snr_gamma

        # Coordinate noise augmentation: small Gaussian noise on all input coords during training
        self.coord_noise_std = coord_noise_std

        # Element type disable flag
        self.disable_element_types = disable_element_types

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
        if not self.use_available_volume:
            return None
        R, ca = build_local_frames(backbone_coords, backbone_mask)  # (B,L,3,3), (B,L,3)
        return available_volume_cones(backbone_coords, backbone_mask, target_coords, target_mask, R, ca)

    @staticmethod
    def _align_volumetric_state_dict(
        state_dict: "dict[str, torch.Tensor]", target_keys: set, path: str
    ) -> "dict[str, torch.Tensor]":
        """Strip any common prefix so a loaded state_dict's keys align to ``self.volumetric_head.*``.

        Handles a raw head ``state_dict()`` (keys already == ``target_keys``) AND a
        Lightning/standalone-trainer checkpoint whose head keys are prefixed (e.g. ``volumetric_head.*`` or
        ``model.volumetric_head.*``). The single common prefix that maps the loaded keys ONTO the head's
        REQUIRED keys is stripped; any non-head keys (other model params) are dropped. Raises if no such
        prefix exists -- a partial/misaligned load is a footgun (an un-loaded teacher reads as random guidance).

        OPTIONAL-KEY EXEMPTION: fresh input grafts such as SANDCLOCK ``available_volume_proj.*`` and SINGLE-SITE
        ``field_proj.*`` may be absent from an older pretrained density head. Those keys keep their zero-init
        local values; the load stays STRICT for every REQUIRED head key -- a checkpoint missing anything else,
        or carrying an unexpected key, still raises. A valid checkpoint = its (prefix-stripped) keys are a subset
        of the head's keys AND cover every REQUIRED (non-optional) head key.
        """
        target_keys = set(target_keys)
        required_keys = {k for k in target_keys if not _is_optional_volumetric_head_key(k)}

        def _matches(keys: set) -> bool:
            # Every loaded key belongs to the head (no unexpected) AND every REQUIRED head key is present.
            # Optional fresh-graft keys may be present or absent.
            return keys <= target_keys and required_keys <= keys

        if _matches(set(state_dict.keys())):
            return dict(state_dict)
        # Anchor on a REQUIRED head key (guaranteed present in any valid checkpoint -- an optional key may be
        # absent, so anchoring on it could miss the prefix). The _matches check below rejects a spurious prefix
        # (e.g. a substring collision); shortest prefix first keeps us closest to the raw head keys.
        anchor = next(iter(required_keys))
        candidate_prefixes = sorted({k[: len(k) - len(anchor)] for k in state_dict if k.endswith(anchor)}, key=len)
        for prefix in candidate_prefixes:
            stripped = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if _matches(set(stripped.keys())):
                return stripped
        raise ValueError(
            f"could not align the pretrained volumetric-head state_dict from {path!r} to self.volumetric_head: "
            f"after stripping any common prefix, the loaded keys do not cover the head's {len(required_keys)} "
            f"REQUIRED parameters/buffers (or carry unexpected keys). Pass a raw VolumetricOccupancyHead "
            f"state_dict, or a checkpoint whose head keys share a single '<prefix>volumetric_head.*'-style "
            f"prefix. Loaded {len(state_dict)} keys."
        )

    def _validate_volumetric_config(self, config: "dict", path: str) -> None:
        """Guard a pretrained head's saved ``config`` against the constructed ``self.volumetric_head`` geometry.

        Because ``query_local`` is a NON-persistent buffer and ``sigma``/``n_query``/``context_radius``/
        ``hidden_dim``/``num_element_types`` are plain attributes (not tensors in the state_dict), a strict
        ``load_state_dict`` does NOT catch a lattice/sigma/geometry mismatch: a head trained with e.g.
        ``n_query=24, sigma=2.0`` would silently load into an ``n_query=128, sigma=1.0`` head, quietly changing
        the field the "warm-started" weights were trained to produce. The Phase-1 trainer now saves ``config``,
        so validate it here and raise on the first mismatch. Only the keys the HEAD actually exposes are checked
        (``reserved_slot0`` and other provenance keys are carried but not enforced against the head).
        """
        head = self.volumetric_head
        # (config_key, head_attr, is_float): the geometry knobs the strict state_dict load cannot catch.
        # The two PER-ELEMENT-SIGMA knobs are here too: the sigma table is a NON-persistent buffer and the
        # param shapes are identical whether sigma is uniform or per-element, so a strict state_dict load would
        # silently accept a per-element-sigma head loaded into a uniform-sigma FT model (or vice versa),
        # changing the splat every "warm-started" weight was trained to match. per_element_sigma is a bool
        # (int compare); sigma_element_scale is a float.
        checks = (
            ("hidden_dim", "hidden_dim", False),
            ("n_query", "n_query", False),
            ("sigma", "sigma", True),
            ("context_radius", "context_radius", True),
            ("num_element_types", "num_element_types", False),
            ("volumetric_per_element_sigma", "per_element_sigma", False),
            ("volumetric_sigma_element_scale", "sigma_element_scale", True),
            # Fourier lift / softplus (6f7648f04). fourier_frequencies DOES change the state_dict (the
            # b_matrix buffer + a wider density_mlp input), so a strict load would already catch a mismatch;
            # use_softplus does NOT (it is an activation), so a legacy-vs-softplus warm-start would silently
            # change the field this "teacher" was trained to produce -> validate both here for a loud, clear
            # error. A pretrain checkpoint that stored NEITHER key simply skips the check (`cfg_key not in`).
            ("volumetric_fourier_frequencies", "fourier_frequencies", False),
            ("volumetric_use_softplus", "use_softplus", False),
            # PER-QUERY kNN CONTEXT. use_per_query_context BUILDS the context_encoder submodule => the strict
            # state_dict load already catches a mismatch, but validate it (+ context_k / context_heads, which
            # change the kNN/attention behavior WITHOUT changing param shapes and so would load silently) for a
            # loud, clear error. A pretrain checkpoint that stored none of these keys simply skips the check.
            ("volumetric_per_query_context", "use_per_query_context", False),
            ("volumetric_context_k", "context_k", False),
            ("volumetric_context_heads", "context_heads", False),
        )
        # A stored scale of None (the hparam default / per_element_sigma off) resolves to the head's own
        # default, exactly as the head does at build time -- so a None-scale checkpoint compares equal to a
        # default-scale head instead of crashing float(None), while a genuine numeric mismatch still raises.
        if config.get("volumetric_sigma_element_scale", 0.0) is None:
            config = {**config, "volumetric_sigma_element_scale": default_sigma_element_scale()}
        for cfg_key, attr, is_float in checks:
            if cfg_key not in config:
                continue
            saved = config[cfg_key]
            current = getattr(head, attr)
            if is_float:
                mismatch = not math.isclose(float(saved), float(current), rel_tol=1e-6, abs_tol=1e-9)
            else:
                mismatch = int(saved) != int(current)
            if mismatch:
                raise ValueError(
                    f"pretrained volumetric-head config from {path!r} is INCOMPATIBLE with the constructed head: "
                    f"{cfg_key}={saved!r} (checkpoint) != {cfg_key}={current!r} (this model). The head's query "
                    f"lattice / sigma / geometry are not in the state_dict, so loading across this mismatch would "
                    f"silently corrupt the field. Rebuild the head with matching volumetric_* settings, or "
                    f"pretrain a head with the current geometry."
                )

    @staticmethod
    def _synthesize_volumetric_config_from_hparams(hparams: dict) -> dict | None:
        """Build a :meth:`_validate_volumetric_config`-shaped config from a Lightning checkpoint's ``hyper_parameters``.

        A ``--volumetric-pretrain-only`` run is an ordinary Lightning module, so its checkpoint stores the
        head geometry (and the two per-element-sigma knobs) under ``hyper_parameters`` via
        ``save_hyperparameters`` -- NOT the sibling ``config`` block that the standalone
        ``train_volumetric_head.py`` writes. Map the hparam names onto the config keys the validator checks so
        a pretrain-only checkpoint is actually validated instead of falling to the config-absent warning path
        (which validates nothing). ``num_element_types`` is intentionally omitted: it is fixed by the
        ``ATOMWEAVER_ELEMENT_VOCAB`` env at build time and a vocab mismatch is already caught by the strict
        state_dict load (the ``element_embed`` embedding-weight shape is a persistent param). Returns ``None``
        if none of the keys are present (=> warn-but-load, unchanged).
        """
        hparam_to_cfg = {
            "hidden_dim": "hidden_dim",
            "volumetric_n_query": "n_query",
            "volumetric_sigma": "sigma",
            "volumetric_context_radius": "context_radius",
            "volumetric_per_element_sigma": "volumetric_per_element_sigma",
            "volumetric_sigma_element_scale": "volumetric_sigma_element_scale",
            # Fourier lift / softplus geometry knobs so a pretrain-only Lightning checkpoint's head config is
            # validated on these too (the config keys match the _validate_volumetric_config checks above).
            "volumetric_fourier_frequencies": "volumetric_fourier_frequencies",
            "volumetric_use_softplus": "volumetric_use_softplus",
            # Per-query kNN context geometry so a pretrain-only Lightning checkpoint is validated on these too.
            "volumetric_per_query_context": "volumetric_per_query_context",
            "volumetric_context_k": "volumetric_context_k",
            "volumetric_context_heads": "volumetric_context_heads",
        }
        config = {cfg_key: hparams[hp_key] for hp_key, cfg_key in hparam_to_cfg.items() if hp_key in hparams}
        return config or None

    def _load_pretrained_volumetric_head(self, path: str) -> set[str]:
        """Load a separately-pretrained ``VolumetricOccupancyHead`` state_dict into ``self.volumetric_head``.

        Accepts a raw head ``state_dict`` OR a checkpoint wrapper: ``{"head_state_dict": ...}`` (exactly what
        ``scripts/joint_diffusion/train_volumetric_head.py`` saves), ``{"state_dict": ...}`` (Lightning), or
        ``{"model": ...}`` (some savers) -- each with possibly-prefixed keys. Any common prefix is stripped
        (see ``_align_volumetric_state_dict``) and the load is ``strict=True`` -- a silent partial load would
        leave the teacher head partly random, which is worse than an error. When the checkpoint carries a
        sibling ``config`` block (standalone Phase-1 trainer) OR -- for a ``--volumetric-pretrain-only``
        Lightning checkpoint -- a ``hyper_parameters`` block, its geometry (+ the per-element-sigma knobs) is
        validated against the constructed head BEFORE the load (see ``_validate_volumetric_config`` /
        ``_synthesize_volumetric_config_from_hparams``); only a checkpoint with NEITHER skips that check with a
        warning.

        Returns the set of OPTIONAL head param names that were MISSING from the checkpoint (kept at their
        zero-init values) -- the fresh grafts the caller must leave trainable even under ``freeze_volumetric_head``
        (an optional module that WAS loaded is a trained teacher tensor and freezes with the rest).
        """
        ckpt = torch.load(path, map_location="cpu")
        config = ckpt.get("config") if isinstance(ckpt, dict) else None
        # A --volumetric-pretrain-only checkpoint has no `config` block; its geometry lives in Lightning's
        # `hyper_parameters`. Synthesize a config from those so the geometry + per-element-sigma guard runs.
        if config is None and isinstance(ckpt, dict) and isinstance(ckpt.get("hyper_parameters"), dict):
            config = self._synthesize_volumetric_config_from_hparams(ckpt["hyper_parameters"])
        if isinstance(ckpt, dict) and isinstance(ckpt.get("head_state_dict"), dict):
            sd = ckpt["head_state_dict"]  # standalone volumetric-head trainer checkpoint
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
            sd = ckpt["state_dict"]  # Lightning / standalone-trainer checkpoint
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
            sd = ckpt["model"]  # some savers nest under "model"
        else:
            sd = ckpt  # assume a raw state_dict
        if not isinstance(sd, dict):
            raise ValueError(
                f"pretrained volumetric-head file {path!r} did not contain a state_dict (got {type(sd)!r}); "
                f"expected a raw head state_dict or a checkpoint dict with a "
                f"'head_state_dict'/'state_dict'/'model' entry."
            )
        # Geometry guard (the strict load below cannot catch lattice/sigma/hidden-dim mismatches).
        if isinstance(config, dict):
            self._validate_volumetric_config(config, path)
        else:
            os.environ.get("ATOMWEAVER_VERBOSE") and print(
                f"[volumetric-head] WARNING: pretrained checkpoint {path!r} has neither a 'config' block nor a "
                f"'hyper_parameters' block; skipping the geometry-compatibility check. A lattice/sigma/hidden_dim "
                f"mismatch will NOT be caught -- ensure this head was trained with matching volumetric_* settings."
            )
        target_keys = set(self.volumetric_head.state_dict().keys())
        aligned = self._align_volumetric_state_dict(sd, target_keys, path)
        # The aligner guarantees: aligned keys are a subset of the head's keys (no unexpected) AND cover every
        # REQUIRED (non-optional) key. So load with strict=False and then re-assert strictness EXCEPT for the
        # OPTIONAL fresh input grafts (available_volume_proj.* / field_proj.*), which an older pretrained head
        # may legitimately lack (they keep their zero-init values). Any OTHER missing key -- or any unexpected key
        # -- still raises, so a genuine partial load (which would read as random teacher guidance) is caught loudly.
        result = self.volumetric_head.load_state_dict(aligned, strict=False)
        non_optional_missing = [k for k in result.missing_keys if not _is_optional_volumetric_head_key(k)]
        if non_optional_missing or result.unexpected_keys:
            raise ValueError(
                f"pretrained volumetric-head state_dict from {path!r} did not strict-load into "
                f"self.volumetric_head: missing (non-optional) keys={non_optional_missing!r}, unexpected "
                f"keys={list(result.unexpected_keys)!r}. Only optional fresh input grafts "
                f"('available_volume_proj.*' / 'field_proj.*') may be absent (kept zero-init); everything else "
                f"must match exactly."
            )
        kept_zero_init = [k for k in result.missing_keys if _is_optional_volumetric_head_key(k)]
        os.environ.get("ATOMWEAVER_VERBOSE") and print(
            f"[volumetric-head] loaded {len(aligned)} pretrained tensors into self.volumetric_head from {path!r}"
            + (f" (kept {len(kept_zero_init)} zero-init optional tensors: {kept_zero_init})" if kept_zero_init else "")
        )
        # The optional grafts absent from the checkpoint (kept zero-init): the caller exempts EXACTLY these
        # from freeze so they stay trainable, while optional grafts that WERE loaded freeze like any other
        # teacher tensor. state_dict keys of these Linear grafts == their named_parameters names.
        return set(kept_zero_init)

    def _element_slot_powers(
        self,
        sidechain_mask: torch.Tensor,
    ) -> torch.Tensor:
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

        # Compute per-slot real power (proximal->distal interpolation)
        slot_idx = torch.arange(max_sc, device=device, dtype=torch.float32)
        depth = slot_idx / (max_sc - 1) if max_sc > 1 else torch.zeros_like(slot_idx)
        real_power = self.flow_real_proximal_power + depth * (
            self.flow_real_distal_power - self.flow_real_proximal_power
        )  # (max_sc,)
        real_power = real_power.view(1, 1, max_sc).expand(batch_size, seq_len, max_sc)

        # Ghost atoms: power=1.0 (default schedule, no delay)
        ghost_power = torch.ones(batch_size, seq_len, max_sc, device=device)

        # Blend based on GT mask
        mask_f = sidechain_mask.float()
        return mask_f * real_power + (1.0 - mask_f) * ghost_power

    def _existence_slot_powers(self, max_sc: int, device: torch.device) -> torch.Tensor:
        """Per-slot existence powers: same schedule as coords (lockstep).

        Uses the same proximal->distal interpolation as the coordinate flow so
        existence resolves in sync with coordinates for proper EVC coupling.
        """
        slot_idx = torch.arange(max_sc, device=device, dtype=torch.float32)
        depth = slot_idx / (max_sc - 1) if max_sc > 1 else torch.zeros_like(slot_idx)
        return self.flow_real_proximal_power + depth * (self.flow_real_distal_power - self.flow_real_proximal_power)

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
            # Per-slot effective source variance (Gaussian approximation of shell + thickness).
            # Shape (1, 1, max_sc, 1) to broadcast with the unsqueeze(-1) pattern used below.
            sigma_sq = self.coord_flow._shell_source_var[:max_sc].view(1, 1, max_sc, 1)
        else:
            noise_scale = self.coord_flow.noise_scale  # σ (4.0 Å)
            sigma_sq = noise_scale**2

        # Normalized time s ∈ [0,1]
        tau = self.coord_flow._tau(t)  # (B,)
        s = tau.view(batch_size, 1, 1)  # (B, 1, 1)

        # Per-residue centroid: learned (Stage 2) or zero-centered (Stage 1).
        # Zero-centered = purely distance-based ghost/real discrimination at high noise.
        if learned_centroid is not None:
            mu_res = learned_centroid  # (B, L, 3)
        else:
            mu_res = torch.zeros(batch_size, seq_len, 3, device=device)

        if learned_cloud_logvar is not None:
            var_res = learned_cloud_logvar.exp()  # (B, L)
        else:
            var_res = self._slot_coord_var.mean().expand(batch_size, seq_len)

        # Broadcast per-residue -> per-slot (shared centroid for all slots in a residue)
        mu_slot = mu_res.unsqueeze(2).expand(-1, -1, max_sc, -1)  # (B, L, max_sc, 3)
        var_slot = var_res.unsqueeze(2).expand(-1, -1, max_sc)  # (B, L, max_sc)

        # Prior: slot fill rates π_real per slot (stays per-slot)
        pi_real = self._slot_fill_rate.view(1, 1, max_sc).expand(batch_size, seq_len, -1).clamp(0.01, 0.99)

        ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, max_sc, -1)  # (B, L, max_sc, 3)
        x_rel = noised_coords - ca_expanded  # (B, L, max_sc, 3) offset from CA

        # Ghost: x_t ~ N(0, ghost_var I) in CA-relative space
        # sigma_sq is scalar (legacy) or (1, 1, max_sc, 1) (donut: per-slot source variance)
        ghost_var = s.unsqueeze(-1) ** 2 * sigma_sq + self.ghost_var_floor
        ghost_var = ghost_var.squeeze(-1)  # (B, 1, 1) or (B, 1, max_sc)
        ghost_mahal = (x_rel**2).sum(dim=-1) / ghost_var  # (B, L, max_sc)
        ghost_log_norm = 3.0 * torch.log(ghost_var)
        log_p_ghost = -0.5 * (ghost_mahal + ghost_log_norm)

        # Real: x_t ~ N((1-s)·μ_res, real_var I) -- shared centroid per residue
        one_minus_s = 1.0 - s  # (B, 1, 1)
        real_var = one_minus_s.unsqueeze(-1) ** 2 * var_slot.unsqueeze(-1) + s.unsqueeze(-1) ** 2 * sigma_sq
        if self.mixture_real_var_floor > 0:
            real_var = real_var + s.unsqueeze(-1) ** 2 * self.mixture_real_var_floor
        real_var = real_var.squeeze(-1).clamp(min=1e-4)  # (B, L, max_sc)
        real_mean = one_minus_s.unsqueeze(-1) * mu_slot  # (B, L, max_sc, 3)
        diff = x_rel - real_mean  # (B, L, max_sc, 3)
        real_mahal = (diff**2).sum(dim=-1) / real_var  # (B, L, max_sc)
        real_log_norm = 3.0 * torch.log(real_var)  # (B, L, max_sc)
        log_p_real = -0.5 * (real_mahal + real_log_norm)

        # Bayes: P(real|x) = σ(log(P(x|real)/P(x|ghost)) + log(π_real/π_ghost) - threshold)
        # threshold > 0 requires stronger coordinate evidence before granting "real" status
        log_prior_ratio = torch.log(pi_real / (1.0 - pi_real))
        log_likelihood_ratio = log_p_real - log_p_ghost
        # Per-residue LRT: global threshold + backbone-predicted delta per residue
        # Positive delta -> stricter (fewer atoms), negative -> permissive (more atoms)
        threshold = self.mixture_lr_threshold
        if residue_lrt_delta is not None and self.dynamic_lrt:
            delta = residue_lrt_delta
            if self.dynamic_lrt_clamp > 0:
                delta = delta.clamp(-self.dynamic_lrt_clamp, self.dynamic_lrt_clamp)
            # Scale delta at sampling time to dampen late-training overshoot
            if not self.training and self.dlrt_sample_scale != 1.0:
                delta = delta * self.dlrt_sample_scale
            # delta is (B, L), broadcast to (B, L, max_sc)
            threshold = threshold + delta.unsqueeze(-1)
        elif self.dlrt_analytical_scale > 0 and residue_count_pred is not None:
            # Analytical threshold: use count head prediction directly.
            # Higher predicted count -> lower threshold (more permissive).
            # delta = -scale * (count_pred - mean_count) / std_count
            # mean ~4.0, std ~2.3 from dataset statistics
            delta = -self.dlrt_analytical_scale * (residue_count_pred - 4.0) / 2.3
            delta = delta.clamp(-0.5, 0.5)  # Same clamp range as learned dlrt
            threshold = threshold + delta.unsqueeze(-1)
        logit = log_likelihood_ratio + log_prior_ratio - threshold
        if temperature != 1.0:
            logit = logit / temperature
        return torch.sigmoid(logit)

    def _canonicalize_cluster_ids(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        """Remap each residue's cluster labels to the minimum member slot index."""
        batch_size, seq_len, max_sc = cluster_ids.shape
        canonical = cluster_ids.clone()
        for b in range(batch_size):
            for seq_idx in range(seq_len):
                residue_ids = cluster_ids[b, seq_idx]
                remapped = residue_ids.clone()
                for cluster_idx in torch.unique(residue_ids, sorted=False).tolist():
                    members = residue_ids == cluster_idx
                    min_slot = torch.where(members)[0].min()
                    remapped[members] = min_slot
                canonical[b, seq_idx] = remapped
        return canonical

    def _compute_cluster_occupancy_probs(self, cluster_logits: torch.Tensor) -> torch.Tensor:
        """Compute differentiable expected occupancy of each cluster label within a residue."""
        cluster_probs = torch.softmax(cluster_logits, dim=-1)  # (B, L, particle, cluster)
        return 1.0 - torch.prod(1.0 - cluster_probs, dim=2)  # (B, L, cluster)

    def _merge_cluster_predictions(
        self,
        coords: torch.Tensor,
        element_types: torch.Tensor,
        predicted_mask: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge predicted particles by cluster id for diagnostics."""
        batch_size, seq_len, max_sc, _ = coords.shape
        cluster_ids = self._canonicalize_cluster_ids(cluster_ids)
        merged_coords = coords.clone()
        merged_elements = element_types.clone()
        merged_mask = predicted_mask.clone()
        for b in range(batch_size):
            for seq_idx in range(seq_len):
                for cluster_idx in range(max_sc):
                    members = cluster_ids[b, seq_idx] == cluster_idx
                    active_members = members & predicted_mask[b, seq_idx]
                    if not active_members.any():
                        merged_mask[b, seq_idx, members] = False
                        continue
                    center = coords[b, seq_idx, active_members].mean(dim=0)
                    merged_coords[b, seq_idx, active_members] = center
                    element_votes = element_types[b, seq_idx, active_members]
                    if len(element_votes) > 0:
                        merged_element = torch.mode(element_votes).values
                        merged_elements[b, seq_idx, active_members] = merged_element
                    first_idx = torch.where(active_members)[0].min()
                    keep_mask = torch.zeros_like(active_members)
                    keep_mask[first_idx] = True
                    merged_mask[b, seq_idx, members] = keep_mask[members]
        return merged_coords, merged_elements, merged_mask

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
        if self.use_polarity_head:
            polarity_logits = self.polarity_head(backbone_features)  # (B, L, K-1) over real classes
            log_pol = F.log_softmax(polarity_logits, dim=-1)
            pad_logit = element_logits[..., :1]  # untouched -> P(PAD) invariant
            real = element_logits[..., 1:]
            lse_before = torch.logsumexp(real, dim=-1, keepdim=True)
            real_biased = real + log_pol.unsqueeze(2)  # per-residue bias broadcast over slots
            lse_after = torch.logsumexp(real_biased, dim=-1, keepdim=True)
            real_renorm = real_biased - (lse_after - lse_before)  # restore real block lse -> P(PAD) exact
            element_logits = torch.cat([pad_logit, real_renorm], dim=-1)
            aux["polarity_logits"] = polarity_logits
            aux["log_pol"] = log_pol
        if self.use_glycine_head:
            dih = self.denoiser.backbone_encoder._backbone_dihedral_features(backbone_coords, backbone_mask)  # (B,L,8)
            glycine_logit = self.glycine_head(dih)  # (B, L, 1)
            gate = torch.tanh(F.softplus(self.glycine_pad_gate))  # nonnegative (softplus) & bounded (tanh) in [0,1)
            if self.training:
                # PROBE (train-only, detached, one logsumexp): P(PAD) the element head assigns BEFORE the
                # glycine push. Paired with the GT-glycine mask in forward() this is the "is there any
                # incentive left for the gate to open?" number -- if it is already ~1.0 at GT-glycine
                # positions the expert is redundant and a shut gate is CORRECT, not starved.
                with torch.no_grad():
                    aux["pad_prob_pre_glycine"] = torch.softmax(element_logits.detach(), dim=-1)[..., 0]  # (B,L,K)
            pad_push = self.glycine_pad_cap * gate * torch.sigmoid(glycine_logit)  # (B,L,1) in [0, cap], ADD-only
            pad_logit_g = element_logits[..., :1] + pad_push.unsqueeze(2)  # broadcast over slots
            element_logits = torch.cat([pad_logit_g, element_logits[..., 1:]], dim=-1)  # no in-place
            aux["glycine_logit"] = glycine_logit
        return element_logits, aux

    # ------------------------------------------------------------------ neighbour-x0 packing

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
        if self.coord_process_type == "flow_matching":
            x0 = self.coord_flow.predict_x0_from_velocity(x_flat, t, v_flat, ca_coords=ca_coords, mask_probs=mp)
        elif self.prediction_type == "v":
            x0 = self.diffusion.predict_x0_from_v(x_flat, t, v_flat, ca_coords=ca_coords)
        elif self.prediction_type == "epsilon":
            x0 = self.diffusion.predict_x0_from_noise(x_flat, t, v_flat, ca_coords=ca_coords)
        else:
            x0 = v_flat
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
        prog = 1.0 if (epoch is None or ramp <= 0) else min(float(epoch) / ramp, 1.0)
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
        if self.neighbor_x0_packing_random_recycles:
            return max(int(self.neighbor_x0_packing_max_recycles), int(self.neighbor_x0_packing_recycles))
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
        if self.split_element_existence:
            # 3-track: existence lives on its own head.
            return torch.sigmoid(denoiser_out["occupancy_logits"]).detach().clamp(0.0, 1.0)
        probs = torch.softmax(denoiser_out["element_logits"], dim=-1)
        p_real = 1.0 - probs[..., ELEMENT_PAD]
        if self.num_element_classes > NUM_ELEMENT_TYPES:
            from .diffusion import ELEMENT_MASK

            # An unresolved MASK token is not an atom either.
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
        trust = (1.0 - t_original_res.float() / max(self.timesteps, 1)).clamp(0.0, 1.0)  # (B, L)
        if keep_mask is not None:
            keep = keep_mask.to(device=coords.device, dtype=torch.bool)
            if clean_context_coords is not None:
                coords = torch.where(keep[:, :, None, None], clean_context_coords.detach(), coords)
            if clean_context_mask is not None:
                mask = torch.where(keep[:, :, None], clean_context_mask.detach().bool(), mask)
            trust = torch.where(keep, torch.ones_like(trust), trust)
        return coords, mask, trust

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
            return x, element_types, noised_mask, evc_sampling
        if self.split_element_existence:
            # Mirror forward()'s guard: sample() must also fail loud for 3-track. The per-step existence
            # state is NOT pinned here, so context residues would get free-evolving (MASK/noisy) existence
            # contradicting their pinned chemistry -- silent train/sample divergence. Use 2-track.
            raise NotImplementedError(
                "K-mask (design_mask) sampling under split_element_existence (3-track) is not supported: "
                "the per-step existence state is not pinned. Use 2-track, or implement existence-state pinning."
            )
        batch_size, seq_len = design_mask.shape[0], design_mask.shape[1]
        keep = (~design_mask.bool()).view(batch_size, seq_len, 1)  # (B,L,1) True = pinned to GT
        if inpaint_mode == "clean":
            pin_coords, pin_elems = inpaint_gt_coords, inpaint_gt_elements
        elif inpaint_mode == "noised":
            # The in-loop pin runs AFTER the reverse update (x is now x_{t-1}) but re-noises GT to the
            # OUTGOING t -> off-by-one for noised pinning. clean inpaint is t-independent and correct.
            # Fail loud until the step index is fixed rather than leave a silently-wrong mode callable.
            raise NotImplementedError(
                "inpaint_mode='noised' is not supported (post-update pin re-noises GT to the wrong step). "
                "Use inpaint_mode='clean' (the validated path); fix the in-loop step index to re-enable."
            )
        else:
            raise ValueError(f"inpaint_mode must be 'clean' or 'noised', got {inpaint_mode!r}")
        x = torch.where(keep.unsqueeze(-1), pin_coords, x)
        if pin_elems is not None:
            element_types = torch.where(keep, pin_elems, element_types)
            noised_mask = (element_types != ELEMENT_PAD).float()
            if evc_sampling is not None and inpaint_gt_mask is not None:
                evc_sampling = torch.where(keep, inpaint_gt_mask.float(), evc_sampling)
        return x, element_types, noised_mask, evc_sampling

    def _stereochem_gated_pd(
        self,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
        target_backbone_coords: torch.Tensor | None = None,
        target_backbone_mask: torch.Tensor | None = None,
        target_residue_types: torch.Tensor | None = None,
        target_seq_mask: torch.Tensor | None = None,
        target_coords: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_atom_element_type: torch.Tensor | None = None,
        target_atom_is_backbone: torch.Tensor | None = None,
        gt_sidechain_coords: torch.Tensor | None = None,
        gt_sidechain_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """-1: P(D) prior for the sigmoid-gated cone init, computed BEFORE ``x_T`` is sampled.

        Runs the v2 residue-frame stream + stereochem head (and the volumetric head first when it is on, so
        ``vol_hidden`` is available as the stereo head's auxiliary input) on the STATIC backbone + target, and
        returns ``sigmoid(rfv2_stereo_logit)`` = P(D) (shape ``(B, L)``, label 1 = D). Returns ``None`` when the
        gated-init flag is off (byte-identical no-op) or the head emits no logit.

        Detached (``no_grad``): this is a source-distribution steering only. The stereo head is trained solely
        by its own BCE inside the main forward pass (unchanged); the init-gating introduces NO new gradient
        coupling to the coord flow (that feedback is a later chunk). The v2 stream is cheap + residue-level, so
        this recompute is the deliberate "denoiser may recompute frame-v2 as it does today" simplification.
        """
        if not self.use_stereochem_gated_init:
            return None
        with torch.no_grad():
            vol_hidden = None
            if self.use_volumetric_head:
                vol_out = self.volumetric_head(
                    backbone_coords=backbone_coords,
                    backbone_mask=backbone_mask,
                    seq_mask=seq_mask,
                    target_coords=target_coords,
                    target_mask=target_mask,
                    target_element=target_atom_element_type,
                    target_is_backbone=target_atom_is_backbone,
                    gt_sidechain_coords=gt_sidechain_coords,
                    gt_sidechain_mask=gt_sidechain_mask,
                    available_volume=self._compute_available_volume(
                        backbone_coords, backbone_mask, target_coords, target_mask
                    ),
                )
                vol_hidden = vol_out["vol_hidden"]
            rfs = self.denoiser.residue_frame_stream_v2
            b, length = backbone_coords.shape[:2]
            # backbone_features is IGNORED by the v2 stream under frame_v2_clean_input=True (the coherence guard
            # requires clean_input), so a zeros placeholder is faithful; it never touches the P(D) output.
            bb_feats = backbone_coords.new_zeros(b, length, rfs.hidden_dim)
            v2_out = rfs(
                bb_feats,
                backbone_coords,
                backbone_mask,
                seq_mask,
                target_backbone_coords=target_backbone_coords,
                target_backbone_mask=target_backbone_mask,
                target_residue_types=target_residue_types,
                target_seq_mask=target_seq_mask,
                vol_hidden=(vol_hidden.detach() if vol_hidden is not None else None),
            )
            stereo_logit = v2_out["rfv2_stereo_logit"]  # (B, L) or None
            if stereo_logit is None:
                return None
            return torch.sigmoid(stereo_logit)

    def _stereochem_gate_cone_direction(
        self,
        direction: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor | None,
        pd: torch.Tensor,
        ramp_s: float = 1.0,
    ) -> torch.Tensor:
        """-1: steer the analytic cone direction's out-of-plane (e3) sign by P(D).

        In the residue's local backbone frame ``R`` (columns e1,e2,e3 from :func:`build_local_frames`), write
        ``d_local = Rᵀ·d = (a, b, c)`` (``c`` = e3 = plane-normal component). Canonical L has ``c < 0``. The
        gate keeps the in-plane part ``(a, b)`` and replaces the out-of-plane part with the head-first-RAMPED
        blend ``c_effective = (1-s)·c + s·c_gated`` where ``c_gated = (2·P(D) - 1)·|c|`` and ``s`` is the
        gated-init ramp weight (:meth:`_gated_init_ramp`), then maps back to global ``d = R·(a, b, c_effective)``:

        * s=0 (ramp not started / fresh graft): c_effective = c -> d = d EXACTLY (byte-identical L cone; an
                                                   UNTRAINED head with P(D)≈0.5 does NOT flatten the cone).
        * s=1 (inference / ramp complete), P(D)=0 (L): c_effective = -|c| = c (c<0) -> d EXACTLY (sign-error catcher).
        * s=1, P(D)=1 (D): c_effective = +|c| = -c -> mirror of d across the backbone plane.
        * s=1, P(D)=0.5: c_effective = 0 -> d lies IN the backbone plane (non-renormalized => a
                                                            shorter, near-plane init -- intended near-plane flattening).

        Parameters
        ----------
        direction : (B, L, 3) analytic pseudo-Cβ cone directions ``d``.
        pd : (B, L) P(D) = sigmoid(stereo logit), detached (source-dist steering only).
        ramp_s : head-first ramp weight ``s`` ∈ [0, 1]; 0 = ungated L cone, 1 = full P(D)-gating. Defaults
            to 1.0 (inference / full gating), matching the ramp convention (None training_progress -> full).
        """
        R, _ = build_local_frames(backbone_coords, backbone_mask)  # (B, L, 3, 3); columns = e1,e2,e3
        # d_local = Rᵀ·d (R columns are the basis vectors, so Rᵀ maps global -> local).
        d_local = torch.einsum("blij,blj->bli", R.transpose(-1, -2), direction)  # (B, L, 3) = (a, b, c)
        c = d_local[..., 2]
        c_gated = (2.0 * pd.to(d_local.dtype) - 1.0) * c.abs()  # (B, L)
        # Head-first ramp: blend the ungated L value `c` -> the P(D)-gated value over training. At s=0 this is
        # EXACTLY `c` (byte-identical L cone) regardless of P(D); at s=1 it is fully P(D)-gated.
        s = float(ramp_s)
        c_effective = (1.0 - s) * c + s * c_gated  # (B, L)
        d_gated_local = torch.stack([d_local[..., 0], d_local[..., 1], c_effective], dim=-1)  # (B, L, 3)
        # d_gated = R·(a, b, c_effective) back to the global frame.
        return torch.einsum("blij,blj->bli", R, d_gated_local)

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
        use_ddim: bool = False,  # Use DDPM (stochastic) by default - performs better than DDIM
        ddim_eta: float = 1.0,  # Full stochasticity (DDPM equivalent)
        return_intermediates: bool = False,  # Return atom counts at each step
        return_coord_trajectory: bool = False,  # Capture full per-step coords/elements/mask (for viz)
        element_sampling_temp_max: float = 25.0,  # Max temperature for element logits at t=T
        element_sampling_temp_power: float = 5.0,  # Power for temp schedule (higher = drops faster)
        disc_count_guidance: object | None = None,  # DiscretizationLoss for count guidance
        disc_count_guidance_k: int = 2,  # Run disc guidance every k steps
        max_count_delta: int = 0,  # Max atom count change per reverse step (0=no clamp, 1=±1 per step)
        oracle_sidechain_mask: torch.Tensor | None = None,  # GT mask for oracle ablation
        count_head_element_bias: float = 0.0,  # Strength of count-head->element bias (0=disabled)
        non_pad_logit_bias: float = 0.0,  # Non-PAD boost (0=disabled). Counteracts PAD-heavy prior.
        non_pad_bias_nres_scale: float = 0.0,  # Scale bias by (1 + scale*(n_res - 11)); 0=off, 0.05=mild
        non_pad_bias_time_decay: str = "none",  # Time schedule: "none", "linear" (bias->0), "cosine"
        element_sampling_mode: str = "posterior",  # "posterior", "reflow", "reflow_cond", "greedy" (argmax x_0)
        reflow_start_frac: float = 1.0,  # Fraction of timesteps to use reflow (1.0=all, 0.5=top half only)
        element_stride: int = 1,  # Update elements every N coord steps (1=every step, 10=every 10th step)
        element_early_stop_frac: float = 1.0,  # Stop element updates after this fraction of steps (1.0=never stop, 0.8=stop at 80%)
        posterior_pad_squash: float = 1.0,  # Squash posterior PAD probability (<1 reduces PAD bias)
        squash_schedule: str = "constant",  # "constant" or "linear" (time-dependent: aggressive at high noise)
        evc_mask_init_value: float = 0.0,  # EVC conditioning for MASK/PAD tokens at init (0.0=ghost, 0.5=neutral, 1.0=real)
        evc_dist_override_frac: float = 0.0,  # Use dist-from-CA as EVC for first N% of steps (0=disabled, 0.5=first half)
        element_dist_logit_bias: float = 0.0,  # Bias element logits by distance from CA (>0: far->non-PAD, near->PAD)
        mask_persistence_bias: float = 0.0,  # Suppress PAD logit when element is MASK (prevent premature PAD collapse)
        reserved_slot0_prefix_exempt: bool = False,  # exempt reserved-slot0 (N-connecting) from the hard prefix constraint
        count_ghost_velocity: float = 0.0,  # Ghost velocity override strength (0=off, >0: blend by sigmoid(k*(count-slot-0.5)), 999=hard)
        gt_sidechain_coords: torch.Tensor | None = None,  # GT sidechain coords for leak positive controls
        gt_sidechain_mask: torch.Tensor | None = None,  # GT sidechain mask for leak positive controls
        freeze_count_at_frac: float = -1.0,  # Freeze model's own count prediction at this trajectory fraction (-1=off, 0.05=5%, 0.25=25%)
        logit_anchor_frac: float = -1.0,  # Snapshot logit ranking at this trajectory fraction (-1=off, 0.05=5%)
        logit_anchor_alpha: float = 0.5,  # Anchor bias strength (0=off, 0.5=sweet spot, >1=overconstrained)
        # --- Replacement-inpainting (K-mask done on-manifold, NOT fold-to-target) ---
        design_mask: torch.Tensor | None = None,  # (B, L) bool: True = positions to DESIGN (free); False = pinned to GT
        inpaint_gt_coords: torch.Tensor | None = None,  # (B, L, max_sc, 3) GT sidechain coords for pinned positions
        inpaint_gt_elements: torch.Tensor | None = None,  # (B, L, max_sc) GT element types (PAD=0,C=1,N=2,O=3,S=4)
        inpaint_gt_mask: torch.Tensor | None = None,  # (B, L, max_sc) GT atom-existence mask for pinned positions
        inpaint_mode: str = "clean",  # "clean" = pin GT x0 every step; "noised" = pin GT noised-to-t (RePaint-style)
        chirality: torch.Tensor | None = None,  # (B,L) L/D sign for the pseudo-CB cone (+1 L / -1 D); None=+1 (L)
        noised_count_override: torch.Tensor
        | None = None,  # (B,L) override self-fed noised_count at designed positions (count-conditioning ablation)
        neighbor_x0_packing_recycles: int
        | None = None,  # ARBITRARY N, independent of the training value (each pass is told its absolute index j); 1 = cheap mode (no extra pass, t_cond == t_orig), >1 = recycled (t_cond = 0)
        forced_atom_counts: torch.Tensor
        | None = None,  # (B,L) int: force EXACTLY K real atoms per residue (slots 0..K-1 REAL->shell, K..GHOST->CA; reserved-slot0 prefix layout). Reuses the leak_gt_count donut-source + per-step oracle path with a K-derived mask instead of GT. NO GT leak. Used by the count-search eval.
        **kwargs,  # noqa: ARG002 -- Accept legacy mask-related kwargs for backward compat
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
        # Alias so the per-step recycle loop below is not shadowed by the kwarg name.
        # SAMPLING RECYCLE DEFAULT: an EXPLICIT `neighbor_x0_packing_recycles` arg always wins. When the arg
        # is None (the common eval path) we no longer inherit the checkpoint's TRAINING value
        # (`self.neighbor_x0_packing_recycles`, typically 2 => a wasteful 2-pass recycle at inference); we fall
        # back to the `ATOMWEAVER_SAMPLE_RECYCLES` env var (default "1") so sampling is a genuine SINGLE forward
        # pass unless the caller/driver overrides it. With the single-site inference context (below) the
        # prev-step predicted-x0 now supplies what the recycle used to provide. `_n_rc = max(1, _sample_recycles)`
        # downstream, and `_n_rc == 1` => the coord-flow neighbour-x0 recycle loop never fires (single pass).
        _sample_recycles = neighbor_x0_packing_recycles
        if _sample_recycles is None:
            _sample_recycles = max(1, int(os.environ.get("ATOMWEAVER_SAMPLE_RECYCLES", "1")))
        _sample_absorbing = self.absorbing_mask

        if self.coord_process_type == "flow_matching" and self.use_cluster_particle_diffusion:
            raise NotImplementedError("flow_matching coordinates are not yet supported with cluster particle diffusion")

        # Start from CA-relative noise
        # At full noise (t=T), each sidechain forms an isotropic Gaussian cloud
        # around its CA position, rather than being scattered globally
        ca_coords = backbone_coords[:, :, 1, :]  # (B, L, 3) - CA is index 1
        ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, max_sc, -1)  # (B, L, max_sc, 3)

        # Positive control: compute per_atom_mask and direction_override for leak flags
        leak_per_atom_mask = None
        leak_direction = None
        leak_mask_for_elements = None
        if self.leak_gt_count and gt_sidechain_mask is not None:
            leak_per_atom_mask = gt_sidechain_mask.reshape(batch_size, -1, 1).float()
            leak_mask_for_elements = gt_sidechain_mask  # Also used for element init
        # Count-search (research eval): force an ARBITRARY per-residue atom count K by building the
        # SAME real/ghost per-atom mask leak_gt_count uses, but derived from K instead of GT. Slots
        # 0..K-1 = REAL (donut shell), slots K.. = GHOST (collapsed to CA) -- the reserved-slot0 prefix
        # layout. This drives BOTH the donut source (leak_per_atom_mask -> sample_prior) AND the
        # per-step oracle mask (leak_mask_for_elements -> oracle_sidechain_mask below), so the count is
        # pinned at EVERY reverse step exactly like the GT-count leak. Independent of self.leak_gt_count
        # (production checkpoints have it False) and uses NO ground truth. When forcing is active the
        # donut source depends only on the shared t=T RNG draw and the deterministic K-gating, so seeding
        # identically across the K passes makes the shell displacement byte-identical (count = only var).
        if forced_atom_counts is not None:
            _slot_idx = torch.arange(max_sc, device=device).view(1, 1, max_sc)  # (1,1,max_sc)
            _forced_k = forced_atom_counts.to(device).long().clamp(min=0, max=max_sc)  # (B,L)
            forced_mask = (_slot_idx < _forced_k.unsqueeze(-1)).float()  # (B,L,max_sc): 1 = real, 0 = ghost
            if seq_mask is not None:
                # padded residues stay all-ghost (no spurious forced atoms outside the peptide)
                forced_mask = forced_mask * seq_mask.to(device).view(batch_size, seq_len, 1).float()
            leak_per_atom_mask = forced_mask.reshape(batch_size, -1, 1)  # donut source: real->shell, ghost->CA
            leak_mask_for_elements = forced_mask  # element init + per-step ghost->PAD oracle pin
        if self.pseudo_cb_direction:
            # Chirality-aware cone (+1 L / -1 D): for single-site NCAA design the target
            # residue identity is known, so D-amino acids get the correct hemisphere.
            # Unknown/de-novo (chirality=None) defaults to +1 (L), the common case.
            # SAMPLING respects the caller-provided chirality: de-novo passes chirality=None -> +1/L (the
            # "surprise-D" flip mode); a known-D hint passes -1 -> D-init (conditioned mode). The training-only
            # d_aa_l_source_prob prior does NOT force anything here -- the init cone is the caller's mode switch.
            _src_chir = chirality
            leak_direction = compute_pseudo_cb_direction(backbone_coords, chirality=_src_chir)
            # -1: sigmoid-gated cone init (SAMPLING side -- mirrors the training q_sample prior above so
            # train/inference match). Steer the cone's e3 sign by the stereochem head's P(D). No-op /
            # byte-identical when the flag is off. gt_sidechain_* are None for de-novo sampling; the volumetric
            # head handles that (its GT branch is optional).
            if self.use_stereochem_gated_init:
                pd = self._stereochem_gated_pd(
                    backbone_coords,
                    backbone_mask,
                    seq_mask=seq_mask,
                    target_backbone_coords=target_backbone_coords,
                    target_backbone_mask=target_backbone_mask,
                    target_residue_types=target_residue_types,
                    target_seq_mask=target_seq_mask,
                    target_coords=target_coords,
                    target_mask=target_mask,
                    target_atom_element_type=target_atom_element_type,
                    target_atom_is_backbone=target_atom_is_backbone,
                    gt_sidechain_coords=gt_sidechain_coords,
                    gt_sidechain_mask=gt_sidechain_mask,
                )
                if pd is not None:
                    # Inference = FULL gating (ramp_s=1.0, the _gated_init_ramp None->1.0 convention). The
                    # L-prior stereo-head bias keeps an untrained head (fresh graft) on the L cone here.
                    leak_direction = self._stereochem_gate_cone_direction(
                        leak_direction, backbone_coords, backbone_mask, pd, ramp_s=1.0
                    )
        elif self.leak_gt_direction and gt_sidechain_coords is not None and gt_sidechain_mask is not None:
            real_mask_f = gt_sidechain_mask.float()  # (B, L, max_sc)
            weighted = gt_sidechain_coords * real_mask_f.unsqueeze(-1)  # (B, L, max_sc, 3)
            n_real = real_mask_f.sum(dim=2, keepdim=True).clamp(min=1)  # (B, L, 1)
            centroid = weighted.sum(dim=2) / n_real  # (B, L, 3)
            gt_dir = torch.nn.functional.normalize(centroid - ca_coords, dim=-1)  # (B, L, 3)
            no_real = real_mask_f.sum(dim=2) == 0  # (B, L)
            rand_dir = torch.nn.functional.normalize(torch.randn_like(gt_dir), dim=-1)
            leak_direction = torch.where(no_real.unsqueeze(-1), rand_dir, gt_dir)

        if self.coord_process_type == "flow_matching":
            x, _ = self.coord_flow.sample_prior(
                (batch_size, seq_len * max_sc, 3),
                ca_coords=ca_coords,
                per_atom_mask=leak_per_atom_mask,
                direction_override=leak_direction,
            )
            x = x.view(batch_size, seq_len, max_sc, 3)
        else:
            # Use the same noise_scale as training (default 4.0Å)
            # CRITICAL: This must match the diffusion's noise_scale or sampling will fail!
            local_noise = torch.randn(batch_size, seq_len, max_sc, 3, device=device) * self.diffusion.noise_scale
            x = ca_expanded + local_noise

        # Reverse diffusion - start from t=T-1 (full noise)
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        # Start element types: all-PAD for "bare" mode, or sample from prior
        from .diffusion import ELEMENT_MASK, ELEMENT_PAD

        # Split existence mode: initialize existence at source=self.existence_threshold, elements at all-MASK.
        # Note: existence=source_value means all slots start at the source (default 0.5 = max entropy).
        # Slots with existence >= self.existence_threshold are classified as "real".
        # The element track starts all-MASK independently. During sampling, existence flows
        # toward 0 (ghost) or 1 (real) while elements resolve from MASK to {C,N,O,S}.
        if self.split_element_existence:
            from .diffusion import EXISTENCE_GHOST, EXISTENCE_MASK, EXISTENCE_REAL, SPLIT_ELEMENT_MASK

            if self.existence_absorbing:
                # Absorbing existence: start all slots as MASK (will resolve to GHOST or REAL)
                existence = torch.full((batch_size, seq_len, max_sc), EXISTENCE_MASK, device=device, dtype=torch.long)
                noised_mask = torch.zeros(batch_size, seq_len, max_sc, device=device)  # All MASK -> ghost initially
            else:
                existence = torch.full((batch_size, seq_len, max_sc), self.existence_flow.source_value, device=device)
                noised_mask = (existence >= self.existence_threshold).float()  # Classify from source value
            if self.split_element_flow:
                # Flow matching: initialize soft probs from prior at t=T
                _split_soft_probs = (
                    self.split_element_flow_proc.prior.view(1, 1, 1, -1)
                    .expand(batch_size, seq_len, max_sc, -1)
                    .clone()
                    .to(device)
                )
                element_types = _split_soft_probs.argmax(dim=-1)
                if self.split_element_flow_temp > 0:
                    log_probs = (_split_soft_probs + 1e-8).log()
                    soft_element_probs = torch.softmax(log_probs / self.split_element_flow_temp, dim=-1)
            else:
                _split_soft_probs = None
                element_types = torch.full(
                    (batch_size, seq_len, max_sc), SPLIT_ELEMENT_MASK, device=device, dtype=torch.long
                )

        # leak_gt_count also initializes elements from GT mask (oracle-style)
        # AND overrides oracle_sidechain_mask so ghost->PAD is enforced at every sampling step
        if not self.split_element_existence and leak_mask_for_elements is not None and oracle_sidechain_mask is None:
            oracle_sidechain_mask = leak_mask_for_elements
        effective_oracle_mask = oracle_sidechain_mask if oracle_sidechain_mask is not None else leak_mask_for_elements
        if self.split_element_existence:
            pass  # Already initialized above
        elif effective_oracle_mask is not None:
            # Oracle ablation: initialize from GT mask -- non-PAD slots get Carbon (1), PAD slots get PAD (0)
            gt_mask_bool = effective_oracle_mask.bool()
            element_types = torch.where(
                gt_mask_bool,
                torch.ones(batch_size, seq_len, max_sc, device=device, dtype=torch.long),  # C=1
                torch.zeros(batch_size, seq_len, max_sc, device=device, dtype=torch.long),  # PAD=0
            )
        elif self.all_carbon_sampling or self.late_element_resolution or self.non_pad_element_sampling:
            # All-Carbon init: denoiser sees all atoms as real Carbon.
            # Ghost/real determined purely from coordinates via mixture posterior.
            # Breaks the PAD->zero-velocity death spiral.
            # Late element resolution also needs all-Carbon init: PAD/non-PAD (ghost/real)
            # comes from mixture posterior at t=0, not from element diffusion.
            element_types = torch.ones(batch_size, seq_len, max_sc, device=device, dtype=torch.long)  # C=1
        elif self.disable_element_types:
            element_types = torch.full((batch_size, seq_len, max_sc), ELEMENT_PAD, device=device, dtype=torch.long)
        elif self.pad_sampling_init == "bare" and self.donut_element_init == "carbon":
            # All-Carbon init -- matches donut source where all atoms start at shell positions (look real).
            # Forward process drives toward all-Carbon at t=T. Ghost/real emerges via mask BCE head.
            element_types = torch.ones(batch_size, seq_len, max_sc, device=device, dtype=torch.long)  # C=1
        elif self.pad_sampling_init == "bare" and self.donut_element_init == "mask":
            # All-MASK init -- "unknown" occupancy at t=T. Forward drives toward MASK.
            # Model must resolve MASK->{PAD,C,N,O,S} during reverse. Ghost/real via mask BCE head.
            from .diffusion import ELEMENT_MASK

            element_types = torch.full((batch_size, seq_len, max_sc), ELEMENT_MASK, device=device, dtype=torch.long)
        elif self.pad_sampling_init == "bare":
            # All-PAD init (bare backbone) -- matches time-varying prior at t=T
            element_types = torch.full((batch_size, seq_len, max_sc), ELEMENT_PAD, device=device, dtype=torch.long)
        else:
            # Sample from data prior distribution (PAD~71%, C~22%, N~2.8%, O~3.9%, S~0.2%)
            prior = self.element_diffusion.prior
            element_types = torch.multinomial(
                prior.expand(batch_size * seq_len * max_sc, -1),
                num_samples=1,
            ).view(batch_size, seq_len, max_sc)

        # Element flow matching: initialize soft probs from prior at t=T
        soft_element_probs = None
        if self.element_flow_matching:
            soft_element_probs = (
                self.element_flow.prior.view(1, 1, 1, -1).expand(batch_size, seq_len, max_sc, -1).clone()
            )
            element_types = soft_element_probs.argmax(dim=-1)

        if not self.split_element_existence:
            # Mask derived from element types (PAD=0 -> absent, others -> present)
            noised_mask = (element_types != ELEMENT_PAD).float()

            # For MASK init: resolve MASK tokens using CA distance for initial mask/count features.
            if self.donut_element_init == "mask" and hasattr(self.coord_flow, "_shell_radii"):
                from .diffusion import ELEMENT_MASK

                is_mask = element_types == ELEMENT_MASK
                if is_mask.any():
                    ca_exp = ca_coords.unsqueeze(2).expand(-1, -1, max_sc, -1)
                    dist_to_ca = (x - ca_exp).norm(dim=-1)  # (B, L, max_sc)
                    shell_radii = self.coord_flow._shell_radii
                    # Sample site: epoch not available -> fully-ramped cap (see _distal_read_threshold).
                    threshold = self._distal_read_threshold(shell_radii, epoch=None).view(1, 1, max_sc)
                    noised_mask = torch.where(is_mask, (dist_to_ca >= threshold).float(), noised_mask)

        # Existence flow: continuous scalar for ghost/real determination
        # Starts at source_value=0.5 (max entropy), flows toward 0 (ghost) or 1 (real)
        # In split mode, existence was already initialized above -- don't overwrite.
        if not self.split_element_existence:
            existence = None
        if self.use_existence_flow and not self.split_element_existence:
            # Non-split existence flow (legacy mode). Split mode initializes existence above.
            existence = torch.full((batch_size, seq_len, max_sc), self.existence_flow.source_value, device=device)

        # Self-conditioning: track previous predictions for use in next step
        prev_element_pred = None
        prev_cluster_pred = None
        noised_cluster_ids = (
            torch.arange(max_sc, device=device).view(1, 1, max_sc).expand(batch_size, seq_len, max_sc).clone()
            if self.use_cluster_particle_diffusion
            else None
        )

        # Freeze-count: compute the step at which to snapshot model's count prediction
        freeze_count_step = -1
        if freeze_count_at_frac >= 0:
            freeze_count_step = max(1, int(freeze_count_at_frac * len(timesteps)))

        # Logit anchoring: snapshot per-slot ranking at anchor_frac, then bias toward it
        logit_anchor_step = -1
        logit_anchor_scores = None  # Will be (B, L, max_sc) real-vs-PAD scores
        if logit_anchor_frac >= 0:
            logit_anchor_step = max(1, int(logit_anchor_frac * len(timesteps)))

        # Track final-step mixture head outputs for diagnostics
        final_residue_centroid = None
        final_residue_cloud_logvar = None
        final_mixture_posterior = None

        # Cache per-residue LRT delta and count prediction from first step --
        # both depend only on backbone/target features, fixed across sampling steps.
        cached_lrt_delta = None
        cached_count_pred = None

        # Track intermediate count proxies if requested
        intermediate_atom_counts = [] if return_intermediates else None
        coord_traj = [] if return_coord_trajectory else None  # per-step (L, max_sc, 3) for viz
        elem_traj = [] if return_coord_trajectory else None
        mask_traj = [] if return_coord_trajectory else None
        intermediate_mixture_counts = [] if return_intermediates else None
        # Per-residue trajectory: soft count, hard count, dist from CA at each step
        # "pre" = before reverse step (consistent: model input x, t, denoiser_outputs)
        # "post" = after reverse step (updated x, but stale t -- for hard count / CA dist)
        intermediate_per_res_soft_pre = [] if return_intermediates else None  # pre-update P(real) sum
        intermediate_per_res_hard = [] if return_intermediates else None  # post-update non-PAD count
        intermediate_per_res_ca_dist_real = [] if return_intermediates else None  # p_real-weighted mean dist
        intermediate_per_res_ca_dist_ghost = [] if return_intermediates else None  # (1-p_real)-weighted mean dist
        intermediate_slot_pad_logit_pre = [] if return_intermediates else None  # pre-update PAD logit
        intermediate_slot_non_pad_post = [] if return_intermediates else None  # post-update hard non-PAD state
        track_cluster_counts = return_intermediates and self.use_cluster_particle_diffusion
        intermediate_cluster_expected_counts = [] if track_cluster_counts else None
        intermediate_cluster_unique_counts = [] if track_cluster_counts else None
        # Record initial atom counts (before any reverse step)
        if return_intermediates and not self.use_cluster_particle_diffusion:
            init_counts = noised_mask.float().view(batch_size, -1).sum(dim=-1).tolist()
            intermediate_atom_counts.append(init_counts)

        if track_cluster_counts and noised_cluster_ids is not None:
            valid_seq_mask = (
                seq_mask if seq_mask is not None else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            )
            initial_unique_counts = []
            initial_expected_counts = []
            for b in range(batch_size):
                total_unique = 0
                total_expected = 0.0
                for seq_idx in range(seq_len):
                    if not valid_seq_mask[b, seq_idx]:
                        continue
                    total_unique += torch.unique(noised_cluster_ids[b, seq_idx]).numel()
                    total_expected += float(max_sc)
                initial_unique_counts.append(float(total_unique))
                initial_expected_counts.append(total_expected)
            intermediate_atom_counts.append(initial_unique_counts)
            intermediate_cluster_unique_counts.append(initial_unique_counts)
            intermediate_cluster_expected_counts.append(initial_expected_counts)

        # Element-velocity coupling: initialize from current element types.
        # MASK/PAD tokens default to 0.0 (ghost), but evc_mask_init_value overrides
        # to allow neutral (0.5) or real-biased (1.0) initialization.
        evc_sampling = None
        if self.use_element_velocity_coupling:
            from .diffusion import ELEMENT_MASK, ELEMENT_PAD

            if self.split_element_existence and self.existence_absorbing:
                # Absorbing existence at SAMPLING: derive P(real) from the discrete state
                # REAL->1.0, GHOST->0.0, MASK->0.5 (uncertain). Sampling uses the existence state because
                # there is no GT mask here; TRAINING now feeds the GT binary mask to EVC instead (the
                # MASK->0.5 path silenced FiLM at noise -- see the training-side note ~L5535).
                evc_sampling = torch.where(
                    existence == EXISTENCE_REAL,
                    torch.ones(batch_size, seq_len, max_sc, device=device),
                    torch.where(
                        existence == EXISTENCE_GHOST,
                        torch.zeros(batch_size, seq_len, max_sc, device=device),
                        torch.full((batch_size, seq_len, max_sc), 0.5, device=device),
                    ),
                )
            elif self.split_element_existence:
                # Split mode at SAMPLING: use the continuous existence value directly (no GT mask at
                # sampling). TRAINING no longer uses this -- it feeds the GT binary mask to EVC.
                evc_sampling = existence.clone()  # (B, L, max_sc) ∈ [0, 1]
            elif getattr(self, "evc_ss_noised_element_prob", 0.0) > 0:
                # Noised-element EVC (trained with evc_ss_noised_element_prob>0): honest 3-way read of the
                # absorbing element state -- REAL->1.0, MASK->0.5, PAD->0.0 -- so at t=T (all MASK) EVC reads
                # 0.5 (uncertain) instead of 0.0 (confirmed ghost), avoiding the all-PAD collapse spiral.
                evc_sampling = evc_from_element_state(element_types)
            else:
                is_resolved_real = ((element_types != ELEMENT_PAD) & (element_types != ELEMENT_MASK)).float()
                if evc_mask_init_value > 0.0:
                    is_unresolved = ((element_types == ELEMENT_PAD) | (element_types == ELEMENT_MASK)).float()
                    evc_sampling = is_resolved_real + evc_mask_init_value * is_unresolved
                else:
                    evc_sampling = is_resolved_real

        # Count-velocity coupling: initialized from cached_count_pred after first step.
        # None until first denoiser call produces count_pred.
        cvc_sampling = None

        # K-mask: pin the binder context BEFORE the first denoiser (and before frame-0 traj) so the first
        # reverse step conditions on it -- matching training, which pins before the forward. Subsequent
        # steps are pinned after each reverse update inside the loop.
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

        # BUG-A fix (2026-06-25): build the per-residue design/context state ONCE (design_mask is static across
        # reverse steps), via the SAME helper as the training forward so the denoiser conditions on the pinned
        # clean context rather than "denoising" it. Only for partial-design inpaint (legs B/C/D); full-design
        # sampling passes no design_mask -> stays None, matching training's full-design path.
        inpaint_ar_state = None
        if design_mask is not None and inpaint_gt_coords is not None:
            inpaint_ar_state = build_kmask_ar_state(design_mask, seq_mask, x.shape[2], device)

        if return_coord_trajectory:  # frame 0 = initial donut state
            coord_traj.append(x.detach().to("cpu", torch.float32).clone())
            elem_traj.append(element_types.detach().to("cpu").clone())
            mask_traj.append((element_types != ELEMENT_PAD).detach().to("cpu").clone())

        # === SAMPLE-PATH PARITY -- compute the volumetric latent ONCE, BEFORE the reverse loop ===
        # The volumetric head is t-INDEPENDENT (reads only the fixed backbone + fixed target; no side chains,
        # no noise), so its per-residue `vol_hidden` is CONSTANT across the whole trajectory. We compute it a
        # single time here and pass the SAME tensor into every denoiser step below via `_denoise_kwargs`, so
        # the deep injection is applied at INFERENCE exactly as in training (NOT silent at sampling -- this is
        # the "expert heads sample-path silent" trap the deep-inject must avoid), and is never recomputed
        # per step. None (=> denoiser injection skipped) unless deep-inject is on. No GT side chains at
        # sampling => the head returns `vol_hidden` (+ vol_density_pred) but no GT field, which is all we need.
        # SINGLE-SITE POCKET CONTEXT is deliberately NOT wired at sampling (the accepted train-with-context /
        # sample-without-context bet). OPTION (ii) routes the neighbour-occupancy field into BOTH vol_density_pred
        # (field_proj) AND the RETURNED vol_hidden (field_to_hidden_proj), so at inference vol_hidden WOULD carry
        # the pocket context IF a context source were fed. It is not: no context_sidechain_* is passed here, so
        # `vol_density_neighbor` is None => neither the density path nor the vol_hidden add fires => the head is
        # byte-identical to context-off, and no GT ever leaks (none exists at inference). This runs sampling at 0
        # recycles (no predicted x0 to serve as context). FUTURE OPTION: feed the reverse loop's running x0
        # estimate as the per-step context source here so vol_hidden tracks the resolving pocket at inference.
        # the existence coupling's element-head PAD bias ALSO consumes `vol_hidden` at sampling. And
        # the stereochem head reads `vol_hidden` (threaded into the v2 stream via the denoiser) even
        # with NO deep-inject and NO existence coupling. So compute the (t-independent) latent whenever ANY
        # consumer is active -- deep-inject OR (stereo head AND volumetric head) OR existence coupling -- EXACTLY
        # matching the forward trigger. Without the stereo term a use_volumetric_head + use_stereochem_head run
        # with no other vol consumer would train vol-informed but sample frame-only (train/sample mismatch, the
        # "expert heads sample-path silent" trap). Threaded into every denoiser call this step via
        # `_denoise_kwargs["vol_hidden"]` below.
        # Is ANY consumer of `vol_hidden` active this run? (deep-inject OR stereo-head+vol-head OR existence
        # coupling.) Identical predicate to the pre-forward trigger. `_avail_vol` (backbone+target only,
        # t-independent) is computed ONCE and reused across steps.
        _vol_consumer_active = (
            self.use_volumetric_deep_inject
            or (self.use_stereochem_head and self.use_volumetric_head)
            or self.use_volumetric_existence_coupling
        )
        _avail_vol = (
            self._compute_available_volume(backbone_coords, backbone_mask, target_coords, target_mask)
            if _vol_consumer_active
            else None
        )
        # SINGLE-SITE INFERENCE CONTEXT (option-ii at sampling): the PREVIOUS reverse step's predicted x0
        # (coords + existence mask + element ids) fed to the volumetric head as the neighbour-occupancy source.
        # The head's `_neighbor_occupancy_field` self-excludes each query residue, so each site sees only the
        # OTHER sites' prev-step x0 = the single-site pattern. `None` here marks the FIRST step (t=T), which has
        # no previous x0 and is instead handled by the 2-pass bootstrap in the loop. NEVER GT -- model x0 only.
        x0_prev_coords: torch.Tensor | None = None
        x0_prev_mask: torch.Tensor | None = None
        x0_prev_element: torch.Tensor | None = None

        # NON-single-site path: `vol_hidden` is t-INDEPENDENT (backbone+target only, no side chains, no context)
        # => compute it ONCE here and reuse the SAME tensor every step (byte-identical to the historical sample
        # path). When `use_single_site_context` is ON we instead RECOMPUTE it per step in the loop (just below the
        # recycle block) from the prev-step x0 (step-1: bootstrap x0), so leave it None here for that path.
        vol_hidden_sample = None
        vol_density_inject_sample = None  # sample-path parity (H1): mirror the training forward's density-inject
        if _vol_consumer_active and not self.use_single_site_context:
            _vh_out = self.volumetric_head(
                backbone_coords=backbone_coords,
                backbone_mask=backbone_mask,
                seq_mask=seq_mask,
                target_coords=target_coords,
                target_mask=target_mask,
                target_element=target_atom_element_type,
                target_is_backbone=target_atom_is_backbone,
                available_volume=_avail_vol,
            )
            vol_hidden_sample = _vh_out["vol_hidden"]
            vol_density_inject_sample = (
                self.volumetric_density_inject_proj(_vh_out["vol_density_pred"].detach())
                if self.use_volumetric_density_inject
                else None
            )

        for step_i, t_idx in enumerate(timesteps):
            t = torch.full((batch_size,), t_idx.item(), device=device, dtype=torch.long)

            # Mixture temperature sharpening: anneal from 1.0 (soft) to temp_min (sharp)
            if self.sharpen_temperature_min < 1.0:
                tau_frac = t_idx.float() / max(self.timesteps - 1, 1)
                mix_temperature = (
                    self.sharpen_temperature_min
                    + (1.0 - self.sharpen_temperature_min) * tau_frac**self.sharpen_temperature_power
                )
            else:
                mix_temperature = 1.0

            # Compute noised count (non-PAD atoms per residue)
            noised_count = noised_mask.sum(dim=-1)  # (B, L)
            # Count-conditioning ablation: override the self-fed count at DESIGNED positions
            # (pinned/context positions keep self-fed, which already equals GT via _pin_inpaint).
            if noised_count_override is not None:
                _ovr = noised_count_override.to(noised_count.dtype)
                noised_count = torch.where(design_mask, _ovr, noised_count) if design_mask is not None else _ovr

            _denoise_kwargs = {
                "noised_element_types": element_types,
                "noised_cluster_ids": noised_cluster_ids,
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
                "count_velocity_conditioning": cvc_sampling,
                "soft_element_probs": soft_element_probs,
                # BUG-A fix: flag designed(1)/clean-context(2) residues for inpaint
                "ar_state": inpaint_ar_state,
                # the volumetric latent deep-injected into every SE(3) layer of BOTH the recycle and
                # main denoiser calls this step. Non-single-site: the once-computed constant latent (None unless
                # on). Single-site-context: None here, OVERWRITTEN just below the recycle block with the
                # prev-step-x0 (or step-1 bootstrap-x0) conditioned latent before the main denoiser call.
                "vol_hidden": vol_hidden_sample,
                "vol_density_inject": vol_density_inject_sample,
            }

            # Neighbour-x0 packing at SAMPLING: same mechanism, same recycle semantics as training.
            # `neighbor_x0_packing_recycles=1` at the call site is the CHEAP mode (no extra pass,
            # t_conditioning == t_original); >1 pays ~N x wall-clock for the sharpened context.
            # N here is ARBITRARY and independent of whatever the model trained at: every pass is
            # told its absolute index j via `recycle_index=`, and j is a function of the refinements
            # already applied, not of the total, so pass 2 is encoded identically whether N is 2 or
            # 50. (Train with neighbor_x0_packing_random_recycles so the larger j values are not
            # pure extrapolation -- the encoding saturates, but coverage still helps.)
            nx0_coords = nx0_mask = nx0_trust = nx0_apply = None
            t_orig_s = t_cond_s = None
            nx0_recycle_index = None
            # geom-reconcile stash of a SAME-STEP FRESH predicted x0 produced by an rc>1 recycle
            # (`_rc_x0`) or the nx0-mix fresh pass (`_mix_x0`) this step. Reset to None every step; the reconcile
            # below prefers it over the previous step's `x0_prev_coords`. Unused when the lever is off.
            _geom_src_x0 = None
            _geom_src_mask = None  # bond-inject: hard predicted-existence mask (P(real)>0.5) of _geom_src_x0
            if self.use_neighbor_x0_packing:
                _n_rc = self.neighbor_x0_packing_recycles if _sample_recycles is None else _sample_recycles
                _n_rc = max(1, int(_n_rc))
                _keep_s = (~design_mask.to(device=device, dtype=torch.bool)) if design_mask is not None else None
                t_orig_s = t.float().view(-1, 1).expand(batch_size, seq_len).clone()
                if _keep_s is not None:
                    t_orig_s = torch.where(_keep_s, torch.zeros_like(t_orig_s), t_orig_s)
                t_cond_s = t_orig_s.clone()
                # Pass 1 unless the recycle loop below runs, in which case the graded pass is j=N.
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
                        # Model-owned on BOTH channels, exactly as in training (see
                        # _predicted_existence_probs).
                        _rc_pexist = self._predicted_existence_probs(_rc_out)
                        _rc_x0 = self._x0_from_model_output(
                            _rc_out["noise_pred"], x, t, ca_coords, mask_probs=_rc_pexist
                        )
                        _geom_src_x0 = _rc_x0  # geom-reconcile: freshest same-step x0 (last recycle iter wins)
                        _geom_src_mask = _rc_pexist.detach() > 0.5  # bond-inject: real-atom mask of this x0
                        nx0_coords, nx0_mask, nx0_trust = self._build_neighbor_x0_inputs(
                            _rc_x0,
                            _rc_pexist,
                            t_orig_s,
                            keep_mask=_keep_s,
                            clean_context_coords=inpaint_gt_coords,
                            clean_context_mask=inpaint_gt_mask,
                        )

            # === SINGLE-SITE INFERENCE CONTEXT (DEFAULT for single-site-context volumetric models) ===
            # The volumetric head's option-(ii) pocket context is fed here from the flow's PREDICTED x0 (never
            # GT), so the deep-inject / stereo / existence consumers see the resolving pocket at inference --
            # automatically, with no env var or explicit arg. It REPLACES the `_denoise_kwargs["vol_hidden"]`
            # (None for a single-site model out of the pre-loop) that the MAIN denoiser call below consumes.
            # * Steps >= 2: condition on the PREVIOUS step's x0 (`x0_prev_*`, captured post-reverse-step
            # below) -- a single forward pass, the neighbour context already available for free.
            # * FIRST step (t=T, `x0_prev_*` is None): 2-PASS BOOTSTRAP mirroring the training recycle's
            # pass 1 -> pass 2. PASS 1 runs the denoiser on the no-context latent to get `x0_bootstrap`;
            # the head is then rebuilt WITH that x0 as context and PASS 2 (the main call below) consumes it.
            # So step 1 = 2 denoiser passes, every later step = 1. The main-call output (pass 2) is what the
            # coord/element reverse uses AND what becomes `x0_prev` for step 2.
            # Each site self-excludes its own x0 in the head, so the context is the single-site neighbour
            # pattern. Off / non-volumetric => this whole block is skipped (byte-identical).
            if _vol_consumer_active and self.use_single_site_context:
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
                    _vd_bootstrap = (
                        self.volumetric_density_inject_proj(_vh_boot_out["vol_density_pred"].detach())
                        if self.use_volumetric_density_inject
                        else None
                    )
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
                    _ctx_coords, _ctx_mask, _ctx_element = x0_prev_coords, x0_prev_mask, x0_prev_element
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
                _denoise_kwargs["vol_density_inject"] = (
                    self.volumetric_density_inject_proj(_vh_ss_out["vol_density_pred"].detach())
                    if self.use_volumetric_density_inject
                    else None
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
                # x0 bond-inject: sample-path parity with the training final call (bond_prev_x0=ba_prev_x0 +
                # bond_prev_mask). Source is the freshest SAME-STEP predicted x0, `_geom_src_x0`, and its hard
                # predicted-existence mask `_geom_src_mask`, produced this step by the recycle (rc>1) or nx0-mix
                # fresh pass and reset to None every step; detached one-way read-out. Both None unless
                # use_bond_angle_deep_inject AND such a pass ran this step => byte-identical when the flag is off.
                # x0 bond-inject source: prefer the same-step fresh x0 (`_geom_src_x0`); else fall back to the
                # PREVIOUS step's x0 (`x0_prev_coords`) -- the SAME fallback geom-reconcile uses at ~14215 -- so the
                # inject FIRES even at the default eval recycles=1 (no recycle/nx0-mix pass) instead of being
                # silently off . Fallback mask = prev-step existence thresholded to the same
                # hard (P(real)>0.5) format as `_geom_src_mask`. Both None (flag off, or step-0 with no source)
                # => byte-identical.
                bond_prev_x0=(
                    (_geom_src_x0 if _geom_src_x0 is not None else x0_prev_coords).detach()
                    if (self.use_bond_angle_deep_inject and (_geom_src_x0 is not None or x0_prev_coords is not None))
                    else None
                ),
                bond_prev_mask=(
                    (
                        _geom_src_mask
                        if _geom_src_x0 is not None
                        else ((x0_prev_mask > 0.5) if x0_prev_mask is not None else None)
                    )
                    if (self.use_bond_angle_deep_inject and (_geom_src_x0 is not None or x0_prev_coords is not None))
                    else None
                ),
                t_original_res=t_orig_s,
                t_conditioning_res=t_cond_s,
                recycle_index=nx0_recycle_index,
                **_denoise_kwargs,
            )
            model_output = denoiser_outputs["noise_pred"]
            element_logits = denoiser_outputs["element_logits"]
            # Train/eval parity: apply the SAME supervised expert modulations (polarity PoE + glycine PAD
            # push) the model was trained with -- without this the heads never touch de-novo sampling.
            if self.use_polarity_head or self.use_glycine_head:
                element_logits, _ = self._apply_expert_logit_biases(
                    element_logits, denoiser_outputs["backbone_features"], backbone_coords, backbone_mask
                )
            cluster_logits = denoiser_outputs["cluster_logits"]
            cluster_occupancy_probs = (
                self._compute_cluster_occupancy_probs(cluster_logits) if self.use_cluster_particle_diffusion else None
            )

            # Prior cloud blending: replace denoiser's residue centroid/logvar with blended version
            if self.use_prior_cloud:
                bb_feats = denoiser_outputs["backbone_features"]
                pc = self.prior_centroid_head(bb_feats)  # (B, L, 3)
                plv = self.prior_cloud_logvar_head(bb_feats).squeeze(-1)  # (B, L)
                tau = t.float() / max(self.timesteps - 1, 1)
                w = ((1.0 - tau) ** self.prior_blend_power).view(batch_size, 1)
                rc = denoiser_outputs.get("residue_centroid")
                rlv = denoiser_outputs.get("residue_cloud_logvar")
                if rc is not None:
                    denoiser_outputs["residue_centroid"] = (1.0 - w.unsqueeze(-1)) * pc + w.unsqueeze(-1) * rc
                if rlv is not None:
                    denoiser_outputs["residue_cloud_logvar"] = (1.0 - w) * plv + w * rlv

            # Cache LRT delta and count prediction from first step -- backbone-only, constant across steps
            if cached_lrt_delta is None and self.dynamic_lrt:
                cached_lrt_delta = denoiser_outputs.get("residue_lrt_delta")
            if cached_count_pred is None and (self.dlrt_analytical_scale > 0 or count_ghost_velocity > 0):
                cached_count_pred = denoiser_outputs.get("residue_count_pred")
            # CVC: set from count_pred on first step (stable, backbone-only signal)
            if cvc_sampling is None and self.use_count_velocity_coupling:
                cvc_sampling = denoiser_outputs.get("residue_count_pred")

            # Per-residue trajectory: pre-update soft count from consistent state (x, t, denoiser_outputs)
            if return_intermediates:
                _lm = denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None
                _lv = denoiser_outputs.get("residue_cloud_logvar") if self.mixture_loss_weight > 0 else None
                pre_mix_post = self.compute_mixture_posterior(
                    noised_coords=x,
                    ca_coords=ca_coords,
                    t=t,
                    learned_centroid=_lm,
                    learned_cloud_logvar=_lv,
                    residue_lrt_delta=cached_lrt_delta,
                    residue_count_pred=cached_count_pred,
                    temperature=mix_temperature,
                )  # (B, L, max_sc)
                if seq_mask is not None:
                    pre_mix_post = pre_mix_post * seq_mask.unsqueeze(-1).float()
                # Per-residue soft count: sum of P(real) across slots
                intermediate_per_res_soft_pre.append(pre_mix_post.sum(dim=-1).detach().cpu())
                # Split CA distance: real-like vs ghost-like slots
                slot_dist_pre = (x - ca_expanded).norm(dim=-1)  # (B, L, max_sc)
                p_real = pre_mix_post.detach()
                p_ghost = 1.0 - p_real
                # Weighted mean dist for likely-real slots (high p_real)
                real_denom = p_real.sum(dim=-1).clamp(min=1e-6)
                intermediate_per_res_ca_dist_real.append(
                    (slot_dist_pre * p_real).sum(dim=-1).div(real_denom).detach().cpu()
                )
                # Weighted mean dist for likely-ghost slots (high 1-p_real)
                ghost_denom = p_ghost.sum(dim=-1).clamp(min=1e-6)
                intermediate_per_res_ca_dist_ghost.append(
                    (slot_dist_pre * p_ghost).sum(dim=-1).div(ghost_denom).detach().cpu()
                )

            # Extract occupancy logits for existence flow (velocity prediction)
            existence_velocity = denoiser_outputs["occupancy_logits"] if self.use_existence_flow else None

            # Occupancy-gated element logits (mirrors training gating)
            if self.occupancy_gate_elements or self.mixture_gate_weight > 0:
                occupancy_logits = denoiser_outputs["occupancy_logits"]
                occ_gate = torch.sigmoid(occupancy_logits)  # No detach needed at sampling time

                # Blend with mixture posterior if active
                if self.mixture_gate_weight > 0:
                    learned_centroid = denoiser_outputs["residue_centroid"] if self.mixture_loss_weight > 0 else None
                    learned_cloud_logvar = (
                        denoiser_outputs["residue_cloud_logvar"] if self.mixture_loss_weight > 0 else None
                    )
                    mixture_posterior = self.compute_mixture_posterior(
                        noised_coords=x,
                        ca_coords=ca_coords,
                        t=t,
                        learned_centroid=learned_centroid,
                        learned_cloud_logvar=learned_cloud_logvar,
                        residue_lrt_delta=cached_lrt_delta,
                        temperature=mix_temperature,
                    )
                    # Disable mixture posterior above max noise fraction
                    if self.mixture_gate_max_noise < 1.0:
                        t_frac = t.float() / self.timesteps
                        high_noise_mask = t_frac > self.mixture_gate_max_noise
                        if high_noise_mask.any():
                            mixture_posterior = mixture_posterior.clone()
                            mixture_posterior[high_noise_mask] = 0.0
                    # Store final-step outputs for diagnostics
                    final_residue_centroid = learned_centroid
                    final_residue_cloud_logvar = learned_cloud_logvar
                    final_mixture_posterior = mixture_posterior

                    w = self.mixture_gate_weight
                    gate_signal = (1.0 - w) * occ_gate + w * mixture_posterior
                else:
                    gate_signal = occ_gate

                s = self.occupancy_gate_strength
                element_logits = element_logits.clone()
                element_logits[..., 0] += (1.0 - gate_signal) * s
                element_logits[..., 1:] += gate_signal.unsqueeze(-1) * s

            # Count-head element bias: use per-residue count prediction to bias PAD/non-PAD
            # Slots below predicted count get non-PAD boost, slots above get PAD boost.
            # Uses smooth sigmoid step function centered at predicted count boundary.
            if count_head_element_bias > 0 and self.residue_count_loss_weight > 0:
                rc_pred = denoiser_outputs["residue_count_pred"].detach()  # (B, L)
                rc_pred = rc_pred.clamp(0, max_sc)
                # Per-slot signal: sigmoid(strength * (count - slot_idx - 0.5))
                # High for slots below count, low for slots above
                slot_idx = torch.arange(max_sc, device=device).float().view(1, 1, -1)  # (1, 1, max_sc)
                count_gate = torch.sigmoid(4.0 * (rc_pred.unsqueeze(-1) - slot_idx - 0.5))  # (B, L, max_sc)
                if seq_mask is not None:
                    count_gate = count_gate * seq_mask.unsqueeze(-1).float()
                s_count = count_head_element_bias
                element_logits = (
                    element_logits.clone()
                    if not (self.occupancy_gate_elements or self.mixture_gate_weight > 0)
                    else element_logits
                )
                element_logits[..., 0] += (1.0 - count_gate) * s_count  # PAD boost above count
                element_logits[..., 1:] += count_gate.unsqueeze(-1) * s_count  # non-PAD boost below count

            # Non-PAD logit bias: boost non-PAD logits during sampling.
            # Counteracts the PAD-heavy prior (71%) that suppresses atom creation.
            if non_pad_logit_bias > 0:
                element_logits = (
                    element_logits.clone()
                    if not (
                        self.occupancy_gate_elements
                        or self.mixture_gate_weight > 0
                        or (count_head_element_bias > 0 and self.residue_count_loss_weight > 0)
                    )
                    else element_logits
                )
                effective_bias = non_pad_logit_bias
                if non_pad_bias_nres_scale > 0:
                    # Use actual residue count (not padded seq_len)
                    n_res = int(seq_mask.sum().item()) if seq_mask is not None else seq_len
                    # Scale up for large peptides, never below base bias
                    effective_bias *= max(1.0, 1.0 + non_pad_bias_nres_scale * (n_res - 11))
                if non_pad_bias_time_decay == "linear":
                    # Full bias at high noise (step 0), zero at low noise (last step)
                    frac = 1.0 - step_i / max(num_steps - 1, 1)
                    effective_bias *= frac
                elif non_pad_bias_time_decay == "cosine":
                    import math

                    frac = 0.5 * (1.0 + math.cos(math.pi * step_i / max(num_steps - 1, 1)))
                    effective_bias *= frac
                element_logits[..., 1:] += effective_bias  # Boost C, N, O, S equally

            # Distance-based element logit bias: far from CA -> non-PAD, near CA -> PAD.
            # Coords->elements coupling at sampling time.
            if element_dist_logit_bias > 0:
                dist_from_ca = (x - ca_expanded).norm(dim=-1)  # (B, L, max_sc)
                # Sigmoid centered at 2.5Å: atoms >2.5Å -> positive (non-PAD), <2.5Å -> negative (PAD)
                dist_signal = torch.sigmoid((dist_from_ca - 2.5) * 2.0) * 2.0 - 1.0  # range [-1, 1]
                bias = dist_signal * element_dist_logit_bias
                element_logits = (
                    element_logits.clone() if not isinstance(element_logits, torch.Tensor) else element_logits
                )
                element_logits[..., 0] -= bias  # suppress PAD for far atoms, boost for near
                element_logits[..., 1:] += (bias / max(self.num_element_classes - 1, 1)).unsqueeze(-1)

            # MASK persistence bias: suppress PAD logit for MASK tokens to prevent premature collapse.
            if mask_persistence_bias > 0 and self.donut_element_init == "mask":
                from .diffusion import ELEMENT_MASK

                is_mask = (element_types == ELEMENT_MASK).float()
                if is_mask.any():
                    element_logits = element_logits.clone()
                    element_logits[..., 0] -= mask_persistence_bias * is_mask  # suppress PAD for MASK tokens

            # Logit anchoring: snapshot per-slot real-vs-PAD score at anchor step
            # Not applicable in split mode (no PAD class in element logits)
            if not self.split_element_existence and logit_anchor_step >= 0 and step_i == logit_anchor_step:
                pad_logits = element_logits[:, :, :, ELEMENT_PAD]  # (B, L, max_sc)
                nonpad_logits = element_logits[:, :, :, 1:].max(dim=-1).values  # max over C, N, O, S
                logit_anchor_scores = (nonpad_logits - pad_logits).detach()  # positive = non-PAD

            # Logit anchoring: bias toward early ranking while allowing global count evolution
            if not self.split_element_existence and logit_anchor_scores is not None and step_i > logit_anchor_step:
                element_logits = element_logits.clone()
                element_logits[:, :, :, ELEMENT_PAD] -= logit_anchor_alpha * logit_anchor_scores

            if return_intermediates and not self.split_element_existence:
                intermediate_slot_pad_logit_pre.append(element_logits[..., ELEMENT_PAD].detach().cpu())

            # Reverse diffusion step for element types (includes PAD=0 -> determines mask)
            # Element stride: only update elements every element_stride coord steps.
            # With stride > 1, each element step covers a larger noise range, giving larger
            # effective beta and making PAD->non-PAD transitions easier in the posterior.
            do_element_step = (element_stride <= 1) or (step_i % element_stride == 0)
            # Element early stop: freeze element/mask after element_early_stop_frac of trajectory
            if element_early_stop_frac < 1.0 and step_i >= int(element_early_stop_frac * num_steps):
                do_element_step = False
            # === Split existence/element reverse step ===
            if self.split_element_existence:
                from .diffusion import SPLIT_ELEMENT_MASK

                # Compute per-slot power schedules for reverse steps
                # Existence: inverted (distal resolves first -- collapse outer shells)
                # Elements: same as coords (proximal resolves first)
                exist_samp_slot_powers = None
                elem_samp_slot_powers = None
                exist_samp_sched_kwargs = {}
                if self.position_based_element_powers:
                    exist_power_1d = self._existence_slot_powers(max_sc, device)
                    exist_samp_slot_powers = exist_power_1d.view(1, 1, max_sc).expand(batch_size, seq_len, max_sc)
                    if not self.split_element_uniform_power:
                        mask_for_powers = torch.ones(batch_size, seq_len, max_sc, dtype=torch.bool, device=device)
                        elem_samp_slot_powers = self._element_slot_powers(mask_for_powers)

                if self.existence_absorbing:
                    # 1. Absorbing existence: construct 3-class logits from scalar occupancy_logits
                    occ_logits = denoiser_outputs["occupancy_logits"]  # (B, L, max_sc)

                    # Existence anchor: snapshot early real-vs-ghost ranking, bias toward it
                    if logit_anchor_step >= 0 and step_i == logit_anchor_step:
                        logit_anchor_scores = occ_logits.detach().clone()  # positive = real
                    if logit_anchor_scores is not None and step_i > logit_anchor_step:
                        occ_logits = occ_logits + logit_anchor_alpha * logit_anchor_scores

                    exist_x0_logits = torch.stack(
                        [
                            torch.zeros_like(occ_logits),  # GHOST logit (reference)
                            occ_logits,  # REAL logit (anchored)
                            torch.full_like(occ_logits, -1e9),  # MASK logit (never predict MASK)
                        ],
                        dim=-1,
                    )  # (B, L, max_sc, 3)

                    if exist_samp_slot_powers is not None:
                        sab, sabp, sb = self.existence_diffusion.compute_slot_schedule(t, exist_samp_slot_powers)
                        exist_samp_sched_kwargs = {
                            "slot_alpha_bar": sab,
                            "slot_alpha_bar_prev": sabp,
                            "slot_beta": sb,
                        }
                    existence = self.existence_diffusion.p_sample(
                        existence,
                        t,
                        exist_x0_logits,
                        absorbing_mask=self.split_absorbing_element,
                        **exist_samp_sched_kwargs,
                    )
                    # Resolve: REAL->real, GHOST/MASK->ghost (conservative: unresolved MASK = ghost)
                    is_real = existence == EXISTENCE_REAL
                else:
                    existence_velocity = denoiser_outputs["occupancy_logits"]
                    if self.existence_velocity_scale != 1.0:
                        existence_velocity = existence_velocity * self.existence_velocity_scale

                    # 1. Existence flow Euler step
                    if t_idx > 0:
                        t_prev_idx_e = timesteps[step_i + 1]
                        t_prev_e = torch.full((batch_size,), t_prev_idx_e.item(), device=device, dtype=torch.long)
                        existence = self.existence_flow.flow_step(
                            existence.unsqueeze(-1), t, t_prev_e, existence_velocity.unsqueeze(-1)
                        ).squeeze(-1)
                    else:
                        # Final step: predict e0 directly
                        existence = self.existence_flow.predict_e0_from_velocity(
                            existence.unsqueeze(-1), t, existence_velocity.unsqueeze(-1)
                        ).squeeze(-1)
                    is_real = existence >= self.existence_threshold

                # 2. Element reverse step on real slots only
                if do_element_step:
                    if self.split_element_flow and _split_soft_probs is not None:
                        # Flow matching: OT-path jump on {C,N,O,S,MASK} simplex
                        if step_i + 1 < len(timesteps):
                            t_elem_prev = torch.full(
                                (batch_size,), timesteps[step_i + 1].item(), device=device, dtype=torch.long
                            )
                        else:
                            t_elem_prev = torch.zeros(batch_size, device=device, dtype=torch.long)
                        _split_soft_probs = self.split_element_flow_proc.reverse_step(
                            _split_soft_probs,
                            t,
                            t_elem_prev,
                            element_logits,
                            slot_powers=elem_samp_slot_powers,
                        )
                        # Ghost slots -> all-MASK probability
                        ghost_probs = torch.zeros_like(_split_soft_probs)
                        ghost_probs[..., SPLIT_ELEMENT_MASK] = 1.0
                        _split_soft_probs = torch.where(
                            (~is_real).unsqueeze(-1).expand_as(_split_soft_probs),
                            ghost_probs,
                            _split_soft_probs,
                        )
                        element_types = _split_soft_probs.argmax(dim=-1)
                        if self.split_element_flow_temp > 0:
                            log_probs = (_split_soft_probs + 1e-8).log()
                            soft_element_probs = torch.softmax(log_probs / self.split_element_flow_temp, dim=-1)
                    else:
                        # Discrete diffusion: p_sample reverse step
                        element_types_new = self.split_element_diffusion.p_sample(
                            element_types,
                            t,
                            element_logits,
                            absorbing_mask=self.split_absorbing_element,
                        )
                        # Ghost slots forced to MASK (their element is undefined)
                        element_types = torch.where(
                            is_real,
                            element_types_new,
                            torch.full_like(element_types, SPLIT_ELEMENT_MASK),
                        )

                # 3. Update mask from existence
                noised_mask = is_real.float()

                # 4. At final step: collapse lingering MASK, convert split -> original encoding
                if t_idx == 0:
                    # For absorbing existence: any remaining MASK -> GHOST (conservative)
                    if self.existence_absorbing:
                        is_real = existence == EXISTENCE_REAL

                    # Collapse any remaining MASK on real slots to Carbon (most common element)
                    from .diffusion import SPLIT_ELEMENT_C

                    element_types = torch.where(
                        is_real & (element_types == SPLIT_ELEMENT_MASK),
                        torch.full_like(element_types, SPLIT_ELEMENT_C),
                        element_types,
                    )
                    # Real slots: split+1 (C=0->1, N=1->2, O=2->3, S=3->4)
                    # Ghost slots: PAD=0
                    element_types = torch.where(
                        is_real,
                        element_types + 1,  # shift to original {C=1, N=2, O=3, S=4}
                        torch.zeros_like(element_types),  # PAD=0
                    )
                    # Enforce prefix constraint
                    element_types = apply_prefix_constraint(element_types, exempt_slot0=reserved_slot0_prefix_exempt)

                # Update EVC from existence (if active)
                if evc_sampling is not None:
                    if self.existence_absorbing:
                        # Derive P(real) from discrete state: REAL->1.0, GHOST->0.0, MASK->0.5
                        evc_sampling = torch.where(
                            existence == EXISTENCE_REAL,
                            torch.ones_like(noised_mask),
                            torch.where(
                                existence == EXISTENCE_GHOST,
                                torch.zeros_like(noised_mask),
                                torch.full_like(noised_mask, 0.5),
                            ),
                        )
                    else:
                        evc_sampling = is_real.float()

            # Late element resolution: block PAD transitions during element reverse.
            # Ghost/real comes from mixture posterior at t=0, not from element diffusion.
            # Without this, p_sample's 71% PAD prior drags slots back to PAD mid-trajectory.
            elif self.late_element_resolution:
                element_logits = element_logits.clone()
                element_logits[..., ELEMENT_PAD] = -1e9
            if self.all_carbon_sampling:
                # All-carbon mode: skip element diffusion entirely during sampling.
                # Denoiser sees all-Carbon -> predicts real-atom velocities for all slots.
                # Ghost/real determined purely from coordinates via mixture posterior.
                learned_centroid = denoiser_outputs["residue_centroid"] if self.mixture_loss_weight > 0 else None
                learned_cloud_logvar = (
                    denoiser_outputs["residue_cloud_logvar"] if self.mixture_loss_weight > 0 else None
                )
                mix_post = self.compute_mixture_posterior(
                    noised_coords=x,
                    ca_coords=ca_coords,
                    t=t,
                    learned_centroid=learned_centroid,
                    learned_cloud_logvar=learned_cloud_logvar,
                    residue_lrt_delta=cached_lrt_delta,
                    residue_count_pred=cached_count_pred,
                    temperature=mix_temperature,
                )
                final_residue_centroid = learned_centroid
                final_residue_cloud_logvar = learned_cloud_logvar
                final_mixture_posterior = mix_post

                # At final step, apply mixture to determine ghost/real
                if t_idx == 0:
                    is_real = mix_post > 0.5
                    element_types = torch.where(
                        is_real,
                        element_types,  # Keep resolved element type for real slots
                        torch.zeros_like(element_types),  # PAD=0 for ghost
                    )
                # element_types stays all-Carbon during high-noise steps
            elif self.non_pad_element_sampling and do_element_step:
                # Non-PAD element sampling: run element reverse among {C,N,O,S} only.
                # PAD logit is blocked so all 14 slots stay non-PAD throughout.
                # Ghost/real determined from mixture posterior at t=0.
                chem_logits = element_logits.clone()
                chem_logits[..., ELEMENT_PAD] = -1e9
                element_types = self.element_diffusion.p_sample(
                    element_types,
                    t,
                    chem_logits,
                )

                # Chemistry collapse: same cosine schedule as training, compressed into [0, cutoff].
                # Above cutoff -> guaranteed all-Carbon. Below cutoff -> progressively resolve.
                t_frac = t_idx.float() / max(self.timesteps - 1, 1)
                t_chem_int = (t_frac / self.late_element_cutoff).clamp(0, 1) * (self.timesteps - 1)
                t_chem_int = t_chem_int.round().long().clamp(0, self.timesteps - 1)
                p_keep_chem = self.chem_retention[t_chem_int].item()
                if p_keep_chem < 1.0:
                    collapse_rand = torch.rand(element_types.shape, device=device)
                    should_collapse = collapse_rand >= p_keep_chem
                    element_types = torch.where(
                        should_collapse,
                        torch.ones_like(element_types),  # C=1
                        element_types,
                    )

                # Track mixture posterior at every step (for diagnostics)
                learned_centroid = denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None
                learned_cloud_logvar = (
                    denoiser_outputs.get("residue_cloud_logvar") if self.mixture_loss_weight > 0 else None
                )
                mix_post = self.compute_mixture_posterior(
                    noised_coords=x,
                    ca_coords=ca_coords,
                    t=t,
                    learned_centroid=learned_centroid,
                    learned_cloud_logvar=learned_cloud_logvar,
                    residue_lrt_delta=cached_lrt_delta,
                    residue_count_pred=cached_count_pred,
                    temperature=mix_temperature,
                )
                final_residue_centroid = learned_centroid
                final_residue_cloud_logvar = learned_cloud_logvar
                final_mixture_posterior = mix_post

                # At final step: mixture posterior determines ghost/real
                if t_idx == 0:
                    is_real = mix_post > 0.5
                    element_types = torch.where(
                        is_real,
                        element_types,  # Keep resolved chemistry for real slots
                        torch.zeros_like(element_types),  # PAD=0 for ghost
                    )
            elif self.element_flow_matching and do_element_step:
                # Element flow matching: OT-path step on probability simplex
                if step_i + 1 < len(timesteps):
                    t_elem_prev = torch.full(
                        (batch_size,), timesteps[step_i + 1].item(), device=device, dtype=torch.long
                    )
                else:
                    t_elem_prev = torch.zeros(batch_size, device=device, dtype=torch.long)
                soft_element_probs = self.element_flow.reverse_step(soft_element_probs, t, t_elem_prev, element_logits)
                element_types = soft_element_probs.argmax(dim=-1)
                # Enforce prefix constraint
                element_types = apply_prefix_constraint(element_types, exempt_slot0=reserved_slot0_prefix_exempt)
            elif not self.disable_element_types and do_element_step and not self.split_element_existence:
                # Temperature schedule: hot at high t to escape all-PAD, cool to 1.0 at t=0
                if element_sampling_temp_max != 1.0:
                    frac = t_idx.float() / max(self.timesteps - 1, 1)
                    elem_temp = 1.0 + (element_sampling_temp_max - 1.0) * frac**element_sampling_temp_power
                else:
                    elem_temp = 1.0

                if self.mixture_override_pad:
                    # === Mixture-derived occupancy reverse ===
                    # PAD/non-PAD from mixture posterior on current coords.
                    # Chemistry uses proper p_sample with PAD zeroed, then occupancy override.
                    learned_centroid = denoiser_outputs["residue_centroid"] if self.mixture_loss_weight > 0 else None
                    learned_cloud_logvar = (
                        denoiser_outputs["residue_cloud_logvar"] if self.mixture_loss_weight > 0 else None
                    )
                    mix_post = self.compute_mixture_posterior(
                        noised_coords=x,
                        ca_coords=ca_coords,
                        t=t,
                        learned_centroid=learned_centroid,
                        learned_cloud_logvar=learned_cloud_logvar,
                        residue_lrt_delta=cached_lrt_delta,
                        temperature=mix_temperature,
                    )
                    # Store for diagnostics
                    final_residue_centroid = learned_centroid
                    final_residue_cloud_logvar = learned_cloud_logvar
                    final_mixture_posterior = mix_post

                    # Oracle override: use GT mask instead of mixture posterior
                    if oracle_sidechain_mask is not None:
                        is_real = oracle_sidechain_mask.bool()
                    else:
                        is_real = mix_post > 0.5  # (B, L, max_sc)

                    # Chemistry reverse: use p_sample with PAD logit zeroed so the
                    # posterior and x_t participate (matching the uniform {C,N,O,S}
                    # forward corruption schedule).
                    chem_logits = element_logits.clone()
                    chem_logits[..., ELEMENT_PAD] = -1e9
                    element_types_new = self.element_diffusion.p_sample(
                        element_types,
                        t,
                        chem_logits,
                        temperature=elem_temp,
                    )

                    # Apply mixture occupancy: real -> chemistry from p_sample, ghost -> PAD
                    element_types = torch.where(is_real, element_types_new, torch.zeros_like(element_types_new))

                    # Enforce prefix constraint
                    element_types = apply_prefix_constraint(element_types, exempt_slot0=reserved_slot0_prefix_exempt)
                else:
                    # Oracle mode: force ghost slot logits to PAD-dominant before p_sample
                    # to avoid degenerate multinomial distributions
                    if oracle_sidechain_mask is not None:
                        gt_mask_bool = oracle_sidechain_mask.bool()
                        element_logits = element_logits.clone()
                        # Ghost slots: set PAD logit high, all others low
                        ghost_mask = ~gt_mask_bool  # (B, L, max_sc)
                        ghost_logit_vals = torch.full((self.num_element_classes,), -1e9, device=device)
                        ghost_logit_vals[ELEMENT_PAD] = 0.0
                        element_logits[ghost_mask] = ghost_logit_vals

                    # For element_stride > 1, compute the target timestep for this element step.
                    # The posterior needs alpha_bar_prev from the target (not just t-1).
                    if element_stride > 1:
                        next_elem_step_i = min(step_i + element_stride, len(timesteps) - 1)
                        t_elem_prev_idx = timesteps[next_elem_step_i]
                        t_elem_prev = torch.full((batch_size,), t_elem_prev_idx.item(), device=device, dtype=torch.long)
                        # Override alphas_cumprod_prev for this larger step
                        elem_alpha_bar_prev = self.element_diffusion.alphas_cumprod[t_elem_prev]
                        for _ in range(element_types.dim() - 1):
                            elem_alpha_bar_prev = elem_alpha_bar_prev.unsqueeze(-1)
                        # Compute effective beta for this larger step:
                        # beta_eff = 1 - alpha_bar_t / alpha_bar_{t_target}
                        # (probability of being absorbed between t_target and t)
                        elem_alpha_bar = self.element_diffusion.alphas_cumprod[t]
                        for _ in range(element_types.dim() - 1):
                            elem_alpha_bar = elem_alpha_bar.unsqueeze(-1)
                        elem_beta = 1.0 - elem_alpha_bar / (elem_alpha_bar_prev + 1e-8)
                        elem_beta = elem_beta.clamp(min=0, max=0.999)

                    # Compute per-slot element schedule for reverse step if groupwise enabled
                    elem_sched_kwargs = {}
                    if self.groupwise_element_schedule:
                        if self.position_based_element_powers:
                            # Use all-real mask so every slot gets proximal->distal schedule
                            # regardless of current PAD state.
                            mask_for_powers = torch.ones(batch_size, seq_len, max_sc, dtype=torch.bool, device=device)
                        else:
                            # Default: use current element state to determine powers (PAD slots get ghost power)
                            mask_for_powers = element_types != 0  # non-PAD = real
                        elem_slot_powers = self._element_slot_powers(mask_for_powers)
                        sab, sabp, sb = self.element_diffusion.compute_slot_schedule(t, elem_slot_powers)
                        elem_sched_kwargs = {
                            "slot_alpha_bar": sab,
                            "slot_alpha_bar_prev": sabp,
                            "slot_beta": sb,
                        }

                    # Sequential slot sampling with prefix constraint:
                    # Sample slot 0 first. If PAD -> all remaining are PAD. Otherwise continue.
                    use_reflow = element_sampling_mode in ("reflow", "reflow_cond")
                    # Threshold hybrid: use reflow only above reflow_start_frac of timesteps
                    if use_reflow and reflow_start_frac < 1.0:
                        t_frac = t_idx.float() / max(self.timesteps - 1, 1)
                        use_reflow = t_frac >= (1.0 - reflow_start_frac)

                    if element_sampling_mode == "greedy":
                        # Greedy: use model's argmax x_0 prediction directly as x_{t-1}
                        # No posterior, no re-corruption -- bypasses PAD bias completely.
                        # Model sees clean element types at each step (slight train/sample mismatch
                        # at intermediate noise, but elements carry less info than coordinates).
                        element_types_new = torch.softmax(element_logits, dim=-1).argmax(dim=-1)
                    elif use_reflow:
                        # Reflow: predict x_0, re-corrupt to t-1 (bypasses PAD-biased posterior)
                        reflow_kwargs = {}
                        if "slot_alpha_bar_prev" in elem_sched_kwargs:
                            reflow_kwargs["slot_alpha_bar_prev"] = elem_sched_kwargs["slot_alpha_bar_prev"]
                        conditional = element_sampling_mode == "reflow_cond"
                        element_types_new = self.element_diffusion.p_sample_reflow(
                            element_types,
                            t,
                            element_logits,
                            temperature=elem_temp,
                            conditional=conditional,
                            **reflow_kwargs,
                        )
                    else:
                        # For element_stride > 1 (non-groupwise), override schedule for larger step
                        if element_stride > 1 and not self.groupwise_element_schedule:
                            elem_sched_kwargs = {
                                "slot_alpha_bar": elem_alpha_bar.expand_as(element_types),
                                "slot_alpha_bar_prev": elem_alpha_bar_prev.expand_as(element_types),
                                "slot_beta": elem_beta.expand_as(element_types),
                            }
                        # Time-dependent squash schedules
                        if squash_schedule != "constant" and posterior_pad_squash != 1.0:
                            t_norm = t_idx.item() / max(self.timesteps - 1, 1)
                            if squash_schedule == "linear":
                                # Linear: squash_base at t=T, 1.0 at t=0
                                effective_squash = 1.0 - (1.0 - posterior_pad_squash) * t_norm
                            elif squash_schedule.startswith("cosine"):
                                # Cosine: stays near squash_base longer, only relaxes near t=0
                                import math

                                effective_squash = posterior_pad_squash + (1.0 - posterior_pad_squash) * (
                                    1.0 - math.cos(math.pi / 2 * (1.0 - t_norm))
                                )
                            elif squash_schedule == "step":
                                # Step: full squash for t > T/2, no squash below
                                effective_squash = posterior_pad_squash if t_norm > 0.5 else 1.0
                            else:
                                effective_squash = posterior_pad_squash
                        else:
                            effective_squash = posterior_pad_squash
                        element_types_new = self.element_diffusion.p_sample(
                            element_types,
                            t,
                            element_logits,
                            temperature=elem_temp,
                            posterior_pad_squash=effective_squash,
                            absorbing_mask=_sample_absorbing,
                            **elem_sched_kwargs,
                        )

                    if oracle_sidechain_mask is not None:
                        # Oracle ablation: enforce GT PAD/non-PAD mask
                        gt_mask_bool = oracle_sidechain_mask.bool()
                        # Non-PAD slots that became PAD: restore to C (most common non-PAD element)
                        element_types_new = torch.where(
                            gt_mask_bool & (element_types_new == ELEMENT_PAD),
                            torch.ones_like(element_types_new),  # C=1
                            element_types_new,
                        )
                        # PAD slots must stay PAD
                        element_types_new = torch.where(
                            gt_mask_bool, element_types_new, torch.zeros_like(element_types_new)
                        )
                    else:
                        # Enforce prefix constraint: if slot i is PAD, all j>i must be PAD
                        # Vectorized: cummin on (element != PAD) gives 1,1,...,1,0,0,...,0
                        element_types_new = apply_prefix_constraint(
                            element_types_new, exempt_slot0=reserved_slot0_prefix_exempt
                        )

                        # Enforce ±max_count_delta atom count change per step (mirrors forward process)
                        # max_count_delta=1 is standard, 0=no clamp (unlimited count changes)
                        old_count = noised_count.long()  # (B, L) -- count before this step
                        new_count = (element_types_new != ELEMENT_PAD).sum(dim=-1)  # (B, L)
                        if max_count_delta > 0:
                            clamped_count = new_count.clamp(
                                min=(old_count - max_count_delta).clamp(min=0), max=old_count + max_count_delta
                            )
                        else:
                            clamped_count = new_count  # no clamping
                        # If count needs to decrease, zero out the last occupied slot
                        needs_decrease = new_count > clamped_count  # (B, L)
                        if needs_decrease.any():
                            last_occ = (clamped_count).clamp(min=0).long()  # index of first slot to zero
                            # Zero all slots from clamped_count onward
                            slot_indices = torch.arange(max_sc, device=device).view(1, 1, max_sc)
                            decrease_mask = slot_indices < last_occ.unsqueeze(-1)
                            element_types_new = torch.where(
                                needs_decrease.unsqueeze(-1),
                                element_types_new * decrease_mask.long(),
                                element_types_new,
                            )
                        # If count needs to increase, restore slots from old element_types
                        needs_increase = new_count < clamped_count  # (B, L)
                        if needs_increase.any():
                            target_count = clamped_count  # how many slots should be non-PAD
                            slot_indices = torch.arange(max_sc, device=device).view(1, 1, max_sc)
                            restore_mask = (slot_indices < target_count.unsqueeze(-1)) & (
                                element_types_new == ELEMENT_PAD
                            )
                            # Use old element types for restored slots; if old was also PAD, use C (most common)
                            restore_values = torch.where(
                                element_types != ELEMENT_PAD, element_types, torch.ones_like(element_types)
                            )
                            element_types_new = torch.where(
                                needs_increase.unsqueeze(-1) & restore_mask,
                                restore_values,
                                element_types_new,
                            )

                    element_types = element_types_new

            # Existence flow: Euler step on continuous existence variable, then override mask
            # Skip in split mode: existence is handled in the split block above.
            if (
                self.use_existence_flow
                and not self.split_element_existence
                and existence is not None
                and existence_velocity is not None
            ):
                if t_idx > 0:
                    t_prev_idx_e = timesteps[step_i + 1]
                    t_prev_e = torch.full((batch_size,), t_prev_idx_e.item(), device=device, dtype=torch.long)
                    existence = self.existence_flow.flow_step(
                        existence.unsqueeze(-1), t, t_prev_e, existence_velocity.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    # Final step: predict e0 directly from current state and velocity
                    existence = self.existence_flow.predict_e0_from_velocity(
                        existence.unsqueeze(-1), t, existence_velocity.unsqueeze(-1)
                    ).squeeze(-1)
                # Override element types based on existence threshold
                is_real = existence >= self.existence_threshold
                # Real slots that element diffusion set to PAD: restore to C (most common non-PAD)
                element_types = torch.where(
                    is_real & (element_types == ELEMENT_PAD),
                    torch.ones_like(element_types),  # C=1
                    element_types,
                )
                # Ghost slots: force to PAD regardless of element diffusion
                element_types = torch.where(is_real, element_types, torch.zeros_like(element_types))
                # Re-enforce prefix constraint after existence override
                element_types = apply_prefix_constraint(element_types, exempt_slot0=reserved_slot0_prefix_exempt)

            # Late element resolution: smooth Carbon collapse + mixture posterior at t=0.
            # Standard element reverse already ran above; now stochastically collapse
            # non-PAD chemistry to Carbon using the same cosine schedule as training.
            if self.late_element_resolution:
                t_frac = t_idx.float() / max(self.timesteps - 1, 1)
                t_chem_int = (t_frac / self.late_element_cutoff).clamp(0, 1) * (self.timesteps - 1)
                t_chem_int = t_chem_int.round().long().clamp(0, self.timesteps - 1)
                p_keep_chem = self.chem_retention[t_chem_int].item()
                if p_keep_chem < 1.0:
                    collapse_rand = torch.rand(element_types.shape, device=device)
                    should_collapse = collapse_rand >= p_keep_chem
                    is_non_pad = element_types > 0
                    element_types = torch.where(
                        should_collapse & is_non_pad,
                        torch.ones_like(element_types),  # C=1
                        element_types,
                    )
                # At final step: mixture posterior determines ghost/real
                if t_idx == 0:
                    learned_centroid = (
                        denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None
                    )
                    learned_cloud_logvar = (
                        denoiser_outputs.get("residue_cloud_logvar") if self.mixture_loss_weight > 0 else None
                    )
                    mix_post = self.compute_mixture_posterior(
                        noised_coords=x,
                        ca_coords=ca_coords,
                        t=t,
                        learned_centroid=learned_centroid,
                        learned_cloud_logvar=learned_cloud_logvar,
                        residue_lrt_delta=cached_lrt_delta,
                        temperature=mix_temperature,
                    )
                    final_mixture_posterior = mix_post
                    final_residue_centroid = learned_centroid
                    final_residue_cloud_logvar = learned_cloud_logvar
                    is_real = mix_post > 0.5
                    element_types = torch.where(
                        is_real,
                        element_types,  # Keep resolved chemistry for real slots
                        torch.zeros_like(element_types),  # PAD=0 for ghost
                    )

            # Freeze-count: at the specified step, snapshot the model's predicted count
            # and enforce as oracle mask for all remaining steps. Uses the model's LOGITS
            # (argmax != PAD) rather than hard element types, because absorbing diffusion
            # p_sample lags behind the logit predictions -- logits know which slots should
            # be non-PAD well before the actual types transition.
            if freeze_count_step >= 0 and step_i == freeze_count_step and oracle_sidechain_mask is None:
                if self.split_element_existence and self.existence_absorbing:
                    # Absorbing existence: REAL=1 is real, GHOST/MASK are ghost
                    frozen_mask = (existence == EXISTENCE_REAL).long()
                elif self.split_element_existence:
                    # Split mode: use existence variable directly (>= threshold = real)
                    frozen_mask = (existence >= self.existence_threshold).long()
                else:
                    # Use logit argmax: the model's best prediction of what each slot should be
                    pred_types = element_logits.argmax(dim=-1)  # (B, L, max_sc)
                    frozen_mask = ((pred_types != ELEMENT_PAD) & (pred_types != ELEMENT_MASK)).long()
                # Enforce prefix constraint on frozen mask
                prefix_mask, _ = frozen_mask.cummin(dim=-1)
                oracle_sidechain_mask = prefix_mask

            # Update mask from element types (skip in split mode -- already updated from existence)
            if not self.split_element_existence:
                noised_mask = (element_types != ELEMENT_PAD).float()

            # Update EVC conditioning from current element state (feedback loop).
            # Exclude MASK tokens -- only resolved non-PAD elements are "real".
            if evc_sampling is not None and self.split_element_existence and not self.existence_absorbing:
                # Split mode (continuous flow): use existence value directly (updated by flow step above)
                evc_sampling = existence.clone()
            elif evc_sampling is not None:
                # Distance-from-CA override: for early steps, use geometric signal
                # instead of (noisy) element predictions to avoid death spiral.
                step_frac = 1.0 - step_i / max(num_steps - 1, 1)  # 1.0 at first step -> 0.0 at last
                if evc_dist_override_frac > 0 and step_frac > (1.0 - evc_dist_override_frac):
                    # Use sigmoid of distance from CA: far atoms -> ~1.0 (real), close -> ~0.0 (ghost)
                    dist_from_ca = (x - ca_expanded).norm(dim=-1)  # (B, L, max_sc)
                    # Threshold at 2.0Å with sharpness 2.0: atoms >2Å from CA are likely real
                    evc_sampling = torch.sigmoid((dist_from_ca - 2.0) * 2.0)
                elif self.evc_soft_conditioning:
                    # Soft: use P(non-PAD) from element logits for smoother gradients
                    elem_probs = torch.softmax(element_logits, dim=-1)
                    evc_sampling = 1.0 - elem_probs[..., ELEMENT_PAD]
                    if self.num_element_classes > NUM_ELEMENT_TYPES:
                        evc_sampling = evc_sampling - elem_probs[..., ELEMENT_MASK]
                    evc_sampling = evc_sampling.clamp(0.0, 1.0)
                elif getattr(self, "evc_ss_noised_element_prob", 0.0) > 0:
                    # Honest 3-way read: REAL->1.0, MASK->0.5 (unknown/absorbing), PAD->0.0 (ghost). Matches
                    # the noised-element EVC training signal; MASK is NOT conflated with confirmed ghost.
                    evc_sampling = evc_from_element_state(element_types)
                else:
                    evc_sampling = ((element_types != ELEMENT_PAD) & (element_types != ELEMENT_MASK)).float()

            # Self-conditioning for next iteration
            prev_element_pred = torch.softmax(element_logits, dim=-1).detach()
            prev_cluster_pred = torch.softmax(cluster_logits, dim=-1).detach()
            if noised_cluster_ids is not None:
                noised_cluster_ids = cluster_logits.argmax(dim=-1)

            # SINGLE-SITE INFERENCE CONTEXT: snapshot THIS step's predicted x0 as the NEXT step's neighbour-
            # occupancy source. `x` here is still the x_t the denoiser consumed (the coord step below rebinds
            # it), and `model_output` its raw velocity, so `_x0_from_model_output` inverts to the flow's own x0
            # endpoint -- the SAME quantity the recycle path feeds the training single-site context, and NO extra
            # denoiser pass. Existence mask + element ids come from the current running discrete state
            # (`noised_mask` / `element_types`, already updated by the element reverse step above). Detached: a
            # frozen conditioning input, never a gradient path (sampling is no-grad anyway). Only when
            # single-site is on; off => never fires, no GT ever referenced.
            if (
                (_vol_consumer_active and self.use_single_site_context)
                # bond-inject needs this step's x0 as the NEXT step's recycles=1 fallback source .
                or self.use_bond_angle_deep_inject
            ):
                x0_prev_coords = self._x0_from_model_output(
                    model_output, x, t, ca_coords, mask_probs=noised_mask
                ).detach()
                x0_prev_mask = noised_mask.detach()
                x0_prev_element = element_types.detach()

            # Denoise step using either DDIM or DDPM
            if self.coord_process_type == "flow_matching":
                x_flat = x.view(batch_size, -1, 3)
                model_output_flat = model_output.view(batch_size, -1, 3)

                # Count-based ghost velocity override: blend learned velocity with analytical
                # ghost velocity (-> CA) based on predicted atom count per residue.
                # k controls sharpness: k=4 soft sigmoid, k=999 hard threshold.
                if count_ghost_velocity > 0 and cached_count_pred is not None:
                    v_ghost = self.coord_flow.analytical_ghost_velocity(x_flat, t, ca_coords)
                    count_pred = cached_count_pred.detach()  # (B, L)
                    slot_idx = torch.arange(max_sc, device=device).float().view(1, 1, -1)  # (1, 1, max_sc)
                    # P(real) per slot via sigmoid: smooth for small k, hard for large k
                    p_real_slot = torch.sigmoid(count_ghost_velocity * (count_pred.unsqueeze(-1) - slot_idx - 0.5))
                    # Reshape to flat atom dim: (B, L, max_sc) -> (B, L*max_sc, 1)
                    p_real_flat = p_real_slot.reshape(batch_size, -1, 1)
                    model_output_flat = p_real_flat * model_output_flat + (1.0 - p_real_flat) * v_ghost

                # EVC-gated velocity: blend learned velocity with ghost velocity using EVC state
                if evc_sampling is not None and getattr(self, "evc_velocity_blend", False):
                    v_ghost = self.coord_flow.analytical_ghost_velocity(x_flat, t, ca_coords)
                    p_real_flat = evc_sampling.reshape(batch_size, -1, 1)  # (B, L*max_sc, 1)
                    model_output_flat = p_real_flat * model_output_flat + (1.0 - p_real_flat) * v_ghost

                # Split velocity: blend learned v_real with analytical v_ghost
                if self.use_split_velocity or self.split_velocity_sampling_only:
                    v_ghost = self.coord_flow.analytical_ghost_velocity(x_flat, t, ca_coords)  # (B, num_atoms, 3)
                    # Get P(real) from mixture posterior (compute if not already available)
                    if final_mixture_posterior is not None:
                        sv_p_real = final_mixture_posterior
                    else:
                        sv_lc = denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None
                        sv_lv = denoiser_outputs.get("residue_cloud_logvar") if self.mixture_loss_weight > 0 else None
                        sv_p_real = self.compute_mixture_posterior(
                            noised_coords=x,
                            ca_coords=ca_coords,
                            t=t,
                            learned_centroid=sv_lc,
                            learned_cloud_logvar=sv_lv,
                            residue_lrt_delta=cached_lrt_delta,
                            residue_count_pred=cached_count_pred,
                            temperature=mix_temperature,
                        )
                    # Expand P(real) to match flat atom dim: (B, L, max_sc) -> (B, L*max_sc, 1)
                    sv_p_real_flat = sv_p_real.reshape(batch_size, -1, 1)
                    # Keep unblended v_real for final-step x0 inversion
                    sv_v_real_flat = model_output_flat
                    blended_v = sv_p_real_flat * model_output_flat + (1.0 - sv_p_real_flat) * v_ghost
                    model_output_flat = blended_v

                if t_idx > 0:
                    t_prev_idx = timesteps[step_i + 1]
                    t_prev = torch.full((batch_size,), t_prev_idx.item(), device=device, dtype=torch.long)
                    x = self.coord_flow.flow_step(x_flat, t, t_prev, model_output_flat).view_as(x)
                else:
                    if self.use_split_velocity or self.split_velocity_sampling_only:
                        # Final step: predict x0_real from v_real, then blend with CA for ghost slots.
                        # Cannot invert blended velocity because ghost and real components use
                        # different power schedules.
                        if self.use_split_velocity:
                            # Full split velocity: model was trained with all-real powers
                            sv_x0_mask_probs = torch.ones(batch_size, seq_len * max_sc, 1, device=device)
                        else:
                            # Sampling-only split: model was trained with standard per-slot powers
                            sv_x0_mask_probs = noised_mask.view(batch_size, -1, 1)
                        x0_real = self.coord_flow.predict_x0_from_velocity(
                            x_flat,
                            t,
                            sv_v_real_flat,
                            ca_coords=ca_coords,
                            mask_probs=sv_x0_mask_probs,
                        )
                        x0_ca = ca_expanded.reshape(batch_size, seq_len * max_sc, 3)
                        x = (sv_p_real_flat * x0_real + (1.0 - sv_p_real_flat) * x0_ca).view_as(x)
                    else:
                        x = self.coord_flow.predict_x0_from_velocity(
                            x_flat,
                            t,
                            model_output_flat,
                            ca_coords=ca_coords,
                            mask_probs=noised_mask.view(batch_size, -1, 1),
                        ).view_as(x)
            elif use_ddim:
                # DDIM sampling: deterministic when eta=0, avoids error accumulation
                # ddim_step handles prediction_type internally
                if t_idx > 0:
                    t_prev_idx = timesteps[step_i + 1]
                    t_prev = torch.full((batch_size,), t_prev_idx.item(), device=device, dtype=torch.long)
                    x = self.diffusion.ddim_step(
                        x.view(batch_size, -1, 3),
                        t,
                        t_prev,
                        model_output.view(batch_size, -1, 3),
                        eta=ddim_eta,
                        ca_coords=ca_coords,
                    ).view_as(x)
                else:
                    # Final step: just predict x_0 using appropriate method
                    if self.prediction_type == "v":
                        x = self.diffusion.predict_x0_from_v(
                            x.view(batch_size, -1, 3),
                            t,
                            model_output.view(batch_size, -1, 3),
                            ca_coords=ca_coords,
                        ).view_as(x)
                    elif self.prediction_type == "epsilon":
                        x = self.diffusion.predict_x0_from_noise(
                            x.view(batch_size, -1, 3),
                            t,
                            model_output.view(batch_size, -1, 3),
                            ca_coords=ca_coords,
                        ).view_as(x)
                    else:  # x0
                        x = model_output
            else:
                # DDPM sampling: stochastic posterior sampling
                # First, get x_0 prediction using appropriate method
                if self.prediction_type == "v":
                    x_0_pred = self.diffusion.predict_x0_from_v(
                        x.view(batch_size, -1, 3),
                        t,
                        model_output.view(batch_size, -1, 3),
                        ca_coords=ca_coords,
                    ).view_as(x)
                elif self.prediction_type == "epsilon":
                    x_0_pred = self.diffusion.predict_x0_from_noise(
                        x.view(batch_size, -1, 3),
                        t,
                        model_output.view(batch_size, -1, 3),
                        ca_coords=ca_coords,
                    ).view_as(x)
                else:  # x0
                    x_0_pred = model_output

                if t_idx > 0:
                    # CRITICAL: q_posterior_mean_variance must operate in the same space
                    # as q_sample. Coordinates are diffused CA-relatively, so computing the
                    # posterior directly in global coordinates introduces a CA drift term at
                    # high noise. Compute posterior in CA-relative space, then add CA back.
                    x_0_pred_rel = (x_0_pred - ca_expanded).view(batch_size, -1, 3)
                    x_rel = (x - ca_expanded).view(batch_size, -1, 3)
                    posterior_mean_rel, posterior_var, _ = self.diffusion.q_posterior_mean_variance(
                        x_0_pred_rel,
                        x_rel,
                        t,
                    )
                    posterior_mean_rel = posterior_mean_rel.view_as(x)
                    # Posterior variance must be scaled by noise_scale² since forward process uses noise * noise_scale
                    posterior_std = torch.sqrt(posterior_var).unsqueeze(-1) * self.diffusion.noise_scale

                    noise = torch.randn_like(x)
                    x_rel = posterior_mean_rel + posterior_std * noise
                    x = x_rel + ca_expanded
                else:
                    x = x_0_pred

            # Disc-guided count adjustment every k steps
            if disc_count_guidance is not None and (t_idx.item() % disc_count_guidance_k == 0):
                # Get predicted x0 coords for scoring
                if self.coord_process_type == "flow_matching":
                    x0_for_disc = self.coord_flow.predict_x0_from_velocity(
                        x.view(batch_size, -1, 3),
                        t,
                        model_output.view(batch_size, -1, 3),
                        ca_coords=ca_coords,
                        mask_probs=noised_mask.view(batch_size, -1, 1),
                    ).view(batch_size, seq_len, max_sc, 3)
                elif self.prediction_type == "v":
                    x0_for_disc = self.diffusion.predict_x0_from_v(
                        x.view(batch_size, -1, 3),
                        t,
                        model_output.view(batch_size, -1, 3),
                        ca_coords=ca_coords,
                    ).view(batch_size, seq_len, max_sc, 3)
                elif self.prediction_type == "epsilon":
                    x0_for_disc = self.diffusion.predict_x0_from_noise(
                        x.view(batch_size, -1, 3),
                        t,
                        model_output.view(batch_size, -1, 3),
                        ca_coords=ca_coords,
                    ).view(batch_size, seq_len, max_sc, 3)
                else:
                    x0_for_disc = model_output.view(batch_size, seq_len, max_sc, 3)

                # Pass ghost_weight so PAD slots contribute softly to NDM (matching training),
                # and element types so element-aware references can be used.
                disc_ghost_w = self.ghost_weight if self.ghost_weight > 0 else 0.1
                best_variant = disc_count_guidance.select_best_count(
                    x0_for_disc,
                    noised_mask,
                    seq_mask=seq_mask,
                    backbone_coords=backbone_coords,
                    backbone_mask=backbone_mask,
                    predicted_element_types=element_types - 1,  # model PAD=0,C=1.. -> disc C=0,N=1..
                    max_plus=self.multi_count_max_plus,
                    ghost_weight=disc_ghost_w,
                )
                # Apply: variant 0->delta=-1, 1->delta=0, 2->delta=+1, 3->delta=+2, etc.
                current_count = (element_types != ELEMENT_PAD).sum(dim=-1)  # (B, L)
                target_count = current_count + (best_variant - 1)  # variant index 0 maps to delta -1
                target_count = target_count.clamp(min=0, max=max_sc)
                # Adjust element types to match target count
                slot_indices = torch.arange(max_sc, device=device).view(1, 1, max_sc)
                new_mask = slot_indices < target_count.unsqueeze(-1)
                # For slots being added, use C (most common element)
                element_types = torch.where(
                    new_mask & (element_types == ELEMENT_PAD),
                    torch.ones_like(element_types),  # C=1
                    element_types,
                )
                # For slots being removed, set to PAD
                element_types = torch.where(~new_mask, torch.zeros_like(element_types), element_types)
                noised_mask = (element_types != ELEMENT_PAD).float()

            # --- Replacement-inpainting: re-pin non-designed positions to GT after each reverse update
            # (keeps the joint denoiser ON-manifold; the pre-loop pin already covered the first step's
            # denoiser, so every denoiser call sees the binder context) ---
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

            if return_coord_trajectory:  # frame per reverse step
                coord_traj.append(x.detach().to("cpu", torch.float32).clone())
                elem_traj.append(element_types.detach().to("cpu").clone())
                mask_traj.append((element_types != ELEMENT_PAD).detach().to("cpu").clone())

            # Track atom counts if requested
            if return_intermediates:
                if self.use_cluster_particle_diffusion and noised_cluster_ids is not None:
                    valid_seq_mask = (
                        seq_mask
                        if seq_mask is not None
                        else torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
                    )
                    cluster_expected_counts = cluster_occupancy_probs.sum(dim=-1)
                    cluster_expected_counts = cluster_expected_counts.sum(dim=-1).tolist()
                    unique_counts = []
                    for b in range(batch_size):
                        total_unique = 0
                        for seq_idx in range(seq_len):
                            if not valid_seq_mask[b, seq_idx]:
                                continue
                            total_unique += torch.unique(noised_cluster_ids[b, seq_idx]).numel()
                        unique_counts.append(float(total_unique))
                    intermediate_atom_counts.append(unique_counts)
                    intermediate_cluster_expected_counts.append(cluster_expected_counts)
                    intermediate_cluster_unique_counts.append(unique_counts)
                else:
                    # Count atoms per sample in current mask state: (B,)
                    current_counts = noised_mask.float().view(batch_size, -1).sum(dim=-1).tolist()
                    intermediate_atom_counts.append(current_counts)

                # Mixture-model soft count: sum of P(real|x_t) across all slots per sample.
                # This counts atoms by their likelihood of being in the non-ghost component,
                # independent of the discrete PAD/non-PAD element state.
                mix_post = self.compute_mixture_posterior(
                    noised_coords=x,
                    ca_coords=ca_coords,
                    t=t,
                    learned_centroid=denoiser_outputs.get("residue_centroid") if self.mixture_loss_weight > 0 else None,
                    learned_cloud_logvar=denoiser_outputs.get("residue_cloud_logvar")
                    if self.mixture_loss_weight > 0
                    else None,
                    residue_lrt_delta=cached_lrt_delta,
                    residue_count_pred=cached_count_pred,
                    temperature=mix_temperature,
                )  # (B, L, max_sc)
                if seq_mask is not None:
                    mix_post = mix_post * seq_mask.unsqueeze(-1).float()
                mix_counts = mix_post.view(batch_size, -1).sum(dim=-1).tolist()
                intermediate_mixture_counts.append(mix_counts)

                # Post-update hard count: non-PAD elements per residue -> (B, L)
                # In split mode, use noised_mask (from existence) rather than element_types != PAD
                if self.split_element_existence:
                    per_res_hard = noised_mask.sum(dim=-1)  # (B, L)
                    intermediate_per_res_hard.append(per_res_hard.detach().cpu())
                    intermediate_slot_non_pad_post.append(noised_mask.bool().detach().cpu())
                else:
                    per_res_hard = (element_types != ELEMENT_PAD).float().sum(dim=-1)  # (B, L)
                    intermediate_per_res_hard.append(per_res_hard.detach().cpu())
                    intermediate_slot_non_pad_post.append((element_types != ELEMENT_PAD).detach().cpu())

        # Collapse any remaining MASK tokens after sampling.
        # MASK should resolve to {PAD,C,N,O,S} during reverse, but if any linger:
        # use per-slot shell radii as threshold -- atoms within half the slot's shell
        # radius from CA are likely ghost (-> PAD), others are real (-> Carbon).
        # Skip in split mode: element types already converted to original encoding.
        if self.donut_element_init == "mask" and not self.split_element_existence:
            from .diffusion import ELEMENT_MASK

            is_mask = element_types == ELEMENT_MASK
            n_mask = is_mask.sum().item()
            n_total = element_types.numel()
            if is_mask.any():
                ca_exp = ca_coords.unsqueeze(2).expand_as(x)
                dist_to_ca = (x - ca_exp).norm(dim=-1)  # (B, L, max_sc)
                # Per-slot threshold: half the shell radius (atoms that collapsed toward CA are ghost),
                # with the FEATURE 4 distal cap. Sample site -> epoch=None -> fully-ramped cap; shares the
                # helper with the forward + sample-init sites so existence classification stays consistent.
                shell_radii = self.coord_flow._shell_radii  # (max_sc,)
                threshold = (
                    self._distal_read_threshold(shell_radii, epoch=None).view(1, 1, max_sc).expand_as(dist_to_ca)
                )
                collapse_to_pad = dist_to_ca < threshold
                n_to_pad = (is_mask & collapse_to_pad).sum().item()
                n_to_carbon = (is_mask & ~collapse_to_pad).sum().item()
                print(
                    f"  MASK collapse: {n_mask}/{n_total} ({100 * n_mask / n_total:.1f}%) remaining -> "
                    f"{n_to_pad} PAD + {n_to_carbon} Carbon"
                )
                element_types = torch.where(is_mask & collapse_to_pad, torch.zeros_like(element_types), element_types)
                element_types = torch.where(
                    is_mask & ~collapse_to_pad, torch.ones_like(element_types), element_types
                )  # Carbon=1
            else:
                print(f"  MASK collapse: 0/{n_total} remaining (all resolved)")

        # Predicted mask: derived from final element types (PAD=0 -> absent)
        predicted_mask = element_types != ELEMENT_PAD
        if self.use_cluster_particle_diffusion and noised_cluster_ids is not None:
            raw_particle_coords = x
            if cluster_occupancy_probs is None:
                raw_particle_mask = torch.ones_like(predicted_mask, dtype=torch.bool)
            else:
                active_cluster_mask = cluster_occupancy_probs > 0.5
                raw_particle_mask = torch.gather(active_cluster_mask, 2, noised_cluster_ids)
            merged_coords, merged_elements, merged_mask = self._merge_cluster_predictions(
                raw_particle_coords,
                element_types,
                raw_particle_mask,
                noised_cluster_ids,
            )
            element_types = torch.where(merged_mask, merged_elements, torch.full_like(merged_elements, -1))
            x = merged_coords * merged_mask.unsqueeze(-1).float()
            predicted_mask = merged_mask
            result = {
                "sidechain_coords": x,
                "element_types": element_types,
                "predicted_mask": predicted_mask,
                "raw_particle_coords": raw_particle_coords,
                "raw_particle_mask": raw_particle_mask,
                "predicted_cluster_ids": noised_cluster_ids,
            }
        else:
            # Capture un-zeroed final x0 BEFORE masking: ghost/PAD slots have flowed to ~Ca
            # here, but the next line zeroes them to the origin. raw_coords preserves the true
            # positions for the ghost/real -> Ca two-component-mixture diagnostic.
            raw_coords = x.clone()
            # PAD slots already have element_type=0; zero out their coords
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
        if noised_cluster_ids is not None and not self.use_cluster_particle_diffusion:
            result["predicted_cluster_ids"] = noised_cluster_ids

        # Add intermediate atom counts if requested
        if return_intermediates:
            result["intermediate_atom_counts"] = intermediate_atom_counts
            result["intermediate_mixture_counts"] = intermediate_mixture_counts
            if intermediate_cluster_expected_counts is not None:
                result["intermediate_cluster_expected_counts"] = intermediate_cluster_expected_counts
            if intermediate_cluster_unique_counts is not None:
                result["intermediate_cluster_unique_counts"] = intermediate_cluster_unique_counts
            # Per-residue trajectory data (list of (B, L) cpu tensors, one per step)
            if intermediate_per_res_soft_pre:
                result["intermediate_per_res_soft_pre"] = intermediate_per_res_soft_pre  # pre-update, consistent
                result["intermediate_per_res_hard"] = intermediate_per_res_hard  # post-update
                result["intermediate_per_res_ca_dist_real"] = intermediate_per_res_ca_dist_real  # p_real-weighted
                result["intermediate_per_res_ca_dist_ghost"] = intermediate_per_res_ca_dist_ghost  # (1-p_real)-weighted
            if intermediate_slot_pad_logit_pre:
                result["intermediate_slot_pad_logit_pre"] = intermediate_slot_pad_logit_pre
            if intermediate_slot_non_pad_post:
                result["intermediate_slot_non_pad_post"] = intermediate_slot_non_pad_post

        # Add final-step mixture head outputs for diagnostics
        # Always compute a one-shot mixture posterior on final coords for diagnostic use,
        # even when mixture gating was not active during sampling (runs without mixture_gate_weight).
        # If the model has trained mixture heads (mixture_loss_weight > 0), use the learned
        # centroid/logvar from the last denoiser call for a more accurate posterior.
        if return_intermediates and final_mixture_posterior is None:
            t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
            # Use learned params from last denoiser call if available
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
                residue_lrt_delta=cached_lrt_delta,
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

        # Add final existence values for diagnostics
        if existence is not None:
            result["existence"] = existence.detach()

        # Add cached LRT delta for sampling-time ranking diagnostic
        if cached_lrt_delta is not None:
            result["cached_lrt_delta"] = cached_lrt_delta.detach()

        return result
