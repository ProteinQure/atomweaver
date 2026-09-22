"""
Diffusion process utilities for SE(3)-equivariant denoising.

This module implements the forward diffusion process (adding noise) and provides
utilities for the reverse process (denoising) used in diffusion models.

Standard Diffusion
------------------
The standard DDPM formulation adds Gaussian noise to data x_0 according to:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

where epsilon ~ N(0, I) and alpha_bar_t is the cumulative product of (1 - beta_t).

For SE(3)-equivariant diffusion on coordinates, we apply this noise independently
to each coordinate dimension, which preserves equivariance since Gaussian noise
is isotropic.

Future Work: Groupwise / Hierarchical Diffusion
-----------------------------------------------
A key research direction is to use DIFFERENT noise schedules for different atoms
based on their distance from the backbone. The intuition is:

**Backbone-proximal atoms (Cb carbons):**
- Define the "broad direction" of the side chain
- Should be denoised FIRST (less noise in forward, emerge early in reverse)
- More constrained by backbone geometry, so resolve them first

**Distal atoms (side chain tips, aromatic rings, etc.):**
- Fine structural details
- Should be denoised LAST (more noise in forward, stay noisy longer in reverse)
- Depend on positions of proximal atoms, so resolve them after Cb is placed

This creates a coarse-to-fine hierarchy:
1. Early reverse steps: resolve coarse structure (Cb positions)
2. Late reverse steps: refine fine details (distal atoms conditioned on Cb)

Implementation approaches:

1. **Blockwise schedules**: Partition atoms into groups (e.g., by bond distance
   from backbone), use different beta schedules per group.

2. **Continuous depth-based schedules**: Define noise level as a function of
   atom "depth" d (bonds from backbone):
       beta_t(d) = beta_min + (beta_max - beta_min) * f(d, t)
   where f increases noise for distal atoms at early timesteps.

3. **Cascaded diffusion**: Two-stage model:
   - Stage 1: Predict Cβ positions given backbone
   - Stage 2: Predict full side chain conditioned on Cβ

4. **Learned schedules**: Let the model learn optimal per-atom noise levels
   via a small auxiliary network.

Advantages of hierarchical diffusion:
- Better tractability for atom-level clouds
- Physically motivated inductive bias
- Potentially faster convergence (easier problem at each stage)
- Natural curriculum learning

For now, we implement standard uniform diffusion. The infrastructure is designed
to be extensible to per-atom schedules via the `atom_depths` parameter.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models".

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    s : float
        Small offset to prevent beta from being too small at t=0.

    Returns
    -------
    betas : torch.Tensor
        Beta values of shape (timesteps,).
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear schedule for beta values.

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    beta_start : float
        Starting beta value.
    beta_end : float
        Ending beta value.

    Returns
    -------
    betas : torch.Tensor
        Beta values of shape (timesteps,).
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Quadratic schedule for beta values.

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    beta_start : float
        Starting beta value.
    beta_end : float
        Ending beta value.

    Returns
    -------
    betas : torch.Tensor
        Beta values of shape (timesteps,).
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


class GaussianDiffusion(nn.Module):
    """
    Gaussian diffusion process for coordinates.

    Implements the forward process (adding noise) and provides utilities for
    training and sampling.

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    schedule : str
        Noise schedule type: 'linear', 'cosine', or 'quadratic'.
    beta_start : float
        Starting beta for linear/quadratic schedules.
    beta_end : float
        Ending beta for linear/quadratic schedules.
    noise_scale : float
        Scale factor for noise in Ångströms. Multiplies the unit-variance noise
        to control the magnitude of coordinate perturbations. Default is 4.0Å.
    prediction_type : str
        Type of prediction target: 'epsilon' (noise), 'v' (velocity), or 'x0' (clean data).
        'v' is recommended for numerical stability at high noise levels.
    """

    def __init__(
        self,
        timesteps: int = 250,
        schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        noise_scale: float = 4.0,
        prediction_type: str = "v",
    ):
        super().__init__()

        if prediction_type not in ("epsilon", "v", "x0"):
            msg = f"Unknown prediction_type: {prediction_type}. Must be 'epsilon', 'v', or 'x0'."
            raise ValueError(msg)

        self.timesteps = timesteps
        self.noise_scale = noise_scale
        self.prediction_type = prediction_type

        # Compute beta schedule
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            msg = f"Unknown schedule: {schedule}"
            raise ValueError(msg)

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register as buffers (not parameters, but should move with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )


class GhostRealFlowMatching(nn.Module):
    """
    CA-centered flow-matching coordinate process for real and ghost atom slots.

    Two source modes:

    **Donut shell** (``use_donut_source=True``):
      Each atom slot samples from a per-slot radial shell ("donut") at its
      characteristic distance from CA. All slots within a residue share a single
      random direction, producing a coherent proto-sidechain at t=T. Small
      per-atom isotropic noise (proportional to shell radius) provides angular
      deviation. Power schedule is slot-only (no ghost-specific power), so no
      GT occupancy leaks into the noise schedule.

    **Isotropic Gaussian** (``use_donut_source=False``, legacy):
      All slots share isotropic Gaussian noise around CA. Ghost atoms may use
      a separate power schedule (``ghost_power``).
    """

    def __init__(
        self,
        timesteps: int = 250,
        noise_scale: float = 4.0,
        ghost_power: float = 2.0,
        real_proximal_power: float = 2.0,
        real_distal_power: float = 1.0,
        use_conditional_groupwise: bool = False,
        use_donut_source: bool = False,
        donut_thickness_ratio: float = 0.2,
        use_empirical_shell_thickness: bool = False,  # data-driven radial std sqrt(Var(||x-CA||)) instead of tau*r jitter
        occupancy_weighted_source: bool = False,
        max_sidechain_atoms: int = 14,
        distal_jitter_cap: float = 0.0,  # FEATURE 5: cap the absolute source jitter (tau*r) at this Angstrom; 0.0 = OFF (bit-exact)
        distal_shell_ramp_epochs: int = 0,  # shared distal-shell warmup window; 0 = instant/no-ramp
        shell_target_var_scale: float = 1.0,  # scale the EMPIRICAL per-slot radial variance (target shells) DOWN for tighter, less-overlapping shells; 1.0 = OFF (byte-identical)
    ):
        super().__init__()
        self.timesteps = timesteps
        self.noise_scale = noise_scale
        self.ghost_power = ghost_power
        self.real_proximal_power = real_proximal_power
        self.real_distal_power = real_distal_power
        self.use_conditional_groupwise = use_conditional_groupwise
        self.use_donut_source = use_donut_source
        self.donut_thickness_ratio = donut_thickness_ratio
        self.use_empirical_shell_thickness = use_empirical_shell_thickness
        self.occupancy_weighted_source = occupancy_weighted_source
        self.distal_jitter_cap = float(distal_jitter_cap)
        self.distal_shell_ramp_epochs = int(distal_shell_ramp_epochs)
        self._distal_ramp_epoch = 0  # updated each epoch via set_distal_shell_epoch (mirrors shell-interp)
        # Fail-loud: the scale multiplies a variance, so it must be a finite, strictly-positive number.
        # Validated at construction so every entry path (CLI, distributed/Ray, eval-loader) funnels through here.
        self.shell_target_var_scale = float(shell_target_var_scale)
        if not math.isfinite(self.shell_target_var_scale) or self.shell_target_var_scale <= 0.0:
            raise ValueError(f"shell_target_var_scale must be finite and > 0, got {shell_target_var_scale!r}")

        # Donut shell buffers -- populated by set_donut_shell_radii() after
        # dataset coord stats are loaded. Defaults are empirical RMS distances
        # from CA per slot (peptidempnn training set) so visualization / testing
        # without a checkpoint produces sensible geometry.
        _base_radii = torch.tensor(
            [1.67, 2.61, 3.40, 4.03, 4.89, 5.52, 6.08, 6.62, 6.37, 7.37, 8.34, 9.77, 8.82, 12.10]
        )
        # Size shell buffers to max_sidechain_atoms (NCAA/max16 needs >14). Extra slots get the last
        # empirical radius as a placeholder -- overridden by set_donut_shell_radii() from real coord stats.
        if max_sidechain_atoms <= _base_radii.numel():
            _default_radii = _base_radii[:max_sidechain_atoms].clone()
        else:
            _pad = _base_radii[-1:].repeat(max_sidechain_atoms - _base_radii.numel())
            _default_radii = torch.cat([_base_radii, _pad])
        self.register_buffer("_shell_radii", _default_radii)
        self.register_buffer("_shell_thickness", _default_radii * donut_thickness_ratio)
        self.register_buffer(
            "_shell_source_var", _default_radii**2 / 3.0 + (_default_radii * donut_thickness_ratio) ** 2
        )
        # Empirical per-slot radial variance Var(||x-CA||). Always present in the state_dict so a
        # checkpoint round-trips regardless of the flag. Default mirrors the fixed tau*r thickness
        # squared so a flag-OFF run's buffer is a faithful (unused) placeholder.
        self.register_buffer("_shell_radial_var", (_default_radii * donut_thickness_ratio) ** 2)

        # Per-slot occupancy rate from training set -- used for occupancy-weighted source.
        # When enabled, scales shell displacement by P(real|slot_j) so ghost-heavy slots
        # start near CA. Populated by set_slot_occupancy() at training startup.
        self.register_buffer("_slot_occupancy", torch.ones(max_sidechain_atoms))  # default: all-real (no effect)

    def _distal_ramp_progress(self) -> float:
        """0->1 warmup fraction for the distal-shell caps. 1.0 (fully ramped) when no window set."""
        ramp = int(getattr(self, "distal_shell_ramp_epochs", 0) or 0)
        if ramp <= 0:
            return 1.0
        return min(float(getattr(self, "_distal_ramp_epoch", 0)) / ramp, 1.0)

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        denom = max(self.timesteps - 1, 1)
        return (t.float() / denom).clamp(0.0, 1.0)

    def _slot_powers(
        self,
        num_atoms: int,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = ca_coords.shape
        atoms_per_residue = num_atoms // seq_len
        if not self.use_conditional_groupwise:
            slot_power = torch.full(
                (batch_size, num_atoms, 1),
                self.real_distal_power,
                device=ca_coords.device,
                dtype=ca_coords.dtype,
            )
        else:
            slot_idx = torch.arange(atoms_per_residue, device=ca_coords.device, dtype=ca_coords.dtype)
            depth = slot_idx / (atoms_per_residue - 1) if atoms_per_residue > 1 else torch.zeros_like(slot_idx)
            per_slot = self.real_proximal_power + depth * (self.real_distal_power - self.real_proximal_power)
            slot_power = per_slot.view(1, 1, atoms_per_residue, 1).expand(batch_size, seq_len, atoms_per_residue, 1)
            slot_power = slot_power.reshape(batch_size, num_atoms, 1)
        # Donut source: slot-only power, no ghost-specific schedule (avoids GT occupancy leak)
        if self.use_donut_source or mask_probs is None:
            return slot_power
        return mask_probs * slot_power + (1.0 - mask_probs) * self.ghost_power

    def _interp_terms(
        self,
        t: torch.Tensor,
        num_atoms: int,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tau = self._tau(t).view(-1, 1, 1)
        powers = self._slot_powers(num_atoms, ca_coords, mask_probs)
        s = tau.pow(powers)
        s_dot = powers * tau.clamp_min(1e-8).pow(powers - 1.0)
        s_dot = torch.where(tau > 0, s_dot, powers)
        return s, s_dot

    def sample_prior(
        self,
        x_shape: torch.Size,
        ca_coords: torch.Tensor,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        per_atom_mask: torch.Tensor | None = None,
        direction_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the source state at high noise.

        **Donut mode**: per-slot radial shells with shared direction per residue.
        **Legacy mode**: isotropic Gaussian around CA.

        Parameters
        ----------
        per_atom_mask : (B, N, 1) float, optional
            Per-atom scale for shell displacement. 1.0 = full shell (real), 0.0 = at CA (ghost).
            Used for GT count leak positive control.
        direction_override : (B, L, 3) unit vectors, optional
            Override the random donut direction with a specific direction per residue.
            Used for GT direction leak positive control.
        """
        batch_size, num_atoms, _ = x_shape
        num_residues = ca_coords.shape[1]
        atoms_per_residue = num_atoms // num_residues
        device, dtype = ca_coords.device, ca_coords.dtype

        if noise is None:
            noise = torch.randn(x_shape, device=device, dtype=dtype, generator=generator)

        ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1).reshape(batch_size, num_atoms, 3)

        if self.use_donut_source:
            if direction_override is not None:
                direction = direction_override  # (B, L, 3)
            else:
                # One shared random direction per residue (uniform on S²)
                dir_noise = torch.randn(batch_size, num_residues, 3, device=device, dtype=dtype, generator=generator)
                direction = torch.nn.functional.normalize(dir_noise, dim=-1)  # (B, L, 3)
            dir_expanded = (
                direction.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1).reshape(batch_size, num_atoms, 3)
            )  # (B, N, 3)

            # Per-slot shell radii and thickness, broadcast to (B, N, 1)
            radii = self._shell_radii[:atoms_per_residue].view(1, 1, atoms_per_residue, 1)
            radii = radii.expand(batch_size, num_residues, -1, -1).reshape(batch_size, num_atoms, 1)
            thickness = self._shell_thickness[:atoms_per_residue].view(1, 1, atoms_per_residue, 1)

            # FEATURE 5: distal source-jitter cap (warmup). The absolute jitter is ``thickness`` = tau*r
            # (per-slot), which blows up for distal slots and over-spreads the source -> ghost-collapse /
            # undercount. When ``distal_jitter_cap > 0`` we tighten the DISTAL tail only:
            # jitter(slot) = lerp(tau*r, min(tau*r, cap), ramp_progress)
            # Proximal slots (tau*r < cap) are untouched at every epoch (min = tau*r); distal slots
            # tighten from tau*r toward ``cap`` over the shared distal-shell ramp. Training uses the
            # epoch-aware ramp; eval/sampling (module not in train mode) uses the fully-ramped value
            # min(tau*r, cap) -- correct for any ep>=ramp_epochs checkpoint, matching the read-threshold
            # sample sites. Bit-exact no-op when cap == 0.0 (block skipped; no RNG touched either way).
            if self.distal_jitter_cap > 0.0:
                _prog = self._distal_ramp_progress() if self.training else 1.0
                _capped = torch.minimum(thickness, torch.full_like(thickness, self.distal_jitter_cap))
                thickness = torch.lerp(thickness, _capped, thickness.new_tensor(_prog))
            thickness = thickness.expand(batch_size, num_residues, -1, -1).reshape(batch_size, num_atoms, 1)

            shell_displacement = radii * dir_expanded + thickness * noise

            if per_atom_mask is not None:
                # GT count leak: real atoms get full shell, ghost atoms at CA
                source = ca_expanded + per_atom_mask * shell_displacement
            elif self.occupancy_weighted_source:
                # Occupancy-weighted source: scale displacement by P(real|slot_j).
                occ = self._slot_occupancy[:atoms_per_residue].view(1, 1, atoms_per_residue, 1)
                occ = occ.expand(batch_size, num_residues, -1, -1).reshape(batch_size, num_atoms, 1)
                source = ca_expanded + occ * shell_displacement
            else:
                source = ca_expanded + shell_displacement
        else:
            source = ca_expanded + self.noise_scale * noise

        return source, noise

    def predict_x0_from_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        velocity: torch.Tensor,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Recover x_0 from x_t and velocity for the linear interpolant.
        """
        s, s_dot = self._interp_terms(t, x_t.shape[1], ca_coords, mask_probs)
        ratio = s / s_dot.clamp_min(1e-8)
        return x_t - ratio * velocity

    def flow_step(
        self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """
        Take one reverse step from t to t_prev using an Euler update.
        """
        tau = self._tau(t)
        tau_prev = self._tau(t_prev)
        delta = (tau - tau_prev).view(-1, 1, 1)
        return x_t - delta * velocity

    def analytical_ghost_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ca_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Exact velocity field for ghost slots (target = CA).

        From the interpolant x_t = (1-s)*CA + s*source:
          source - CA = (x_t - CA) / s
          v_ghost = s_dot * (source - CA) = (s_dot / s) * (x_t - CA)

        **Donut mode**: uses per-slot powers (same schedule as forward process).
        **Legacy mode**: uses fixed ``ghost_power`` for all slots.
        """
        tau = self._tau(t).view(-1, 1, 1)  # (B, 1, 1)
        batch_size, num_atoms, _ = x_t.shape
        num_residues = ca_coords.shape[1]
        atoms_per_residue = num_atoms // num_residues

        if self.use_donut_source:
            # Per-slot powers (same as training forward -- no ghost leak)
            powers = self._slot_powers(num_atoms, ca_coords)  # (B, N, 1)
            s = tau.pow(powers)
            s_dot = powers * tau.clamp_min(1e-8).pow(powers - 1.0)
            s_dot = torch.where(tau > 0, s_dot, powers)
        else:
            s = tau.pow(self.ghost_power)
            s_dot = self.ghost_power * tau.clamp_min(1e-8).pow(self.ghost_power - 1.0)
            s_dot = torch.where(tau > 0, s_dot, torch.tensor(self.ghost_power, device=tau.device, dtype=tau.dtype))

        ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1).reshape(batch_size, num_atoms, 3)
        x_rel = x_t - ca_expanded
        return (s_dot / s.clamp_min(1e-8)) * x_rel


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.

    Embeds scalar timesteps into a high-dimensional space using sinusoidal
    functions, similar to positional encodings in transformers.

    Parameters
    ----------
    dim : int
        Output dimension of embeddings.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Parameters
        ----------
        t : torch.Tensor
            Timesteps of shape (batch_size,).

        Returns
        -------
        emb : torch.Tensor
            Embeddings of shape (batch_size, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding with MLP projection.

    Combines sinusoidal embeddings with learnable MLP layers.

    Parameters
    ----------
    dim : int
        Intermediate dimension for sinusoidal embeddings.
    out_dim : int
        Output dimension.
    """

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed timesteps.

        Parameters
        ----------
        t : torch.Tensor
            Timesteps of shape (batch_size,).

        Returns
        -------
        emb : torch.Tensor
            Embeddings of shape (batch_size, out_dim).
        """
        emb = self.sinusoidal(t)
        return self.mlp(emb)


# =============================================================================
# Discrete Diffusion for Element Types
# =============================================================================

# Element type constants (matching datasets.py ELEMENT_TO_TYPE)
# PAD=0 encodes "no atom" (replaces separate mask diffusion track)
#
# ELEMENT VOCAB KNOB (env ATOMWEAVER_ELEMENT_VOCAB) -- toggles the whole vocab5<->vocab12 ladder
# without editing constants or rebuilding logic:
# 5 = (PAD, C, N, O, S) + MASK -- vocab5 canonical / faithful replicate
# 12 = (PAD, C,N,O,S, P,F,Cl,Br,I,Se,B) + MASK -- NCAA-aware extended vocab (default)
# The 12-order (...,Br,I,Se,B) matches the v1 rotamer-DB encoding (DB = model-1), so no DB
# rebuild/migration is needed. ELEMENT_MASK = NUM_ELEMENT_TYPES (the post-vocab "unknown" token).
ELEMENT_VOCAB = 5  # The released model uses PAD, C, N, O, and X.

ELEMENT_PAD = 0
ELEMENT_C = 1
ELEMENT_N = 2
ELEMENT_O = 3
ELEMENT_S = 4
if ELEMENT_VOCAB >= 12:
    ELEMENT_P = 5
    ELEMENT_F = 6
    ELEMENT_CL = 7
    ELEMENT_BR = 8
    ELEMENT_I = 9
    ELEMENT_SE = 10
    ELEMENT_B = 11
NUM_ELEMENT_TYPES = ELEMENT_VOCAB  # 5 or 12 (incl PAD)
ELEMENT_MASK = NUM_ELEMENT_TYPES  # "Unknown" token for MASK init mode (never in GT)

# Split existence/element encoding: element space WITHOUT PAD.
# Used when split_element_existence=True. Existence (PAD vs non-PAD) is handled
# by a separate flow process; element diffusion only resolves elements + MASK.
SPLIT_ELEMENT_C = 0
SPLIT_ELEMENT_N = 1
SPLIT_ELEMENT_O = 2
SPLIT_ELEMENT_S = 3
if ELEMENT_VOCAB >= 12:
    SPLIT_ELEMENT_P = 4
    SPLIT_ELEMENT_F = 5
    SPLIT_ELEMENT_CL = 6
    SPLIT_ELEMENT_BR = 7
    SPLIT_ELEMENT_I = 8
    SPLIT_ELEMENT_SE = 9
    SPLIT_ELEMENT_B = 10
    SPLIT_ELEMENT_MASK = 11
    NUM_SPLIT_ELEMENT_TYPES = 12  # C, N, O, S, P, F, Cl, Br, I, Se, B, MASK
    # Among existing (non-PAD) atoms: C=75.7%, N=9.7%, O=13.6%, S=0.7%; halogens/I/Se/B near-zero.
    SPLIT_ELEMENT_DATA_PRIOR = torch.tensor(
        [0.750, 0.096, 0.135, 0.007, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003]
    )  # C, N, O, S, P, F, Cl, Br, I, Se, B (no MASK)
else:
    SPLIT_ELEMENT_MASK = 4
    NUM_SPLIT_ELEMENT_TYPES = 5  # C, N, O, S, MASK
    # Among existing (non-PAD) atoms: C=75.7%, N=9.7%, O=13.6%, S=0.7%
    SPLIT_ELEMENT_DATA_PRIOR = torch.tensor([0.757, 0.097, 0.136, 0.007])  # C, N, O, S (no MASK)

# Existence absorbing diffusion: 3-class {GHOST=0, REAL=1, MASK=2}
EXISTENCE_GHOST = 0
EXISTENCE_REAL = 1
EXISTENCE_MASK = 2
NUM_EXISTENCE_CLASSES = 3


# Default class weights/prior for element types - log-based for gentle rebalancing. Among existing atoms:
# C=75.7%, N=9.7%, O=13.6%, S=0.7% (slot 4 is S in plain vocab5, or X=non-CNO ~1.2% under vocab5-X).
# NOTE: under absorbing-mask (v60) these are OVERRIDDEN in models.py by (1/prior.clamp(0.01)).clamp(max=20) then
# mean-normalized -> they only feed the non-absorbing ClassWeighted path. 12-vocab adds rare P/F/Cl/Br/I/Se/B at 3.0.
if ELEMENT_VOCAB >= 12:
    DEFAULT_ELEMENT_CLASS_WEIGHTS = torch.tensor(
        [0.25, 0.50, 1.10, 0.98, 2.10, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00, 3.00]
    )  # PAD, C, N, O, S, P, F, Cl, Br, I, Se, B
    DEFAULT_ELEMENT_PRIOR = torch.tensor(
        [0.710, 0.218, 0.028, 0.039, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000]
    )  # PAD, C, N, O, S, P, F, Cl, Br, I, Se, B
else:
    # vocab5: slot 4 is "X" (= non-CNO: S+P+halogens+Se+B) under vocab5-X (was S-only). Values kept as trained.
    # IMPORTANT: under absorbing-mask (v60) these are NOT the effective weights -- models.py overrides class_weights
    # with (1/prior.clamp(min=0.01)).clamp(max=20) THEN mean-normalizes. X (prior<0.01) clamps at 20 then normalizes
    # to ~1.79 = the SAME effective weight as N/O/S (all rare classes clamp then normalize equally). So retuning slot 4
    # here is INERT for absorbing/v60; it only feeds ClassWeightedDiscreteDiffusion (if ever used, log-based X ~1.90).
    DEFAULT_ELEMENT_CLASS_WEIGHTS = torch.tensor([0.25, 0.50, 1.10, 0.98, 2.10])  # PAD, C, N, O, X
    DEFAULT_ELEMENT_PRIOR = torch.tensor([0.710, 0.220, 0.028, 0.039, 0.002])  # PAD, C, N, O, X


class ClassWeightedDiscreteDiffusion(nn.Module):
    """
    Class-weighted discrete diffusion for element types.

    Unlike absorbing-state diffusion, this allows transitions BETWEEN element
    types during both forward and reverse processes. The equilibrium distribution
    matches the empirical prior (mostly Carbon), making the model's job easier.

    Forward process (per-step transition):
        P(stay same) = 1 - β_t
        P(transition to class k) = β_t * π_k

    where π is the empirical prior distribution over element types.

    At equilibrium (t -> T), the distribution converges to π.

    Reverse process:
        Given model's prediction of x_0 probabilities, compute posterior
        q(x_{t-1} | x_t, x_0) and sample from it. This allows the model
        to CHANGE its element predictions at each timestep.

    Parameters
    ----------
    timesteps : int
        Number of diffusion timesteps.
    schedule : str
        Noise schedule type: 'linear', 'cosine', or 'quadratic'.
    num_classes : int
        Number of element classes (default 4: C, N, O, S).
    prior : torch.Tensor, optional
        Prior distribution over classes. If None, uses DEFAULT_ELEMENT_PRIOR.
    use_class_weights : bool
        If True, use inverse-frequency class weights in cross-entropy loss.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        num_classes: int = 4,
        prior: torch.Tensor | None = None,
        use_class_weights: bool = True,
        prior_schedule: str | None = None,
        bare_target: int = ELEMENT_PAD,
        decoupled_count: bool = False,
        count_ramp_threshold: float = 0.9,
        count_overdispersion: float = 1.5,
        max_sidechain_atoms: int = 14,
    ):
        super().__init__()

        self.timesteps = timesteps
        self.num_classes = num_classes
        self.num_tokens = num_classes  # No MASK token
        self.use_class_weights = use_class_weights
        self.prior_schedule = prior_schedule
        self.decoupled_count = decoupled_count
        self.count_ramp_threshold = count_ramp_threshold
        self.count_overdispersion = count_overdispersion
        self.max_sidechain_atoms = max_sidechain_atoms
        self.bare_target = bare_target

        # Prior distribution (equilibrium)
        if prior is None:
            prior = DEFAULT_ELEMENT_PRIOR[:num_classes].clone()
        prior = prior / prior.sum()  # Normalize
        self.register_buffer("prior", prior)

        # Non-PAD prior: for sampling random non-PAD elements
        non_pad_prior = prior.clone()
        non_pad_prior[ELEMENT_PAD] = 0.0
        if non_pad_prior.sum() > 0:
            non_pad_prior = non_pad_prior / non_pad_prior.sum()
        self.register_buffer("non_pad_prior", non_pad_prior)

        # Equilibrium count for decoupled count mode
        if decoupled_count:
            eq_count = max_sidechain_atoms * (1.0 - prior[ELEMENT_PAD].item())
            self.register_buffer("eq_count", torch.tensor(eq_count))
        else:
            self.register_buffer("eq_count", None)

        # Time-varying prior for "bare" mode: interpolates from data prior toward
        # a target class as t->T. Default target is PAD (bare backbone); when
        # bare_target=ELEMENT_C, drives toward all-Carbon instead.
        # At intermediate t, all transitions remain free (non-degenerate prior).
        if prior_schedule == "bare":
            bare_prior = torch.zeros(num_classes)
            bare_prior[bare_target] = 1.0  # Target class at t=T
            t_norm = torch.linspace(0, 1, timesteps)
            w = t_norm**2  # Quadratic ramp: gentle near t=0, steep near t=T
            # prior_t[t] = (1 - w[t]) * prior_data + w[t] * bare_prior
            prior_t = (1 - w).unsqueeze(1) * prior.unsqueeze(0) + w.unsqueeze(1) * bare_prior.unsqueeze(0)
            self.register_buffer("prior_t", prior_t)  # (T, num_classes)
        else:
            self.register_buffer("prior_t", None)

        # Class weights for cross-entropy loss
        if use_class_weights:
            # Inverse frequency weighting, capped to avoid instability
            weights = (1.0 / prior.clamp(min=0.01)).clamp(max=20.0)
            weights = weights / weights.mean()  # Normalize to mean=1
            self.register_buffer("class_weights", weights)
        else:
            self.register_buffer("class_weights", None)

        # Compute beta schedule
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        else:
            msg = f"Unknown schedule: {schedule}"
            raise ValueError(msg)

        self.register_buffer("betas", betas)

        # Compute cumulative retention (probability of staying at original value)
        # retention_t = prod_{i=0}^{t} (1 - beta_i)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # Also store alphas_cumprod_prev for reverse process
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

    def compute_slot_schedule(
        self,
        t: torch.Tensor,
        slot_powers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-slot (alpha_bar_t, alpha_bar_{t-1}, beta_t) for reverse process.

        Returns the three schedule values needed by q_posterior and p_sample.
        """
        tau = (t.float() / max(self.timesteps - 1, 1)).clamp(0.0, 1.0)  # (B,)
        tau = tau.view(-1, 1, 1)  # (B, 1, 1)
        effective_tau = tau.pow(slot_powers)  # (B, L, max_sc)
        effective_t = (effective_tau * (self.timesteps - 1)).round().long().clamp(0, self.timesteps - 1)

        # For t-1: use tau for (t-1) then apply power
        tau_prev = ((t.float() - 1).clamp(min=0) / max(self.timesteps - 1, 1)).clamp(0.0, 1.0)
        tau_prev = tau_prev.view(-1, 1, 1)
        effective_tau_prev = tau_prev.pow(slot_powers)
        effective_t_prev = (effective_tau_prev * (self.timesteps - 1)).round().long().clamp(0, self.timesteps - 1)

        alpha_bar_t = self.alphas_cumprod[effective_t]
        alpha_bar_t_prev = self.alphas_cumprod[effective_t_prev]
        # Beta: 1 - alpha_bar_t / alpha_bar_{t-1}, clamped for safety
        beta_t = (1.0 - alpha_bar_t / alpha_bar_t_prev.clamp(min=1e-8)).clamp(0.0, 0.999)

        return alpha_bar_t, alpha_bar_t_prev, beta_t

    def q_posterior(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        slot_alpha_bar: torch.Tensor | None = None,
        slot_alpha_bar_prev: torch.Tensor | None = None,
        slot_beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute posterior q(x_{t-1} | x_t, x_0) as a categorical distribution.

        Using Bayes' rule:
            q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

        Parameters
        ----------
        x_0 : torch.Tensor
            Predicted clean element types of shape (...).
        x_t : torch.Tensor
            Current noised element types of shape (...).
        t : torch.Tensor
            Current timesteps of shape (batch_size,).
        slot_alpha_bar : torch.Tensor, optional
            Per-slot alpha_bar_t matching x_0 shape. Overrides uniform schedule.
        slot_alpha_bar_prev : torch.Tensor, optional
            Per-slot alpha_bar_{t-1} matching x_0 shape.
        slot_beta : torch.Tensor, optional
            Per-slot beta_t matching x_0 shape.

        Returns
        -------
        posterior_probs : torch.Tensor
            Posterior probabilities of shape (..., num_classes).
        """
        # Get schedule values -- use per-slot overrides if provided
        if slot_alpha_bar is not None:
            alpha_bar_t = slot_alpha_bar
            alpha_bar_t_prev = slot_alpha_bar_prev
            beta_t = slot_beta
        else:
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_t_prev = self.alphas_cumprod_prev[t]
            beta_t = self.betas[t]
            # Reshape for broadcasting
            for _ in range(x_0.dim() - 1):
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)
                alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)
                beta_t = beta_t.unsqueeze(-1)

        # Get prior (time-varying or fixed), shaped for broadcasting
        if self.prior_t is not None:
            # Use prior_t[t] for this timestep: (batch, num_classes)
            prior_vec = self.prior_t[t]
            for _ in range(x_0.dim() - 1):
                prior_vec = prior_vec.unsqueeze(1)
            # prior_vec: (batch, 1, ..., num_classes) -- broadcasts with (..., num_classes)
        else:
            prior_vec = self.prior
            for _ in range(x_0.dim()):
                prior_vec = prior_vec.unsqueeze(0)
            # prior_vec: (1, 1, ..., num_classes)

        # Get prior (time-varying or fixed), shaped for broadcasting
        if self.prior_t is not None:
            # Use prior_t[t] for this timestep: (batch, num_classes)
            prior_vec = self.prior_t[t]
            for _ in range(x_0.dim() - 1):
                prior_vec = prior_vec.unsqueeze(1)
            # prior_vec: (batch, 1, ..., num_classes) -- broadcasts with (..., num_classes)
        else:
            prior_vec = self.prior
            for _ in range(x_0.dim()):
                prior_vec = prior_vec.unsqueeze(0)
            # prior_vec: (1, 1, ..., num_classes)

        # q(x_{t-1} | x_0): prob of being at each class at t-1 given x_0
        # P(x_{t-1} = k | x_0) = alpha_bar_{t-1} * 1[k = x_0] + (1 - alpha_bar_{t-1}) * prior[k]
        x_0_onehot = torch.nn.functional.one_hot(x_0.clamp(min=0), self.num_classes).float()
        q_t_minus_1_given_x0 = (
            alpha_bar_t_prev.unsqueeze(-1) * x_0_onehot + (1 - alpha_bar_t_prev.unsqueeze(-1)) * prior_vec
        )

        # q(x_t | x_{t-1}): transition from t-1 to t
        # P(x_t = j | x_{t-1} = k) = (1 - beta_t) * 1[j = k] + beta_t * prior[j]
        # We need P(x_t = x_t_observed | x_{t-1} = k) for each k
        x_t_onehot = torch.nn.functional.one_hot(x_t.clamp(min=0), self.num_classes).float()
        # prior[x_t]: sum out all classes except the observed x_t
        prior_at_xt = (prior_vec * x_t_onehot).sum(dim=-1, keepdim=True)  # (..., 1)
        q_xt_given_t_minus_1 = (1 - beta_t.unsqueeze(-1)) * x_t_onehot + beta_t.unsqueeze(-1) * prior_at_xt

        # Posterior: q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
        posterior_unnorm = q_xt_given_t_minus_1 * q_t_minus_1_given_x0
        return posterior_unnorm / (posterior_unnorm.sum(dim=-1, keepdim=True) + 1e-8)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        slot_alpha_bar: torch.Tensor | None = None,
        slot_alpha_bar_prev: torch.Tensor | None = None,
        slot_beta: torch.Tensor | None = None,
        posterior_pad_squash: float = 1.0,
        absorbing_mask: bool = False,
    ) -> torch.Tensor:
        """
        Reverse diffusion step: sample x_{t-1} given x_t and model's x_0 prediction.

        Parameters
        ----------
        x_t : torch.Tensor
            Current noised element types of shape (...).
        t : torch.Tensor
            Current timesteps of shape (batch_size,).
        x_0_logits : torch.Tensor
            Model's predicted logits for x_0 of shape (..., num_classes).
        mask : torch.Tensor, optional
            Mask for valid positions.
        generator : torch.Generator, optional
            Random generator for reproducibility.
        slot_alpha_bar : torch.Tensor, optional
            Per-slot alpha_bar_t for groupwise element schedules.
        slot_alpha_bar_prev : torch.Tensor, optional
            Per-slot alpha_bar_{t-1}.
        slot_beta : torch.Tensor, optional
            Per-slot beta_t.
        posterior_pad_squash : float
            Multiply posterior PAD probability by this factor before sampling.
            Values < 1 reduce PAD bias (e.g. 0.5 halves PAD probability).
            Default 1.0 (no modification).

        Returns
        -------
        x_t_minus_1 : torch.Tensor
            Sampled element types at t-1 of same shape as x_t.
        """
        # In decoupled count mode, zero out PAD logits so elements never transition to PAD.
        # PAD assignment is handled entirely by the multi-count discretization module.
        if self.decoupled_count:
            x_0_logits = x_0_logits.clone()
            x_0_logits[..., ELEMENT_PAD] = -1e9

        # Convert logits to predicted x_0
        x_0_probs = torch.softmax(x_0_logits, dim=-1)  # (..., num_classes)
        if temperature != 1.0:
            # Sample x_0 from tempered distribution instead of argmax
            tempered_probs = torch.softmax(x_0_logits / temperature, dim=-1)
            flat_tempered = tempered_probs.view(-1, self.num_classes)
            # Numerical stability: prevent zero-probability rows from crashing multinomial
            flat_tempered = flat_tempered.clamp(min=1e-10)
            flat_tempered = flat_tempered / flat_tempered.sum(dim=-1, keepdim=True)
            x_0_pred = torch.multinomial(flat_tempered, 1, generator=generator).view(x_t.shape)
        else:
            x_0_pred = x_0_probs.argmax(dim=-1)  # (...)

        # Compute posterior probabilities
        posterior_probs = self.q_posterior(
            x_0_pred,
            x_t,
            t,
            slot_alpha_bar=slot_alpha_bar,
            slot_alpha_bar_prev=slot_alpha_bar_prev,
            slot_beta=slot_beta,
        )  # (..., num_classes)

        # Posterior PAD squash: reduce PAD probability to counteract absorbing state bias
        if posterior_pad_squash != 1.0:
            posterior_probs = posterior_probs.clone()
            posterior_probs[..., ELEMENT_PAD] *= posterior_pad_squash
            posterior_probs = posterior_probs / (posterior_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Absorbing MASK: once resolved (x_t ≠ absorbing state), cannot return.
        # Zero out absorbing state probability for already-resolved slots, then renormalize.
        # Works for both standard mode (bare_target=ELEMENT_MASK=5) and split mode (bare_target=SPLIT_ELEMENT_MASK=4).
        if absorbing_mask and self.bare_target is not None:
            absorbing_class = self.bare_target
            already_resolved = x_t != absorbing_class  # (...) bool
            posterior_probs = posterior_probs.clone()
            posterior_probs[..., absorbing_class] = torch.where(
                already_resolved,
                torch.zeros_like(posterior_probs[..., absorbing_class]),
                posterior_probs[..., absorbing_class],
            )
            posterior_probs = posterior_probs / (posterior_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Sample from posterior
        flat_probs = posterior_probs.view(-1, self.num_classes)
        # Numerical stability: posterior can underflow to zero at high noise with time-varying prior
        flat_probs = flat_probs.clamp(min=1e-10)
        flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
        flat_samples = torch.multinomial(flat_probs, num_samples=1, generator=generator)
        x_t_minus_1 = flat_samples.view(x_t.shape)

        # At t=0, just use the argmax prediction (no noise)
        t_broadcast = t
        for _ in range(x_t.dim() - 1):
            t_broadcast = t_broadcast.unsqueeze(-1)
        x_t_minus_1 = torch.where(t_broadcast == 0, x_0_pred, x_t_minus_1)

        # Preserve padding and invalid positions
        x_t_minus_1 = torch.where(x_t == -1, x_t, x_t_minus_1)
        if mask is not None:
            x_t_minus_1 = torch.where(mask, x_t_minus_1, x_t)

        return x_t_minus_1


# =============================================================================
# Placeholder for future groupwise diffusion implementation
# =============================================================================
