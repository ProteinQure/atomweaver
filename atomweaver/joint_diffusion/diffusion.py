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
import os

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

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        ca_coords: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample x_t given x_0.

        Parameters
        ----------
        x_0 : torch.Tensor
            Clean data of shape (..., 3) or (..., num_atoms, 3).
        t : torch.Tensor
            Timesteps of shape (batch_size,).
        noise : torch.Tensor, optional
            Pre-sampled noise. If None, will sample from N(0, I).
        mask : torch.Tensor, optional
            Mask for valid positions. Noise is zeroed where mask is False.
        generator : torch.Generator, optional
            Random generator for reproducible noise sampling.
        ca_coords : torch.Tensor, optional
            CA coordinates for CA-relative noising. Shape (B, L, 3).
            If provided, noise is added relative to CA positions.

        Returns
        -------
        x_t : torch.Tensor
            Noised data of same shape as x_0.
        noise : torch.Tensor
            The noise that was added.
        """
        if noise is None:
            noise = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype, generator=generator)

        if mask is not None:
            # Zero out noise for invalid positions
            noise = noise * mask.unsqueeze(-1).float()

        # Get coefficients for this timestep
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # CA-relative diffusion: noise sidechains relative to their CA positions
        # This keeps atoms tethered to their residue at high noise instead of
        # scattering globally, making denoising easier and more physically realistic
        if ca_coords is not None:
            # x_0 shape: (B, L*max_sc, 3), ca_coords shape: (B, L, 3)
            # Need to expand CA to match sidechain atoms
            batch_size, num_atoms, _ = x_0.shape
            num_residues = ca_coords.shape[1]
            atoms_per_residue = num_atoms // num_residues

            # Expand CA: (B, L, 3) -> (B, L, 1, 3) -> (B, L, max_sc, 3) -> (B, L*max_sc, 3)
            ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
            ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)

            # Compute CA-relative coordinates
            x_0_relative = x_0 - ca_expanded

            # Add noise to CA-relative coordinates (scaled by noise_scale)
            x_t_relative = sqrt_alpha_cumprod * x_0_relative + sqrt_one_minus_alpha_cumprod * noise * self.noise_scale

            # Convert back to global coordinates
            x_t = x_t_relative + ca_expanded
        else:
            # Standard global diffusion (scaled by noise_scale)
            x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise * self.noise_scale

        return x_t, noise

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        Parameters
        ----------
        x_0 : torch.Tensor
            Predicted clean data.
        x_t : torch.Tensor
            Current noised data.
        t : torch.Tensor
            Current timestep.

        Returns
        -------
        posterior_mean : torch.Tensor
            Mean of posterior distribution.
        posterior_variance : torch.Tensor
            Variance of posterior distribution.
        posterior_log_variance : torch.Tensor
            Log variance of posterior distribution.
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def get_v_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        ca_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the v-prediction target: v = sqrt(ᾱ_t) * ε - sqrt(1-ᾱ_t) * x_0.

        V-prediction is numerically stable at all timesteps because recovering
        x_0 and ε from v only requires multiplication, not division.

        The target is NORMALIZED by noise_scale so the model outputs have
        approximately unit variance (matching typical NN initialization).
        The inverse operation in predict_x0_from_v/predict_noise_from_v
        multiplies by noise_scale to recover the correct scale.

        Parameters
        ----------
        x_0 : torch.Tensor
            Clean data (CA-relative if ca_coords provided).
        noise : torch.Tensor
            The noise ε ~ N(0, I) (unit variance, before noise_scale).
        t : torch.Tensor
            Timesteps.
        ca_coords : torch.Tensor, optional
            CA coordinates. If provided, x_0 should be in global coords and
            will be converted to CA-relative for v computation.

        Returns
        -------
        v : torch.Tensor
            V-prediction target (normalized to have ~unit variance).
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        if ca_coords is not None:
            # Convert x_0 to CA-relative
            batch_size, num_atoms, _ = x_0.shape
            num_residues = ca_coords.shape[1]
            atoms_per_residue = num_atoms // num_residues

            ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
            ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)
            x_0_relative = x_0 - ca_expanded

            # v = (sqrt(alpha_bar) * eps * sigma - sqrt(1-alpha_bar) * x_0_relative) / sigma
            # Normalized so model outputs have ~unit variance
            v = sqrt_alpha * noise * self.noise_scale - sqrt_one_minus_alpha * x_0_relative
            return v / self.noise_scale

        # Global diffusion
        v = sqrt_alpha * noise * self.noise_scale - sqrt_one_minus_alpha * x_0
        return v / self.noise_scale

    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v_pred: torch.Tensor,
        ca_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted v. Numerically stable at all timesteps.

        Formula: x̂_0 = sqrt(ᾱ_t) * x_t - sqrt(1-ᾱ_t) * v̂

        This only uses multiplication, avoiding the division by sqrt(ᾱ) that
        causes instability in ε-prediction at high noise levels.

        Parameters
        ----------
        x_t : torch.Tensor
            Noised data.
        t : torch.Tensor
            Timestep.
        v_pred : torch.Tensor
            Predicted v.
        ca_coords : torch.Tensor, optional
            CA coordinates for CA-relative diffusion.

        Returns
        -------
        x_0 : torch.Tensor
            Predicted clean data.
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # v_pred is normalized (divided by noise_scale during training)
        # so we multiply by noise_scale to recover the correct scale
        v_scaled = v_pred * self.noise_scale

        if ca_coords is not None:
            batch_size, num_atoms, _ = x_t.shape
            num_residues = ca_coords.shape[1]
            atoms_per_residue = num_atoms // num_residues

            ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
            ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)

            x_t_relative = x_t - ca_expanded
            # x̂_0_relative = sqrt(ᾱ) * x_t_relative - sqrt(1-ᾱ) * v̂_scaled
            x_0_relative = sqrt_alpha * x_t_relative - sqrt_one_minus_alpha * v_scaled
            return x_0_relative + ca_expanded

        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v_scaled

    def predict_noise_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v_pred: torch.Tensor,
        ca_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict noise ε from x_t and predicted v. For logging/debugging.

        Formula: ε̂ = sqrt(1-ᾱ_t) * x_t + sqrt(ᾱ_t) * v̂

        Parameters
        ----------
        x_t : torch.Tensor
            Noised data.
        t : torch.Tensor
            Timestep.
        v_pred : torch.Tensor
            Predicted v.
        ca_coords : torch.Tensor, optional
            CA coordinates for CA-relative diffusion.

        Returns
        -------
        noise : torch.Tensor
            Predicted noise (scaled by noise_scale).
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # v_pred is normalized, scale it back
        v_scaled = v_pred * self.noise_scale

        if ca_coords is not None:
            batch_size, num_atoms, _ = x_t.shape
            num_residues = ca_coords.shape[1]
            atoms_per_residue = num_atoms // num_residues

            ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
            ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)

            x_t_relative = x_t - ca_expanded
            # ε̂ = sqrt(1-ᾱ) * x_t_relative + sqrt(ᾱ) * v̂_scaled
            return sqrt_one_minus_alpha * x_t_relative + sqrt_alpha * v_scaled

        return sqrt_one_minus_alpha * x_t + sqrt_alpha * v_scaled

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        ca_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.

        Parameters
        ----------
        x_t : torch.Tensor
            Noised data.
        t : torch.Tensor
            Timestep.
        noise : torch.Tensor
            Predicted noise.
        ca_coords : torch.Tensor, optional
            CA coordinates for CA-relative diffusion. Shape (B, L, 3).
            If provided, assumes x_t was noised using CA-relative diffusion.

        Returns
        -------
        x_0 : torch.Tensor
            Predicted clean data.
        """
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if ca_coords is not None:
            # CA-relative diffusion: x_t = ca + sqrt(alpha_t)*(x_0 - ca) + sqrt(1-alpha_t)*epsilon*sigma
            # To recover: x_0 = ca + (x_t - ca - sqrt(1-alpha_t)*epsilon*sigma) / sqrt(alpha_t)
            batch_size, num_atoms, _ = x_t.shape
            num_residues = ca_coords.shape[1]
            atoms_per_residue = num_atoms // num_residues

            # Expand CA: (B, L, 3) -> (B, L*max_sc, 3)
            ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
            ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)

            x_t_relative = x_t - ca_expanded
            x_0_relative = (x_t_relative - sqrt_one_minus_alpha_cumprod * noise * self.noise_scale) / sqrt_alpha_cumprod
            return x_0_relative + ca_expanded
        # Global diffusion
        return (x_t - sqrt_one_minus_alpha_cumprod * noise * self.noise_scale) / sqrt_alpha_cumprod

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        model_output: torch.Tensor,
        eta: float = 0.0,
        ca_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single DDIM sampling step from t to t_prev.

        DDIM (Denoising Diffusion Implicit Models) allows deterministic sampling
        when eta=0, which avoids error accumulation from stochastic noise.

        Parameters
        ----------
        x_t : torch.Tensor
            Current noised data.
        t : torch.Tensor
            Current timestep.
        t_prev : torch.Tensor
            Previous timestep (t-1 for sequential, or larger jumps for accelerated).
        model_output : torch.Tensor
            Model prediction. Interpretation depends on self.prediction_type:
            - 'epsilon': predicted noise
            - 'v': predicted velocity
            - 'x0': predicted clean data
        eta : float
            Stochasticity parameter. eta=0 gives deterministic sampling (DDIM),
            eta=1 gives stochastic sampling (equivalent to DDPM).
        ca_coords : torch.Tensor, optional
            CA coordinates for CA-relative diffusion. Shape (B, L, 3).

        Returns
        -------
        x_prev : torch.Tensor
            Denoised data at timestep t_prev.
        """
        # Get x_0 and noise predictions based on prediction type
        if self.prediction_type == "v":
            # V-prediction: numerically stable at all timesteps
            x_0_pred = self.predict_x0_from_v(x_t, t, model_output, ca_coords=ca_coords)
            noise_pred = self.predict_noise_from_v(x_t, t, model_output, ca_coords=ca_coords)
        elif self.prediction_type == "epsilon":
            # Epsilon-prediction: can be unstable at high t
            x_0_pred = self.predict_x0_from_noise(x_t, t, model_output, ca_coords=ca_coords)
            noise_pred = model_output * self.noise_scale
        else:  # x0
            x_0_pred = model_output
            # Derive noise from x_0 and x_t (needed for DDIM update)
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            if ca_coords is not None:
                batch_size, num_atoms, _ = x_t.shape
                num_residues = ca_coords.shape[1]
                atoms_per_residue = num_atoms // num_residues
                ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
                ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)
                x_t_rel = x_t - ca_expanded
                x_0_rel = x_0_pred - ca_expanded
                noise_pred = (x_t_rel - sqrt_alpha * x_0_rel) / (sqrt_one_minus_alpha + 1e-8)
            else:
                noise_pred = (x_t - sqrt_alpha * x_0_pred) / (sqrt_one_minus_alpha + 1e-8)

        # Apply radial clamp to x_0_pred as safety net
        # Physical constraint: sidechain atoms within ~15Å of CA
        if ca_coords is not None:
            batch_size, num_atoms, _ = x_0_pred.shape
            num_residues = ca_coords.shape[1]
            atoms_per_residue = num_atoms // num_residues

            ca_expanded = ca_coords.unsqueeze(2).expand(-1, -1, atoms_per_residue, -1)
            ca_expanded = ca_expanded.reshape(batch_size, num_atoms, 3)

            x_0_relative = x_0_pred - ca_expanded
            radius = torch.norm(x_0_relative, dim=-1, keepdim=True)

            max_radius = 15.0  # Å - covers sidechains + NCAAs + margin
            scale = torch.clamp(max_radius / (radius + 1e-8), max=1.0)
            x_0_relative = x_0_relative * scale
            x_0_pred = x_0_relative + ca_expanded
        else:
            x_0_pred = x_0_pred.clamp(-100, 100)

        # Get alpha values
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        # Compute variance for stochastic sampling
        sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))

        # Compute x_{t-1} = sqrt(ᾱ_{t-1}) * x̂_0_rel + sqrt(1-ᾱ_{t-1}-σ²) * ε̂
        # CRITICAL: Must do DDIM update in CA-relative space!
        # x_0_pred is in global coords (includes CA), noise_pred is CA-relative.
        # Mixing them would cause coordinates to drift from CA at high noise.
        sqrt_alpha_prev = torch.sqrt(alpha_prev)
        sqrt_one_minus_alpha_prev_minus_sigma = torch.sqrt(1.0 - alpha_prev - sigma_t**2)

        if ca_coords is not None:
            # Convert x_0_pred to CA-relative (ca_expanded already computed above)
            x_0_pred_rel = x_0_pred - ca_expanded
            # noise_pred is already CA-relative from predict_noise_from_v
            x_prev_rel = sqrt_alpha_prev * x_0_pred_rel + sqrt_one_minus_alpha_prev_minus_sigma * noise_pred

            # Add stochastic component if eta > 0
            if eta > 0:
                noise = torch.randn_like(x_t) * self.noise_scale
                x_prev_rel = x_prev_rel + sigma_t * noise

            x_prev = x_prev_rel + ca_expanded
        else:
            x_prev = sqrt_alpha_prev * x_0_pred + sqrt_one_minus_alpha_prev_minus_sigma * noise_pred

            # Add stochastic component if eta > 0
            if eta > 0:
                noise = torch.randn_like(x_t) * self.noise_scale
                x_prev = x_prev + sigma_t * noise

        return x_prev

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract coefficients at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))


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

    def set_donut_shell_radii(
        self,
        slot_coord_var: torch.Tensor,
        slot_coord_mean: torch.Tensor,
        slot_radial_var: torch.Tensor | None = None,
    ) -> None:
        """Compute shell radii from dataset per-slot coordinate statistics.

        Parameters
        ----------
        slot_coord_var : (max_sc,) isotropic variance of (x0 - CA) per slot
        slot_coord_mean : (max_sc, 3) mean offset from CA per slot
        slot_radial_var : (max_sc,) optional, empirical Var(||x0 - CA||) per slot.
            When ``use_empirical_shell_thickness`` is set and this is supplied, the
            per-axis source jitter std becomes the data-driven radial std sqrt(Var(||x-CA||))
            instead of the fixed ``donut_thickness_ratio * r``.
        """
        # Single ingestion point for --shell-target-var-scale: scale the EMPIRICAL per-slot radial
        # variance (the TARGET data's shell thickness) DOWN before it is used. This is the ONLY place
        # the target radial variance enters the model state (_shell_radial_var / _shell_thickness), and
        # the shell-interpolation TARGET is derived from that post-calibration state, so scaling here
        # also tightens the interp target -- while the resumed PRETRAIN variance stays untouched. The
        # radii (slot_coord_var / slot_coord_mean) are NOT scaled; only the radial variance moves.
        # scale == 1.0 multiplies by exactly 1.0 (numerically identical / byte-identical no-op).
        if slot_radial_var is not None:
            slot_radial_var = slot_radial_var * self.shell_target_var_scale
        r = (slot_coord_var + (slot_coord_mean**2).sum(dim=-1)).sqrt()  # RMS distance from CA
        n = min(len(r), len(self._shell_radii))
        self._shell_radii[:n] = r[:n]
        if self.use_empirical_shell_thickness and slot_radial_var is not None:
            self._shell_radial_var[:n] = slot_radial_var[:n]
            self._shell_thickness[:n] = slot_radial_var[:n].clamp(min=0.01).sqrt()  # empirical radial std
        else:
            self._shell_thickness[:n] = self.donut_thickness_ratio * r[:n]  # current behavior (tau*r)
        self._shell_source_var[:n] = r[:n] ** 2 / 3.0 + self._shell_thickness[:n] ** 2

    def setup_shell_interpolation(
        self,
        pretrain_radii: torch.Tensor,
        target_radii: torch.Tensor,
        warmup_epochs: int = 12,
        n_target_slots: int | None = None,
        pretrain_radial_var: torch.Tensor | None = None,
        target_radial_var: torch.Tensor | None = None,
    ) -> None:
        """Configure gradual shell radii transition from pretrained to target.

        When ``use_empirical_shell_thickness`` is set and both radial-var tensors are
        supplied, the per-slot radial variance is interpolated alongside the radii so the
        empirical source jitter also warms up from pretrain to target.
        """
        self.register_buffer("_pretrain_radii", pretrain_radii.clone())
        self.register_buffer("_target_radii", target_radii.clone())
        self._shell_warmup_epochs = warmup_epochs
        self._n_target_slots = n_target_slots or len(target_radii)
        if pretrain_radial_var is not None and target_radial_var is not None:
            self.register_buffer("_pretrain_radial_var", pretrain_radial_var.clone())
            self.register_buffer("_target_radial_var", target_radial_var.clone())

    def update_shell_interpolation(self, current_epoch: int) -> None:
        """Update shell radii based on epoch progress during fine-tuning."""
        if not hasattr(self, "_pretrain_radii"):
            return  # No interpolation configured

        alpha = min(1.0, current_epoch / max(1, self._shell_warmup_epochs))
        n_tgt = self._n_target_slots
        max_sc = len(self._shell_radii)

        # Slots 0..n_tgt-1: linear interpolation
        for k in range(min(n_tgt, max_sc)):
            self._shell_radii[k] = (1 - alpha) * self._pretrain_radii[k] + alpha * self._target_radii[k]

        # Slots n_tgt..max_sc-1: proportional scaling based on last target slot's ratio
        if n_tgt < max_sc and n_tgt > 0:
            pretrain_last = self._pretrain_radii[n_tgt - 1].clamp(min=1e-6)
            target_last = self._target_radii[n_tgt - 1] if n_tgt - 1 < len(self._target_radii) else pretrain_last
            ratio = target_last / pretrain_last
            for k in range(n_tgt, max_sc):
                scaled = self._pretrain_radii[k] * ratio
                self._shell_radii[k] = (1 - alpha) * self._pretrain_radii[k] + alpha * scaled

        # Update thickness and source variance to match new radii.
        if self.use_empirical_shell_thickness and hasattr(self, "_target_radial_var"):
            # Interpolate the empirical radial variance with the SAME alpha/scaling structure as radii.
            for k in range(min(n_tgt, max_sc)):
                self._shell_radial_var[k] = (1 - alpha) * self._pretrain_radial_var[
                    k
                ] + alpha * self._target_radial_var[k]
            if n_tgt < max_sc and n_tgt > 0:
                pretrain_last = self._pretrain_radial_var[n_tgt - 1].clamp(min=1e-6)
                target_last = (
                    self._target_radial_var[n_tgt - 1] if n_tgt - 1 < len(self._target_radial_var) else pretrain_last
                )
                ratio = target_last / pretrain_last
                for k in range(n_tgt, max_sc):
                    scaled = self._pretrain_radial_var[k] * ratio
                    self._shell_radial_var[k] = (1 - alpha) * self._pretrain_radial_var[k] + alpha * scaled
            self._shell_thickness[:] = self._shell_radial_var.clamp(min=0.01).sqrt()  # empirical radial std
        else:
            self._shell_thickness[:] = self.donut_thickness_ratio * self._shell_radii  # current behavior (tau*r)
        self._shell_source_var[:] = self._shell_radii**2 / 3.0 + self._shell_thickness**2

    def set_distal_shell_epoch(self, current_epoch: int) -> None:
        """Record the current (offset-adjusted) epoch for the distal source-jitter warmup.

        Mirrors the ``update_shell_interpolation`` plumbing: called once per epoch from the
        LightningModule's ``on_train_epoch_start``. Only used when ``distal_jitter_cap > 0``.
        """
        self._distal_ramp_epoch = int(current_epoch)

    def _distal_ramp_progress(self) -> float:
        """0->1 warmup fraction for the distal-shell caps. 1.0 (fully ramped) when no window set."""
        ramp = int(getattr(self, "distal_shell_ramp_epochs", 0) or 0)
        if ramp <= 0:
            return 1.0
        return min(float(getattr(self, "_distal_ramp_epoch", 0)) / ramp, 1.0)

    def set_slot_occupancy(self, slot_occupancy: torch.Tensor) -> None:
        """Set per-slot occupancy rates for occupancy-weighted source.

        Parameters
        ----------
        slot_occupancy : (max_sc,) P(real) for each slot position from training data
        """
        n = min(len(slot_occupancy), len(self._slot_occupancy))
        self._slot_occupancy[:n] = slot_occupancy[:n].clamp(0.0, 1.0)

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

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        per_atom_mask: torch.Tensor | None = None,
        direction_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an interpolated state x_t and return its source state and unit noise.
        """
        source, noise = self.sample_prior(
            x_0.shape,
            ca_coords=ca_coords,
            noise=noise,
            generator=generator,
            per_atom_mask=per_atom_mask,
            direction_override=direction_override,
        )
        s, _ = self._interp_terms(t, x_0.shape[1], ca_coords, mask_probs)
        x_t = (1.0 - s) * x_0 + s * source
        return x_t, source, noise

    def get_velocity_target(
        self,
        x_0: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the constant target velocity for the linear interpolant.
        """
        _, s_dot = self._interp_terms(t, x_0.shape[1], ca_coords, mask_probs)
        return s_dot * (source - x_0)

    def recompute_target_for_corrupted_xt(
        self,
        x_0: torch.Tensor,
        x_t_corrupted: torch.Tensor,
        t: torch.Tensor,
        ca_coords: torch.Tensor,
        mask_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recompute the flow-matching velocity target after x_t was corrupted by an arbitrary map.

        The linear interpolant is ``x_t = (1 - s) * x_0 + s * x_1`` with ``s = s(t)`` from
        :meth:`_interp_terms`, and the velocity target is ``v = s_dot * (x_1 - x_0)``
        (:meth:`get_velocity_target`). A training-time *coordinate corruption* replaces the noised
        state ``x_t`` with ``x_t'`` while the true clean endpoint ``x_0`` is unchanged. Leaving the
        target built from the ORIGINAL source ``x_1`` is a bug: the process x0-inverse
        (:meth:`predict_x0_from_velocity`) applied to ``(x_t', v_stale)`` reconstructs
        ``x_0 + s * (x_1' - x_1)`` -- i.e. the model learns to PRESERVE the corruption, not undo it.

        This method rebuilds the target so that the x0-inverse of ``(x_t', v)`` returns ``x_0``. It
        inverts the interpolant for the *implied corrupted source*

            x_1' = (x_t' - (1 - s) * x_0) / s,

        then returns ``get_velocity_target(x_0, x_1')`` == ``s_dot * (x_1' - x_0)``. The derivation
        reads only ``x_t'``, ``x_0`` and ``t``, so it is GENERAL over any coordinate corruption of
        ``x_t`` (rotation, mirror/reflection, in-plane shift, compression, ...).

        Note the singularity: ``s -> 0`` as ``t -> 0``, so the ``1/s`` inversion blows up near
        ``t = 0``. The caller MUST gate corruption away from small ``t`` (see
        ``rotation_corruption_min_tau`` in the model); ``s`` is clamped here only as a defensive
        numeric guard, not as a substitute for that gate.

        Parameters
        ----------
        x_0 : (B, N, 3) true clean endpoint (side-chain x0), flattened over atoms.
        x_t_corrupted : (B, N, 3) the corrupted noised state x_t'.
        t : (B,) integer timesteps.
        ca_coords : (B, L, 3) CA positions (drives the per-slot power schedule).
        mask_probs : (B, N, 1) optional per-slot real/ghost mix -- MUST match what the paired
            :meth:`get_velocity_target` uses so ``s``/``s_dot`` are consistent.
        """
        s, s_dot = self._interp_terms(t, x_0.shape[1], ca_coords, mask_probs)
        implied_source = (x_t_corrupted - (1.0 - s) * x_0) / s.clamp_min(1e-6)
        return s_dot * (implied_source - x_0)

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


class ExistenceFlowMatching(nn.Module):
    """
    Flow-matched existence process for per-slot ghost/real determination.

    Each atom slot has a scalar existence variable e ∈ [0,1]:
    - e=0 means ghost/PAD (slot is empty)
    - e=1 means real (slot has an atom)

    The flow linearly interpolates between the clean existence (0 or 1 from GT)
    and a source at 0.5 (maximum entropy). This eliminates the all-PAD cold start
    problem of absorbing-state diffusion.

    Parameters
    ----------
    timesteps : int
        Total diffusion timesteps.
    source_value : float
        Source existence value at t=T. Default 0.5 (maximum entropy).
    power : float
        Schedule power. Higher values keep existence near source longer.
    """

    def __init__(self, timesteps: int = 250, source_value: float = 0.5, power: float = 1.0):
        super().__init__()
        self.timesteps = timesteps
        self.source_value = source_value
        self.power = power

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        denom = max(self.timesteps - 1, 1)
        return (t.float() / denom).clamp(0.0, 1.0)

    def _interp_terms(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (s, s_dot) for the power schedule."""
        tau = self._tau(t).view(-1, 1, 1)  # (B, 1, 1)
        s = tau.pow(self.power)
        s_dot = self.power * tau.clamp_min(1e-8).pow(self.power - 1.0)
        s_dot = torch.where(tau > 0, s_dot, torch.full_like(s_dot, self.power))
        return s, s_dot

    def q_sample(self, e_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Noise existence values from clean to interpolated state.

        Parameters
        ----------
        e_0 : torch.Tensor
            Clean existence values (0 or 1) of shape (B, N_atoms, 1) or (B, L, max_sc, 1).
        t : torch.Tensor
            Timesteps of shape (B,).

        Returns
        -------
        e_t : torch.Tensor
            Noised existence values.
        """
        s, _ = self._interp_terms(t)
        # Expand s to match e_0 shape (handles both (B, N, 1) and (B, L, S, 1))
        while s.dim() < e_0.dim():
            s = s.unsqueeze(-1)
        source = torch.full_like(e_0, self.source_value)
        return (1.0 - s) * e_0 + s * source

    def get_velocity_target(self, e_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Velocity target for the existence flow: de/dt = s_dot * (source - e_0).
        """
        _, s_dot = self._interp_terms(t)
        while s_dot.dim() < e_0.dim():
            s_dot = s_dot.unsqueeze(-1)
        source = torch.full_like(e_0, self.source_value)
        return s_dot * (source - e_0)

    def predict_e0_from_velocity(self, e_t: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """Recover clean existence from current state and predicted velocity."""
        s, s_dot = self._interp_terms(t)
        while s.dim() < e_t.dim():
            s = s.unsqueeze(-1)
            s_dot = s_dot.unsqueeze(-1)
        ratio = s / s_dot.clamp_min(1e-8)
        return (e_t - ratio * v_pred).clamp(0.0, 1.0)

    def flow_step(self, e_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """Take one reverse Euler step from t to t_prev."""
        tau = self._tau(t)
        tau_prev = self._tau(t_prev)
        delta = (tau - tau_prev).view(-1, 1, 1)
        while delta.dim() < e_t.dim():
            delta = delta.unsqueeze(-1)
        return (e_t - delta * v_pred).clamp(0.0, 1.0)


class ElementFlowMatching(nn.Module):
    """
    Flow matching on the probability simplex for discrete element types.

    Replaces absorbing diffusion with continuous interpolation between one-hot
    ground truth and a prior distribution (e.g. all-MASK or dataset prior).

    Forward: p_t = (1-tau)*one_hot(x_0) + tau*prior, where tau=0 is clean, tau=1 is noise.
    Reverse: given model's predicted clean distribution, jump to the OT path at t_prev.
    """

    def __init__(self, timesteps: int, num_classes: int, prior: torch.Tensor):
        super().__init__()
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.register_buffer("prior", prior / prior.sum())

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() / max(self.timesteps - 1, 1)).clamp(0.0, 1.0)

    def _apply_slot_powers(self, tau: torch.Tensor, slot_powers: torch.Tensor | None) -> torch.Tensor:
        """Warp tau per-slot: tau_eff = tau^power. Higher power = stays clean longer."""
        if slot_powers is None:
            return tau
        # tau is (B, 1, 1, 1), slot_powers is (B, L, max_sc)
        return tau.squeeze(-1).pow(slot_powers).unsqueeze(-1)  # (B, L, max_sc, 1)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, slot_powers: torch.Tensor | None = None) -> torch.Tensor:
        """Forward: interpolate one_hot(x_0) toward prior. Returns soft probs (B, L, max_sc, C)."""
        tau = self._tau(t).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        tau = self._apply_slot_powers(tau, slot_powers)
        one_hot = torch.nn.functional.one_hot(x_0.clamp(min=0), self.num_classes).float()
        prior = self.prior.view(1, 1, 1, -1)
        return (1.0 - tau) * one_hot + tau * prior

    def reverse_step(
        self,
        p_t: torch.Tensor,  # noqa: ARG002
        t: torch.Tensor,  # noqa: ARG002
        t_prev: torch.Tensor,
        element_logits: torch.Tensor,
        slot_powers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OT-path jump: given predicted clean, re-evaluate path at t_prev."""
        tau_prev = self._tau(t_prev).view(-1, 1, 1, 1)
        tau_prev = self._apply_slot_powers(tau_prev, slot_powers)
        pred_clean = torch.softmax(element_logits, dim=-1)
        prior = self.prior.view(1, 1, 1, -1)
        p_new = (1.0 - tau_prev) * pred_clean + tau_prev * prior
        # Project to valid simplex
        p_new = p_new.clamp(min=0.0)
        return p_new / p_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)


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
ELEMENT_VOCAB = int(os.environ.get("ATOMWEAVER_ELEMENT_VOCAB", "12"))
if ELEMENT_VOCAB not in (5, 12):
    raise ValueError(f"ATOMWEAVER_ELEMENT_VOCAB must be 5 or 12, got {ELEMENT_VOCAB}")

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

    def compute_slot_alpha_bar(
        self,
        t: torch.Tensor,
        slot_powers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-slot cumulative retention probability using power-warped timesteps.

        Maps each slot's power to an effective timestep: t_eff = round(tau^power * T),
        then looks up alphas_cumprod at that effective time. Higher power = element stays
        original longer (slower corruption), matching the coordinate flow convention.

        Parameters
        ----------
        t : torch.Tensor
            Timesteps of shape (batch_size,).
        slot_powers : torch.Tensor
            Per-slot powers of shape (batch_size, seq_len, max_sc).

        Returns
        -------
        slot_alpha_bar : torch.Tensor
            Per-slot retention probabilities of shape (batch_size, seq_len, max_sc).
        """
        tau = (t.float() / max(self.timesteps - 1, 1)).clamp(0.0, 1.0)  # (B,)
        tau = tau.view(-1, 1, 1)  # (B, 1, 1)
        # Effective tau per slot: tau^power
        effective_tau = tau.pow(slot_powers)  # (B, L, max_sc)
        # Map back to integer timestep for schedule lookup
        effective_t = (effective_tau * (self.timesteps - 1)).round().long().clamp(0, self.timesteps - 1)
        return self.alphas_cumprod[effective_t]  # (B, L, max_sc)

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

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        slot_alpha_bar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t given x_0.

        Each element either stays at its original value (with prob alpha_bar_t)
        or transitions to a random element sampled from the prior.

        Parameters
        ----------
        x_0 : torch.Tensor
            Clean element types of shape (...). Values in [0, num_classes-1].
            Value -1 indicates padding.
        t : torch.Tensor
            Timesteps of shape (batch_size,).
        mask : torch.Tensor, optional
            Mask for valid positions. Invalid positions are not corrupted.
        generator : torch.Generator, optional
            Random generator for reproducibility.
        slot_alpha_bar : torch.Tensor, optional
            Per-slot retention probabilities of shape matching x_0.
            When provided, overrides the uniform alpha_bar_t schedule.

        Returns
        -------
        x_t : torch.Tensor
            Noised element types of same shape as x_0.
        """
        # Get alpha_bar_t for each sample in batch
        alpha_bar_t = self.alphas_cumprod[t]  # (batch_size,)

        if self.decoupled_count and x_0.dim() == 3:
            # Decoupled count mode: separate atom count noise from element identity noise.
            # 1. Sample count_t from NegBin(mu, phi) -- controls how many slots are active
            # 2. Active slots get element noise among {C, N, O, S} only (no PAD transitions)
            # 3. Inactive slots are forced to PAD
            return self._q_sample_decoupled(x_0, t, alpha_bar_t, mask, generator)

        # Use per-slot alpha_bar if provided, otherwise broadcast uniform
        if slot_alpha_bar is not None:
            alpha_bar_t = slot_alpha_bar
        else:
            for _ in range(x_0.dim() - 1):
                alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        # Sample uniform random for stay/transition decision
        rand = torch.rand(x_0.shape, device=x_0.device, dtype=torch.float32, generator=generator)

        # Get prior for transition destinations (time-varying or fixed)
        if self.prior_t is not None:
            # Per-batch time-varying prior: (batch, num_classes)
            prior_for_t = self.prior_t[t]
            # Expand to match x_0 spatial dims: (batch, 1, ..., num_classes)
            for _ in range(x_0.dim() - 1):
                prior_for_t = prior_for_t.unsqueeze(1)
            # Broadcast to (batch, *spatial, num_classes) then flatten
            flat_prior = prior_for_t.expand(*x_0.shape, -1).reshape(-1, self.num_classes)
        else:
            flat_prior = self.prior.expand(x_0.numel(), -1)

        prior_samples = torch.multinomial(
            flat_prior,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).view(x_0.shape)

        # Stay at original if rand < alpha_bar_t, else transition to prior sample
        x_t = torch.where(rand < alpha_bar_t, x_0, prior_samples)

        # Preserve padding (-1) and invalid positions
        x_t = torch.where(x_0 == -1, x_0, x_t)
        if mask is not None:
            x_t = torch.where(mask, x_t, x_0)

        return x_t

    def _q_sample_decoupled(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        alpha_bar_t: torch.Tensor,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Forward diffusion with decoupled count: NegBin count + non-PAD element noise.

        Count mean follows the same alpha_bar schedule as element noise:
            mu(t) = c_0 * alpha_bar_t + eq_count * (1 - alpha_bar_t)
        At t=0: mu = c_0 (clean). At t=T: mu ≈ eq_count (~4).
        NegBin variance = mu * phi allows overshoot in both directions.
        """
        batch_size, seq_len, max_sc = x_0.shape
        device = x_0.device
        phi = self.count_overdispersion

        # GT atom count per residue: count of non-PAD, non-padding slots
        gt_count = ((x_0 > 0) & (x_0 != -1)).sum(dim=-1).float()  # (B, L)

        # Smooth interpolation: mu follows the same schedule as element noise
        # alpha_bar_t: 1 at t=0 (clean), ~0 at t=T (fully noised)
        alpha_expanded = alpha_bar_t.unsqueeze(-1).expand_as(gt_count)  # (B, L)
        mu = gt_count * alpha_expanded + self.eq_count * (1.0 - alpha_expanded)  # (B, L)
        mu = mu.clamp(min=0.1)  # Avoid mu=0 for NegBin

        # Sample count from NegBin(mu, phi): Var = mu * phi
        # PyTorch NegativeBinomial(total_count=r, probs=p): mean = r*p/(1-p)
        # With r = mu/(phi-1), p = (phi-1)/phi
        r = mu / (phi - 1)
        # Use scipy-style: total_count=r, probs=1-p (prob of "failure" = 1/phi)
        count_dist = torch.distributions.NegativeBinomial(total_count=r, probs=1 - 1 / phi)
        count_t = count_dist.sample().long()  # (B, L)
        count_t = count_t.clamp(0, max_sc)

        # At t=0 (clean), force exact GT count
        is_t0 = (t == 0).view(batch_size, 1).expand_as(gt_count)
        count_t = torch.where(is_t0, gt_count.long(), count_t)

        # Build slot activation mask: slots 0..count_t-1 are active
        slot_indices = torch.arange(max_sc, device=device).view(1, 1, max_sc)  # (1, 1, max_sc)
        active_mask = slot_indices < count_t.unsqueeze(-1)  # (B, L, max_sc)

        # Element noise on active slots: stay at x_0 with prob alpha_bar_t,
        # else transition to random non-PAD element
        alpha_bar_t_expanded = alpha_bar_t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        rand = torch.rand(x_0.shape, device=device, dtype=torch.float32, generator=generator)

        # Sample from non-PAD prior {C, N, O, S}
        non_pad_prior = self.non_pad_prior.expand(x_0.numel(), -1)
        prior_samples = torch.multinomial(
            non_pad_prior,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).view(x_0.shape)

        # For active slots: keep x_0 element with prob alpha, else random non-PAD
        # For slots where x_0 is PAD (in active range due to overshoot), use prior sample
        x_0_for_noise = torch.where(x_0 > 0, x_0, prior_samples)
        x_t = torch.where(rand < alpha_bar_t_expanded, x_0_for_noise, prior_samples)

        # Apply count mask: active slots keep noised element, inactive slots get PAD
        x_t = torch.where(active_mask, x_t, torch.zeros_like(x_t))

        # Preserve dataset padding (-1)
        x_t = torch.where(x_0 == -1, x_0, x_t)
        if mask is not None:
            x_t = torch.where(mask, x_t, x_0)

        return x_t

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

    def p_sample_reflow(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_0_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        slot_alpha_bar_prev: torch.Tensor | None = None,
        conditional: bool = True,
    ) -> torch.Tensor:
        """
        Reverse diffusion step using "reflow" (predict-and-recorrupt) instead of posterior.

        Instead of computing q(x_{t-1}|x_t, x_0) via Bayes' rule (which is PAD-biased),
        directly re-corrupt the model's x_0 prediction to noise level t-1:
            x_{t-1}[i] = x_0_pred[i]  with prob alpha_bar_{t-1}
            x_{t-1}[i] ~ prior        with prob (1 - alpha_bar_{t-1})

        When conditional=True (default), only apply reflow to PAD positions. Non-PAD
        positions in x_t are preserved since they carry useful information about x_0
        (either correctly preserved from data or informative prior samples). This hybrid
        avoids both the posterior's PAD->non-PAD suppression AND reflow's information loss.

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
        temperature : float
            Temperature for x_0 prediction sampling (1.0 = argmax).
        slot_alpha_bar_prev : torch.Tensor, optional
            Per-slot alpha_bar_{t-1}. Overrides uniform schedule.
        conditional : bool
            If True (default), only reflow PAD positions; preserve existing non-PAD.
            If False, reflow all positions independently.
        """
        # In decoupled count mode, zero out PAD logits
        if self.decoupled_count:
            x_0_logits = x_0_logits.clone()
            x_0_logits[..., ELEMENT_PAD] = -1e9

        # Get x_0 prediction (argmax or tempered)
        if temperature != 1.0:
            tempered_probs = torch.softmax(x_0_logits / temperature, dim=-1)
            flat_tempered = tempered_probs.view(-1, self.num_classes)
            flat_tempered = flat_tempered.clamp(min=1e-10)
            flat_tempered = flat_tempered / flat_tempered.sum(dim=-1, keepdim=True)
            x_0_pred = torch.multinomial(flat_tempered, 1, generator=generator).view(x_t.shape)
        else:
            x_0_pred = torch.softmax(x_0_logits, dim=-1).argmax(dim=-1)

        # Get alpha_bar_{t-1} for re-corruption level
        if slot_alpha_bar_prev is not None:
            alpha_bar_prev = slot_alpha_bar_prev
        else:
            alpha_bar_prev = self.alphas_cumprod_prev[t]
            for _ in range(x_t.dim() - 1):
                alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)

        # Re-corrupt x_0_pred to noise level t-1:
        # Each position independently: keep x_0_pred with prob alpha_bar_{t-1}, else sample from prior
        keep_mask = torch.rand_like(alpha_bar_prev) < alpha_bar_prev  # (B, L, max_sc) or broadcast

        # Sample from prior for corrupted positions
        prior = self.prior if self.prior_t is None else self.prior_t[t]
        if self.prior_t is not None:
            for _ in range(x_t.dim() - 1):
                prior = prior.unsqueeze(1)
        else:
            for _ in range(x_t.dim()):
                prior = prior.unsqueeze(0)
        flat_prior = prior.expand(*x_t.shape, -1).reshape(-1, self.num_classes)
        flat_prior = flat_prior.clamp(min=1e-10)
        flat_prior = flat_prior / flat_prior.sum(dim=-1, keepdim=True)
        prior_samples = torch.multinomial(flat_prior, 1, generator=generator).view(x_t.shape)

        # Combine: keep x_0_pred where keep_mask, else use prior sample
        x_t_minus_1 = torch.where(keep_mask, x_0_pred, prior_samples)

        # Conditional reflow: only reflow PAD positions, preserve existing non-PAD
        if conditional:
            is_pad = x_t == ELEMENT_PAD
            x_t_minus_1 = torch.where(is_pad, x_t_minus_1, x_t)

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

    def compute_loss(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        mask: torch.Tensor | None = None,
        fn_weight: float = 1.0,
        non_pad_only: bool = False,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for element type prediction.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits of shape (..., num_classes).
        x_0 : torch.Tensor
            Ground truth element types of shape (...).
        mask : torch.Tensor, optional
            Mask for valid positions (binary or soft).
        fn_weight : float
            Extra multiplier on loss for non-PAD GT positions. Values > 1 penalize
            false negatives (predicting PAD when atom exists) more than false positives.
        non_pad_only : bool
            If True, only compute loss on GT non-PAD positions. PAD/non-PAD existence
            decisions are deferred to the atom importance agreement loss.

        Returns
        -------
        loss : torch.Tensor
            Scalar cross-entropy loss.
        """
        valid_mask = (x_0 >= 0).float()
        if mask is not None:
            mask_float = mask.float() if mask.dtype == torch.bool else mask
            valid_mask = valid_mask * mask_float

        # Exclude GT-PAD positions: element CE only teaches "what type?" not "should exist?"
        if non_pad_only:
            valid_mask = valid_mask * (x_0 != ELEMENT_PAD).float()

        if valid_mask.sum() < 1e-8:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_flat = logits.view(-1, self.num_classes)
        targets_flat = x_0.view(-1)
        weights_flat = valid_mask.view(-1)

        valid_positions = targets_flat >= 0
        if non_pad_only:
            valid_positions = valid_positions & (targets_flat != ELEMENT_PAD)
        if not valid_positions.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss_per_sample = torch.nn.functional.cross_entropy(
            logits_flat[valid_positions],
            targets_flat[valid_positions],
            weight=self.class_weights,
            reduction="none",
        )

        loss_full = torch.zeros(targets_flat.shape[0], device=logits.device)
        loss_full[valid_positions] = loss_per_sample

        # Asymmetric weighting: boost loss for non-PAD GT positions (false negative penalty)
        if fn_weight != 1.0:
            non_pad = (targets_flat != ELEMENT_PAD).float()
            sample_weight = 1.0 + (fn_weight - 1.0) * non_pad  # 1.0 for PAD, fn_weight for non-PAD
            weights_flat = weights_flat * sample_weight

        return (loss_full * weights_flat).sum() / (weights_flat.sum() + 1e-8)

    def compute_accuracy(
        self,
        logits: torch.Tensor,
        x_0: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute prediction accuracy for element types.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits of shape (..., num_classes).
        x_0 : torch.Tensor
            Ground truth element types of shape (...).
        mask : torch.Tensor, optional
            Mask for valid positions (binary or soft).

        Returns
        -------
        accuracy : torch.Tensor
            Scalar accuracy in [0, 1].
        """
        valid_mask = (x_0 >= 0).float()
        if mask is not None:
            mask_float = mask.float() if mask.dtype == torch.bool else mask
            valid_mask = valid_mask * mask_float

        if valid_mask.sum() < 1e-8:
            return torch.tensor(0.0, device=logits.device)

        predictions = logits.argmax(dim=-1)
        correct = (predictions == x_0).float() * valid_mask

        return correct.sum() / (valid_mask.sum() + 1e-8)


# =============================================================================
# Placeholder for future groupwise diffusion implementation
# =============================================================================


def compute_atom_depths(atom_names: list[str]) -> torch.Tensor:
    """
    Compute "depth" of each atom (distance from backbone in bonds).

    This is a placeholder for future hierarchical diffusion implementation.

    Parameters
    ----------
    atom_names : list[str]
        List of atom names (e.g., ['CA', 'CB', 'CG', 'CD', ...]).

    Returns
    -------
    depths : torch.Tensor
        Depth of each atom (0 for backbone, 1 for Cβ, etc.).

    Notes
    -----
    For hierarchical diffusion, atoms with lower depth should be denoised
    last (more stable), while atoms with higher depth should be denoised
    first (finer details resolved later).

    Example depth assignments:
    - CA, C, N, O (backbone): depth 0
    - CB: depth 1
    - CG, CG1, CG2: depth 2
    - CD, CD1, CD2: depth 3
    - CE, CE1, CE2: depth 4
    - CZ, NZ, etc.: depth 5+
    """
    # Atom depth computation is not wired up (unused on the inference path).
    # For now, return uniform depths
    return torch.zeros(len(atom_names))


class GroupwiseDiffusion(GaussianDiffusion):
    """
    Placeholder for groupwise diffusion with per-atom noise schedules.

    This extends GaussianDiffusion to support different noise levels for
    different atom groups based on their distance from the backbone.

    NOT YET IMPLEMENTED - this is a research direction for future work.
    See module docstring for design discussion.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        num_depth_groups: int = 5,
        depth_schedule_scale: float = 0.5,
    ):
        super().__init__(timesteps=timesteps, schedule=schedule)
        self.num_depth_groups = num_depth_groups
        self.depth_schedule_scale = depth_schedule_scale
        # Per-group beta schedules are not implemented.
        # Depth-aware q_sample is not implemented.
        raise NotImplementedError("GroupwiseDiffusion is not yet implemented")
