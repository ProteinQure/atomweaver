"""Explicit sampling settings and one-time shell-prior configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingConfig:
    """Runtime settings separate from the fixed checkpoint architecture."""

    shell_var_scale: float | None = None
    recycles: int = 1

    def __post_init__(self):
        if self.shell_var_scale is not None and self.shell_var_scale < 0:
            raise ValueError(f"shell_var_scale must be >= 0, got {self.shell_var_scale}")

    @classmethod
    def from_env(cls) -> SamplingConfig:
        """Resolve CLI environment overrides before constructing the model."""
        raw_scale = os.environ.get("ATOMWEAVER_SHELL_VAR_SCALE")
        return cls(
            shell_var_scale=float(raw_scale) if raw_scale else None,
            recycles=max(1, int(os.environ.get("ATOMWEAVER_SAMPLE_RECYCLES", "1"))),
        )


def apply_sampling_config(model, config: SamplingConfig) -> None:
    """Scale shell jitter once, keeping the shell-radius variance unchanged."""
    if getattr(model, "_sampling_env_overrides_applied", False):
        return
    model._sampling_env_overrides_applied = True
    if config.shell_var_scale is None:
        return
    scale = float(config.shell_var_scale)
    flow = model.coord_flow
    before = flow._shell_source_var.clone()
    with torch.no_grad():
        flow._shell_thickness.mul_(scale**0.5)
        flow._shell_radial_var.mul_(scale)
        flow._shell_source_var.copy_(flow._shell_radii**2 / 3.0 + flow._shell_thickness**2)
    ratio = float((flow._shell_source_var.sum() / before.sum()).item())
    print(f"[shell-var-scale] jitter variance x{scale}; source variance x{ratio:.3f}")
