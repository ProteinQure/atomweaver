"""Production sampling configuration for AtomWeaver.

``design.py`` sets the production recipe in the environment as ``ATOMWEAVER_*`` variables; this module
resolves them into a :class:`SamplingConfig` (a command-line value first, then the environment, then the
built-in default) and validates the result before the model is built. Only the production knobs live
here.

The recognised variables are:

======================================  ==========================================================
``ATOMWEAVER_ELEMENT_VOCAB``            Element vocabulary size. Must match the checkpoint (production
                                        is 5; ``diffusion.py`` defaults to 12).
``ATOMWEAVER_SHELL_VAR_SCALE``          Source-shell variance scale (production 0.25).
``ATOMWEAVER_DISC_REPACK``              Re-pack predicted clouds before discretization (production on).
``ATOMWEAVER_SAMPLE_RECYCLES``          Neighbour-x0 recycle passes at sampling (production 1).
``ATOMWEAVER_LEARNED_READOUT``          Score the learned logistic-regression read-out (production on).
``ATOMWEAVER_READOUT_CACHE``            Path to the fitted read-out cache.
``ATOMWEAVER_LEARNED_CHIRALITY_GATE``   Gate the learned read-out by predicted handedness (production on).
``ATOMWEAVER_INDEX_CACHE_DISABLE``      Dataset index cache: bypass it.
``ATOMWEAVER_INDEX_CACHE_DIR``          Dataset index cache: where to store it.
``ATOMWEAVER_INDEX_BUILD_WORKERS``      Dataset index build: worker count.
``ATOMWEAVER_VERBOSE``                  Print the resolved recipe and extra diagnostics.
======================================  ==========================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

_FALSEY = frozenset({"", "0", "false", "False", "no", "off"})

#: Every ``ATOMWEAVER_*`` variable the package understands. Anything else found in the environment is
#: reported by :func:`undeclared_atomweaver_vars` (only surfaced under ``ATOMWEAVER_VERBOSE``).
KNOWN_KNOBS: frozenset[str] = frozenset(
    {
        "ATOMWEAVER_ELEMENT_VOCAB",
        "ATOMWEAVER_SHELL_VAR_SCALE",
        "ATOMWEAVER_DISC_REPACK",
        "ATOMWEAVER_SAMPLE_RECYCLES",
        "ATOMWEAVER_LEARNED_READOUT",
        "ATOMWEAVER_READOUT_CACHE",
        "ATOMWEAVER_LEARNED_CHIRALITY_GATE",
        "ATOMWEAVER_INDEX_CACHE_DISABLE",
        "ATOMWEAVER_INDEX_CACHE_DIR",
        "ATOMWEAVER_INDEX_BUILD_WORKERS",
        "ATOMWEAVER_VERBOSE",
    }
)


def _truthy(raw: str | None) -> bool:
    return raw is not None and raw not in _FALSEY


def resolve(env: str, cli_value: Any, kind: str = "str") -> tuple[Any, str]:
    """Resolve one value from the command line, then the environment, then the default.

    Returns ``(value, source)`` where ``source`` is ``"cli"``, ``"env"`` or ``"default"``. A value
    read from the environment is coerced to ``kind`` (``bool`` | ``int`` | ``float`` | ``str``), raising
    a named ``ValueError`` if it does not parse -- better than a bad string reaching the model.
    """
    if cli_value is not None:
        return cli_value, "cli"
    raw = os.environ.get(env)
    if raw is not None and raw != "":
        if kind == "bool":
            return _truthy(raw), "env"
        try:
            return (int(raw) if kind == "int" else float(raw) if kind == "float" else raw), "env"
        except ValueError as exc:
            raise ValueError(f"{env}={raw!r} is not a valid {kind}") from exc
    return None, "default"


@dataclass
class SamplingConfig:
    """The resolved production sampling recipe for one run."""

    shell_var_scale: float | None = None
    learned_readout: bool = False
    learned_chirality_gate: bool = True
    readout_cache: str | None = None

    #: Per-field provenance: field name -> "cli" | "env" | "default".
    sources: dict[str, str] = field(default_factory=dict)

    _FIELD_TO_ENV: ClassVar[dict[str, str]] = {
        "shell_var_scale": "ATOMWEAVER_SHELL_VAR_SCALE",
        "learned_readout": "ATOMWEAVER_LEARNED_READOUT",
        "learned_chirality_gate": "ATOMWEAVER_LEARNED_CHIRALITY_GATE",
        "readout_cache": "ATOMWEAVER_READOUT_CACHE",
    }
    _FIELD_KIND: ClassVar[dict[str, str]] = {
        "shell_var_scale": "float",
        "learned_readout": "bool",
        "learned_chirality_gate": "bool",
        "readout_cache": "str",
    }

    @classmethod
    def resolve_all(cls, **cli: Any) -> SamplingConfig:
        """Resolve every field from the supplied CLI values, then the environment, then the default."""
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for name, env in cls._FIELD_TO_ENV.items():
            value, source = resolve(env, cli.get(name), cls._FIELD_KIND[name])
            sources[name] = source
            if source != "default":  # otherwise fall back to the dataclass field default
                values[name] = value
        cfg = cls(**values)
        cfg.sources = sources
        cfg.validate()
        return cfg

    @classmethod
    def from_env(cls) -> SamplingConfig:
        """Resolve purely from the environment. For library callers."""
        return cls.resolve_all()

    def validate(self) -> None:
        """Reject values that cannot mean what they say."""
        if self.shell_var_scale is not None and float(self.shell_var_scale) < 0:
            raise ValueError(f"shell_var_scale must be >= 0, got {self.shell_var_scale}")

    def describe(self) -> list[str]:
        """Human-readable lines naming every resolved value and where it came from."""
        out = []
        for f in fields(self):
            if f.name == "sources":
                continue
            src = self.sources.get(f.name, "default")
            out.append(f"  [sampling-config] {f.name} = {getattr(self, f.name)!r} ({src})")
        return out


def undeclared_atomweaver_vars() -> list[str]:
    """``ATOMWEAVER_*`` variables set in the environment but not recognised by this package."""
    return sorted(
        name
        for name in os.environ
        if name.startswith("ATOMWEAVER_") and name not in KNOWN_KNOBS and not name.endswith("_NOTE")
    )
