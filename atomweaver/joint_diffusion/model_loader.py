"""Strict checkpoint loading for the released AtomWeaver inference architecture."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class LoadedModel(nn.Module):
    """Model and residue lookup used by the sampling CLI."""

    def __init__(self, model: nn.Module, name_to_idx: "dict[str, int] | None"):
        super().__init__()
        self.model = model
        self.name_to_idx = name_to_idx
        self._ckpt_reserved_slot0 = None


_CANONICAL_NAME_TO_CODE = [
    ("Alanine", "ALA"),
    ("Cysteine", "CYS"),
    ("Aspartic", "ASP"),
    ("Glutamic acid", "GLU"),
    ("Phenylalanine", "PHE"),
    ("Glycine", "GLY"),
    ("Histidine", "HIS"),
    ("Isoleucine", "ILE"),
    ("Lysine", "LYS"),
    ("Leucine", "LEU"),
    ("Methionine", "MET"),
    ("Asparagine", "ASN"),
    ("Proline", "PRO"),
    ("Glutamine", "GLN"),
    ("Arginine", "ARG"),
    ("Serine", "SER"),
    ("Threonine", "THR"),
    ("Valine", "VAL"),
    ("Tryptophan", "TRP"),
    ("Tyrosine", "TYR"),
]


def build_name_to_idx(db_metadata: list) -> dict[str, int]:
    """Map known residue names and PDB identifiers to reference-library type indices."""
    name_to_idx: dict[str, int] = {}
    for i, meta in enumerate(db_metadata):
        for pdb_id in meta.get("pdb_ids", []):
            if pdb_id and pdb_id not in name_to_idx:
                name_to_idx[pdb_id] = i
        name = meta.get("name", "")
        for aa_name, code in _CANONICAL_NAME_TO_CODE:
            if aa_name in name and code not in name_to_idx:
                name_to_idx[code] = i
    return name_to_idx


def load_model(checkpoint_path: str | Path, residue_db: str, device: str = "cpu"):
    """Load the released architecture, weights, and residue lookup for inference."""
    from atomweaver.joint_diffusion.models import InverseFoldingDiffusion

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    hparams = checkpoint["hyper_parameters"]
    expected = {"hidden_dim": 432, "num_layers": 10, "coord_process_type": "flow_matching", "reserved_slot0": True}
    for name, value in expected.items():
        if hparams.get(name) != value:
            raise ValueError(f"Unsupported checkpoint: expected {name}={value!r}, got {hparams.get(name)!r}")
    model = InverseFoldingDiffusion()
    state = {
        key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    db = torch.load(residue_db, map_location="cpu", weights_only=True)
    loaded = LoadedModel(model, build_name_to_idx(db["metadata"]))
    loaded._ckpt_reserved_slot0 = True
    return loaded.to(device).eval()
