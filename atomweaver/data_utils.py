"""
Data loading utilities for residue databases.

Provides functions to load pre-built residue databases for use with
ResidueDatabaseMatcher and other US-align components.
"""

from __future__ import annotations

from pathlib import Path

import torch

# Default data directory (relative to project root)
DATA_DIR = Path(__file__).parent.parent / "data"


def load_residue_database(
    variant: str = "toy",
    data_dir: Path | None = None,
) -> dict:
    """
    Load a pre-built residue database.

    Parameters
    ----------
    variant : str
        Which database to load: 'toy' (60 residues) or 'full' (all ~3800)
    data_dir : Path, optional
        Custom data directory. Defaults to package data directory.

    Returns
    -------
    dict
        Dictionary with:
        - coords: (num_residues, max_atoms, 3) tensor
        - masks: (num_residues, max_atoms) boolean tensor
        - metadata: list of dicts with ccd_code, name, smiles, etc.
        - max_atoms: int
        - num_residues: int
    """
    if data_dir is None:
        data_dir = DATA_DIR

    filename = f"residue_database_{variant}.pt"
    path = data_dir / filename

    if not path.exists():
        msg = f"Residue database not found: {path}. Expected a reference library .pt (e.g. data/reference_library.pt)."
        raise FileNotFoundError(msg)

    return torch.load(path, weights_only=False)


def get_id_to_index_mapping(variant: str = "toy", data_dir: Path | None = None) -> dict[str, int]:
    """
    Get mapping from ccd_code to index in the database.

    Useful for creating target labels from residue IDs.

    Parameters
    ----------
    variant : str
        Which database to use
    data_dir : Path, optional
        Custom data directory

    Returns
    -------
    dict[str, int]
        Mapping from ccd_code to tensor index
    """
    db = load_residue_database(variant=variant, data_dir=data_dir)
    return {m["ccd_code"]: i for i, m in enumerate(db["metadata"])}


def get_index_to_id_mapping(variant: str = "toy", data_dir: Path | None = None) -> dict[int, str]:
    """
    Get mapping from index to ccd_code.

    Useful for converting predictions back to residue IDs.

    Parameters
    ----------
    variant : str
        Which database to use
    data_dir : Path, optional
        Custom data directory

    Returns
    -------
    dict[int, str]
        Mapping from tensor index to ccd_code
    """
    db = load_residue_database(variant=variant, data_dir=data_dir)
    return {i: m["ccd_code"] for i, m in enumerate(db["metadata"])}
