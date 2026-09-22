"""Preserve the released PDB parsing, cropping, slotization, and batching."""

from pathlib import Path

import torch

from atomweaver.joint_diffusion.datasets import PeptideDataset, collate_peptides
from tests.test_inference import assert_equivalent

ROOT = Path(__file__).resolve().parents[1]


def test_example_structure():
    """Compare all parsed tensors and metadata with the original dataset."""
    expected = torch.load(ROOT / "tests/data/dataset.pt", weights_only=True)
    dataset = PeptideDataset(ROOT / "examples", max_sidechain_atoms=14, reserved_slot0=True)
    item = dataset[0]
    assert_equivalent(item, expected["item"])
    assert_equivalent(collate_peptides([item]), expected["batch"])
