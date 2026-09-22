"""Preserve the released PDB parsing, cropping, slotization, and batching."""

from pathlib import Path

import pytest
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


@pytest.mark.parametrize("reserved_slot0", [False, True])
def test_zero_sidechain_atoms(tmp_path, reserved_slot0):
    """Accept an all-glycine peptide without requiring deposited side-chain atoms."""
    backbone = [
        line
        for line in (ROOT / "examples/9RA5_MK8.pdb").read_text().splitlines()
        if line.startswith("ATOM")
        and line[21] == "B"
        and line[22:26].strip() == "1"
        and line[12:16].strip() in {"N", "CA", "C", "O"}
    ]
    lines = [
        f"{line[:17]}GLY {chain}{residue:4d}{line[26:]}"
        for chain, residue in [("A", 1), ("B", 1), ("B", 2)]
        for line in backbone
    ]
    (tmp_path / "glycine.pdb").write_text("\n".join([*lines, "END"]) + "\n")

    dataset = PeptideDataset(tmp_path, reserved_slot0=reserved_slot0)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["res_names"] == ["GLY"]
    assert item["backbone_mask"].all()
    assert not item["sidechain_mask"].any()
    assert item["design_mask"].all()
    batch = collate_peptides([item])
    assert batch["sidechain_mask"].shape == (1, 1, 14)
    assert not batch["sidechain_mask"].any()
