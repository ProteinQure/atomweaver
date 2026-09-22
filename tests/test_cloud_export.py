"""Preserve the input target independently of model conditioning and cloud read-out."""

from pathlib import Path

import pytest
import torch

from scripts.joint_diffusion.apply_hybrid_readout import parse_cloud
from scripts.joint_diffusion.sample import _make_cloud_pdb, _target_pdb_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("extra_target", [False, True])
def test_target_atom_records_preserved(tmp_path, extra_target):
    """Retain full targets, including hydrogens, heteroatoms, and colliding chain IDs."""
    source = (ROOT / "examples/9RA5_MK8.pdb").read_text()
    if extra_target:
        # A target chain named C, with an alternate location and insertion code.
        atom = (
            f"HETATM{90000:5d} {'CL1':<4s}A{'LIG':>3s} C{17:4d}A   "
            f"{12.345:8.3f}{-6.789:8.3f}{4.321:8.3f}{0.5:6.2f}{27.65:6.2f}          CL1-"
        )
        anisou = "ANISOU" + atom[6:28] + "".join(f"{value:7d}" for value in (1000, 1000, 1000, 0, 0, 0)) + atom[70:]
        source += atom + "\n" + anisou + "\n"
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(source)
    expected = [
        line for line in source.splitlines() if line.startswith(("ATOM  ", "HETATM", "ANISOU")) and line[21] != "A"
    ]
    records = _target_pdb_records(pdb_path, "A")
    assert records == expected
    batch = torch.load(ROOT / "tests/data/dataset.pt", weights_only=True)["batch"]
    output = torch.load(ROOT / "tests/data/joint.pt", weights_only=True)["outputs"]
    if extra_target:
        # Export must not reconstruct or truncate the target from its model tensors.
        for key, value in batch.items():
            if key.startswith("target_") and torch.is_tensor(value):
                batch[key] = value[:, :1]
    exported = _make_cloud_pdb(
        batch,
        output,
        [0, 1, 2],
        batch["res_names"][0],
        "",
        target_records=records,
    )
    target_chains = {line[21] for line in expected}
    actual = [
        line
        for line in exported.splitlines()
        if line.startswith(("ATOM  ", "HETATM", "ANISOU")) and line[21] in target_chains
    ]
    assert actual == expected
    serials = [int(line[6:11]) for line in exported.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    assert len(serials) == len(set(serials))
    cloud_path = tmp_path / "cloud.pdb"
    cloud_path.write_text(exported)
    generated, input_peptide = parse_cloud(cloud_path)
    reference = parse_cloud(ROOT / "tests/data/clouds/reference_s0.pdb")[0]
    assert generated == [residue | {"rn": "UNK"} for residue in reference]
    assert [residue["rn"] for residue in input_peptide] == batch["res_names"][0][:3]
