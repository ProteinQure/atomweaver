"""Preserve the input target independently of model conditioning and cloud read-out."""

from pathlib import Path

import pytest
import torch

from scripts.joint_diffusion.apply_hybrid_readout import parse_cloud
from scripts.joint_diffusion.sample import _make_cloud_pdb, _target_pdb_records

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("extra_target", [False, True])
@pytest.mark.parametrize("target_chain", ["B", "A", "Z", None])
def test_target_atom_records_preserved(tmp_path, extra_target, target_chain):
    """Keep A=target, B=input, C=generated without changing atom data or merging targets."""
    source = (
        "\n".join(
            line[:21] + ("P" if line[21] == "A" else target_chain) + line[22:]
            for line in (ROOT / "examples/9RA5_MK8.pdb").read_text().splitlines()
            if line.startswith(("ATOM  ", "HETATM", "ANISOU")) and (line[21] == "A" or target_chain is not None)
        )
        + "\n"
    )
    target_mapping = {target_chain: "A"} if target_chain is not None else {}
    if extra_target:
        # A target chain named C, with an alternate location and insertion code.
        atom = (
            f"HETATM{90000:5d} {'CL1':<4s}A{'LIG':>3s} C{17:4d}A   "
            f"{12.345:8.3f}{-6.789:8.3f}{4.321:8.3f}{0.5:6.2f}{27.65:6.2f}          CL1-"
        )
        anisou = "ANISOU" + atom[6:28] + "".join(f"{value:7d}" for value in (1000, 1000, 1000, 0, 0, 0)) + atom[70:]
        source += atom + "\n" + anisou + "\n"
        target_mapping["C"] = "D" if target_chain is not None else "A"
    pdb_path = tmp_path / "input.pdb"
    pdb_path.write_text(source)
    expected = [
        line for line in source.splitlines() if line.startswith(("ATOM  ", "HETATM", "ANISOU")) and line[21] != "P"
    ]
    records = _target_pdb_records(pdb_path, "P")
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
    assert "REMARK  ATOMWEAVER_CHAINS B C\n" in exported
    for original_chain, output_chain in target_mapping.items():
        assert f"REMARK  ATOMWEAVER_TARGET_CHAIN {original_chain!r} -> {output_chain}\n" in exported
    actual = [
        line
        for line in exported.splitlines()
        if line.startswith(("ATOM  ", "HETATM", "ANISOU")) and line[21] not in "BC"
    ]
    assert actual == [line[:21] + target_mapping[line[21]] + line[22:] for line in expected]
    serials = [int(line[6:11]) for line in exported.splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    assert len(serials) == len(set(serials))
    cloud_path = tmp_path / "cloud.pdb"
    cloud_path.write_text(exported)
    generated, input_peptide = parse_cloud(cloud_path)
    reference = parse_cloud(ROOT / "tests/data/clouds/reference_s0.pdb")[0]
    assert generated == [residue | {"rn": "UNK"} for residue in reference]
    assert [residue["rn"] for residue in input_peptide] == batch["res_names"][0][:3]
