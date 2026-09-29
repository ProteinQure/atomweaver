"""Exact read-out features and distributions from the original release."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.joint_diffusion import apply_hybrid_readout, design
from scripts.joint_diffusion.apply_hybrid_readout import build_tensors, label_cloud, learned_features, main, parse_cloud
from scripts.joint_diffusion.design import postprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"


@pytest.fixture(params=[False, True], ids=["original-chains", "explicit-chains"])
def clouds(request, tmp_path):
    """Include targets named B/C alongside explicitly identified peptide chains."""
    lines = (DATA / "clouds/reference_s0.pdb").read_text().splitlines()
    if request.param:
        target = [line for line in lines if line.startswith("ATOM") and line[21] == "B"]
        target += [line[:21] + "C" + line[22:] for line in target]
        peptide = [
            line[:21] + {"B": "D", "C": "E"}[line[21]] + line[22:] if line.startswith("ATOM") else line
            for line in lines
        ]
        lines = ["REMARK  ATOMWEAVER_CHAINS D E", *target, *peptide]
    directory = tmp_path / "clouds"
    directory.mkdir()
    (directory / "reference_s0.pdb").write_text("\n".join(lines) + "\n")
    return directory


def test_vocab_registries_agree():
    """Keep design.py's --vocab choices and ship heads identical to the read-out's."""
    assert list(design.VOCABS) == list(apply_hybrid_readout.VOCABS)
    assert design.DEF_VOCAB == apply_hybrid_readout.DEF_VOCAB == "full300"
    for name, spec in apply_hybrid_readout.VOCABS.items():
        # design.py stores absolute paths, apply_hybrid_readout repo-relative ones.
        assert Path(design.VOCABS[name]["head"]) == ROOT / spec["head"]
        pinned = design.VOCABS[name]["ref_db"]
        assert (pinned is None) == (spec["ref_db"] is None)
        if pinned is not None:
            assert Path(pinned) == ROOT / spec["ref_db"]


def test_released_features():
    """Keep slot order, element encoding, geometry, and dihedrals unchanged."""
    residues, _ = parse_cloud(DATA / "clouds/reference_s0.pdb")
    actual = learned_features(residues, *build_tensors(residues))
    np.testing.assert_array_equal(actual, np.load(DATA / "features.npy"))


@pytest.mark.parametrize("vocab", ["full300", "canon20"])
def test_released_readout(tmp_path, vocab, clouds):
    """Use hybrid labels in the PDB while preserving probabilities and all other atom data."""
    output = tmp_path / "preds.json"
    cloud = clouds / "reference_s0.pdb"
    before = cloud.read_bytes().splitlines(keepends=True)
    main(
        head=str(ROOT / f"data/readout_head_{vocab}.joblib"),
        ref_db=str(ROOT / "data/reference_library.pt"),
        clouds=str(clouds),
        out=str(output),
        preset="b2_balanced",
        vocab=vocab,
        atom_penalty=0.5,
        elem_penalty=0.3,
        chir_penalty=50.0,
        dump_distributions=None,
    )
    actual = json.loads(output.read_text())
    expected = json.loads((DATA / f"{vocab}_uniform.json").read_text())
    assert actual["classes"] == expected["classes"]
    assert actual["designs"] == expected["designs"]
    generated_chain = b"E" if before[0].startswith(b"REMARK  ATOMWEAVER_CHAINS") else b"C"
    for original, labeled in zip(before, cloud.read_bytes().splitlines(keepends=True), strict=True):
        if original.startswith(b"ATOM") and original[21:22] == generated_chain:
            assert original[:17] == labeled[:17]
            assert original[20:] == labeled[20:]
        else:
            assert original == labeled
    residues, _ = parse_cloud(cloud)
    assert [residue["rn"] for residue in residues] == actual["designs"][0]["argmax"]
    np.testing.assert_array_equal(learned_features(residues, *build_tensors(residues)), np.load(DATA / "features.npy"))
    postprocess(output, tmp_path)
    sequence = (tmp_path / "designs.fasta").read_text().splitlines()[1]
    expected_sequence = "[DI7][TRF][LAV]" if vocab == "full300" else "DWD"
    assert sequence == expected_sequence
    labeled = cloud.read_bytes()
    label_cloud(cloud, actual["designs"][0]["argmax"])
    assert cloud.read_bytes() == labeled


@pytest.mark.parametrize("labels", [["ALA"], ["ALA", "TRP", "TOOLONG"]])
def test_invalid_labels_leave_cloud_unchanged(clouds, labels):
    """Reject inconsistent labels before modifying any atom records."""
    cloud = clouds / "reference_s0.pdb"
    before = cloud.read_bytes()
    with pytest.raises(ValueError, match=r"read-out labels|PDB residue names"):
        label_cloud(cloud, labels)
    assert cloud.read_bytes() == before


def test_labels_follow_residue_order_and_insertion_codes(tmp_path):
    """Preserve CRLF and nonsequential/insertion-coded residue IDs while assigning labels."""
    lines = (DATA / "clouds/reference_s0.pdb").read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("ATOM") and line[21] == "C":
            residue_id = {1: "   7 ", 2: "   3 ", 3: "   3A"}[int(line[22:26])]
            lines[index] = line[:22] + residue_id + line[27:]
    cloud = tmp_path / "cloud.pdb"
    cloud.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    label_cloud(cloud, ["DI7", "MK8", "A"])
    assert [residue["rn"] for residue in parse_cloud(cloud)[0]] == ["DI7", "MK8", "A"]
    for before, after in zip(lines, cloud.read_bytes().splitlines(keepends=True), strict=True):
        assert after.endswith(b"\r\n")
        if before.startswith("ATOM") and before[21] == "C":
            assert after[:17] == before[:17].encode()
            assert after[20:-2] == before[20:].encode()
        else:
            assert after == (before + "\r\n").encode()
