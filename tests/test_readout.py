"""Exact read-out features and distributions from the original release."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.joint_diffusion.apply_hybrid_readout import build_tensors, learned_features, main, parse_cloud

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"


def test_released_features():
    """Keep slot order, element encoding, geometry, and dihedrals unchanged."""
    residues, _ = parse_cloud(DATA / "clouds/reference_s0.pdb")
    actual = learned_features(residues, *build_tensors(residues))
    np.testing.assert_array_equal(actual, np.load(DATA / "features.npy"))


@pytest.mark.parametrize("vocab", ["full300", "canon20"])
@pytest.mark.parametrize("natfreq", [False, True])
def test_released_readout(tmp_path, vocab, natfreq):
    """Preserve every exported probability for both vocabularies and priors."""
    output = tmp_path / "preds.json"
    main(
        head=str(ROOT / f"data/readout_head_{vocab}.joblib"),
        ref_db=str(ROOT / "data/reference_library.pt"),
        clouds=str(DATA / "clouds"),
        out=str(output),
        preset="b2_balanced",
        natfreq=natfreq,
        canon20=vocab == "canon20",
        atom_penalty=0.5,
        elem_penalty=0.3,
        chir_penalty=50.0,
        dump_distributions=None,
    )
    actual = json.loads(output.read_text())
    prior = "natfreq" if natfreq else "uniform"
    expected = json.loads((DATA / f"{vocab}_{prior}.json").read_text())
    assert actual["classes"] == expected["classes"]
    assert actual["designs"] == expected["designs"]
