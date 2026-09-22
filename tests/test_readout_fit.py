"""Preserve deterministic custom-vocabulary fitting."""

from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.joint_diffusion.fit_learned_readout import main

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data"


def test_custom_vocabulary_fit(tmp_path):
    """Match fitted coefficients, scaling, class order, and prior exactly."""
    output = tmp_path / "custom.joblib"
    torch.manual_seed(456)
    with threadpool_limits(limits=2):
        main(
            ckpt_fit_cache=str(DATA / "fit_clouds.npz"),
            ref_db=str(ROOT / "data/reference_library.pt"),
            phipsi_table=str(ROOT / "data/phipsi_populations.pt"),
            residues=str(DATA / "residues.txt"),
            out=str(output),
            scope="full",
            popweight="a6",
            warmstart_from=str(ROOT / "data/readout_head_full300.joblib"),
            ckpt_name="regression",
        )
    bundle = joblib.load(output)
    scaler = bundle["clf"].named_steps["standardscaler"]
    classifier = bundle["clf"].named_steps["logisticregression"]
    actual = {
        "classes": bundle["classes"],
        "prior": bundle["prior"],
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coef": classifier.coef_,
        "intercept": classifier.intercept_,
    }
    with np.load(DATA / "fit_expected.npz") as expected:
        for name, value in actual.items():
            np.testing.assert_array_equal(value, expected[name])
