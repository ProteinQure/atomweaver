"""Numerical regressions captured from the original released inference code."""

from pathlib import Path

import pytest
import torch

from atomweaver.joint_diffusion.model_loader import _load_lightning_module_for_eval
from scripts.joint_diffusion.sample import apply_sampling_config
from scripts.joint_diffusion.sampling_knobs import SamplingConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    """Load the released checkpoint once for the CPU sampling cases."""
    checkpoint = ROOT / "weights/atomweaver.pt"
    if not checkpoint.exists():
        pytest.skip("Download the released weights/atomweaver.pt to run numerical regressions")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    loaded = _load_lightning_module_for_eval(checkpoint, str(ROOT / "data/sampling_library.pt"), "cpu").model
    reference = torch.load(ROOT / "tests/data/initialization.pt", weights_only=True)
    assert torch.equal(torch.random.get_rng_state(), reference["rng"])
    assert list(loaded.state_dict()) == reference["keys"]
    apply_sampling_config(loaded, SamplingConfig(shell_var_scale=0.25))
    return loaded


def assert_equivalent(actual, expected):
    """Compare every returned tensor, including intermediate sampling states."""
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_equivalent(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected, strict=True):
            assert_equivalent(result, reference)
    else:
        assert actual == expected


@pytest.mark.parametrize("case", ["joint", "subset", "recycled"])
def test_released_sampling(model, case):
    """Preserve joint design, clean context pinning, and optional recycling."""
    reference = torch.load(ROOT / "tests/data" / f"{case}.pt", weights_only=True)
    torch.manual_seed(123)
    actual = model.sample(**reference["inputs"])
    assert_equivalent(actual, reference["outputs"])
