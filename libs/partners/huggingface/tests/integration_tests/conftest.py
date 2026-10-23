import os
from collections.abc import Callable

import pytest
import torch


@pytest.fixture
def accelerator_device() -> int | str:
    """Device to test on, from `HF_TEST_DEVICE` (e.g. `0` or `xpu:1`).

    Tests using this fixture are skipped unless `HF_TEST_DEVICE` is set.
    """
    value = os.getenv("HF_TEST_DEVICE")
    if value is None:
        pytest.skip(
            "Set HF_TEST_DEVICE=<index or device, e.g. 0 or xpu:1> on an "
            "accelerator box to run."
        )
    return int(value) if value.isdigit() else value


@pytest.fixture
def assert_on_accelerator(
    accelerator_device: int | str,
) -> Callable[[torch.device], None]:
    """Return a check that a model was placed on the requested accelerator.

    A green run with a silent CPU fallback would prove nothing about accelerator
    support, so the device type and index are both checked.
    """

    def check(model_device: torch.device) -> None:
        assert model_device.type != "cpu", (
            f"Expected model on an accelerator, but it loaded on {model_device}. "
            "Check torch device ordering or the device index."
        )
        expected_index = (
            accelerator_device
            if isinstance(accelerator_device, int)
            else torch.device(accelerator_device).index
        )
        if expected_index is not None:
            assert model_device.index == expected_index

    return check
