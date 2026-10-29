import time
from importlib.metadata import requires

import pytest
from blockbuster import BlockingError
from packaging.requirements import Requirement

from langchain_core import sys_info


def test_certifi_minimum_version() -> None:
    requirements = [Requirement(req) for req in requires("langchain-core") or []]
    certifi = next(req for req in requirements if req.name == "certifi")
    assert "2026.10.18" not in certifi.specifier
    assert "2026.10.19" in certifi.specifier


async def test_blockbuster_setup() -> None:
    """Check if blockbuster is correctly setup."""
    # Blocking call outside of langchain_core is allowed.
    time.sleep(0.01)  # noqa: ASYNC251
    with pytest.raises(BlockingError):
        # Blocking call from langchain_core raises BlockingError.
        sys_info.print_sys_info()
