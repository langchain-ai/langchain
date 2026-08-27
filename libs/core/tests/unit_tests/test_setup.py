import time

import pytest
from blockbuster import BlockingError

from langchain_core import sys_info


async def test_blockbuster_setup() -> None:
    """Check if blockbuster is correctly setup."""
    # Blocking call outside of langchain_core is allowed.
    time.sleep(0.01)  # noqa: ASYNC251
    with pytest.raises(BlockingError):
        # Blocking call from langchain_core raises BlockingError.
        sys_info.print_sys_info()
