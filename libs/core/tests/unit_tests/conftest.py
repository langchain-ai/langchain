"""Configuration for unit tests."""

from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from importlib import util
from typing import Any, Optional
from uuid import UUID

import pytest
from pytest_mock import MockerFixture

# Conditional import for blockbuster - not available in all test environments
try:
    from blockbuster import BlockBuster, blockbuster_ctx
    HAS_BLOCKBUSTER = True
except ImportError:
    HAS_BLOCKBUSTER = False
    # Create mock types for type hints
    BlockBuster = Any
    blockbuster_ctx = nullcontext


@pytest.fixture(autouse=True)
def blockbuster() -> Iterator[Optional[BlockBuster]]:
    """Blockbuster fixture for test isolation - gracefully handles missing dependency."""
    if not HAS_BLOCKBUSTER:
        # When blockbuster is not available, provide a no-op fixture
        yield None
        return
        
    with blockbuster_ctx("langchain_core") as bb:
        for func in ["os.stat", "os.path.abspath"]:
            (
                bb.functions[func]
                .can_block_in("langchain_core/_api/internal.py", "is_caller_internal")
                .can_block_in("langchain_core/runnables/base.py", "__repr__")
                .can_block_in(
                    "langchain_core/beta/runnables/context.py", "aconfig_with_context"
                )
            )

        for func in ["os.stat", "io.TextIOWrapper.read"]:
            bb.functions[func].can_block_in(
                "langsmith/client.py", "_default_retry_config"
            )

        for bb_function in bb.functions.values():
            bb_function.can_block_in(
                "freezegun/api.py", "_get_cached_module_attributes"
            )

        yield bb


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options to pytest."""
    parser.addoption(
        "--only-extended",
        action="store_true",
        help="Only run extended tests. Does not allow skipping any extended tests.",
    )
    parser.addoption(
        "--only-core",
        action="store_true",
        help="Only run core tests. Never runs any extended tests.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Function]
) -> None:
    """Add implementations for handling custom markers.

    At the moment, this adds support for a custom `requires` marker.

    The `requires` marker is used to denote tests that require one or more packages
    to be installed to run. If the package is not installed, the test is skipped.

    The `requires` marker syntax is:

    .. code-block:: python

        @pytest.mark.requires("package1", "package2")
        def test_something():
            ...

    """
    # Mapping from the name of a package to whether it is installed or not.
    # Used to avoid repeated calls to `util.find_spec`
    required_pkgs_info: dict[str, bool] = {}

    only_extended = config.getoption("--only-extended") or False
    only_core = config.getoption("--only-core") or False

    if only_extended and only_core:
        msg = "Cannot specify both `--only-extended` and `--only-core`."
        raise ValueError(msg)

    for item in items:
        requires_marker = item.get_closest_marker("requires")
        if requires_marker is not None:
            if only_core:
                item.add_marker(pytest.mark.skip(reason="Skipping not a core test."))
                continue

            # Iterate through the list of required packages
            required_pkgs = requires_marker.args
            for pkg in required_pkgs:
                # If we haven't yet checked whether the pkg is installed
                # let's check it and store the result.
                if pkg not in required_pkgs_info:
                    try:
                        installed = util.find_spec(pkg) is not None
                    except Exception:
                        installed = False
                    required_pkgs_info[pkg] = installed

                if not required_pkgs_info[pkg]:
                    if only_extended:
                        pytest.fail(
                            f"Package `{pkg}` is not installed but is required for "
                            f"extended tests. Please install the given package and "
                            f"try again.",
                        )

                    else:
                        # If the package is not installed, we immediately break
                        # and mark the test as skipped.
                        item.add_marker(
                            pytest.mark.skip(reason=f"Requires pkg: `{pkg}`")
                        )
                        break
        elif only_extended:
            item.add_marker(pytest.mark.skip(reason="Skipping not an extended test."))


@pytest.fixture
def deterministic_uuids(mocker: MockerFixture) -> MockerFixture:
    side_effect = (
        UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10000)
    )
    return mocker.patch("uuid.uuid4", side_effect=side_effect)
