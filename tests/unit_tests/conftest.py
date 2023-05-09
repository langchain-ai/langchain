"""Configuration for unit tests."""
from importlib import util
from typing import Dict, Sequence

import pytest
from pytest import Config, Function


def pytest_collection_modifyitems(config: Config, items: Sequence[Function]) -> None:
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
    required_pkgs_info: Dict[str, bool] = {}

    for item in items:
        requires_marker = item.get_closest_marker("requires")
        if requires_marker is not None:
            # Iterate through the list of required packages
            required_pkgs = requires_marker.args
            for pkg in required_pkgs:
                # If we haven't yet checked whether the pkg is installed
                # let's check it and store the result.
                if pkg not in required_pkgs_info:
                    required_pkgs_info[pkg] = util.find_spec(pkg) is not None

                if not required_pkgs_info[pkg]:
                    # If the package is not installed, we immediately break
                    # and mark the test as skipped.
                    item.add_marker(pytest.mark.skip(reason=f"requires pkg: `{pkg}`"))
                    break
