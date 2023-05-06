import pytest
import importlib

from importlib import util
from typing import Dict


def pytest_collection_modifyitems(config, items):
    """Adds a marker to tests that require a library to be installed to run."""
    # Mapping from the name of a package to whether it is installed or not.
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

                # If the package is installed, we can continue, if not
                # we break immediately and set all_exist to False
                if not required_pkgs_info[pkg]:
                    item.add_marker(pytest.mark.skip(reason=f"requires pkg: `{pkg}`"))
                    break
