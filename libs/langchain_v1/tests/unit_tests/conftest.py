"""Configuration for unit tests."""

from collections.abc import Sequence
from importlib import util
from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer
from langchain_tests.conftest import (
    _base_vcr_config as _base_vcr_config,
)
from vcr import VCR

_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("x-openai-client-user-agent", "PLACEHOLDER"),
]


def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers from the request."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    return request


def remove_response_headers(response: dict) -> dict:
    """Remove sensitive headers from the response."""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config(_base_vcr_config: dict) -> dict:  # noqa: F811
    """Extend the default configuration coming from langchain_tests."""
    config = _base_vcr_config.copy()
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:  # noqa: ARG001
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())


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


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence[pytest.Function]) -> None:
    """Add implementations for handling custom markers.

    At the moment, this adds support for a custom `requires` marker.

    The `requires` marker is used to denote tests that require one or more packages
    to be installed to run. If the package is not installed, the test is skipped.

    The `requires` marker syntax is:

    ```python
    @pytest.mark.requires("package1", "package2")
    def test_something(): ...
    ```
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
                            pytest.mark.skip(reason=f"Requires pkg: `{pkg}`"),
                        )
                        break
        elif only_extended:
            item.add_marker(
                pytest.mark.skip(reason="Skipping not an extended test."),
            )
