"""Configuration for unit tests."""

import json
from collections.abc import Iterator, Sequence
from importlib import util
from typing import Any

import pytest
from blockbuster import BlockBuster, blockbuster_ctx
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR

_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("x-openai-client-user-agent", "PLACEHOLDER"),
]


@pytest.fixture(autouse=True)
def blockbuster() -> Iterator[BlockBuster]:
    with blockbuster_ctx() as bb:
        yield bb


def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers from the request."""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    return request


def remove_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive headers from the response."""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["match_on"] = [m if m != "body" else "json_body" for m in config.get("match_on", [])]
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config


def _json_body_matcher(r1: Any, r2: Any) -> None:
    """Match request bodies as parsed JSON, ignoring key order."""
    b1 = r1.body or b""
    b2 = r2.body or b""
    if isinstance(b1, bytes):
        b1 = b1.decode("utf-8")
    if isinstance(b2, bytes):
        b2 = b2.decode("utf-8")
    try:
        j1 = json.loads(b1)
        j2 = json.loads(b2)
    except (json.JSONDecodeError, ValueError):
        assert b1 == b2, f"body mismatch (non-JSON):\n{b1}\n!=\n{b2}"
        return
    assert j1 == j2, f"body mismatch:\n{j1}\n!=\n{j2}"


def pytest_recording_configure(config: dict[str, Any], vcr: VCR) -> None:  # noqa: ARG001
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    vcr.register_matcher("json_body", _json_body_matcher)


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

    only_extended = config.getoption("--only-extended", default=False)
    only_core = config.getoption("--only-core", default=False)

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
