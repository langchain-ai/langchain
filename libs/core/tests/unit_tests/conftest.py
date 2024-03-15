"""Configuration for unit tests."""
from importlib import util
from typing import Dict, Sequence

import pytest
from _pytest.config import Config
from _pytest.terminal import TerminalReporter
from pytest import Function, Parser

from langchain_core.pydantic import _PYDANTIC_VERSION
from langchain_core.pydantic.config import USE_PYDANTIC_V2

# The maximum number of failed tests to allow when running with
# This number should only be decreased over time until we're at 0!
MAX_FAILED_LC_PYDANTIC_2_MIGRATION = 100


def pytest_addoption(parser: Parser) -> None:
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
    parser.addoption(
        "--max-fail",
        type=int,
        default=0,
        help="Maximum number of failed tests to allow. "
        "Should only be set for LC_PYDANTIC_V2_EXPERIMENTAL=true.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Initialize the count of passed and failed tests."""
    session.count_failed = 0  # type: ignore[attr-defined]


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore
    outcome = yield
    result = outcome.get_result()

    if result.when == "call" and result.failed:
        item.session.count_failed += 1  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Exit with a non-zero status if not enough tests pass."""
    max_fail = session.config.getoption(
        "--max-fail", default=MAX_FAILED_LC_PYDANTIC_2_MIGRATION
    )
    if max_fail > 0 and not USE_PYDANTIC_V2:
        raise ValueError(
            "The `--max-fail` option should only be set when "
            "running with `LC_PYDANTIC_V2_EXPERIMENTAL=true`."
        )
    # This will set up a ratchet approach so that the number of failures
    # has to go down over time.
    if session.count_failed > max_fail:  # type: ignore[attr-defined]
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        reporter.section("Session errors", sep="-", red=True, bold=True)  # type: ignore[union-attr]
        reporter.line(  # type: ignore[union-attr]
            f"Regression in pydantic v2 migration. Expected at most {max_fail} failed "
            f"tests. Instead found {session.count_failed} failed tests."  # type: ignore[attr-defined]
        )
    else:
        session.exitstatus = 0


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int, config: Config
) -> None:
    """Add custom information to the terminal summary."""
    terminalreporter.write_sep("-", title="Pydantic Configuration")
    terminalreporter.write_line(f"Testing with pydantic version {_PYDANTIC_VERSION}.")
    # Let's print out the value of USE_PYDANTIC_V2
    terminalreporter.write_line(
        f"USE_PYDANTIC_V2: {USE_PYDANTIC_V2}. "
        f"Enable with `LC_PYDANTIC_V2_EXPERIMENTAL=true` env variable "
        f"and pydantic>=2 installed."
    )


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

    only_extended = config.getoption("--only-extended") or False
    only_core = config.getoption("--only-core") or False

    if only_extended and only_core:
        raise ValueError("Cannot specify both `--only-extended` and `--only-core`.")

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
        else:
            if only_extended:
                item.add_marker(
                    pytest.mark.skip(reason="Skipping not an extended test.")
                )
