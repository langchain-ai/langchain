from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]

# Pristine, unpatched httpx transport handlers captured at import time (before
# any cassette is installed). VCR monkeypatches these while a cassette is
# active and restores them on teardown -- unless saving the cassette raises,
# in which case `vcr.cassette.CassetteContextDecorator.__exit__` skips its
# un-patch step and leaves the transport patched. That leak then intercepts
# every later request in the same worker (e.g. live integration tests get a
# stale cassette and fail with confusing connection errors).
_PRISTINE_HTTPX_TRANSPORTS = {
    (httpx.HTTPTransport, "handle_request"): httpx.HTTPTransport.handle_request,
    (
        httpx.AsyncHTTPTransport,
        "handle_async_request",
    ): httpx.AsyncHTTPTransport.handle_async_request,
}


def _restore_httpx_transports() -> None:
    """Undo any VCR monkeypatching left behind on httpx transports."""
    for (cls, attribute), original in _PRISTINE_HTTPX_TRANSPORTS.items():
        if getattr(cls, attribute) is not original:
            setattr(cls, attribute, original)


@pytest.fixture(autouse=True)
def _guard_httpx_transport_patches(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Stop leaked VCR cassette patches from breaking live tests.

    A `@pytest.mark.vcr` test whose cassette fails to save can leave httpx
    patched for the rest of the worker session. Tests that hit the network
    (those without a `vcr` marker) then replay against the wrong cassette and
    fail. Restoring the pristine transports around every non-VCR test keeps
    such a leak from cascading, without interfering with VCR's own patching
    during cassette-backed tests.
    """
    uses_vcr = request.node.get_closest_marker("vcr") is not None
    if not uses_vcr:
        _restore_httpx_transports()
    yield
    if not uses_vcr:
        _restore_httpx_transports()


def remove_request_headers(request: Any) -> Any:
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    return request


def remove_response_headers(response: dict) -> dict:
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    config["ignore_hosts"] = [
        *config.get("ignore_hosts", []),
        "api.smith.langchain.com",
    ]

    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
