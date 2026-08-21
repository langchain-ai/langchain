"""Regression tests for the httpx transport guard in `tests/conftest.py`.

vcrpy leaves httpx patched if saving a cassette raises during teardown, which
otherwise leaks the patch into unrelated (live) tests running on the same
worker. The `_restore_httpx_transports` helper and its autouse fixture undo
that leak so it can't cascade.
"""

from pathlib import Path

import httpx
import vcr  # type: ignore[import-untyped]
from vcr.persisters.filesystem import (  # type: ignore[import-untyped]
    CassetteNotFoundError,
)

from tests.conftest import _restore_httpx_transports


class _BoomPersister:
    """Persister that fails to load and errors while saving a cassette."""

    @staticmethod
    def load_cassette(*_args: object, **_kwargs: object) -> tuple[list, list]:
        raise CassetteNotFoundError

    @staticmethod
    def save_cassette(*_args: object, **_kwargs: object) -> None:
        msg = "serialization boom"
        raise RuntimeError(msg)


def _ok_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="ok")


def test_restore_httpx_transports_is_a_noop_when_unpatched() -> None:
    original = httpx.HTTPTransport.handle_request
    _restore_httpx_transports()
    assert httpx.HTTPTransport.handle_request is original


def test_restore_httpx_transports_reverts_leaked_vcr_patch(tmp_path: Path) -> None:
    original_sync = httpx.HTTPTransport.handle_request
    original_async = httpx.AsyncHTTPTransport.handle_async_request

    my_vcr = vcr.VCR(record_mode="once")
    my_vcr.register_persister(_BoomPersister())

    # Recording a new interaction makes the cassette dirty; the failing save
    # then trips the vcrpy bug that skips un-patching httpx on teardown.
    try:
        with my_vcr.use_cassette(str(tmp_path / "leak.yaml")):
            transport = httpx.MockTransport(_ok_response)
            with httpx.Client(transport=transport) as client:
                client.get("http://example.test/")
    except RuntimeError:
        pass

    # The leak is real: httpx is still patched by VCR.
    assert httpx.HTTPTransport.handle_request is not original_sync

    _restore_httpx_transports()

    assert httpx.HTTPTransport.handle_request is original_sync
    assert httpx.AsyncHTTPTransport.handle_async_request is original_async
