from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxListResponse,
    SandboxNotFoundError,
    SandboxProvider,
)

from langchain_tests.integration_tests.sandboxes import SandboxProviderIntegrationTests


class _InMemorySandboxBackend(BaseSandbox):
    def __init__(self, sandbox_id: str) -> None:
        self._id = sandbox_id

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str) -> ExecuteResponse:
        _ = command
        return ExecuteResponse(output="foo")


class _InMemorySandboxProvider(SandboxProvider[dict[str, Any]]):
    def __init__(self) -> None:
        self._sandboxes: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def list(
        self, *, cursor: str | None = None, **kwargs: Any
    ) -> SandboxListResponse[dict[str, Any]]:
        _ = (cursor, kwargs)
        return {
            "items": [
                {"sandbox_id": sandbox_id, "metadata": metadata}
                for sandbox_id, metadata in self._sandboxes.items()
            ],
            "cursor": None,
        }

    def get_or_create(
        self, *, sandbox_id: str | None = None, **kwargs: Any
    ) -> _InMemorySandboxBackend:
        _ = kwargs
        if sandbox_id is None:
            self._counter += 1
            sandbox_id = f"sb_{self._counter:03d}"
            self._sandboxes[sandbox_id] = {}
            return _InMemorySandboxBackend(sandbox_id=sandbox_id)

        if sandbox_id not in self._sandboxes:
            msg = f"Sandbox {sandbox_id} not found"
            raise SandboxNotFoundError(msg)

        return _InMemorySandboxBackend(sandbox_id=sandbox_id)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        _ = kwargs
        self._sandboxes.pop(sandbox_id, None)


class TestInMemorySandboxProviderStandard(SandboxProviderIntegrationTests):
    @pytest.fixture
    def sandbox_provider(self) -> SandboxProvider[Any]:
        return _InMemorySandboxProvider()
