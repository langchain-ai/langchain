from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

deepagents = pytest.importorskip("deepagents")

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.sandbox import (
    SandboxListResponse,
    SandboxNotFoundError,
    SandboxProvider,
)

from langchain_tests.integration_tests.sandboxes import SandboxProviderIntegrationTests


@dataclass(frozen=True, slots=True)
class _InMemorySandboxBackend:
    sandbox_id: str

    @property
    def id(self) -> str:
        return self.sandbox_id

    def execute(self, command: str) -> ExecuteResponse:
        return ExecuteResponse(output=f"executed: {command}")

    def glob(self, *, path: str, pattern: str) -> list[FileInfo]:
        _ = (path, pattern)
        return []

    def write_file(self, *, path: str, content: str) -> WriteResult:
        _ = (path, content)
        msg = "in-memory backend does not persist files"
        raise NotImplementedError(msg)

    def edit_file(self, *, path: str, old: str, new: str) -> EditResult:
        _ = (path, old, new)
        msg = "in-memory backend does not persist files"
        raise NotImplementedError(msg)

    def grep(self, *, path: str, pattern: str) -> list[GrepMatch]:
        _ = (path, pattern)
        return []

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        _ = files
        return []

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        _ = paths
        return []


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
