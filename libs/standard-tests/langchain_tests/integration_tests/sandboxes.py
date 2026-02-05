"""Integration tests for the deepagents `SandboxProvider` abstraction.

Implementers should subclass this test suite and provide a fixture that returns a
clean `SandboxProvider` instance.

The provider is expected to support:

- `list()` returning all sandboxes visible to the provider
- `get_or_create(sandbox_id=None)` creating a sandbox
- `get_or_create(sandbox_id=...)` reconnecting to an existing sandbox without
  creating a new one
- `delete()` being idempotent

Example:
```python
from __future__ import annotations

from typing import Any

import pytest
from deepagents.backends.sandbox import SandboxProvider
from langchain_tests.integration_tests import SandboxProviderIntegrationTests

from langchain_acme_sandbox import AcmeSandboxProvider


class TestAcmeSandboxProviderStandard(SandboxProviderIntegrationTests):
    @pytest.fixture
    def sandbox_provider(self) -> SandboxProvider[Any]:
        # Return a provider instance in a clean state.
        return AcmeSandboxProvider(api_key="...")

    @property
    def has_async(self) -> bool:
        return True
```

"""

# ruff: noqa: E402

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pytest

deepagents = pytest.importorskip("deepagents")

from deepagents.backends.sandbox import SandboxNotFoundError, SandboxProvider

from langchain_tests.base import BaseStandardTests


class SandboxProviderIntegrationTests(BaseStandardTests):
    """Base class for sandbox provider integration tests."""

    @abstractmethod
    @pytest.fixture
    def sandbox_provider(self) -> SandboxProvider[Any]:
        """Get a clean `SandboxProvider` instance."""

    @property
    def has_sync(self) -> bool:
        """Configurable property to enable or disable sync tests."""
        return True

    @property
    def has_async(self) -> bool:
        """Configurable property to enable or disable async tests."""
        return True

    def test_list_is_empty(self, sandbox_provider: SandboxProvider[Any]) -> None:
        """Test that the `SandboxProvider` starts from a blank slate."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        assert sandbox_provider.list()["items"] == []

    def test_list_schema_empty(self, sandbox_provider: SandboxProvider[Any]) -> None:
        """Test the return schema of `list()` on an empty provider."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        result = sandbox_provider.list()
        assert set(result) == {"items", "cursor"}
        assert isinstance(result["items"], list)
        assert isinstance(result["cursor"], str | type(None))
        assert result["items"] == []

    def test_create_then_list_schema_then_delete_restores_empty(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Test create is visible in list, and delete returns provider to empty."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        before = sandbox_provider.list()
        assert before["items"] == []

        backend = sandbox_provider.get_or_create(sandbox_id=None)
        assert isinstance(backend.id, str)
        created_id = backend.id

        after_create = sandbox_provider.list()
        assert len(after_create["items"]) == 1
        assert {item["sandbox_id"] for item in after_create["items"]} == {created_id}
        assert isinstance(after_create["items"][0]["sandbox_id"], str)

        sandbox_provider.delete(sandbox_id=created_id)

        after_delete = sandbox_provider.list()
        assert after_delete["items"] == []

    def test_get_or_create_existing_does_not_create_new(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Test reconnecting to an existing sandbox does not create a new one."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        assert sandbox_provider.list()["items"] == []

        backend = sandbox_provider.get_or_create(sandbox_id=None)
        created_id = backend.id

        after_create = sandbox_provider.list()
        assert {item["sandbox_id"] for item in after_create["items"]} == {created_id}

        _reconnected = sandbox_provider.get_or_create(sandbox_id=created_id)
        after_reconnect = sandbox_provider.list()
        assert {item["sandbox_id"] for item in after_reconnect["items"]} == {created_id}

        sandbox_provider.delete(sandbox_id=created_id)
        assert sandbox_provider.list()["items"] == []

    def test_get_or_create_missing_id_raises(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Test missing sandbox_id raises a `SandboxNotFoundError`."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        assert sandbox_provider.list()["items"] == []

        missing_id = "definitely-not-a-real-sandbox-id"
        with pytest.raises(SandboxNotFoundError):
            sandbox_provider.get_or_create(sandbox_id=missing_id)

        assert sandbox_provider.list()["items"] == []

    def test_delete_is_idempotent(self, sandbox_provider: SandboxProvider[Any]) -> None:
        """Test `delete()` is idempotent and safe to call for missing IDs."""
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        assert sandbox_provider.list()["items"] == []

        backend = sandbox_provider.get_or_create(sandbox_id=None)
        created_id = backend.id

        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id="definitely-not-a-real-sandbox-id")

        assert sandbox_provider.list()["items"] == []

    async def test_async_list_is_empty(
        self, sandbox_provider: SandboxProvider[Any]
    ) -> None:
        """Async: test that the `SandboxProvider` starts from a blank slate."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        assert (await sandbox_provider.alist())["items"] == []

    async def test_async_list_schema_empty(
        self, sandbox_provider: SandboxProvider[Any]
    ) -> None:
        """Async: test the return schema of `list()` on an empty provider."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        result = await sandbox_provider.alist()
        assert set(result) == {"items", "cursor"}
        assert isinstance(result["items"], list)
        assert isinstance(result["cursor"], str | type(None))
        assert result["items"] == []

    async def test_async_create_then_list_schema_then_delete_restores_empty(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Async: test create is visible in list, and delete restores empty."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        before = await sandbox_provider.alist()
        assert before["items"] == []

        backend = await sandbox_provider.aget_or_create(sandbox_id=None)
        assert isinstance(backend.id, str)
        created_id = backend.id

        after_create = await sandbox_provider.alist()
        assert len(after_create["items"]) == 1
        assert {item["sandbox_id"] for item in after_create["items"]} == {created_id}
        assert isinstance(after_create["items"][0]["sandbox_id"], str)

        await sandbox_provider.adelete(sandbox_id=created_id)

        after_delete = await sandbox_provider.alist()
        assert after_delete["items"] == []

    async def test_async_get_or_create_existing_does_not_create_new(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Async: test reconnecting to an existing sandbox does not create a new one."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        assert (await sandbox_provider.alist())["items"] == []

        backend = await sandbox_provider.aget_or_create(sandbox_id=None)
        created_id = backend.id

        after_create = await sandbox_provider.alist()
        assert {item["sandbox_id"] for item in after_create["items"]} == {created_id}

        _reconnected = await sandbox_provider.aget_or_create(sandbox_id=created_id)
        after_reconnect = await sandbox_provider.alist()
        assert {item["sandbox_id"] for item in after_reconnect["items"]} == {created_id}

        await sandbox_provider.adelete(sandbox_id=created_id)
        assert (await sandbox_provider.alist())["items"] == []

    async def test_async_get_or_create_missing_id_raises(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Async: test missing sandbox_id raises a `SandboxNotFoundError`."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        assert (await sandbox_provider.alist())["items"] == []

        missing_id = "definitely-not-a-real-sandbox-id"
        with pytest.raises(SandboxNotFoundError):
            await sandbox_provider.aget_or_create(sandbox_id=missing_id)

        assert (await sandbox_provider.alist())["items"] == []

    async def test_async_delete_is_idempotent(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        """Async: test `delete()` is idempotent and safe to call for missing IDs."""
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        assert (await sandbox_provider.alist())["items"] == []

        backend = await sandbox_provider.aget_or_create(sandbox_id=None)
        created_id = backend.id

        await sandbox_provider.adelete(sandbox_id=created_id)
        await sandbox_provider.adelete(sandbox_id=created_id)
        await sandbox_provider.adelete(sandbox_id="definitely-not-a-real-sandbox-id")

        assert (await sandbox_provider.alist())["items"] == []
