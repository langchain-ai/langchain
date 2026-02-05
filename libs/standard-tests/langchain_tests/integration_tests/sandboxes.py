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

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pytest
from deepagents.backends.sandbox import SandboxProvider

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

    def test_list_schema(self, sandbox_provider: SandboxProvider[Any]) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        result = sandbox_provider.list()
        assert set(result) == {"items", "cursor"}
        assert isinstance(result["items"], list)
        assert isinstance(result["cursor"], str | type(None))

        for item in result["items"]:
            assert isinstance(item["sandbox_id"], str)

    def test_create_visible_in_list_and_reconnect_does_not_create(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        before = sandbox_provider.list()
        before_ids = {item["sandbox_id"] for item in before["items"]}

        backend = sandbox_provider.get_or_create(sandbox_id=None)
        assert isinstance(backend.id, str)
        created_id = backend.id

        after_create = sandbox_provider.list()
        after_create_ids = {item["sandbox_id"] for item in after_create["items"]}
        assert created_id in after_create_ids
        assert len(after_create_ids) == len(before_ids) + 1

        _reconnected = sandbox_provider.get_or_create(sandbox_id=created_id)
        after_reconnect = sandbox_provider.list()
        after_reconnect_ids = {item["sandbox_id"] for item in after_reconnect["items"]}
        assert after_reconnect_ids == after_create_ids

        sandbox_provider.delete(sandbox_id=created_id)

    def test_delete_is_idempotent(self, sandbox_provider: SandboxProvider[Any]) -> None:
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        backend = sandbox_provider.get_or_create(sandbox_id=None)
        created_id = backend.id

        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id=created_id)
        sandbox_provider.delete(sandbox_id="definitely-not-a-real-sandbox-id")

    async def test_async_list_schema(
        self, sandbox_provider: SandboxProvider[Any]
    ) -> None:
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        result = await sandbox_provider.alist()
        assert set(result) == {"items", "cursor"}
        assert isinstance(result["items"], list)
        assert isinstance(result["cursor"], str | type(None))

        for item in result["items"]:
            assert isinstance(item["sandbox_id"], str)

    async def test_async_create_visible_in_list_and_reconnect_does_not_create(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        before = await sandbox_provider.alist()
        before_ids = {item["sandbox_id"] for item in before["items"]}

        backend = await sandbox_provider.aget_or_create(sandbox_id=None)
        assert isinstance(backend.id, str)
        created_id = backend.id

        after_create = await sandbox_provider.alist()
        after_create_ids = {item["sandbox_id"] for item in after_create["items"]}
        assert created_id in after_create_ids
        assert len(after_create_ids) == len(before_ids) + 1

        _reconnected = await sandbox_provider.aget_or_create(sandbox_id=created_id)
        after_reconnect = await sandbox_provider.alist()
        after_reconnect_ids = {item["sandbox_id"] for item in after_reconnect["items"]}
        assert after_reconnect_ids == after_create_ids

        await sandbox_provider.adelete(sandbox_id=created_id)

    async def test_async_delete_is_idempotent(
        self,
        sandbox_provider: SandboxProvider[Any],
    ) -> None:
        if not self.has_async:
            pytest.skip("Async tests not supported.")

        backend = await sandbox_provider.aget_or_create(sandbox_id=None)
        created_id = backend.id

        await sandbox_provider.adelete(sandbox_id=created_id)
        await sandbox_provider.adelete(sandbox_id=created_id)
        await sandbox_provider.adelete(sandbox_id="definitely-not-a-real-sandbox-id")
