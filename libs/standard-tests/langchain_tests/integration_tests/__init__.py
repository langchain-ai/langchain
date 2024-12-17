# ruff: noqa: E402
import pytest

# Rewrite assert statements for test suite so that implementations can
# see the full error message from failed asserts.
# https://docs.pytest.org/en/7.1.x/how-to/writing_plugins.html#assertion-rewriting
modules = [
    "base_store",
    "cache",
    "chat_models",
    "vectorstores",
    "embeddings",
    "tools",
    "retrievers",
]

for module in modules:
    pytest.register_assert_rewrite(f"langchain_tests.integration_tests.{module}")

from .base_store import BaseStoreAsyncTests, BaseStoreSyncTests
from .cache import AsyncCacheTestSuite, SyncCacheTestSuite
from .chat_models import ChatModelIntegrationTests
from .embeddings import EmbeddingsIntegrationTests
from .retrievers import RetrieversIntegrationTests
from .tools import ToolsIntegrationTests
from .vectorstores import VectorStoreIntegrationTests

__all__ = [
    "ChatModelIntegrationTests",
    "EmbeddingsIntegrationTests",
    "ToolsIntegrationTests",
    "BaseStoreAsyncTests",
    "BaseStoreSyncTests",
    "AsyncCacheTestSuite",
    "SyncCacheTestSuite",
    "VectorStoreIntegrationTests",
    "RetrieversIntegrationTests",
]
