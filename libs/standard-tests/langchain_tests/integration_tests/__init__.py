"""Integration tests for LangChain components."""

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

_HAS_DEEPAGENTS = False
try:
    import deepagents  # noqa: F401
except ImportError:
    _HAS_DEEPAGENTS = False
else:
    _HAS_DEEPAGENTS = True
    pytest.register_assert_rewrite("langchain_tests.integration_tests.sandboxes")

from langchain_tests.integration_tests.base_store import (
    BaseStoreAsyncTests,
    BaseStoreSyncTests,
)
from langchain_tests.integration_tests.cache import (
    AsyncCacheTestSuite,
    SyncCacheTestSuite,
)
from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests
from langchain_tests.integration_tests.embeddings import EmbeddingsIntegrationTests
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests
from langchain_tests.integration_tests.tools import ToolsIntegrationTests
from langchain_tests.integration_tests.vectorstores import VectorStoreIntegrationTests

if _HAS_DEEPAGENTS:
    from langchain_tests.integration_tests.sandboxes import (
        SandboxIntegrationTests,
    )

__all__ = [
    "AsyncCacheTestSuite",
    "BaseStoreAsyncTests",
    "BaseStoreSyncTests",
    "ChatModelIntegrationTests",
    "EmbeddingsIntegrationTests",
    "RetrieversIntegrationTests",
    "SyncCacheTestSuite",
    "ToolsIntegrationTests",
    "VectorStoreIntegrationTests",
]

if _HAS_DEEPAGENTS:
    __all__ += ["SandboxIntegrationTests"]
