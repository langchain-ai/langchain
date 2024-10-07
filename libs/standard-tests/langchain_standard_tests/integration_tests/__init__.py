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
]

for module in modules:
    pytest.register_assert_rewrite(
        f"langchain_standard_tests.integration_tests.{module}"
    )

from langchain_standard_tests.integration_tests.chat_models import (
    ChatModelIntegrationTests,
)
from langchain_standard_tests.integration_tests.embeddings import (
    EmbeddingsIntegrationTests,
)

__all__ = [
    "ChatModelIntegrationTests",
    "EmbeddingsIntegrationTests",
]
