"""Unit tests for LangChain components."""

# ruff: noqa: E402
import pytest

# Rewrite assert statements for test suite so that implementations can
# see the full error message from failed asserts.
# https://docs.pytest.org/en/7.1.x/how-to/writing_plugins.html#assertion-rewriting
modules = [
    "chat_models",
    "embeddings",
    "tools",
]

for module in modules:
    pytest.register_assert_rewrite(f"langchain_tests.unit_tests.{module}")

from langchain_tests.unit_tests.chat_models import ChatModelUnitTests
from langchain_tests.unit_tests.embeddings import EmbeddingsUnitTests
from langchain_tests.unit_tests.tools import ToolsUnitTests

__all__ = ["ChatModelUnitTests", "EmbeddingsUnitTests", "ToolsUnitTests"]
