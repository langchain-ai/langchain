# ruff: noqa: E402
import pytest

# Rewrite assert statements for test suite so that implementations can
# see the full error message from failed asserts.
# https://docs.pytest.org/en/7.1.x/how-to/writing_plugins.html#assertion-rewriting
modules = [
    "chat_models",
    "embeddings",
]

for module in modules:
    pytest.register_assert_rewrite(f"langchain_standard_tests.unit_tests.{module}")

from langchain_standard_tests.unit_tests.chat_models import ChatModelUnitTests

__all__ = ["ChatModelUnitTests", "EmbeddingsUnitTests"]
