import pytest

from langchain_standard_tests.integration_tests.chat_models import (
    ChatModelIntegrationTests,
)

# Rewrite assert statements for test suite so that implementations can
# see the full error message from failed asserts.
# https://docs.pytest.org/en/7.1.x/how-to/writing_plugins.html#assertion-rewriting
pytest.register_assert_rewrite("langchain_standard_tests.integration_tests.indexer")

__all__ = [
    "ChatModelIntegrationTests",
]
