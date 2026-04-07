# 🦜️🔗 langchain-tests

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-tests?label=%20)](https://pypi.org/project/langchain-tests/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-tests)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-tests)](https://pypistats.org/packages/langchain-tests)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchain.svg?style=social&label=Follow%20%40LangChain)](https://x.com/langchain)

Looking for the JS/TS version? Check out [LangChain.js](https://github.com/langchain-ai/langchainjs).

## Quick Install

```bash
pip install langchain-tests
```

## 🤔 What is this?

This is a testing library for LangChain integrations. It contains the base classes for a standard set of tests.

## 📖 Documentation

For full documentation, see the [API reference](https://reference.langchain.com/python/langchain_tests/).

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

We encourage pinning your version to a specific version in order to avoid breaking your CI when we publish new tests. We recommend upgrading to the latest version periodically to make sure you have the latest tests.

Not pinning your version will ensure you always have the latest tests, but it may also break your CI if we introduce tests that your integration doesn't pass.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## Usage

To add standard tests to an integration package (e.g., for a chat model), you need to create

1. A unit test class that inherits from `ChatModelUnitTests`
2. An integration test class that inherits from `ChatModelIntegrationTests`

`tests/unit_tests/test_standard.py`:

```python
"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_parrot_chain import ChatParrotChain


class TestParrotChainStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatParrotChain
```

`tests/integration_tests/test_standard.py`:

```python
"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_parrot_chain import ChatParrotChain


class TestParrotChainStandard(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatParrotChain
```

## Reference

The following fixtures are configurable in the test classes. Anything not marked
as required is optional.

- `chat_model_class` (required): The class of the chat model to be tested
- `chat_model_params`: The keyword arguments to pass to the chat model constructor
- `chat_model_has_tool_calling`: Whether the chat model can call tools. By default, this is set to `hasattr(chat_model_class, 'bind_tools)`
- `chat_model_has_structured_output`: Whether the chat model can structured output. By default, this is set to `hasattr(chat_model_class, 'with_structured_output')`
