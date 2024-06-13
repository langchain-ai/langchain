# langchain-standard-tests

This is an INTERNAL library for the LangChain project. It contains the base classes for
a standard set of tests.

## Installation

This package will NOT be regularly published to pypi. It is intended to be installed
directly from github at test time.

Pip:
    
    ```bash
    pip install git+https://github.com/langchain-ai/langchain.git#subdirectory=libs/standard-tests
    ```

Poetry:
    
    ```bash
    poetry add git+https://github.com/langchain-ai/langchain.git#subdirectory=libs/standard-tests
    ```

## Usage

To add standard tests to an integration package's e.g. ChatModel, you need to create

1. A unit test class that inherits from ChatModelUnitTests
2. An integration test class that inherits from ChatModelIntegrationTests

`tests/unit_tests/test_standard.py`:
    
```python
"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

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
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

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
