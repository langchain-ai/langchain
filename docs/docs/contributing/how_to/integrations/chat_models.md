# Standard Tests (Chat Model)

This guide walks through how to create standard tests for a custom Chat Model that you have developed.

## Setup

First we need to install certain dependencies. These include:

- `pytest`: For running tests
- `pytest-socket`: For running unit tests
- `pytest-asyncio`: For testing async functionality
- `langchain-tests`: For importing standard tests
- `langchain-core`: This should already be installed, but is needed to define our integration.

```shell
pip install -U langchain-core pytest pytest-socket pytest-asyncio langchain-tests
```

## Add and configure standard tests
There are 2 namespaces in the langchain-tests package:

[unit tests](../../../concepts/testing.mdx#unit-tests) (langchain_tests.unit_tests): designed to be used to test the component in isolation and without access to external services
[integration tests](../../../concepts/testing.mdx#integration-tests) (langchain_tests.integration_tests): designed to be used to test the component with access to external services (in particular, the external service that the component is designed to interact with).

Both types of tests are implemented as [pytest class-based test suites](https://docs.pytest.org/en/7.1.x/getting-started.html#group-multiple-tests-in-a-class).

By subclassing the base classes for each type of standard test (see below), you get all of the standard tests for that type, and you can override the properties that the test suite uses to configure the tests.

Here's how you would configure the standard unit tests for the custom chat model:

```python
# title="tests/unit_tests/test_chat_models.py"
from typing import Type

from my_package.chat_models import MyChatModel
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatParrotLinkUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[MyChatModel]:
        return MyChatModel

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
```

```python
# title="tests/integration_tests/test_chat_models.py"
from typing import Type

from my_package.chat_models import MyChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[MyChatModel]:
        return MyChatModel

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
```

## Run standard tests

After setting tests up, you would run these with the following commands from your project root:

```shell
# run unit tests without network access
pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# run integration tests
pytest --asyncio-mode=auto tests/integration_tests
```

## Test suite information and troubleshooting

What tests are run to test this integration?

If a test fails, what does that mean?

You can find information on the tests run for this integration in the [Standard Tests API Reference](https://python.langchain.com/api_reference/standard_tests/index.html).

// TODO: link to exact page for this integration test suite information

## Skipping tests

Sometimes you do not need your integration to pass all tests, as you only rely on specific subsets of functionality.
If this is the case, you can turn off specific tests by...

// TODO: add instructions for how to turn off tests