---
pagination_next: contributing/how_to/integrations/publish/
pagination_prev: contributing/how_to/integrations/
---
# How to implement and test a chat model integration

This guide walks through how to implement and test a custom [chat model](/docs/concepts/chat_models) that you have developed.

For testing, we will rely on the `langchain-tests` dependency we added in the previous [bootstrapping guide](/docs/contributing/how_to/integrations/package).

## Implementation

Let's say you're building a simple integration package that provides a `ChatParrotLink`
chat model integration for LangChain. Here's a simple example of what your project
structure might look like:

```plaintext
langchain-parrot-link/
├── langchain_parrot_link/
│   ├── __init__.py
│   └── chat_models.py
├── tests/
│   ├── __init__.py
│   └── test_chat_models.py
├── pyproject.toml
└── README.md
```

Following the [bootstrapping guide](/docs/contributing/how_to/integrations/package),
all of these files should already exist, except for
`chat_models.py` and `test_chat_models.py`. We will implement these files in this guide.

To implement `chat_models.py`, we copy the [implementation](/docs/how_to/custom_chat_model/#implementation) from our
[Custom Chat Model Guide](/docs/how_to/custom_chat_model). Refer to that guide for more detail.

<details>
    <summary>chat_models.py</summary>
```python title="langchain_parrot_link/chat_models.py"
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field


class ChatParrotLink(BaseChatModel):
    """A custom chat model that echoes the first `parrot_buffer_length` characters
    of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = ChatParrotLink(parrot_buffer_length=2, model="bird-brain-001")
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                 [HumanMessage(content="world")]])
    """

    model_name: str = Field(alias="model")
    """The name of the model"""
    parrot_buffer_length: int
    """The number of characters from the last message of the prompt to be echoed."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        # Replace this with actual logic to generate a response from a list
        # of messages.
        last_message = messages[-1]
        tokens = last_message.content[: self.parrot_buffer_length]
        ct_input_tokens = sum(len(message.content) for message in messages)
        ct_output_tokens = len(tokens)
        message = AIMessage(
            content=tokens,
            additional_kwargs={},  # Used to add additional payload to the message
            response_metadata={  # Use for response metadata
                "time_in_seconds": 3,
            },
            usage_metadata={
                "input_tokens": ct_input_tokens,
                "output_tokens": ct_output_tokens,
                "total_tokens": ct_input_tokens + ct_output_tokens,
            },
        )
        ##

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        last_message = messages[-1]
        tokens = str(last_message.content[: self.parrot_buffer_length])
        ct_input_tokens = sum(len(message.content) for message in messages)

        for token in tokens:
            usage_metadata = UsageMetadata(
                {
                    "input_tokens": ct_input_tokens,
                    "output_tokens": 1,
                    "total_tokens": ct_input_tokens + 1,
                }
            )
            ct_input_tokens = 0
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=token, usage_metadata=usage_metadata)
            )

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(token, chunk=chunk)

            yield chunk

        # Let's add some other information (e.g., response metadata)
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token(token, chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }
```
</details>

## Testing

To implement our test files, we will subclass test classes from the `langchain_tests` package. These test classes contain the tests that will be run. We will just need to configure what model is tested, what parameters it is tested with, and specify any tests that should be skipped.

### Setup

First we need to install certain dependencies. These include:

- `pytest`: For running tests
- `pytest-socket`: For running unit tests
- `pytest-asyncio`: For testing async functionality
- `langchain-tests`: For importing standard tests
- `langchain-core`: This should already be installed, but is needed to define our integration.

If you followed the previous [bootstrapping guide](/docs/contributing/how_to/integrations/package/), these should already be installed.

### Add and configure standard tests
There are two namespaces in the langchain-tests package:

[unit tests](../../../concepts/testing.mdx#unit-tests) (`langchain_tests.unit_tests`): designed to be used to test the component in isolation and without access to external services
[integration tests](../../../concepts/testing.mdx#integration-tests) (`langchain_tests.integration_tests`): designed to be used to test the component with access to external services (in particular, the external service that the component is designed to interact with).

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

And here is the corresponding snippet for integration tests:

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

These two snippets should be written into `tests/unit_tests/test_chat_models.py` and `tests/integration_tests/test_chat_models.py`, respectively.

:::note

LangChain standard tests test a range of behaviors, from the most basic requirements to optional capabilities like multi-modal support. The above implementation will likely need to be updated to specify any tests that should be ignored. See [below](#skipping-tests) for detail.

:::

### Run standard tests

After setting tests up, you would run these with the following commands from your project root:

```shell
# run unit tests without network access
pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests

# run integration tests
pytest --asyncio-mode=auto tests/integration_tests
```

Our objective is for the pytest run to be successful. That is,

1. If a feature is intended to be supported by the model, it passes;
2. If a feature is not intended to be supported by the model, it is skipped.

### Skipping tests

LangChain standard tests test a range of behaviors, from the most basic requirements (generating a response to a query) to optional capabilities like multi-modal support, tool-calling, or support for messages generated from other providers. Tests for "optional" capabilities are controlled via a [set of properties](https://python.langchain.com/api_reference/standard_tests/unit_tests/langchain_tests.unit_tests.chat_models.ChatModelTests.html) that can be overridden on the test model subclass.


### Test suite information and troubleshooting

What tests are run to test this integration?

If a test fails, what does that mean?

You can find information on the tests run for this integration in the [Standard Tests API Reference](https://python.langchain.com/api_reference/standard_tests/index.html).

// TODO: link to exact page for this integration test suite information
