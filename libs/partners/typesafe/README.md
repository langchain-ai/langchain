# langchain-typesafe

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-typesafe?label=%20)](https://pypi.org/project/langchain-typesafe/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-typesafe)](https://opensource.org/licenses/MIT)

## Installation

```bash
uv add langchain-typesafe
```

Set the `TYPESAFE_API_KEY` environment variable before making requests.

## Usage

`TypeSafeClassifier` is a LangChain `Runnable` for probabilistic classification and scoring with TypeSafe.

```python
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

classifier = TypeSafeClassifier(
    questions={
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payment or subscription issues",
                "technical": "Product or integration issues",
            },
        ),
        "urgent": Noul(instructions="Does this message express urgency?"),
        "frustration": Score(
            instructions="How frustrated does the customer appear?",
            criteria=["calm", "frustrated", "angry"],
        ),
    }
)

result = classifier.invoke("Stripe has failed to connect for three days. Help ASAP.")
print(result.choices["department"].choice)
print(result.nouls["urgent"].noul)
print(result.scores["frustration"].score)
```

Use `await classifier.ainvoke(...)` for asynchronous applications. As a `Runnable`, the classifier can also be composed with other LangChain runnables and supports standard batching, callbacks, and tracing.

### Experimental middleware

Install the experimental extra to use TypeSafe-powered agent middleware. For example, `ModelRouterMiddleware` routes an agent to a model selected by a TypeSafe `Choice` question:

```bash
uv add "langchain-typesafe[experimental]"
```

```python
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model="openai:gpt-5-mini",
            criteria="Simple, well-scoped tasks.",
        ),
        "powerful": ModelChoice(
            model=powerful_model,
            criteria="Complex tasks requiring deeper reasoning.",
        ),
    },
    instructions="Choose the least costly model suited to the task.",
)
agent = create_agent("openai:gpt-5-mini", middleware=[router])
```

The middleware classifies the latest human message once per agent run and stores the complete `ChoiceAnswer` in agent state, keeping its probabilities and confidence available to applications and traces. Classifier failures propagate instead of silently selecting another model. APIs under `langchain_typesafe.experimental` may change without notice.

### LangChain messages as state

`BaseMessage` objects and message sequences can appear at the root or anywhere inside JSON state. The integration recursively converts them to objects with `role` and `content` fields while preserving surrounding application data:

```python
from langchain_core.messages import HumanMessage, SystemMessage

response = classifier.invoke(
    {
        "conversation": [
            SystemMessage("You are reviewing a customer support conversation."),
            HumanMessage("My payouts have failed for three days. Help!"),
        ],
        "account_tier": "enterprise",
    }
)
```

### Custom HTTP clients

The classifier creates sync and async `httpx2` clients when they are not supplied. Applications that need custom transports, proxies, or shared connection pools can inject either client independently:

```python
import httpx2

classifier = TypeSafeClassifier(
    questions={"urgent": Noul(instructions="Is this urgent?")},
    client=httpx2.Client(proxy="http://proxy.internal"),
    async_client=httpx2.AsyncClient(proxy="http://proxy.internal"),
)
```

Injected clients are used as-is, and the application retains responsibility for their lifecycle.

### Error handling

Provider errors also inherit from LangChain's standard model-error hierarchy. Applications can therefore catch a TypeSafe-specific error when provider metadata is needed, or a LangChain error when handling several model providers uniformly:

```python
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from langchain_typesafe import TypeSafeRateLimitError

try:
    response = classifier.invoke("Classify this message.")
except TypeSafeRateLimitError as error:
    print(error.request_id, error.retry_after_ms)
except (ModelAuthenticationError, ModelRateLimitError):
    handle_model_error()
```

`TypeSafeAPIError` exposes the response status, parsed body, headers, sanitized endpoint, and request ID. Connection, timeout, response-validation, and status-specific subclasses follow the names used by the TypeSafe Python SDK.

## Documentation

See the [TypeSafe documentation](https://docs.typesafe.ai/) for model and question semantics. LangChain API reference documentation is available at [reference.langchain.com](https://reference.langchain.com/python/integrations/langchain_typesafe/).

## Contributing

For contribution instructions, see the [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview).
