# langchain-typesafe

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-typesafe?label=%20)](https://pypi.org/project/langchain-typesafe/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-typesafe)](https://opensource.org/licenses/MIT)

## Installation

```bash
uv add langchain-typesafe
```

Set the `TYPESAFE_API_KEY` environment variable before making requests. `TYPESAFE_BASE_URL` and `TYPESAFE_DEFAULT_MODEL` are also honored, and explicit constructor arguments take precedence over both.

## Usage

`TypeSafeClassifier` is a LangChain `Runnable` for probabilistic classification and scoring with TypeSafe. It wraps the official [TypeSafe Python SDK](https://docs.typesafe.ai/sdk/python), so questions, answers, retries, and error semantics are the provider's own; this package adds the `Runnable` interface, LangChain tracing, and support for LangChain messages as input state.

```python
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

classifier = TypeSafeClassifier()
result = classifier.invoke(
    {
        "state": "Stripe has failed to connect for three days. Help ASAP.",
        "questions": {
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
        },
    }
)
print(result.choices["department"].choice)
print(result.nouls["urgent"].noul)
print(result.scores["frustration"].score)
```

Use `await classifier.ainvoke(...)` for asynchronous applications. As a `Runnable`, the classifier can also be composed with other LangChain runnables and supports standard batching, callbacks, and tracing.

`invoke` returns the SDK's `SystemOneResponse`. This package exports only `TypeSafeClassifier`, `ClassificationRequest`, the three question types, and the `State` type; everything else — answers, responses, `RetryPolicy`, and the exception types — is imported from `typesafe_sdk`, which is where it is documented.

### LangChain messages as state

A `BaseMessage` or a sequence of them can be passed directly and is converted to objects with `role` and `content` fields, which is the common case when classifying agent context:

```python
response = classifier.invoke(
    {"state": state.messages, "questions": questions}
)
```

To embed messages in a larger structure, convert them where you build it:

```python
from langchain_core.messages import HumanMessage, convert_to_openai_messages

response = classifier.invoke(
    {
        "state": {
            "conversation": convert_to_openai_messages(
                [HumanMessage("My payouts have failed for three days. Help!")]
            ),
            "account_tier": "enterprise",
        },
        "questions": questions,
    }
)
```

### Retries

TypeSafe asks clients to back off and retry on `429 Too Many Requests` and `529 Overloaded`. The SDK's default retry policy does this automatically, retrying HTTP 408, 429, and 5xx responses along with connection and timeout failures, using exponential backoff that honors the provider's retry headers. Pass a policy to change or disable it:

```python
from typesafe_sdk import RetryPolicy

classifier = TypeSafeClassifier(
    retry=RetryPolicy(max_retries=5, timeout=20.0),
)
```

### Client lifecycle and custom clients

TypeSafe clients are created during initialization, so a missing or invalid API key fails immediately rather than on the first request. Keep classifier instances long-lived to benefit from connection pooling, and use the classifier as a context manager, or call `close` and `aclose`, when deterministic cleanup is required:

```python
with TypeSafeClassifier() as classifier:
    result = classifier.invoke(
        {
            "state": "Production is down.",
            "questions": {"urgent": Noul(instructions="Is this urgent?")},
        }
    )
```

Applications that need custom transports, proxies, or shared connection pools can inject either client independently:

```python
import httpx2
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient

classifier = TypeSafeClassifier(
    client=TypeSafeClient(http_client=httpx2.Client(proxy="http://proxy.internal")),
    async_client=AsyncTypeSafeClient(
        http_client=httpx2.AsyncClient(proxy="http://proxy.internal")
    ),
)
```

Injected clients are used as-is; the application retains responsibility for their lifecycle, and `close` and `aclose` leave them open.

### Error handling

Provider errors also inherit from LangChain's standard model-error hierarchy. Applications can therefore catch a TypeSafe-specific error when provider metadata is needed, or a LangChain error when handling several model providers uniformly:

```python
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError
from typesafe_sdk import TypeSafeRateLimitError

try:
    response = classifier.invoke(
        {"state": "Classify this message.", "questions": questions}
    )
except TypeSafeRateLimitError as error:
    print(error.request_id, error.retry_after_ms)
except (ModelAuthenticationError, ModelRateLimitError):
    handle_model_error()
```

Each error raised by this package is a subclass of both the corresponding `typesafe_sdk` exception and the matching LangChain `ModelError`, so both imports above catch it. Because every error is reachable through one of those two, this package does not re-export the exception types itself. `TypeSafeAPIError` exposes the response status, parsed body, headers, sanitized endpoint, and request ID.

## Documentation

See the [TypeSafe documentation](https://docs.typesafe.ai/) for model and question semantics. LangChain API reference documentation is available at [reference.langchain.com](https://reference.langchain.com/python/integrations/langchain_typesafe/).

## Contributing

For contribution instructions, see the [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview).
