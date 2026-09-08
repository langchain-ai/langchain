---
type: Factory
title: Chat Model Initialization with init_chat_model
description: Factory function for instantiating chat models from provider strings with unified configuration and runtime model switching.
tags: [chat-models, factory-pattern, initialization, model-parameters, configuration, provider-registry]
verified:
  - by: openwiki/0.5.0
    at: 2026-09-03T15:18:34.589Z
sources:
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-09-03T15:18:34.589Z" }
---

## Overview

`init_chat_model` is a factory function that creates chat model instances from a unified interface. It centralizes model instantiation across all supported provider integrations (OpenAI, Anthropic, Bedrock, Google Vertex AI, etc.), handles parameter routing to provider-specific constructors, and supports runtime model configuration via LangChain's `Runnable` configuration system.

The factory accepts a **model name with optional provider prefix** (e.g., `"openai:gpt-4"`, `"anthropic:claude-opus-4-7"`), infers the provider when unspecified, retrieves the provider's integration package, and instantiates the corresponding chat model class with translated kwargs.

**Core responsibilities:**
- Accept and parse model identifiers with or without provider prefixes
- Infer providers from model name prefixes using heuristics
- Dynamically import provider integration packages and classes
- Route provider-specific kwargs to provider constructors
- Support both fixed (non-configurable) and runtime-configurable (switch model/provider at invoke time) initialization modes
- Route requests through LangSmith gateway when provider is `langsmith`

## Location

**File**: `repo://libs/langchain_v1/langchain/chat_models/base.py`

**Public export**: `repo://libs/langchain_v1/langchain/chat_models/__init__.py#L5`

## Function Signature

```python
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel
```

**Parameters:**

- `model` (`str | None`): Model identifier, optionally with provider prefix (`"provider:model-name"`). If `None`, returns a configurable model that requires model name at runtime. Examples:
  - `"openai:gpt-5.5"` (explicit prefix)
  - `"gpt-5.5"` (inferred as OpenAI)
  - `None` (configurable at runtime)

- `model_provider` (`str | None`): Provider name as an alternative to prefix format. Used when provider is dynamic or needs to be independently configurable. Normalized to lowercase with underscores (e.g., `"azure-openai"` → `"azure_openai"`).

- `configurable_fields` (`Literal["any"] | list[str] | tuple[str, ...] | None`): 
  - `None`: No fields are configurable (fixed model, default if `model` is specified)
  - `"any"`: All parameters become configurable at runtime (⚠️ security: includes api_key, base_url)
  - `list[str] | tuple[str, ...]`: Specified parameter names (e.g., `("temperature", "max_tokens")`) are configurable
  - Defaults to `("model", "model_provider")` if `model` is `None`

- `config_prefix` (`str | None`): Optional namespace prefix for runtime config keys. Used when multiple configurable models exist in the same application. Config is accessed via `config["configurable"]["{config_prefix}_{param}"]`.

- `**kwargs`: Provider-specific parameters passed to the underlying chat model's constructor. Common parameters:
  - `temperature` (`float`): Randomness control (0–1 or provider-specific range)
  - `max_tokens` (`int`): Maximum output tokens
  - `timeout` (`float`): Request timeout in seconds
  - `max_retries` (`int`): Retry attempt limit
  - `base_url` (`str`): Custom API endpoint (OpenAI-compatible)
  - `rate_limiter` (`BaseRateLimiter`): Rate limiting instance
  - Provider-specific: `openai_api_key`, `anthropic_api_url`, `bedrock_region`, etc.

**Returns:**

- `BaseChatModel`: A fixed (non-configurable) chat model when `configurable_fields` is `None` (the default if `model` is specified)
- `_ConfigurableModel`: A wrapper runnable that defers model instantiation until `invoke`/`stream` is called with configuration, enabling runtime model selection

**Raises:**

- `TypeError`: If `model` is not a string (e.g., a model object is passed)
- `ValueError`: If provider cannot be inferred or is not supported
- `ImportError`: If the provider's integration package is not installed

## Model Name Parsing and Provider Inference

### Explicit Provider Prefix

If a colon (`:`) divides the model string and the prefix is a registered provider, it is extracted:

```python
init_chat_model("openai:gpt-5.5")     # provider='openai', model='gpt-5.5'
init_chat_model("anthropic:claude-opus-4-7")  # provider='anthropic', model='claude-opus-4-7'
```

### Bare Model Name with Inference

Without an explicit prefix, `_attempt_infer_model_provider` uses case-insensitive prefix matching:

| Model Prefix | Inferred Provider |
|---|---|
| `gpt-`, `o1`, `o3`, `chatgpt`, `text-davinci` | `openai` |
| `claude` | `anthropic` |
| `command` | `cohere` |
| `accounts/fireworks` | `fireworks` |
| `gemini` | `google_vertexai` (⚠️ deprecated default; changing to `google_genai` in next major release) |
| `amazon.`, `anthropic.`, `meta.` | `bedrock` |
| `mistral`, `mixtral` | `mistralai` |
| `deepseek` | `deepseek` |
| `grok` | `xai` |
| `sonar` | `perplexity` |
| `solar` | `upstage` |

**Example:**

```python
init_chat_model("gpt-4")  # inferred as openai:gpt-4
init_chat_model("claude-sonnet-4-5-20250929")  # inferred as anthropic
```

If inference fails and `model_provider` is not provided, a `ValueError` lists supported providers and suggests the documentation.

## Built-in Provider Registry

The `_BUILTIN_PROVIDERS` dictionary maps provider names to module paths, class names, and instantiation functions. Each entry is a tuple: `(module_path, class_name, creator_func)`.

**Representative Entries** (repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L97):

| Provider | Package | Class | Module | Notes |
|---|---|---|---|---|
| `openai` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | |
| `anthropic` | `langchain-anthropic` | `ChatAnthropic` | `langchain_anthropic` | |
| `azure_openai` | `langchain-openai` | `AzureChatOpenAI` | `langchain_openai` | |
| `azure_ai` | `langchain-azure-ai` | `AzureAIOpenAIApiChatModel` | `langchain_azure_ai.chat_models` | Submodule import |
| `google_vertexai` | `langchain-google-vertexai` | `ChatVertexAI` | `langchain_google_vertexai` | |
| `google_genai` | `langchain-google-genai` | `ChatGoogleGenerativeAI` | `langchain_google_genai` | |
| `anthropic_bedrock` | `langchain-aws` | `ChatAnthropicBedrock` | `langchain_aws` | Bedrock-hosted Anthropic |
| `bedrock` | `langchain-aws` | `ChatBedrock` | `langchain_aws` | Generic Bedrock models |
| `bedrock_converse` | `langchain-aws` | `ChatBedrockConverse` | `langchain_aws` | Bedrock Converse API |
| `cohere` | `langchain-cohere` | `ChatCohere` | `langchain_cohere` | |
| `deepseek` | `langchain-deepseek` | `ChatDeepSeek` | `langchain_deepseek` | |
| `fireworks` | `langchain-fireworks` | `ChatFireworks` | `langchain_fireworks` | |
| `groq` | `langchain-groq` | `ChatGroq` | `langchain_groq` | |
| `huggingface` | `langchain-huggingface` | `ChatHuggingFace` | `langchain_huggingface` | Uses `from_model_id()` |
| `ibm` | `langchain-ibm` | `ChatWatsonx` | `langchain_ibm` | Uses `model_id=` param |
| `litellm` | `langchain-litellm` | `ChatLiteLLM` | `langchain_litellm` | |
| `mistralai` | `langchain-mistralai` | `ChatMistralAI` | `langchain_mistralai` | |
| `nvidia` | `langchain-nvidia-ai-endpoints` | `ChatNVIDIA` | `langchain_nvidia_ai_endpoints` | |
| `ollama` | `langchain-ollama` | `ChatOllama` | `langchain_ollama` | Fallback to `langchain_community` |
| `openrouter` | `langchain-openrouter` | `ChatOpenRouter` | `langchain_openrouter` | |
| `perplexity` | `langchain-perplexity` | `ChatPerplexity` | `langchain_perplexity` | |
| `together` | `langchain-together` | `ChatTogether` | `langchain_together` | |
| `upstage` | `langchain-upstage` | `ChatUpstage` | `langchain_upstage` | |
| `xai` | `langchain-xai` | `ChatXAI` | `langchain_xai` | |
| `langsmith` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | Routes via LangSmith gateway |

**Design notes:**

- The registry is **not exhaustive**. Unlisted providers can still be used if their integration package is installed, but model name inference will not work; `model_provider` must be specified.
- Most entries use the standard `_call` creator function, which directly instantiates the class.
- Special creators: `huggingface` uses `from_model_id(model_id=...)`, `ibm` uses `model_id=...`, `langsmith` wraps instantiation with gateway configuration.

## Parameter Mapping and Creator Functions

The `_get_chat_model_creator` function retrieves the provider's creator function and returns a partially-applied callable:

```python
@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(provider: str) -> Callable[..., BaseChatModel]:
    # Look up provider in registry
    pkg, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    # Import module and get class
    module = _import_module(pkg, class_name)
    cls = getattr(module, class_name)
    # Return partial with class bound
    return functools.partial(creator_func, cls=cls)
```

**Standard Creator (`_call`)**:

```python
def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    return cls(**kwargs)
```

Forwards all kwargs directly to the provider's `__init__`.

**Special Creators**:

- **HuggingFace**: `lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs)`
  - Uses the class method `from_model_id` instead of direct instantiation
  - `model` parameter becomes `model_id=`

- **IBM**: `lambda cls, model, **kwargs: cls(model_id=model, **kwargs)`
  - Maps `model` to `model_id=` parameter in constructor

- **LangSmith Gateway** (`_init_langsmith`):
  - Calls `_apply_gateway_config` to inject gateway credentials and base URL from environment (or `LANGSMITH_GATEWAY` URL)
  - Sets `use_responses_api=True` to enable compatibility with gateway
  - Falls back to `LANGSMITH_API_KEY` if `LANGSMITH_GATEWAY_API_KEY` not set

```python
def _init_langsmith(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    _apply_gateway_config(
        kwargs,
        cls,
        base_url_field="openai_api_base",
        api_key_field="openai_api_key",
        provider_path="v1",
        api_key_env=("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"),
        default_base_url="https://gateway.smith.langchain.com/v1",
    )
    kwargs["use_responses_api"] = True
    return cls(**kwargs)
```

**Parameter Routing**:

All `**kwargs` passed to `init_chat_model` are forwarded to the provider's constructor. The provider's parameter validation enforces which kwargs are accepted. Common cross-provider kwargs (`temperature`, `max_tokens`, `timeout`, `max_retries`) work on most providers; provider-specific kwargs (e.g., `openai_api_key`, `anthropic_api_url`) are only valid for their target provider.

## Fixed Model Initialization

When `model` is specified and `configurable_fields` is `None` (default), `init_chat_model` immediately instantiates and returns a `BaseChatModel`:

```python
init_chat_model("gpt-4", temperature=0.7, max_tokens=500)
# Returns ChatOpenAI instance, ready to invoke
```

**Control flow** (repo://libs/langchain_v1/langchain/chat_models/base.py#L515-L520):

```python
if not configurable_fields:
    return _init_chat_model_helper(
        cast("str", model),
        model_provider=model_provider,
        **kwargs,
    )
```

The `_init_chat_model_helper` function parses the model, retrieves the creator, and instantiates:

```python
def _init_chat_model_helper(
    model: str,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    model, model_provider = _parse_model(model, model_provider)
    creator_func = _get_chat_model_creator(model_provider)
    return creator_func(model=model, **kwargs)
```

**Error handling:**

- If the package is missing: `ImportError` with suggestion to `pip install <package>`
- If the provider is unknown: `ValueError` listing all supported providers
- If model_provider inference fails: `ValueError` with docs link

## Runtime-Configurable Model Initialization

When `configurable_fields` is not `None` or `model` is `None`, `init_chat_model` returns a `_ConfigurableModel`, a `Runnable` wrapper that defers instantiation until config is provided at invoke time.

**Use cases:**

1. **No default model** – select model at runtime:
   ```python
   model = init_chat_model()  # No model specified
   model.invoke("hello", config={"configurable": {"model": "gpt-4"}})
   model.invoke("hello", config={"configurable": {"model": "claude-opus-4-7"}})
   ```

2. **Default model, switchable parameters** – override specific fields at runtime:
   ```python
   model = init_chat_model(
       "gpt-4",
       configurable_fields=("temperature", "max_tokens"),
       temperature=0.5,
       max_tokens=100
   )
   model.invoke(
       "hello",
       config={"configurable": {"temperature": 0.9, "max_tokens": 500}}
   )
   ```

3. **Default model, fully configurable** – switch model or any parameter at runtime:
   ```python
   model = init_chat_model(
       "gpt-4",
       configurable_fields="any",  # All fields configurable
       config_prefix="my_model"
   )
   model.invoke("hello")  # Uses gpt-4, temperature=None
   model.invoke(
       "hello",
       config={
           "configurable": {
               "my_model_model": "claude-opus-4-7",
               "my_model_temperature": 0.8
           }
       }
   )
   ```

### _ConfigurableModel

**Location**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L657-L1050`

`_ConfigurableModel` is a `Runnable[LanguageModelInput, Any]` that queues model initialization and operations until a config is provided:

**State:**
- `_default_config`: Dictionary of default parameter values (e.g., `{"model": "gpt-4", "temperature": 0.5}`)
- `_configurable_fields`: Which fields can be overridden at runtime (`"any"`, a list of field names)
- `_config_prefix`: Namespace for config keys (e.g., `"my_model_"`)
- `_queued_declarative_operations`: List of method calls (e.g., `bind_tools`, `with_structured_output`) to apply after model instantiation

**Lifecycle:**

1. **Instantiation**: `init_chat_model(...)` creates `_ConfigurableModel` with default config and queued operations
2. **Declarative operations** (e.g., `.bind_tools(...)`): Operations are queued; a new `_ConfigurableModel` is returned without mutation
3. **Invocation** (e.g., `.invoke(..., config=...)`): `_model(config)` is called to:
   - Merge default and runtime config
   - Call `_init_chat_model_helper` to instantiate the actual model
   - Apply all queued operations in order
   - Return the configured model instance
4. **Streaming/batch operations** delegate to the instantiated model

**Config merging** (repo://libs/langchain_v1/langchain/chat_models/base.py#L711-L727):

```python
def _model(self, config: RunnableConfig | None = None) -> Runnable[Any, Any]:
    params = {**self._default_config, **self._model_params(config)}
    model = _init_chat_model_helper(**params)
    for name, args, kwargs in self._queued_declarative_operations:
        model = getattr(model, name)(*args, **kwargs)
    return model

def _model_params(self, config: RunnableConfig | None) -> dict[str, Any]:
    config = ensure_config(config)
    # Extract configurable params and remove prefix
    model_params = {
        _remove_prefix(k, self._config_prefix): v
        for k, v in config.get("configurable", {}).items()
        if k.startswith(self._config_prefix)
    }
    # Filter to only allowed fields if not "any"
    if self._configurable_fields != "any":
        model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
    return model_params
```

**Declarative operations** (repo://libs/langchain_v1/langchain/chat_models/base.py#L681-L702):

Methods like `bind_tools` and `with_structured_output` are intercepted and queued instead of applied immediately:

```python
def __getattr__(self, name: str) -> Any:
    if name in _DECLARATIVE_METHODS:
        def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
            queued_declarative_operations = list(self._queued_declarative_operations)
            queued_declarative_operations.append((name, args, kwargs))
            return _ConfigurableModel(
                default_config=dict(self._default_config),
                configurable_fields=self._configurable_fields,
                config_prefix=self._config_prefix,
                queued_declarative_operations=queued_declarative_operations,
            )
        return queue
    # ... delegate to default model if one exists
```

**Caching**: Creator functions are cached with `@functools.lru_cache` to avoid redundant module imports.

## Common Parameter Mapping Examples

### Temperature and max_tokens

These are nearly universal but have different default values and ranges per provider:

```python
# OpenAI: temperature 0–2 (default 1)
init_chat_model("gpt-4", temperature=0.7, max_tokens=500)

# Anthropic: temperature 0–1 (default 1)
init_chat_model("claude-opus-4-7", temperature=0.7, max_tokens=500)

# Google Vertex AI: temperature 0–2
init_chat_model("google_vertexai:gemini-1.5-pro", temperature=0.7)
```

Check the provider's integration documentation for exact ranges and defaults.

### API Keys and Base URLs

Providers vary in parameter names:

```python
# OpenAI: openai_api_key, openai_api_base
init_chat_model("gpt-4", openai_api_key="...", openai_api_base="https://custom.com/v1")

# Anthropic: anthropic_api_key, anthropic_api_url
init_chat_model("claude-opus-4-7", anthropic_api_key="...", anthropic_api_url="https://custom.com")

# Vertex AI: uses GCP credentials from environment, or project_id, location
init_chat_model("google_vertexai:gemini-1.5-pro", project_id="my-project")
```

Environment variable fallbacks are provider-specific; check the integration package docs.

### Retry and Timeout

Common cross-provider params:

```python
init_chat_model(
    "gpt-4",
    max_retries=3,
    timeout=30.0,
)
```

### Bedrock Region and Model IDs

AWS Bedrock requires region and uses full model IDs:

```python
init_chat_model(
    "bedrock:amazon.titan-text-express-v1",
    region_name="us-east-1",
)
```

## Testing and Example Patterns

### Fixed Model Initialization

```python
from langchain.chat_models import init_chat_model

# Explicit provider prefix
llm = init_chat_model("openai:gpt-4", temperature=0)
response = llm.invoke("What is 2+2?")

# Inferred provider
llm = init_chat_model("gpt-4", temperature=0)
response = llm.invoke("What is 2+2?")

# Separate model_provider parameter
llm = init_chat_model("gpt-4", model_provider="openai", temperature=0)
```

### Configurable Model with Partial Override

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature", "max_tokens"),
    temperature=0.5,
    max_tokens=100,
)

# Use defaults
result = model.invoke("hello")

# Override at runtime
result = model.invoke(
    "hello",
    config={
        "configurable": {
            "temperature": 0.9,
            "max_tokens": 500,
        }
    }
)
```

### Configurable Model with No Default

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(temperature=0.5)  # No model specified

# Select model at runtime
result = model.invoke(
    "hello",
    config={"configurable": {"model": "gpt-4"}}
)

result = model.invoke(
    "hello",
    config={"configurable": {"model": "claude-opus-4-7"}}
)
```

### Chaining with Prompts

```python
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model("gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

chain = prompt | model
response = chain.invoke({"input": "What is 2+2?"})
```

### Binding Tools

```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    """Perform arithmetic."""
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

model = init_chat_model("gpt-4")
model_with_tools = model.bind_tools([Calculator])

result = model_with_tools.invoke("What is 2+2?")
```

### Configurable Model with Tools

```python
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

class Calculator(BaseModel):
    """Perform arithmetic."""
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

model = init_chat_model(
    "gpt-4",
    configurable_fields=("model", "model_provider"),
)
model_with_tools = model.bind_tools([Calculator])

# Use with default gpt-4
result = model_with_tools.invoke("What is 2+2?")

# Switch to Claude at runtime
result = model_with_tools.invoke(
    "What is 2+2?",
    config={"configurable": {"model": "claude-opus-4-7"}}
)
```

## LangSmith Gateway Integration

The `langsmith` provider bridges to LangSmith's unified LLM gateway, allowing multiple non-OpenAI models to be served through a common OpenAI-compatible API:

```python
init_chat_model("langsmith:moonshotai/kimi-k3")
```

**Configuration flow**:

1. `_init_langsmith` is invoked instead of standard `_call`
2. `_apply_gateway_config` reads:
   - `LANGSMITH_GATEWAY` or defaults to `https://gateway.smith.langchain.com`
   - `LANGSMITH_GATEWAY_API_KEY` (preferred) or `LANGSMITH_API_KEY`
   - Injects these as `openai_api_base` and `openai_api_key`
3. Sets `use_responses_api=True` for compatibility
4. Returns a `ChatOpenAI` instance pointing to the gateway

**Example**:

```python
import os
os.environ["LANGSMITH_GATEWAY_API_KEY"] = "..."
model = init_chat_model("langsmith:moonshotai/kimi-k3")
# Routes to https://gateway.smith.langchain.com/v1/chat/completions
```

## Extension and Adding New Providers

To add support for a new provider integration, update `_BUILTIN_PROVIDERS` in `repo://libs/langchain_v1/langchain/chat_models/base.py`:

1. Add an entry: `"provider_name": (module_path, ClassName, creator_func)`
2. If using standard instantiation, use `_call`
3. If the constructor uses a non-standard parameter for model name, create a custom creator
4. Ensure the provider module exports the class at the specified module path
5. The integration package must be pip-installable and named `langchain-<provider-name>` (with underscores converted to hyphens)

Example for a hypothetical "myai" provider:

```python
_BUILTIN_PROVIDERS = {
    ...
    "myai": ("langchain_myai", "ChatMyAI", _call),
}
```

Then install with `pip install langchain-myai` and use:

```python
init_chat_model("myai:my-model-v1")
# or
init_chat_model("my-model-v1", model_provider="myai")
```

Update model name prefix inference in `_attempt_infer_model_provider` if a stable, unambiguous prefix exists (e.g., all MyAI models start with `myai-`).

## Security Considerations

**API Key Exposure**: When `configurable_fields="any"`, all parameters including `api_key`, `openai_api_key`, `anthropic_api_key`, and `base_url` become runtime-configurable. In production, restrict configurable fields to safe parameters:

```python
# ❌ Unsafe: accepts any field, including secrets
model = init_chat_model("gpt-4", configurable_fields="any")

# ✅ Safe: whitelist only model switching and temperature
model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature", "max_tokens"),
)
```

**Runtime Configuration Source**: Validate that config dicts come from trusted sources. If config is derived from user input, filter keys to prevent unexpected parameter injection.
