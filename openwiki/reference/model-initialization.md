---
type: API Reference
title: Model Initialization API Reference
description: Complete API reference for init_chat_model factory function, covering function signature, provider registry, model parsing, configuration, and runtime behavior.
tags: [chat-models, factory-pattern, initialization, api-reference, provider-registry, configuration]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-b09b1477098d69af6abaa5b4
    resource: repo://libs/langchain_v1/langchain/chat_models/__init__.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

## Overview

`init_chat_model` is a factory function exported from `langchain.chat_models` that creates chat model instances from a unified interface. It centralizes model instantiation across all supported provider integrations, handles parameter routing to provider-specific constructors, and supports both fixed (non-configurable) and runtime-configurable models via LangChain's Runnable configuration system.

**Location:**
- **Implementation**: `repo://libs/langchain_v1/langchain/chat_models/base.py`
- **Public export**: `repo://libs/langchain_v1/langchain/chat_models/__init__.py#L5`

## Function Signature and Overloads

The function is defined with type overloads to communicate return types clearly:

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

### Overloads

**Overload 1: Fixed model** (model specified, no configurable fields)
```python
@overload
def init_chat_model(
    model: str,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel: ...
```
Returns a concrete `BaseChatModel` instance ready to use immediately.

**Overload 2: Configurable model without default** (no model specified)
```python
@overload
def init_chat_model(
    model: None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...
```
Returns a `_ConfigurableModel` wrapper that requires `model` name at runtime via config.

**Overload 3: Configurable model** (any configurable_fields specification)
```python
@overload
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ...,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...
```
Returns a `_ConfigurableModel` wrapper for runtime parameter configuration.

## Parameters

### `model` (optional str)

Model identifier, optionally with provider prefix in the format `provider:model-name`.

**Behavior:**
- If `None`: Function returns a configurable model that requires model name at runtime. Defaults `configurable_fields` to `("model", "model_provider")` if not specified.
- If a string: Can be prefixed (e.g., `"openai:gpt-4"`) or bare (e.g., `"gpt-4"`). Bare names trigger provider inference.
- If not a string: Raises `TypeError` if model object instead of string is passed.

**Examples:**
```python
init_chat_model("openai:gpt-4")  # explicit prefix
init_chat_model("gpt-4")  # inferred as openai
init_chat_model("claude-opus-4-7")  # inferred as anthropic
init_chat_model()  # configurable model, no default
```

### `model_provider` (optional str)

Provider name as an alternative to prefix format on `model`. Normalized to lowercase with underscores (e.g., `"azure-openai"` → `"azure_openai"`).

**Use cases:**
- Provider is dynamic (read from environment, config, or function parameter)
- You want `model` and `model_provider` to be independently configurable at runtime
- The provider must be specified separately from the model name

**Supported values:** See [Built-in Provider Registry](#built-in-provider-registry)

**Behavior:**
- If specified, overrides any prefix detection in `model`
- Applied after model name parsing and prefix extraction
- Must match a key in `_BUILTIN_PROVIDERS` or be installable via a custom integration package

**Example:**
```python
# These are equivalent:
init_chat_model("openai:gpt-4", temperature=0)
init_chat_model("gpt-4", model_provider="openai", temperature=0)

# Dynamic provider selection:
provider = os.getenv("LLM_PROVIDER", "openai")
model = init_chat_model("gpt-4", model_provider=provider)
```

### `configurable_fields` (optional)

Specifies which model parameters are configurable at runtime. Controls whether a fixed model or configurable wrapper is returned.

**Values:**
- `None` (default if `model` is specified): No fields are configurable; returns a fixed `BaseChatModel`
- `"any"`: All parameters become configurable at runtime (⚠️ security: includes `api_key`, `base_url`)
- `list[str] | tuple[str, ...]`: Named parameters (e.g., `("temperature", "max_tokens")`) become configurable
- Defaults to `("model", "model_provider")` if `model` is `None`

**Behavior:**
- When configurable fields are specified, returns a `_ConfigurableModel` wrapper instead of a fixed model
- Configurable parameters are resolved at invoke time from `config["configurable"]` dict
- Non-configurable parameters use values from constructor kwargs or defaults
- Field names are relative to `config_prefix` (prefix is stripped before lookup)

**Security considerations:**
```python
# ❌ Unsafe: All fields including api_key are configurable
model = init_chat_model("gpt-4", configurable_fields="any")

# ✅ Safe: Only temperature and max_tokens are configurable
model = init_chat_model("gpt-4", configurable_fields=("temperature", "max_tokens"))
```

**Examples:**
```python
# Fixed model
model = init_chat_model("gpt-4", temperature=0)
model.invoke("hello")  # uses temperature=0

# Configurable model with defaults
model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature",),
    temperature=0
)
model.invoke("hello")  # uses temperature=0
model.invoke("hello", config={"configurable": {"temperature": 0.9}})  # overrides to 0.9

# Configurable model, no default
model = init_chat_model(configurable_fields=("model", "model_provider"))
model.invoke("hello", config={"configurable": {"model": "gpt-4"}})  # must specify model
```

### `config_prefix` (optional str)

Optional namespace prefix for runtime config keys. Used when multiple configurable models exist in the same application.

**Behavior:**
- If non-empty string: Config keys are accessed via `config["configurable"]["{config_prefix}_{param}"]`
- If empty string (`""`): Config keys are accessed via `config["configurable"]["{param}"]`
- If not specified or `None`: Defaults to `""` (empty string, no prefix)
- Automatically appends `_` if prefix doesn't already end with one

**Warning:** If `config_prefix` is set but no fields are configurable, a warning is issued (line 512-517).

**Examples:**
```python
# Two models with different prefixes
gpt_model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature",),
    config_prefix="gpt",
    temperature=0
)
claude_model = init_chat_model(
    "claude-opus-4-7",
    configurable_fields=("temperature",),
    config_prefix="claude",
    temperature=0.5
)

# Invoke with different configs for each model
gpt_model.invoke("hello", config={"configurable": {"gpt_temperature": 0.9}})
claude_model.invoke("hello", config={"configurable": {"claude_temperature": 0.2}})

# Same prefix for all models (default behavior):
model1 = init_chat_model("gpt-4", configurable_fields=("temperature",))
model2 = init_chat_model("claude-opus-4-7", configurable_fields=("temperature",))
# Both use "temperature" in config (no prefix)
```

### `**kwargs` (arbitrary provider-specific parameters)

Additional model-specific keyword arguments passed to the underlying chat model's `__init__` method.

**Common parameters (supported by many providers):**
- `temperature` (float): Randomness control (typical range 0–1 or provider-specific)
- `max_tokens` (int): Maximum output tokens
- `timeout` (float): Request timeout in seconds
- `max_retries` (int): Retry attempt limit
- `base_url` (str): Custom API endpoint (OpenAI-compatible providers)
- `rate_limiter` (BaseRateLimiter): Rate limiting instance

**Provider-specific parameters:**
Examples from provider implementations:
- OpenAI: `openai_api_key`, `organization`, `api_version`
- Anthropic: `anthropic_api_url`, `default_headers`
- AWS Bedrock: `region_name`, `profile_name`

**Behavior:**
- Kwargs are forwarded directly to the provider's chat model class constructor
- Provider-specific parameters not recognized are passed through unchanged (delegation to provider)
- For fixed models: Applied immediately during instantiation
- For configurable models: Stored as defaults in `_default_config`, can be overridden at runtime

**Note:** Refer to the specific provider's integration documentation for all available parameters at https://docs.langchain.com/oss/python/integrations/providers.

## Return Types

### `BaseChatModel` (Fixed model)

A concrete chat model instance from LangChain Core, ready to invoke immediately. Returned when:
- `model` is specified AND `configurable_fields` is `None` (the default)

**Usage:**
```python
model = init_chat_model("gpt-4", temperature=0)
msg = model.invoke("hello world")  # invoke immediately
```

All LangChain chat model methods are available: `invoke`, `ainvoke`, `stream`, `batch`, `bind_tools`, `with_structured_output`, etc.

### `_ConfigurableModel` (Runtime-configurable wrapper)

A wrapper `Runnable` that defers actual model instantiation until `invoke` is called with a config. Returned when:
- `configurable_fields` is specified (even if `model` is provided)
- `model` is `None` (defaults to configurable)

**Key behaviors:**
- Implements the `Runnable` interface (same as `BaseChatModel` for most purposes)
- Lazy-instantiates the actual model on each `invoke`, `stream`, `batch` call
- Supports `bind_tools`, `with_structured_output`, and other declarative methods through queuing
- Can be composed with other Runnables in chains, even before configuration is known

**Core methods:**
- `invoke(input, config)`: Instantiate model from `config`, invoke, return result
- `ainvoke(input, config)`: Async variant
- `stream(input, config)`: Stream tokens from instantiated model
- `astream(input, config)`: Async streaming variant
- `batch(inputs, config)`: Batch invoke (optimized if config is uniform)
- `abatch(inputs, config)`: Async batch variant

**Usage:**
```python
# Configurable model with default
model = init_chat_model("gpt-4", configurable_fields=("temperature",), temperature=0)

# Use default temperature
msg1 = model.invoke("hello world")

# Override at runtime
msg2 = model.invoke("hello world", config={"configurable": {"temperature": 0.9}})

# Compose in a chain before configuration is known
chain = model | another_runnable

# Invoke the chain with config
result = chain.invoke("input", config={"configurable": {"temperature": 0.5}})
```

## Model Name Parsing and Provider Inference

### `_parse_model()` Function

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L623-L651`

Parses model name and provider, inferring provider if necessary. Returns a tuple `(model_name: str, provider: str)`.

**Algorithm:**
1. If `model_provider` is not specified and model contains `:`, check if prefix is in `_BUILTIN_PROVIDERS`
   - If yes: extract prefix as provider, remainder as model name
   - If no: continue to step 2
2. If `model_provider` is not specified, attempt inference via `_attempt_infer_model_provider()`
3. If no provider found, raise `ValueError` with suggestions
4. Normalize provider: convert to lowercase, replace `-` with `_`
5. Return `(model_name, normalized_provider)`

**Raises:**
- `ValueError`: If provider cannot be inferred and is not specified, listing supported providers and documentation URL

**Example:**
```python
# Explicit prefix
_parse_model("openai:gpt-4", None)  # → ("gpt-4", "openai")

# Inference
_parse_model("claude-opus-4-7", None)  # → ("claude-opus-4-7", "anthropic")

# Override with model_provider
_parse_model("gpt-4", "anthropic")  # → ("gpt-4", "anthropic")

# Error case
_parse_model("unknown-model", None)  # → ValueError
```

### `_attempt_infer_model_provider()` Function

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L547-L620`

Attempts to infer model provider from model name using case-insensitive prefix matching.

**Inference rules:**

| Model Prefix | Inferred Provider | Notes |
|---|---|---|
| `gpt-`, `o1`, `o3`, `chatgpt`, `text-davinci` | `openai` | OpenAI models (current & legacy) |
| `claude` | `anthropic` | Anthropic Claude models |
| `command` | `cohere` | Cohere Command models |
| `accounts/fireworks` | `fireworks` | Fireworks routing prefix |
| `gemini` | `google_vertexai` | Google Vertex AI (⚠️ deprecated default) |
| `amazon.`, `anthropic.`, `meta.` | `bedrock` | AWS Bedrock model IDs |
| `mistral`, `mixtral` | `mistralai` | Mistral & Mixtral models |
| `deepseek` | `deepseek` | DeepSeek models |
| `grok` | `xai` | xAI Grok models |
| `sonar` | `perplexity` | Perplexity Sonar models |
| `solar` | `upstage` | Upstage Solar models |

**Deprecation note:** `gemini` inference currently defaults to `google_vertexai` but will change to `google_genai` in the next major release. A `DeprecationWarning` is issued with guidance to use `model_provider` explicitly (line 585-593).

**Returns:** Inferred provider string or `None` if no match found.

## Built-in Provider Registry

### `_BUILTIN_PROVIDERS` Dictionary

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L99`

Maps provider names to module paths, class names, and creator functions. Each entry is a 3-tuple: `(module_path, class_name, creator_func)`.

**Design:**
- Registry is **not exhaustive** of all LangChain integrations, but covers ~30 most popular providers
- Unlisted providers can still be used if integrated package is installed, but model inference won't work
- Standard creator is `_call`, which directly instantiates the class
- Special creators: HuggingFace (`from_model_id`), IBM (`model_id=`), LangSmith (gateway wrapping)

**All Built-in Providers** (30 entries):

| Provider | Package | Class | Module | Notes |
|---|---|---|---|---|
| `openai` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | |
| `anthropic` | `langchain-anthropic` | `ChatAnthropic` | `langchain_anthropic` | |
| `azure_openai` | `langchain-openai` | `AzureChatOpenAI` | `langchain_openai` | |
| `azure_ai` | `langchain-azure-ai` | `AzureAIOpenAIApiChatModel` | `langchain_azure_ai.chat_models` | Submodule import |
| `google_vertexai` | `langchain-google-vertexai` | `ChatVertexAI` | `langchain_google_vertexai` | |
| `google_genai` | `langchain-google-genai` | `ChatGoogleGenerativeAI` | `langchain_google_genai` | |
| `google_anthropic_vertex` | `langchain-google-vertexai` | `ChatAnthropicVertex` | `langchain_google_vertexai.model_garden` | Anthropic via Vertex |
| `anthropic_bedrock` | `langchain-aws` | `ChatAnthropicBedrock` | `langchain_aws` | Bedrock-hosted Anthropic |
| `bedrock` | `langchain-aws` | `ChatBedrock` | `langchain_aws` | Generic Bedrock models |
| `bedrock_converse` | `langchain-aws` | `ChatBedrockConverse` | `langchain_aws` | Bedrock Converse API |
| `bedrock_mantle_anthropic` | `langchain-aws` | `ChatAnthropicMantle` | `langchain_aws` | Bedrock Mantle (Anthropic) |
| `bedrock_mantle_openai` | `langchain-aws` | `ChatOpenAIMantle` | `langchain_aws` | Bedrock Mantle (OpenAI) |
| `cohere` | `langchain-cohere` | `ChatCohere` | `langchain_cohere` | |
| `deepseek` | `langchain-deepseek` | `ChatDeepSeek` | `langchain_deepseek` | |
| `fireworks` | `langchain-fireworks` | `ChatFireworks` | `langchain_fireworks` | |
| `groq` | `langchain-groq` | `ChatGroq` | `langchain_groq` | |
| `huggingface` | `langchain-huggingface` | `ChatHuggingFace` | `langchain_huggingface` | Uses `from_model_id()` |
| `ibm` | `langchain-ibm` | `ChatWatsonx` | `langchain_ibm` | Uses `model_id=` param |
| `litellm` | `langchain-litellm` | `ChatLiteLLM` | `langchain_litellm` | |
| `meta` | `langchain-meta` | `ChatMetaModel` | `langchain_meta` | |
| `mistralai` | `langchain-mistralai` | `ChatMistralAI` | `langchain_mistralai` | |
| `nvidia` | `langchain-nvidia-ai-endpoints` | `ChatNVIDIA` | `langchain_nvidia_ai_endpoints` | |
| `ollama` | `langchain-ollama` | `ChatOllama` | `langchain_ollama` | Fallback to `langchain_community` |
| `openrouter` | `langchain-openrouter` | `ChatOpenRouter` | `langchain_openrouter` | |
| `perplexity` | `langchain-perplexity` | `ChatPerplexity` | `langchain_perplexity` | |
| `together` | `langchain-together` | `ChatTogether` | `langchain_together` | |
| `upstage` | `langchain-upstage` | `ChatUpstage` | `langchain_upstage` | |
| `xai` | `langchain-xai` | `ChatXAI` | `langchain_xai` | |
| `baseten` | `langchain-baseten` | `ChatBaseten` | `langchain_baseten` | |
| `langsmith` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | Gateway routing |

## Creator Functions

### Standard Creator: `_call()`

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L51-L53`

Direct instantiation of the chat model class:

```python
def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    return cls(**kwargs)
```

All kwargs are forwarded directly to the class constructor.

**Used by:** 28 of 30 providers.

### Special Creator: HuggingFace (`from_model_id`)

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L77-L81`

```python
lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs)
```

The HuggingFace integration uses a class method instead of direct instantiation. The `model` parameter is mapped to the `model_id=` argument of `from_model_id()`.

### Special Creator: IBM (`model_id=` mapping)

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L82-L86`

```python
lambda cls, model, **kwargs: cls(model_id=model, **kwargs)
```

The IBM Watsonx integration expects `model_id=` instead of `model=` in the constructor. This creator performs the mapping.

### Special Creator: LangSmith (Gateway routing)

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L27-L38`

```python
def _init_langsmith(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    _apply_gateway_config(
        kwargs,
        cls,
        base_url_field="openai_api_base",
        api_key_field="openai_api_key",
        provider_path="v1",
        api_key_env=("LANGSMITH_GATEWAY_API_KEY", "LANGSMITH_API_KEY"),
        default_base_url=_LANGSMITH_GATEWAY_DEFAULT_BASE,
    )
    kwargs["use_responses_api"] = True
    return cls(**kwargs)
```

The LangSmith provider wraps instantiation with gateway configuration via `_apply_gateway_config()`. The resulting model is a `ChatOpenAI` instance configured to route through the LangSmith gateway.

## `_get_chat_model_creator()` Function

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L152-L193`

Retrieves the provider's creator function and returns a partially-applied callable with the class bound.

**Signature:**
```python
@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(provider: str) -> Callable[..., BaseChatModel]
```

**Algorithm:**
1. Check if `provider` is in `_BUILTIN_PROVIDERS`
2. Extract module path, class name, and creator function from registry
3. Import the module via `_import_module()` (line 125-149)
   - On ImportError for ollama: Fall back to `langchain_community.chat_models` (backward compatibility)
4. Retrieve the class from the module via `getattr()`
5. Return a `functools.partial` with creator function bound to the class

**Caching:** Results are cached via LRU cache to avoid repeated module imports.

**Raises:**
- `ValueError`: If provider not in `_BUILTIN_PROVIDERS`
- `ImportError`: If the provider's integration package is not installed (with helpful message suggesting `pip install`)

**Example:**
```python
creator = _get_chat_model_creator("openai")
# creator is now: functools.partial(_call, cls=ChatOpenAI)

model = creator(model="gpt-4", temperature=0)
# Equivalent to: ChatOpenAI(model="gpt-4", temperature=0)
```

## `_import_module()` Function

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L125-L149`

Safely imports a module by name with helpful error messages.

**Signature:**
```python
def _import_module(module: str, class_name: str) -> ModuleType
```

**Behavior:**
- Uses `importlib.import_module()` to import the given module
- On ImportError: Derives the pip package name from module path (e.g., `langchain_azure_ai` → `langchain-azure-ai`)
- Raises ImportError with a message suggesting the pip install command

**Error message format:**
```
Initializing {class_name} requires the {pkg} package. Please install it with `pip install {pkg}`
```

**Example:**
```python
module = _import_module("langchain_openai", "ChatOpenAI")
# If not installed: ImportError: Initializing ChatOpenAI requires the langchain-openai package...
```

## Internal Implementation Details

### `_init_chat_model_helper()` Function

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L536-L544`

Helper used internally to instantiate a fixed (non-configurable) model. Parses model/provider, retrieves the creator, and instantiates.

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

### `_ConfigurableModel` Class

**Location:** `repo://libs/langchain_v1/langchain/chat_models/base.py#L661-L896`

A `Runnable` subclass that wraps configurable model instantiation.

**Constructor:**
```python
def __init__(
    self,
    *,
    default_config: dict[str, Any] | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
    config_prefix: str = "",
    queued_declarative_operations: Sequence[tuple[str, tuple[Any, ...], dict[str, Any]]] = (),
) -> None
```

**Core attributes:**
- `_default_config`: Default parameters (model name, provider, temperature, etc.)
- `_configurable_fields`: Which fields can be overridden at runtime ("any" or list)
- `_config_prefix`: Namespace prefix for config keys (ends with `_`)
- `_queued_declarative_operations`: Pending operations like `bind_tools`, `with_structured_output` to apply after model instantiation

**Key methods:**

- `_model(config)`: Instantiate the actual chat model from config + defaults
- `_model_params(config)`: Extract configurable fields from `config["configurable"]`
- `invoke(input, config, **kwargs)`: Instantiate model, invoke, return result
- `stream(input, config, **kwargs)`: Instantiate model, stream, yield results
- `batch(inputs, config, **kwargs)`: Optimized batching (uses underlying model's batch if config is uniform)

**Declarative method interception:**
- When `bind_tools()`, `with_structured_output()` are called on a `_ConfigurableModel`, they are queued rather than applied immediately
- These operations are applied in order during model instantiation (see `_model()` line 718-719)

**Example flow:**
```python
model = init_chat_model("gpt-4", configurable_fields=("temperature",), temperature=0)
# Returns _ConfigurableModel with _default_config={"model": "gpt-4", "temperature": 0}

model_with_tools = model.bind_tools([MyTool])
# Queues ("bind_tools", (MyTool,), {}) to _queued_declarative_operations

result = model_with_tools.invoke(
    "hello",
    config={"configurable": {"temperature": 0.9}}
)
# 1. _model_params extracts {"temperature": 0.9}
# 2. Merges with defaults: {"model": "gpt-4", "temperature": 0.9}
# 3. Instantiates ChatOpenAI(model="gpt-4", temperature=0.9)
# 4. Applies queued bind_tools([MyTool])
# 5. Calls invoke on resulting model
```

## Error Handling and Validation

### Type Validation

**Line 501-507:** If `model` is not `None` and not a string, raises `TypeError`:

```python
if model is not None and not isinstance(model, str):
    raise TypeError(
        f"`model` must be a string (e.g., 'openai:gpt-5.5'), got "
        f"{type(model).__name__}. If you've already constructed a chat model "
        f"object, use it directly instead of passing it to init_chat_model()."
    )
```

This catches cases where a user accidentally passes an already-instantiated model object.

### Provider Inference Failures

**Line 637-647:** If provider cannot be inferred and not specified, raises `ValueError`:

```python
if not model_provider:
    supported_list = ", ".join(sorted(_BUILTIN_PROVIDERS))
    raise ValueError(
        f"Unable to infer model provider for {model=}. "
        f"Please specify 'model_provider' directly.\n\n"
        f"Supported providers: {supported_list}\n\n"
        f"For help with specific providers, see: "
        f"https://docs.langchain.com/oss/python/integrations/providers"
    )
```

### Missing Package Installation

**Line 179-190:** If provider package is not installed:

```python
try:
    module = _import_module(pkg, class_name)
except ImportError as e:
    if provider != "ollama":
        raise
    # For backwards compatibility, try langchain_community fallback for ollama
    try:
        module = _import_module("langchain_community.chat_models", class_name)
    except ImportError:
        raise e from None
```

The `_import_module()` function formats helpful error messages suggesting `pip install {package}`.

### Config Prefix Warning

**Line 511-517:** If `config_prefix` is set but no fields are configurable, warns:

```python
if config_prefix and not configurable_fields:
    warnings.warn(
        f"{config_prefix=} has been set but no fields are configurable. Set "
        f"`configurable_fields=(...)` to specify the model params that are "
        f"configurable.",
        stacklevel=2,
    )
```

## Usage Patterns

### Fixed Model (Most Common)

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4", temperature=0, max_tokens=1000)
response = model.invoke("What is 2+2?")
print(response.content)
```

Returns a concrete `ChatOpenAI` instance ready to use immediately.

### Partially Configurable Model

```python
model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature",),
    temperature=0
)

# Default temperature
response1 = model.invoke("hello")

# Override temperature at runtime
response2 = model.invoke(
    "hello",
    config={"configurable": {"temperature": 0.9}}
)
```

Returns a `_ConfigurableModel` wrapper that allows runtime temperature override.

### Fully Configurable Model with Default

```python
model = init_chat_model(
    "gpt-4",
    configurable_fields="any",
    config_prefix="main",
    temperature=0,
    max_tokens=500
)

# Default configuration
response1 = model.invoke("hello")

# Override multiple fields
response2 = model.invoke(
    "hello",
    config={
        "configurable": {
            "main_model": "claude-opus-4-7",
            "main_temperature": 0.7,
            "main_max_tokens": 1000
        }
    }
)
```

### Configurable Model with No Default

```python
model = init_chat_model()

# Must specify model at runtime
response = model.invoke(
    "hello",
    config={"configurable": {"model": "gpt-4"}}
)
```

### Dynamic Provider Selection

```python
provider = os.getenv("LLM_PROVIDER", "openai")
model_name = os.getenv("LLM_MODEL", "gpt-4")

model = init_chat_model(model_name, model_provider=provider)
response = model.invoke("hello")
```

### In a Chain (with runtime configuration)

```python
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature",),
    temperature=0
)

prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = prompt | model | StrOutputParser()

# Use default temperature
result1 = chain.invoke({"question": "What is 2+2?"})

# Override temperature at runtime
result2 = chain.invoke(
    {"question": "What is 2+2?"},
    config={"configurable": {"temperature": 0.9}}
)
```

## Security Considerations

- **API Key Exposure:** Setting `configurable_fields="any"` makes all fields configurable, including `api_key` and `base_url`. This allows untrusted configurations to redirect API calls to malicious services.
- **Safe Pattern:** Always enumerate `configurable_fields` explicitly when accepting untrusted configuration (e.g., from user input or external config files):
  ```python
  model = init_chat_model(
      "gpt-4",
      configurable_fields=("temperature", "max_tokens"),  # Explicit safe list
      temperature=0
  )
  ```
- **Environment Variables:** Use environment variables for sensitive configuration (API keys), not runtime config dictionaries.
