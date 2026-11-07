---
type: Reference
title: init_chat_model API Reference
description: Complete API reference for init_chat_model function parameters, return types, built-in provider registry, and creator function specifications.
tags: [chat-models, initialization, factory-pattern, api-reference, provider-registry, parameters]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-b09b1477098d69af6abaa5b4
    resource: repo://libs/langchain_v1/langchain/chat_models/__init__.py
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

## Overview

This reference documents the `init_chat_model` function's complete API surface: parameter specifications, return type variants, built-in provider registry, error conditions, and creator function behavior. For architectural patterns and examples, see [/openwiki/model-initialization.md](/openwiki/model-initialization.md).

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

**File**: `repo://libs/langchain_v1/langchain/chat_models/base.py`  
**Public export**: `repo://libs/langchain_v1/langchain/chat_models/__init__.py`

## Type Overloads

The function defines three overloads to support different initialization patterns:

### Overload 1: Fixed Model (String model, no configurable fields)

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

When `model` is a string and `configurable_fields` is `None` (default), returns a ready-to-use `BaseChatModel` instance.

### Overload 2: Configurable Model (No model specified)

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

When `model` is `None` and `configurable_fields` is `None`, defaults to `configurable_fields=("model", "model_provider")` and returns a `_ConfigurableModel`.

### Overload 3: Configurable Model (Explicit configurable_fields)

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

When `configurable_fields` is explicitly set to a non-`None` value, returns a `_ConfigurableModel` regardless of whether `model` is specified.

## Parameters

### model: `str | None` (default: `None`)

The model identifier, optionally prefixed with provider name using colon syntax.

**Accepted formats:**
- `"provider:model-name"` (explicit prefix) – e.g., `"openai:gpt-5.5"`, `"anthropic:claude-opus-4-7"`
- `"model-name"` (bare name, provider inferred) – e.g., `"gpt-5.5"`, `"claude-opus-4-7"`
- `None` (no default model, must be provided at runtime config)

**Behavior:**
- If a colon divides the string and the prefix is a registered provider (in `_BUILTIN_PROVIDERS`), the prefix is extracted as the provider and the remainder is the model name.
- If no colon or unrecognized prefix, `_attempt_infer_model_provider` infers the provider using case-insensitive prefix matching.
- If inference fails and `model_provider` is not set, raises `ValueError` listing supported providers.
- Type validation: if `model` is not `None` and not a string, raises `TypeError`.

**Provider inference rules** (case-insensitive prefix matching):

| Prefix | Provider |
|--------|----------|
| `gpt-`, `o1`, `o3`, `chatgpt`, `text-davinci` | `openai` |
| `claude` | `anthropic` |
| `command` | `cohere` |
| `accounts/fireworks` | `fireworks` |
| `gemini` | `google_vertexai` (⚠️ deprecated; will change to `google_genai` in next major) |
| `amazon.`, `anthropic.`, `meta.` | `bedrock` |
| `mistral`, `mixtral` | `mistralai` |
| `deepseek` | `deepseek` |
| `grok` | `xai` |
| `sonar` | `perplexity` |
| `solar` | `upstage` |

### model_provider: `str | None` (default: `None`)

Provider name, an alternative to colon-prefixed format on `model`.

**Behavior:**
- Equivalent to the prefix form – e.g., `model='gpt-4', model_provider='openai'` is the same as `model='openai:gpt-4'`.
- Normalized to lowercase with underscores – e.g., `"azure-openai"` → `"azure_openai"`.
- If both `model` prefix and `model_provider` are specified, the prefix takes precedence.
- Can be independently configured at runtime via `configurable_fields` when separate swappability is needed (e.g., routing the same model name to different hosts).

**Supported values** (see below for full registry):
- `openai`, `anthropic`, `azure_openai`, `azure_ai`, `google_vertexai`, `google_genai`, `google_anthropic_vertex`
- `anthropic_bedrock`, `bedrock`, `bedrock_converse`, `bedrock_mantle_anthropic`, `bedrock_mantle_openai`
- `cohere`, `deepseek`, `fireworks`, `groq`, `huggingface`, `ibm`, `litellm`, `meta`
- `mistralai`, `nvidia`, `ollama`, `openrouter`, `perplexity`, `together`, `upstage`, `xai`, `baseten`, `langsmith`

### configurable_fields: `Literal["any"] | list[str] | tuple[str, ...] | None` (default: `None` if `model` specified, `("model", "model_provider")` if not)

Specifies which parameters can be overridden at runtime via config.

**Valid values:**
- `None` (or omitted if `model` is specified): No fields are configurable; the returned model is fixed.
- `"any"`: All kwargs become configurable at runtime. ⚠️ **Security warning**: Includes `api_key`, `base_url`, and other sensitive parameters that can redirect requests or expose credentials if untrusted config is accepted.
- `list[str] | tuple[str, ...]`: Whitelist of parameter names that are configurable – e.g., `["temperature", "max_tokens"]`, `("model", "model_provider")`.

**Defaults:**
- If `model` is specified and `configurable_fields` is omitted, defaults to `None` (fixed model).
- If `model` is `None` and `configurable_fields` is omitted, defaults to `("model", "model_provider")`.

**Implications:**
- If `configurable_fields` is `None`, returns a `BaseChatModel` (Overload 1).
- If `configurable_fields` is not `None` or `model` is `None`, returns a `_ConfigurableModel` (Overload 2 or 3).

### config_prefix: `str | None` (default: `None`)

Optional namespace for runtime configuration keys when multiple configurable models exist in the same application.

**Behavior:**
- If a non-empty string, config keys are accessed via `config["configurable"]["{config_prefix}_{param}"]`.
- If `None` or empty string `""`, config keys are accessed via `config["configurable"]["{param}"]`.
- The prefix is automatically normalized: if provided and does not end with `_`, an underscore is appended.
- A warning is issued if `config_prefix` is set but no fields are configurable.

**Example:**
```python
model = init_chat_model(
    "gpt-4",
    configurable_fields=("temperature",),
    config_prefix="my_model",
)
# At runtime:
model.invoke("hello", config={"configurable": {"my_model_temperature": 0.8}})
```

### **kwargs: `Any` (provider-specific parameters)

All additional keyword arguments are forwarded to the underlying chat model's constructor.

**Common cross-provider parameters** (supported by most providers):
- `temperature` (`float`): Controls randomness/diversity. Range varies by provider (typically 0–2 or 0–1).
- `max_tokens` (`int`): Maximum number of output tokens to generate.
- `timeout` (`float`): Request timeout in seconds.
- `max_retries` (`int`): Maximum number of retry attempts for failed requests.
- `base_url` (`str`): Custom API endpoint URL (for OpenAI-compatible providers).
- `rate_limiter` (`BaseRateLimiter`): Rate limiting instance.

**Provider-specific parameters** (forwarded as-is; consult provider integration docs):
- OpenAI: `openai_api_key`, `openai_api_base`, `model_name`
- Anthropic: `anthropic_api_key`, `anthropic_api_url`
- Google Vertex AI: `project_id`, `location`, `credentials`
- AWS Bedrock: `region_name`, `client_options`
- Azure OpenAI: `api_version`, `azure_deployment`, `azure_endpoint`
- And more – refer to each provider's integration reference

**Routing behavior:**
- For standard creators (`_call`), all kwargs are passed directly to the provider's `__init__`.
- For special creators (HuggingFace, IBM, LangSmith), parameter transformation occurs (see **Creator Functions** section below).

## Return Types

### BaseChatModel

Returned when `configurable_fields` is `None` (default) and `model` is a non-empty string.

A concrete chat model instance, immediately ready to invoke:

```python
model: BaseChatModel = init_chat_model("gpt-4", temperature=0.5)
response = model.invoke("What is 2+2?")  # Ready to use
```

**Type**: `langchain_core.language_models.BaseChatModel`

### _ConfigurableModel

Returned when `configurable_fields` is not `None` or `model` is `None`.

A `Runnable[LanguageModelInput, Any]` wrapper that defers model instantiation until config is provided at invoke time. Supports:
- `invoke(input, config=...)` – Instantiate model with config, then invoke
- `stream(input, config=...)` – Streaming invocation
- `ainvoke(input, config=...)` – Async invocation
- Declarative operations: `bind_tools(...)`, `with_structured_output(...)` – Queued until model instantiation
- `batch(...)`, `abatch(...)`, `batch_as_completed(...)` – Optimized batching

**Type**: `langchain.chat_models.base._ConfigurableModel[LanguageModelInput, Any]`

## Built-in Provider Registry

The `_BUILTIN_PROVIDERS` dictionary maps provider names to module paths, class names, and creator functions. This enables dynamic import and instantiation without hardcoding provider logic.

**Registry location**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L56-L99`

### Registry Structure

Each entry is a tuple: `(module_path: str, class_name: str, creator_func: Callable)`

```python
_BUILTIN_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "openai": ("langchain_openai", "ChatOpenAI", _call),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    # ... 28 more providers
}
```

### Complete Provider Registry (30 providers)

| Provider | Package | Class | Module | Creator | Notes |
|----------|---------|-------|--------|---------|-------|
| `openai` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | `_call` | Standard OpenAI API |
| `anthropic` | `langchain-anthropic` | `ChatAnthropic` | `langchain_anthropic` | `_call` | Anthropic Claude models |
| `azure_openai` | `langchain-openai` | `AzureChatOpenAI` | `langchain_openai` | `_call` | Azure-hosted OpenAI |
| `azure_ai` | `langchain-azure-ai` | `AzureAIOpenAIApiChatModel` | `langchain_azure_ai.chat_models` | `_call` | Azure AI services |
| `google_vertexai` | `langchain-google-vertexai` | `ChatVertexAI` | `langchain_google_vertexai` | `_call` | Google Vertex AI |
| `google_genai` | `langchain-google-genai` | `ChatGoogleGenerativeAI` | `langchain_google_genai` | `_call` | Google Generative AI (Gemini API) |
| `google_anthropic_vertex` | `langchain-google-vertexai` | `ChatAnthropicVertex` | `langchain_google_vertexai.model_garden` | `_call` | Anthropic Claude via Vertex AI |
| `anthropic_bedrock` | `langchain-aws` | `ChatAnthropicBedrock` | `langchain_aws` | `_call` | Anthropic via AWS Bedrock |
| `bedrock` | `langchain-aws` | `ChatBedrock` | `langchain_aws` | `_call` | Generic AWS Bedrock models |
| `bedrock_converse` | `langchain-aws` | `ChatBedrockConverse` | `langchain_aws` | `_call` | Bedrock Converse API |
| `bedrock_mantle_anthropic` | `langchain-aws` | `ChatAnthropicMantle` | `langchain_aws` | `_call` | Bedrock Mantle (Anthropic) |
| `bedrock_mantle_openai` | `langchain-aws` | `ChatOpenAIMantle` | `langchain_aws` | `_call` | Bedrock Mantle (OpenAI) |
| `cohere` | `langchain-cohere` | `ChatCohere` | `langchain_cohere` | `_call` | Cohere Command models |
| `deepseek` | `langchain-deepseek` | `ChatDeepSeek` | `langchain_deepseek` | `_call` | DeepSeek API |
| `fireworks` | `langchain-fireworks` | `ChatFireworks` | `langchain_fireworks` | `_call` | Fireworks AI hosted models |
| `groq` | `langchain-groq` | `ChatGroq` | `langchain_groq` | `_call` | Groq API (fast inference) |
| `huggingface` | `langchain-huggingface` | `ChatHuggingFace` | `langchain_huggingface` | `from_model_id()` | HuggingFace Hub models |
| `ibm` | `langchain-ibm` | `ChatWatsonx` | `langchain_ibm` | `model_id=` | IBM Watsonx |
| `litellm` | `langchain-litellm` | `ChatLiteLLM` | `langchain_litellm` | `_call` | LiteLLM unified interface |
| `meta` | `langchain-meta` | `ChatMetaModel` | `langchain_meta` | `_call` | Meta's Llama models |
| `mistralai` | `langchain-mistralai` | `ChatMistralAI` | `langchain_mistralai` | `_call` | Mistral AI API |
| `nvidia` | `langchain-nvidia-ai-endpoints` | `ChatNVIDIA` | `langchain_nvidia_ai_endpoints` | `_call` | NVIDIA AI endpoints |
| `ollama` | `langchain-ollama` | `ChatOllama` | `langchain_ollama` | `_call` | Ollama (local/remote) |
| `openrouter` | `langchain-openrouter` | `ChatOpenRouter` | `langchain_openrouter` | `_call` | OpenRouter aggregator |
| `perplexity` | `langchain-perplexity` | `ChatPerplexity` | `langchain_perplexity` | `_call` | Perplexity API |
| `together` | `langchain-together` | `ChatTogether` | `langchain_together` | `_call` | Together AI |
| `upstage` | `langchain-upstage` | `ChatUpstage` | `langchain_upstage` | `_call` | Upstage API |
| `xai` | `langchain-xai` | `ChatXAI` | `langchain_xai` | `_call` | xAI (Grok) |
| `baseten` | `langchain-baseten` | `ChatBaseten` | `langchain_baseten` | `_call` | Baseten serverless |
| `langsmith` | `langchain-openai` | `ChatOpenAI` | `langchain_openai` | `_init_langsmith` | LangSmith gateway (special) |

**Design notes:**
- The registry is **not exhaustive**. Unlisted providers can still be used via direct instantiation if their integration package is installed.
- Most entries use the standard `_call` creator function.
- Special creators exist for HuggingFace (`from_model_id()`), IBM (`model_id=`), and LangSmith (gateway configuration).

## Creator Functions

Creator functions are callables that instantiate a provider's chat model class with given kwargs. They are looked up from the registry and cached by `_get_chat_model_creator()`.

### Standard Creator: `_call`

```python
def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    return cls(**kwargs)
```

**Behavior**: Directly instantiates the class with all provided kwargs. Used by most providers (>25).

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L51-L53`

### Special Creator: HuggingFace (`from_model_id`)

```python
lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs)
```

**Behavior**: Maps the `model` parameter to the class method `from_model_id(model_id=...)` instead of direct instantiation. Allows HuggingFace to download and configure models from the hub.

**Provider**: `huggingface`  
**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L77-L81`

### Special Creator: IBM (`model_id=` parameter)

```python
lambda cls, model, **kwargs: cls(model_id=model, **kwargs)
```

**Behavior**: Maps the `model` parameter to the `model_id=` kwarg in the constructor. IBM's Watsonx expects `model_id=` instead of `model=`.

**Provider**: `ibm`  
**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L82-L86`

### Special Creator: LangSmith Gateway (`_init_langsmith`)

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

**Behavior**: 
- Applies gateway configuration via `_apply_gateway_config()`, injecting credentials and base URL from environment variables or defaults.
- Supports fallback from `LANGSMITH_GATEWAY_API_KEY` to `LANGSMITH_API_KEY`.
- Sets `use_responses_api=True` for gateway compatibility.
- Uses OpenAI's `ChatOpenAI` class as the underlying implementation.

**Provider**: `langsmith`  
**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L27-L38`

### Creator Retrieval and Caching

`_get_chat_model_creator` looks up a provider in the registry, imports the module, retrieves the class, and returns a partial application of the creator:

```python
@functools.lru_cache(maxsize=len(_BUILTIN_PROVIDERS))
def _get_chat_model_creator(
    provider: str,
) -> Callable[..., BaseChatModel]:
    if provider not in _BUILTIN_PROVIDERS:
        supported = ", ".join(_BUILTIN_PROVIDERS.keys())
        msg = f"Unsupported {provider=}.\n\nSupported model providers are: {supported}"
        raise ValueError(msg)

    pkg, class_name, creator_func = _BUILTIN_PROVIDERS[provider]
    module = _import_module(pkg, class_name)
    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)
```

**Returns**: A callable bound to the provider's class, e.g., `functools.partial(_call, cls=ChatOpenAI)`.

**Caching**: LRU cache of size `len(_BUILTIN_PROVIDERS)` avoids repeated module imports.

**Raises**:
- `ValueError` if the provider is not in `_BUILTIN_PROVIDERS`.
- `ImportError` if the provider's integration package is not installed.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L152-L194`

## Helper Functions

### `_import_module(module: str, class_name: str) -> ModuleType`

Imports a Python module by name, with enhanced error messaging.

**Parameters**:
- `module`: Fully qualified module name (e.g., `'langchain_openai'`, `'langchain_azure_ai.chat_models'`)
- `class_name`: Class name being imported (used in error message)

**Returns**: The imported module object.

**Raises**: `ImportError` with a suggestion to `pip install <package>` if the module cannot be imported.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L125-L150`

### `_attempt_infer_model_provider(model_name: str) -> str | None`

Attempts to infer the provider from a bare model name using case-insensitive prefix matching.

**Parameters**: `model_name` – The bare model identifier (e.g., `"gpt-4"`, `"claude-opus"`)

**Returns**: Inferred provider name (e.g., `"openai"`, `"anthropic"`), or `None` if no match.

**Inference rules** (applied in order):
1. OpenAI: Prefix in `("gpt-", "o1", "o3", "chatgpt", "text-davinci")`
2. Anthropic: Prefix `"claude"`
3. Cohere: Prefix `"command"`
4. Fireworks: Prefix `"accounts/fireworks"`
5. Google Vertex AI: Prefix `"gemini"` (⚠️ deprecated; will change to `google_genai`)
6. AWS Bedrock: Prefix in `("amazon.", "anthropic.", "meta.")`
7. Mistral: Prefix in `("mistral", "mixtral")`
8. DeepSeek: Prefix `"deepseek"`
9. xAI: Prefix `"grok"`
10. Perplexity: Prefix `"sonar"`
11. Upstage: Prefix `"solar"`

**Side effects**: Issues a `DeprecationWarning` if "gemini" is inferred.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L547-L621`

### `_parse_model(model: str, model_provider: str | None) -> tuple[str, str]`

Parses model name and provider, extracting explicit prefixes or inferring the provider.

**Parameters**:
- `model`: Model identifier (may include `"provider:model"` format)
- `model_provider`: Optional provider override

**Returns**: Tuple of `(model_name, provider)` with provider normalized to lowercase and underscores.

**Behavior**:
1. If `model` contains `:` and the prefix is a registered provider, extract it as the provider.
2. Otherwise, attempt to infer the provider from the model name.
3. Normalize provider name: lowercase and replace hyphens with underscores.
4. If provider cannot be determined, raise `ValueError` listing supported providers and docs URL.

**Raises**: `ValueError` if provider inference fails.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L623-L652`

### `_remove_prefix(s: str, prefix: str) -> str`

Removes a prefix from a string if present (equivalent to `str.removeprefix`).

**Parameters**:
- `s`: String to remove prefix from
- `prefix`: Prefix to remove

**Returns**: String with prefix removed if it was present, unchanged otherwise.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L654-L655`

### `_init_chat_model_helper(model: str, *, model_provider: str | None = None, **kwargs) -> BaseChatModel`

Internal helper function that performs fixed model instantiation.

**Parameters**:
- `model`: Model identifier (string)
- `model_provider`: Optional provider name
- `**kwargs`: Provider-specific parameters

**Behavior**:
1. Calls `_parse_model()` to parse and infer provider.
2. Calls `_get_chat_model_creator()` to retrieve the creator function.
3. Invokes the creator with `model=model` and `**kwargs`.

**Returns**: Instantiated `BaseChatModel`.

**Raises**: 
- `ValueError` if provider cannot be inferred or is unsupported
- `ImportError` if the integration package is not installed
- Any exception raised by the provider's constructor

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L536-L545`

## Error Handling

### ValueError: Provider Inference Failure

**Condition**: Model name has no recognized prefix and `model_provider` is not specified.

**Message**:
```
Unable to infer model provider for model='...'. Please specify 'model_provider' directly.

Supported providers: anthropic, azure_ai, azure_openai, ...

For help with specific providers, see: https://docs.langchain.com/oss/python/integrations/providers
```

**Resolution**: Explicitly specify `model_provider` or use prefixed format (e.g., `"anthropic:claude-opus"`).

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L637-L647`

### ValueError: Unsupported Provider

**Condition**: `model_provider` is a string not in `_BUILTIN_PROVIDERS`.

**Message**:
```
Unsupported model_provider='...'.

Supported model providers are: anthropic, azure_ai, ...
```

**Resolution**: Use a supported provider from the registry or install the provider's integration package.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L173-L176`

### ImportError: Missing Integration Package

**Condition**: The provider's integration package (e.g., `langchain-openai`) is not installed.

**Message**:
```
Initializing ChatOpenAI requires the langchain-openai package. Please install it with `pip install langchain-openai`
```

**Resolution**: Install the suggested package via `pip install`.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L139-L150`

### TypeError: Non-String Model

**Condition**: `model` is not `None` and not a string (e.g., a model object is passed).

**Message**:
```
`model` must be a string (e.g., 'openai:gpt-5.5'), got <type>. If you've already constructed a chat model object, use it directly instead of passing it to init_chat_model().
```

**Resolution**: Pass a model name string, not a model instance.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L501-L507`

## Configuration Validation and Warnings

### Warning: config_prefix without configurable_fields

**Condition**: `config_prefix` is set but `configurable_fields` is `None` or not specified.

**Message**:
```
config_prefix='...' has been set but no fields are configurable. Set `configurable_fields=(...)` to specify the model params that are configurable.
```

**Resolution**: Explicitly set `configurable_fields` to a list or tuple of parameter names, or remove `config_prefix`.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L508-L517`

### Warning: Deprecated gemini Inference

**Condition**: A model name starting with "gemini" is inferred and no explicit `model_provider` is specified.

**Message**:
```
Inferred `model_provider='google_vertexai'` from 'gemini...'. This default will change to 'google_genai' in the next major release. To keep current behavior, pass `model_provider='google_vertexai'` (or use the prefix form, e.g. 'google_vertexai:gemini...'); for AI Studio / Gemini API, use 'google_genai' instead.
```

**Resolution**: Explicitly specify `model_provider='google_vertexai'` or `model_provider='google_genai'` to lock in behavior.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L585-L593`

## _ConfigurableModel Class

### State

`_ConfigurableModel` stores configuration and runtime state:

- `_default_config: dict[str, Any]` – Default parameter values (e.g., `{"model": "gpt-4", "temperature": 0.5}`)
- `_configurable_fields: Literal["any"] | list[str]` – Fields that can be overridden at runtime
- `_config_prefix: str` – Namespace for config keys (normalized: ends with `_` if non-empty)
- `_queued_declarative_operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]]` – Declarative operations (bind_tools, etc.) to apply after model instantiation

### Initialization

```python
_ConfigurableModel(
    default_config: dict[str, Any] | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
    config_prefix: str = "",
    queued_declarative_operations: Sequence[tuple[str, tuple[Any, ...], dict[str, Any]]] = (),
)
```

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L662-L684`

### Key Methods

#### `_model(config: RunnableConfig | None = None) -> Runnable[Any, Any]`

Instantiates the underlying model with merged config, then applies all queued declarative operations.

**Behavior**:
1. Merges `_default_config` with runtime config via `_model_params()`.
2. Calls `_init_chat_model_helper()` to instantiate the model.
3. Applies queued operations in order (bind_tools, with_structured_output, etc.).
4. Returns the configured model instance.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L715-L721`

#### `_model_params(config: RunnableConfig | None) -> dict[str, Any]`

Extracts configurable parameters from runtime config, respecting the config_prefix and field whitelist.

**Behavior**:
1. Normalizes config via `ensure_config()`.
2. Extracts keys from `config["configurable"]` that start with `_config_prefix`.
3. Removes the prefix from extracted keys.
4. If `_configurable_fields != "any"`, filters to only whitelisted field names.
5. Returns the merged parameters dict.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L722-L732`

#### `with_config(config: RunnableConfig | None = None, **kwargs) -> _ConfigurableModel`

Creates a new `_ConfigurableModel` with config merged into `_default_config`.

**Behavior**: 
- Merges provided config with existing state.
- Non-configurable parts of the config are queued as a `with_config` operation.
- Returns a new instance without mutating the original.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L733-L765`

#### `invoke(input: LanguageModelInput, config: RunnableConfig | None = None, **kwargs) -> Any`

Invokes the model at runtime with optional config.

**Behavior**: Calls `_model(config).invoke(input, config=config, **kwargs)`.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L776-L783`

#### Streaming and Batch Methods

All streaming/batching methods delegate to the instantiated model:
- `stream()` – `repo://libs/langchain_v1/langchain/chat_models/base.py#L794-L801`
- `astream()` – `repo://libs/langchain_v1/langchain/chat_models/base.py#L803-L811`
- `batch()` – Optimized for single/multiple configs – `repo://libs/langchain_v1/langchain/chat_models/base.py#L812-L839`
- `abatch()` – Async batching – `repo://libs/langchain_v1/langchain/chat_models/base.py#L840-L867`
- `batch_as_completed()` – `repo://libs/langchain_v1/langchain/chat_models/base.py#L868-L896`

### Declarative Operations

Declarative methods (bind_tools, with_structured_output) are intercepted and queued instead of applied immediately:

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
    # ... delegate to default model
```

**Queued operations** are applied in order when `_model()` is called.

**Source**: `repo://libs/langchain_v1/langchain/chat_models/base.py#L685-L714`

## Public Export

The `init_chat_model` function is exported from:

**Module**: `repo://libs/langchain_v1/langchain/chat_models/__init__.py#L5`

**Import**:
```python
from langchain.chat_models import init_chat_model
```
