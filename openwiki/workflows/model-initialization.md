---
type: "Workflow"
title: "Model Initialization: init_chat_model Provider Routing"
description: "How init_chat_model routes model identifiers to provider-specific implementations. Covers model identifier syntax, provider discovery, lazy module loading, _ConfigurableModel, model profiles, and capability detection."
tags: [model-initialization, factory, provider-routing, model-discovery, chat-models, configuration, profiles]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-06T08:17:28.425Z
sources:
  - id: openwiki-source-c479d4fffee5cf62576699e4
    resource: repo://libs/langchain_v1/langchain/chat_models/base.py
  - id: openwiki-source-88a2c811f1dfc4c3cc8cabf0
    resource: repo://libs/model-profiles/langchain_model_profiles/_summary.py
  - id: openwiki-source-e0b95eafb4bbd52f491c8cee
    resource: repo://libs/model-profiles/README.md
generated: { by: "openwiki/0.5.0", at: "2026-11-05T08:18:41.280Z" }
---

# Model Initialization: init_chat_model Provider Routing

The `init_chat_model` factory function provides a unified interface for loading any supported language model provider without knowing provider-specific import paths or class names. It handles model identifier parsing, provider discovery, lazy module loading, configuration, and runtime model switching via the configurable model system.

## Core Pattern: Unified Loading Interface

**Two Use Cases:**

1. **Fixed Model**: Specify the model upfront and get a ready-to-use `BaseChatModel` instance immediately.
   ```python
   from langchain.chat_models import init_chat_model
   
   model = init_chat_model("openai:gpt-4-turbo", temperature=0.7, api_key="...")
   response = model.invoke("Hello, world!")
   ```

2. **Configurable Model**: Defer model selection to runtime, enabling model switching without code changes via the `config` parameter.
   ```python
   # Create a configurable model without specifying which model
   model = init_chat_model(temperature=0.7)
   
   # Specify model at invocation time
   response = model.invoke(
       "Hello",
       config={"configurable": {"model": "openai:gpt-4-turbo"}}
   )
   ```

## Model Identifier Syntax

Model identifiers follow the pattern **`provider:model-name`**, where:

- **provider**: One of the registered provider keys (e.g., `openai`, `anthropic`, `ollama`)
- **model-name**: The model identifier recognized by that provider (e.g., `gpt-4-turbo`, `claude-opus-4-7`)

### Examples

```python
# Explicit provider prefix (recommended)
init_chat_model("openai:gpt-4-turbo")
init_chat_model("anthropic:claude-opus-4-7")
init_chat_model("ollama:llama2")

# Bare model name with inference
init_chat_model("gpt-4-turbo")           # → inferred as openai
init_chat_model("claude-opus-4-7")       # → inferred as anthropic
init_chat_model("mixtral-8x7b")          # → inferred as mistralai

# Separate provider parameter
init_chat_model("gpt-4-turbo", model_provider="openai")

# Provider + model_provider (equivalent forms)
init_chat_model("openai:gpt-4-turbo")
# is the same as:
init_chat_model("gpt-4-turbo", model_provider="openai")
```

## Provider Discovery and Routing

The built-in provider registry (`_BUILTIN_PROVIDERS`) is a dictionary mapping provider identifiers to tuples of `(module_path, class_name, creator_func)`. When `init_chat_model` is called with a model identifier, it:

1. **Parses the identifier** to extract or infer the provider name
2. **Looks up the provider** in `_BUILTIN_PROVIDERS`
3. **Lazily imports the module** on first access via `_get_chat_model_creator` (cached with `@functools.lru_cache`)
4. **Instantiates the class** using the provider's `creator_func` with the model name and kwargs

Each registry entry specifies:

- **Module path**: The Python package and module containing the chat model class (e.g., `'langchain_openai'`, `'langchain_anthropic'`, `'langchain_aws'`)
- **Class name**: The exported class to instantiate (e.g., `'ChatOpenAI'`, `'ChatAnthropic'`, `'ChatBedrock'`)
- **Creator function**: A callable that instantiates the class with the provided kwargs. Most use `_call`, a simple wrapper. Custom creators handle provider-specific initialization (e.g., `_init_langsmith` for LangSmith Gateway, or lambda functions for Hugging Face and IBM models that use different parameter names)

### Built-in Provider Registry

The registry includes 30+ popular providers:

| Provider | Module | Class | Creator Function | Example Models |
|----------|--------|-------|------------------|-----------------|
| `openai` | `langchain_openai` | `ChatOpenAI` | `_call` | `gpt-4-turbo`, `gpt-4o`, `o1` |
| `anthropic` | `langchain_anthropic` | `ChatAnthropic` | `_call` | `claude-opus-4-7`, `claude-haiku-4-5` |
| `azure_openai` | `langchain_openai` | `AzureChatOpenAI` | `_call` | `gpt-4-turbo` (on Azure) |
| `google_vertexai` | `langchain_google_vertexai` | `ChatVertexAI` | `_call` | `gemini-1.5-pro`, `gemini-1.5-flash` |
| `google_genai` | `langchain_google_genai` | `ChatGoogleGenerativeAI` | `_call` | `gemini-1.5-pro` (AI Studio) |
| `ollama` | `langchain_ollama` | `ChatOllama` | `_call` | `llama2`, `mistral`, `neural-chat` |
| `mistralai` | `langchain_mistralai` | `ChatMistralAI` | `_call` | `mistral-large`, `mixtral-8x7b` |
| `groq` | `langchain_groq` | `ChatGroq` | `_call` | `mixtral-8x7b-32768` |
| `bedrock` | `langchain_aws` | `ChatBedrock` | `_call` | `amazon.titan-text-express-v1` |
| `bedrock_converse` | `langchain_aws` | `ChatBedrockConverse` | `_call` | (newer Bedrock API) |
| `anthropic_bedrock` | `langchain_aws` | `ChatAnthropicBedrock` | `_call` | Anthropic via AWS Bedrock |
| `cohere` | `langchain_cohere` | `ChatCohere` | `_call` | `command-r-plus`, `command-light` |
| `xai` | `langchain_xai` | `ChatXAI` | `_call` | `grok-beta` |
| `deepseek` | `langchain_deepseek` | `ChatDeepSeek` | `_call` | `deepseek-chat`, `deepseek-coder` |
| `perplexity` | `langchain_perplexity` | `ChatPerplexity` | `_call` | `sonar-small`, `sonar-large` |
| `upstage` | `langchain_upstage` | `ChatUpstage` | `_call` | `solar-pro` |
| `langsmith` | `langchain_openai` | `ChatOpenAI` | `_init_langsmith` | (any OpenAI model via LangSmith Gateway) |
| `fireworks` | `langchain_fireworks` | `ChatFireworks` | `_call` | `accounts/fireworks/models/mixtral-8x7b-instruct` |
| `together` | `langchain_together` | `ChatTogether` | `_call` | `meta-llama/Meta-Llama-3-70B` |
| `huggingface` | `langchain_huggingface` | `ChatHuggingFace` | lambda (`.from_model_id`) | `mistralai/Mistral-7B` |
| `ibm` | `langchain_ibm` | `ChatWatsonx` | lambda (model_id param) | (IBM Watsonx models) |

**Note**: The registry is not exhaustive; ~35 providers are built in. Unlisted providers can still be used if their integration package is installed by passing `model_provider` explicitly.

## Model Identifier Inference

When no provider is specified, `init_chat_model` attempts to infer the provider from model name prefixes:

| Model Pattern | Inferred Provider | Examples |
|---------------|----|----------|
| `gpt-...`, `o1...`, `o3...` | `openai` | `gpt-4-turbo`, `o1`, `o3-mini` |
| `claude...` | `anthropic` | `claude-opus-4-7`, `claude-haiku-4-5` |
| `amazon....`, `anthropic....`, `meta....` | `bedrock` | `amazon.titan-text-express-v1` |
| `command...` | `cohere` | `command-r-plus` |
| `accounts/fireworks...` | `fireworks` | `accounts/fireworks/models/mixtral-8x7b-instruct` |
| `mistral...`, `mixtral...` | `mistralai` | `mistral-large`, `mixtral-8x7b` |
| `deepseek...` | `deepseek` | `deepseek-v3` |
| `grok...` | `xai` | `grok-beta` |
| `sonar...` | `perplexity` | `sonar-small` |
| `solar...` | `upstage` | `solar-pro` |
| `gemini...` | `google_vertexai` | `gemini-1.5-pro` |

**Inference caveats**: Inference is best-effort and not guaranteed. Always use the explicit provider prefix form (`provider:model-name`) when provider identity must be certain.

## Initialization Flow

When `init_chat_model(model, model_provider=None, configurable_fields=None, **kwargs)` is called, the factory function executes these steps:

**Step 1: Determine Fixed vs. Configurable**
- If `model` is provided and `configurable_fields` is `None`, return a fixed model
- If `model` is `None` or `configurable_fields` is non-`None`, return a `_ConfigurableModel` wrapper instead

**Step 2: Parse Model Identifier** (via `_parse_model`)
- Check for `"provider:model"` syntax; if found and provider is in `_BUILTIN_PROVIDERS`, split and extract both
- Normalize provider name: lowercase, replace hyphens with underscores
- If no provider is explicit, call `_attempt_infer_model_provider(model)` which checks against known prefixes
- If inference fails, raise `ValueError` with list of supported providers and docs link

**Step 3: Get Creator Function** (via `_get_chat_model_creator`, cached)
- Look up provider in `_BUILTIN_PROVIDERS`
- Check provider exists; raise `ValueError` if not found
- Import the provider's module via `_import_module(module_path, class_name)`
- On import failure, provide a helpful error message suggesting the package to install (e.g., "pip install langchain-openai")
- Special fallback: if provider is `ollama` and `langchain_ollama` import fails, try `langchain_community.chat_models` for backwards compatibility
- Return `functools.partial(creator_func, cls=<imported_class>)`

**Step 4: Instantiate Model** (via `creator_func`)
- Call the creator function with `model=<model_name>` and `**kwargs`
- Creator functions handle provider-specific initialization:
  - `_call(cls, model, **kwargs)`: Simple instantiation `cls(model=model, **kwargs)`
  - `_init_langsmith(cls, **kwargs)`: Sets OpenAI gateway URL, extracts API key from environment
  - Lambda for HuggingFace: calls `cls.from_model_id(model_id=model, **kwargs)` instead
  - Lambda for IBM: passes `model_id=model` instead of `model=model`
- Return ready-to-use `BaseChatModel` instance

### Error Handling

| Scenario | Error Type | Cause | Message |
|----------|-----------|-------|---------|
| Unknown provider | `ValueError` | Provider not in `_BUILTIN_PROVIDERS` and cannot be inferred | "Unable to infer model provider for model=...\nPlease specify 'model_provider' directly.\n\nSupported providers: ..." |
| Missing provider package | `ImportError` | Provider package not installed | "Initializing ChatOpenAI requires the langchain-openai package. Please install it with `pip install langchain-openai`" |
| Invalid model type | `TypeError` | Passing model object instead of string | "`model` must be a string (e.g., 'openai:gpt-5.5'), got <type>. If you've already constructed a chat model object, use it directly..." |
| config_prefix set but no fields configurable | `UserWarning` | `config_prefix` provided when `configurable_fields=None` | "config_prefix=... has been set but no fields are configurable..." |

## Configuration: Fixed vs. Configurable Models

### Fixed Model

When `configurable_fields=None` (default if model is specified):

```python
model = init_chat_model("openai:gpt-4-turbo", temperature=0.7)

# model is a BaseChatModel subclass instance
response = model.invoke("Hello")
```

- Returns the instantiated `BaseChatModel` immediately
- All parameters frozen at initialization time
- Direct access to model methods: `get_num_tokens()`, `bind_tools()`, etc.

### Configurable Model (`_ConfigurableModel`)

When `configurable_fields` is set to `"any"`, a list, or a tuple (or when model is `None`), `init_chat_model` returns a **`_ConfigurableModel`** wrapper instead of a `BaseChatModel`. This wrapper:

- **Defers model instantiation** until `invoke` time, when configuration is provided
- **Queues declarative operations** (`bind_tools`, `with_structured_output`) without a concrete model, and applies them after instantiation
- **Filters runtime configuration** by field: only fields in `configurable_fields` are extracted from `config["configurable"]` and passed to model initialization (unless `configurable_fields="any"`)
- **Applies config prefix** to isolate multiple configurable models in the same application
- **Implements the `Runnable` interface** (invoke, ainvoke, stream, astream, batch) by delegating to the instantiated model

**Key behaviors:**

```python
# Case 1: Fully configurable with no defaults — model must be specified at invoke time
model = init_chat_model()
response = model.invoke(
    "Hello",
    config={"configurable": {"model": "openai:gpt-4-turbo"}}
)

# Case 2: Configurable with defaults — use defaults if not overridden
model = init_chat_model(
    "openai:gpt-4-turbo",
    configurable_fields=("temperature", "max_tokens"),
    temperature=0.5
)
response = model.invoke(
    "Hello",
    config={"configurable": {"temperature": 0.9}}  # Overrides default 0.5
)

# Case 3: With config_prefix for multi-model setups
primary = init_chat_model(
    "openai:gpt-4-turbo",
    configurable_fields="any",
    config_prefix="primary"
)
fallback = init_chat_model(
    "openai:gpt-3.5-turbo",
    configurable_fields="any",
    config_prefix="fallback"
)
response = primary.invoke(
    "Hello",
    config={
        "configurable": {
            "primary_temperature": 0.8,
            "fallback_temperature": 0.5
        }
    }
)  # Only primary_temperature affects primary model
```

**Implementation details:**

- `_ConfigurableModel.__getattr__` intercepts calls to `bind_tools` and `with_structured_output` (tracked in `_DECLARATIVE_METHODS`) by queueing them as `(method_name, args, kwargs)` tuples
- When the model is instantiated at invoke time via `_model(config)`, all queued operations are applied in order: `model = getattr(model, name)(*args, **kwargs)`
- `_model_params(config)` extracts runtime params by checking `config["configurable"]` for keys starting with `config_prefix`, removes the prefix, and filters by `configurable_fields`
- Returns a `_ConfigurableModel` wrapper (not a `BaseChatModel`)
- Non-configurable methods (e.g., `get_num_tokens()`) only work if a default model is provided

### Field-Level Configurability

Control which parameters are configurable at runtime:

```python
# All fields configurable (⚠️ security: allows api_key changes at runtime)
model = init_chat_model("openai:gpt-4-turbo", configurable_fields="any")

# Only specific fields configurable
model = init_chat_model(
    "openai:gpt-4-turbo",
    configurable_fields=["temperature", "max_tokens", "model"]
)

# Only model provider (enable model switching)
model = init_chat_model(
    configurable_fields=["model", "model_provider"]
)
```

### Config Prefix

The `config_prefix` parameter namespaces configurable fields, enabling multiple independent configurable models in the same application:

```python
fast_model = init_chat_model(
    "openai:gpt-4-turbo",
    configurable_fields="any",
    config_prefix="fast"
)

accurate_model = init_chat_model(
    "openai:gpt-4",
    configurable_fields="any",
    config_prefix="accurate"
)

response = fast_model.invoke(
    "Quick question",
    config={
        "configurable": {
            "fast_temperature": 1.0,
            "accurate_temperature": 0.1
        }
    }
)
```

## Common Configuration Patterns

### Standard Model Parameters

All models support these common parameters via `**kwargs`:

```python
model = init_chat_model(
    "openai:gpt-4-turbo",
    temperature=0.7,          # Randomness: 0.0 (deterministic) to 1.0+
    max_tokens=1000,          # Maximum output tokens
    timeout=30.0,             # Request timeout in seconds
    max_retries=3,            # Automatic retry attempts
)
```

### API Key and Endpoint Configuration

Credentials and custom endpoints:

```python
# OpenAI with custom endpoint
model = init_chat_model(
    "openai:gpt-4-turbo",
    api_key="sk-...",
    base_url="https://api.openai.com/v1"
)

# Anthropic
model = init_chat_model(
    "anthropic:claude-opus-4-7",
    api_key="sk-ant-..."
)

# Ollama (local)
model = init_chat_model(
    "ollama:llama2",
    base_url="http://localhost:11434"
)

# Azure OpenAI
model = init_chat_model(
    "azure_openai:gpt-4-turbo",
    api_key="...",
    azure_endpoint="https://{resource}.openai.azure.com/",
    api_version="2024-10-21"
)
```

### Model Switching at Runtime

Switch between providers without code changes:

```python
model = init_chat_model(configurable_fields="any")

# Use OpenAI
openai_response = model.invoke(
    "Translate to French: Hello",
    config={"configurable": {"model": "openai:gpt-4-turbo"}}
)

# Use Anthropic
anthropic_response = model.invoke(
    "Translate to French: Hello",
    config={"configurable": {"model": "anthropic:claude-opus-4-7"}}
)

# Use Ollama
local_response = model.invoke(
    "Translate to French: Hello",
    config={"configurable": {"model": "ollama:llama2"}}
)
```

## Declarative Operations on Configurable Models

Configurable models queue declarative operations until the underlying model is instantiated:

```python
model = init_chat_model(configurable_fields="any")

# Queue bind_tools before model is selected
model_with_tools = model.bind_tools([
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {"type": "object", "properties": {...}}
    }
])

# Queue with_structured_output
model_with_schema = model_with_tools.with_structured_output(SomeSchema)

# At invocation time, queued operations are applied after model instantiation
response = model_with_schema.invoke(
    "What's the weather in San Francisco?",
    config={"configurable": {"model": "openai:gpt-4-turbo"}}
)
```

## Provider-Specific Initialization

Some providers require custom initialization logic:

### LangSmith Gateway

The `langsmith` provider routes through OpenAI's client to LangSmith's Gateway:

```python
# Environment-based configuration
import os
os.environ["LANGSMITH_GATEWAY_API_KEY"] = "..."

model = init_chat_model("langsmith:model-name")
# Automatically sets:
# - openai_api_base = "https://gateway.smith.langchain.com/v1"
# - use_responses_api = True
# - api_key from LANGSMITH_GATEWAY_API_KEY or LANGSMITH_API_KEY

# Explicit configuration
model = init_chat_model(
    "langsmith:model-name",
    api_key="explicit-key",
    base_url="https://custom-gateway.example.com/v1"
)
```

### Hugging Face Models

Special creator function handles `from_model_id`:

```python
model = init_chat_model(
    "huggingface:gpt2",  # model param → model_id kwarg
    task="text-generation"
)
```

### IBM Watson

Special creator function maps model to `model_id`:

```python
model = init_chat_model(
    "ibm:my-ibm-model",  # model param → model_id kwarg
    project_id="...",
    space_id="..."
)
```

## Model Profiles: Capability Metadata

Chat models optionally expose a `.profile` field containing structured capability data. Profile data is generated by the [`langchain-model-profiles`](https://github.com/langchain-ai/langchain/tree/master/libs/model-profiles) package, which fetches standardized model metadata from [models.dev](https://github.com/sst/models.dev) (an open-source initiative) and augments it with LangChain-specific fields.

### Profile Registry and Storage

- **Model profiles are stored** in each integration package's `data/_profiles.py` file as a `_PROFILES` dictionary mapping model IDs to profile dictionaries
- **Profiles are generated** via the `langchain-model-profiles` CLI tool: `langchain-profiles refresh --provider anthropic --data-dir ./langchain_anthropic/data`
- **The CLI merges** models.dev data with provider-specific augmentations from `profile_augmentations.toml`
- **Chat model instances** access profiles via the `.profile` property, which retrieves the matching profile from the model's integration package

### Profile Fields

Profiles are dictionaries (typed as `ModelProfile` in `langchain_core.language_models.model_profile.ModelProfile`) with these standard fields:

**Metadata:**
- `name`: Display name
- `status`: Stability status (`"stable"`, `"preview"`, `"deprecated"`)
- `release_date`: ISO 8601 date string
- `last_updated`: Last update date
- `open_weights`: Boolean, whether the model weights are publicly available

**Context window:**
- `max_input_tokens`: Maximum input token capacity
- `max_output_tokens`: Maximum output token capacity

**Modalities (boolean flags):**
- Input: `text_inputs`, `image_inputs`, `audio_inputs`, `video_inputs`, `pdf_inputs`, `image_url_inputs`
- Output: `text_outputs`, `image_outputs`, `audio_outputs`, `video_outputs`

**Capabilities (boolean flags):**
- `tool_calling`: Support for function calling
- `tool_choice`: Support for tool choice parameter (force, required)
- `tool_call_streaming`: Support for streaming tool calls
- `structured_output`: Support for structured output mode
- `reasoning_output`: Support for reasoning/chain-of-thought
- `temperature`: Support for temperature control

**Special:**
- `image_tool_message`: Whether model can see images in tool messages
- `pdf_tool_message`: Whether model can see PDFs in tool messages
- `attachment`: Support for file attachments

### Profile Access

```python
model = init_chat_model("openai:gpt-4-turbo")

# Profile is a dict-like object (typed as ModelProfile)
if hasattr(model, 'profile') and model.profile:
    profile = model.profile
    print(f"Model: {profile.get('name')}")
    print(f"Max input tokens: {profile.get('max_input_tokens')}")
    print(f"Supports tools: {profile.get('tool_calling')}")
    print(f"Supports images: {profile.get('image_inputs')}")
    print(f"Supports structured output: {profile.get('structured_output')}")
    print(f"Max output tokens: {profile.get('max_output_tokens')}")

# Use profiles to check capability before calling methods
if model.profile and model.profile.get('tool_calling'):
    model = model.bind_tools([...])

# Some models may not have profiles yet (None or missing)
if not model.profile:
    print("Profile data not available for this model")
```

### Updating Profiles

Model profiles are periodically updated to reflect changes in model capabilities. To manually update profiles for a provider:

```bash
# Update OpenAI profiles
langchain-profiles refresh --provider openai --data-dir ./langchain_openai/data

# Update all providers
langchain-profiles refresh-all
```

The tool automatically:
1. Fetches latest data from models.dev
2. Merges with local `profile_augmentations.toml` overrides
3. Generates `data/_profiles.py` with the merged data
4. Creates a PR with the changes

## Fallback and Compatibility

### Ollama Backwards Compatibility

For historical reasons, if `langchain-ollama` is not available, the system falls back to `langchain-community.chat_models.ChatOllama`:

```python
# Tries langchain_ollama.ChatOllama first
# Falls back to langchain_community.chat_models.ChatOllama if needed
model = init_chat_model("ollama:llama2")
```

### Unknown Providers

If a provider is not registered but its integration package is installed, pass `model_provider` explicitly:

```python
# Custom/unlisted provider
model = init_chat_model(
    "my-model-name",
    model_provider="my_custom_provider"  # Maps to custom class import
)
```

## Entrypoints and Extension

### Public API Entrypoint

The public API for loading models is:

```python
from langchain.chat_models import init_chat_model

# Fixed model
model = init_chat_model("openai:gpt-4-turbo", temperature=0.7)

# Configurable model
model = init_chat_model(configurable_fields="any")
response = model.invoke("...", config={"configurable": {"model": "gpt-4-turbo"}})
```

The function lives in `/libs/langchain_v1/langchain/chat_models/base.py` and is exported via `/libs/langchain_v1/langchain/chat_models/__init__.py`.

### Adding New Providers to the Built-in Registry

To add a new provider to `_BUILTIN_PROVIDERS`, edit `/libs/langchain_v1/langchain/chat_models/base.py` and append an entry:

```python
_BUILTIN_PROVIDERS["my_provider"] = (
    "langchain_my_provider",           # Module path where ChatMyProvider is defined
    "ChatMyProvider",                  # Class name to instantiate
    _call,                             # Creator function (use _call for standard init)
)
```

**Ensure the entry is appended to the dict** (not inserted in the middle), as documented by the contributor note in the code.

### Custom Creator Functions

For provider-specific initialization logic, define a custom creator function:

```python
def _init_my_provider(cls: type[BaseChatModel], model: str, **kwargs: Any) -> BaseChatModel:
    """Custom initialization for my_provider.
    
    Transforms model name, sets API endpoint, extracts environment variables, etc.
    """
    # Example: transform model identifier
    if model == "small":
        model = "my_provider_small_v2"
    
    # Example: set endpoint from environment
    import os
    if "api_url" not in kwargs:
        kwargs["api_url"] = os.getenv("MY_PROVIDER_API_URL", "https://api.myprovider.com")
    
    # Example: extract API key from environment
    if "api_key" not in kwargs:
        api_key = os.getenv("MY_PROVIDER_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
    
    return cls(model=model, **kwargs)

_BUILTIN_PROVIDERS["my_provider"] = (
    "langchain_my_provider",
    "ChatMyProvider",
    _init_my_provider,  # Use custom creator
)
```

**Example from codebase**: The `langsmith` provider uses `_init_langsmith` to apply LangSmith Gateway configuration (API base URL, API key extraction from environment variables, enabling responses API mode). The `huggingface` provider uses a lambda to call `.from_model_id()` instead of `__init__()`.

### Extending Without Built-in Registry

If a provider's integration package is installed but not in `_BUILTIN_PROVIDERS`, users can still use it via the `model_provider` parameter:

```python
# If langchain_my_provider is installed but not registered
model = init_chat_model(
    "my-model-name",
    model_provider="my_provider"  # Must map to a package following standard structure
)
```

This assumes the integration package follows the partner pattern: module path = `langchain_<provider>`, class name = `Chat<Provider>` or similar. Refer to [Partner Pattern documentation](/openwiki/integrations/partner-pattern.md) for details.

## Security Considerations

### Configuration Exposure

Setting `configurable_fields="any"` allows untrusted configuration to modify sensitive fields at runtime:

```python
# ⚠️ WARNING: This allows api_key to be changed at runtime
model = init_chat_model("openai:gpt-4-turbo", configurable_fields="any")

# Malicious config could redirect requests
config = {
    "configurable": {
        "api_key": "attacker-key",
        "base_url": "https://attacker.example.com"
    }
}
```

**Best practice**: Enumerate allowed fields explicitly:

```python
# ✓ Only allow safe parameters to be reconfigured
model = init_chat_model(
    "openai:gpt-4-turbo",
    configurable_fields=["temperature", "max_tokens"]
)
```

## Testing Patterns

Common test patterns for `init_chat_model`:

```python
import pytest
from langchain.chat_models import init_chat_model

@pytest.mark.requires("langchain_openai", "langchain_anthropic")
def test_init_chat_model_providers():
    """Test model creation across providers."""
    models = [
        init_chat_model("openai:gpt-4-turbo", api_key="test"),
        init_chat_model("anthropic:claude-opus-4-7", api_key="test"),
    ]
    for model in models:
        assert hasattr(model, "invoke")
        assert hasattr(model, "astream")

@pytest.mark.requires("langchain_openai")
def test_configurable_model_switching():
    """Test runtime model switching."""
    model = init_chat_model(configurable_fields="any")
    
    # Model undefined until config is provided
    with pytest.raises(AttributeError):
        model.get_num_tokens("text")
    
    # Works when configured
    response = model.invoke(
        "test",
        config={"configurable": {"model": "gpt-4-turbo"}}
    )
    assert response is not None
```

## Summary

`init_chat_model` abstracts provider routing, module loading, and configuration into a simple unified interface. It supports:

- **Fixed models** for deterministic deployments
- **Configurable models** for runtime flexibility
- **Provider discovery** via explicit prefixes or inference
- **Lazy module loading** with error handling
- **Declarative operations** on configurable models
- **Model profiles** for capability introspection
- **Security controls** via field-level configurability

Use it whenever you need to load a chat model from any LangChain integration without coupling to provider-specific classes.
