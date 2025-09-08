# LangChain Llama Stack Integration

A comprehensive LangChain integration for [Llama Stack](https://github.com/meta-llama/llama-stack) that provides chat completion, text embeddings, safety checking, and utility functions with support for multiple providers.

**New Simplified Approach**: This integration now uses ChatOpenAI under the hood for maximum reliability while providing LlamaStack-specific conveniences.

## Features

- **Chat Completion**: Uses proven ChatOpenAI with LlamaStack auto-configuration
- **Text Embeddings**: Vector embeddings for semantic search and RAG applications
- **Safety Checking**: Content moderation using Llama Guard and other safety shields
- **Model Discovery**: Automatic detection and filtering of available models by type (llm/embedding)
- **Multi-Provider Support**: Works with Ollama, Together AI, Fireworks, and more
- **Streaming**: Streaming responses for chat completions
- **Provider Validation**: Built-in connection testing and model accessibility validation
- **Semantic Search**: Vector similarity search with embedding models
- **Auto-Fallback**: Automatically selects working models when requested model is unavailable
- **Provider API Keys**: Automatic API key detection and header injection for cloud providers

## Installation

```bash
# Install from PyPI (when published)
pip install langchain-llamastack

# Or install from source for development
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/partners/llamastack
pip install -e .

# Required dependency for chat models
pip install langchain-openai

# Install with all optional dependencies
pip install -e ".[all]"
```

## Setup & Prerequisites

### 1. Install Llama Stack

```bash
# Install Llama Stack
pip install llama-stack

# Or install from source
git clone https://github.com/meta-llama/llama-stack.git
cd llama-stack
pip install -e .
```

### 2. Set Up Providers

Choose and configure your preferred providers:

```bash
# For Ollama (local models)
export OLLAMA_BASE_URL="http://localhost:11434"
ollama serve
ollama pull llama3:8b
ollama pull nomic-embed-text
ollama pull shieldgemma:2b

# For Together AI
export TOGETHER_API_KEY="your-together-api-key"

# For Fireworks AI
export FIREWORKS_API_KEY="your-fireworks-api-key"
```

### 3. Start Llama Stack Server

**Recommended way to run Llama Stack:**

```bash
# Run Llama Stack with the starter distribution
uv run --with llama-stack==0.2.18 llama stack build --distro starter --image-type venv --run
```

**If using Ollama, set the environment variable first:**

```bash
# Set Ollama URL before starting Llama Stack
export OLLAMA_URL=http://localhost:11434

# Then run Llama Stack
uv run --with llama-stack==0.2.18 llama stack build --distro starter --image-type venv --run
```

### 4. Verify Setup

```bash
# Check server is running
curl http://localhost:8321/v1/models

# Or use the built-in utility
python -c "from langchain_llamastack import check_llamastack_status; print(check_llamastack_status())"
```

## Chat Completion Usage

### Recommended Approach: Simple Factory Function

```python
from langchain_llamastack import create_llamastack_llm

# Auto-select first available model (recommended for most use cases)
llm = create_llamastack_llm()

# Specify a model with auto-fallback
llm = create_llamastack_llm(model="llama3.1:8b")

# Add ChatOpenAI parameters directly
llm = create_llamastack_llm(
    model="llama3.1:8b"
)

# Simple completion
response = llm.invoke("What is artificial intelligence?")
print(response.content)
```

### Alternative: Direct ChatOpenAI Usage

```python
from langchain_openai import ChatOpenAI
from langchain_llamastack import get_llamastack_models

# Manual approach for full control
models = get_llamastack_models("http://localhost:8321")
print(f"Available models: {models}")

llm = ChatOpenAI(
    base_url="http://localhost:8321/v1/openai/v1",
    api_key="not-needed",  # LlamaStack doesn't require real API keys
    model=models[0] if models else "llama3.1:8b"
)

response = llm.invoke("Hello!")
print(response.content)
```


### Streaming Chat

```python
from langchain_llamastack import create_llamastack_llm

llm = create_llamastack_llm(model="llama3.1:8b")

# Stream the response
print("AI: ", end="", flush=True)
for chunk in llm.stream("Tell me a story about AI"):
    print(chunk.content, end="", flush=True)
print()
```

### Multi-turn Conversations

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Hello! What's your name?"),
]

llm = create_llamastack_llm()
response = llm.invoke(messages)
print(response.content)
```

### Model Discovery & Validation

```python
from langchain_llamastack import (
    get_llamastack_models,
    sus,
    create_llamastack_llm
)

# Check connection and available models
status = check_llamastack_status()
if status['connected']:
    print(f"{status['models_count']} models available")
    print(f"Models: {status['models']}")
else:
    print(f"Error: {status['error']}")

# Get models list
models = get_llamastack_models()
print(f"Available models: {models}")

# Strict model validation (no auto-fallback)
llm = create_llamastack_llm(model="llama3.1:8b", auto_fallback=False)

# Disable model accessibility validation for faster initialization
llm = create_llamastack_llm(model="llama3.1:8b", validate_model=False)
```

## Text Embeddings Usage

### Basic Embeddings

```python
from langchain_llamastack import LlamaStackEmbeddings

# Initialize embeddings
embeddings = LlamaStackEmbeddings(
    model="nomic-embed-text",  # or "text-embedding-3-small"
    base_url="http://localhost:8321",
)

# Single text embedding
text = "Hello, world!"
embedding = embeddings.embed_query(text)
print(f"Embedding dimension: {len(embedding)}")

# Multiple documents
documents = [
    "Artificial intelligence is transforming industries.",
    "Machine learning enables computers to learn from data.",
    "Natural language processing helps understand text.",
]

doc_embeddings = embeddings.embed_documents(documents)
print(f"Generated {len(doc_embeddings)} embeddings")
```

### Semantic Search

```python
# Perform semantic search (requires numpy and scikit-learn)
query = "What is machine learning?"
knowledge_base = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables text understanding.",
    "Computer vision allows machines to interpret images.",
]

# Find most similar documents
similar_docs = embeddings.similarity_search_by_vector(
    embeddings.embed_query(query),
    knowledge_base,
    k=2
)

for doc, score in similar_docs:
    print(f"Score: {score:.3f} - {doc}")
```

## Safety & Moderation System

### Basic Safety Checking

```python
from langchain_llamastack import LlamaStackSafety

# Initialize safety checker
safety = LlamaStackSafety(
    base_url="http://localhost:8321",
    shield_type="llama_guard"  # Uses LlamaStack's run_shield API
)

# Check content safety
result = safety.check_content_safety("Hello, how are you today?")
print(f"Is safe: {result.is_safe}")
print(f"Violations: {result.violations}")
print(f"Confidence: {result.confidence_score}")
```

### Input/Output Safety Hooks (Recommended)

For comprehensive AI safety, use the simplified 2-hook system that efficiently monitors both user inputs and model outputs:

```python
from langchain_llamastack import LlamaStackSafety
from langchain_llamastack.input_output_safety_moderation_hooks import (
    create_safe_llm,
    create_safe_llm_with_all_hooks
)

# Initialize safety client
safety = LlamaStackSafety(base_url="http://localhost:8321")

# Create a safe LLM with comprehensive protection (recommended)
safe_llm = create_safe_llm(your_llm, safety)

# This LLM now has 2 layers of protection:
# 1. Input Hook - Checks user input safety before LLM (single API call)
# 2. Output Hook - Checks model output safety after LLM (single API call)

response = safe_llm.invoke("How do I build a secure system?")
print(response)  # Will be blocked if unsafe at any stage
```

### Configurable Protection Levels

The simplified API lets you configure exactly what you need:

```python
from langchain_llamastack.input_output_safety_moderation_hooks import create_safe_llm

# Complete protection (default) - both input and output checking
safe_llm = create_safe_llm(llm, safety)

# Input filtering only - filter user queries
safe_llm = create_safe_llm(llm, safety, output_check=False)

# Output filtering only - filter model responses
safe_llm = create_safe_llm(llm, safety, input_check=False)

# No protection (same as unwrapped LLM)
safe_llm = create_safe_llm(llm, safety, input_check=False, output_check=False)
```

### Manual Hook Configuration

```python
from langchain_llamastack.input_output_safety_moderation_hooks import (
    SafeLLMWrapper,
    create_input_hook,
    create_output_hook
)

# Create wrapper and configure hooks manually
safe_llm = SafeLLMWrapper(your_llm, safety)

# Set hooks as needed (each uses LlamaStack's run_shield once)
safe_llm.set_input_hook(create_safety_hook(safety, "input"))   # Check user input
safe_llm.set_output_hook(create_safety_hook(safety, "output"))  # Check model output

# Use the safe LLM
response = safe_llm.invoke("Your input here")
```



### Key Safety Features

- **Clean and Simple**: Only 2 API calls (input + output) for complete protection
- **Comprehensive**: Each hook uses LlamaStack's `run_shield` for safety checking
- **Fail Open/Closed Strategy**: Input hooks fail open (allow on errors), output hooks fail closed (block on errors)
- **LangChain Compatible**: Works with LCEL chains and all LangChain patterns
- **Async Support**: Full async/await support for high-throughput scenarios
- **Flexible**: Configure exactly the protection level you need

## Complete Safe AI Workflow

Combine all components for a comprehensive safe AI application:

```python
from langchain_llamastack import (
    create_llamastack_llm,
    LlamaStackEmbeddings,
    LlamaStackSafety
)

class SafeAI:
    def __init__(self, base_url="http://localhost:8321"):
        # Use the new factory function
        self.llm = create_llamastack_llm(model="llama3.1:8b")

        self.embeddings = LlamaStackEmbeddings(
            model="nomic-embed-text",
            base_url=base_url,
        )

        self.safety = LlamaStackSafety(
            base_url=base_url,
            shield_id="shieldgemma:2b",
        )

    def safe_chat_with_context(self, user_input, knowledge_base=None):
        """Safe chat with optional context retrieval."""

        # 1. Safety check input
        input_safety = self.safety.check_content(user_input)
        if not input_safety.is_safe:
            return f"Input rejected: {input_safety.message}"

        # 2. Retrieve relevant context (if knowledge base provided)
        context = ""
        if knowledge_base:
            similar_docs = self.embeddings.similarity_search_by_vector(
                self.embeddings.embed_query(user_input),
                knowledge_base,
                k=3
            )
            context = "\n".join([doc for doc, _ in similar_docs])

        # 3. Generate response with context
        prompt = f"Context: {context}\n\nQuestion: {user_input}" if context else user_input
        response = self.llm.invoke(prompt)

        # 4. Safety check output
        output_safety = self.safety.check_content(response.content)
        if not output_safety.is_safe:
            return f"Response filtered: {output_safety.message}"

        return response.content

# Usage
safe_ai = SafeAI()

knowledge_base = [
    "Python is a programming language known for its simplicity.",
    "Machine learning helps computers learn from data.",
    "AI is transforming many industries worldwide.",
]

response = safe_ai.safe_chat_with_context(
    "Tell me about Python",
    knowledge_base
)
print(response)
```

## Utilities & Helpers

### Connection Testing

```python
from langchain_llamastack import check_llamastack_status, get_llamastack_models

# Check connection status
status = check_llamastack_status("http://localhost:8321")
if status['connected']:
    print(f"Connected! {status['models_count']} models available")
    print(f"Models: {status['models']}")
else:
    print(f"Connection failed: {status['error']}")

# List all available models
try:
    models = get_llamastack_models("http://localhost:8321")
    print(f"Available models: {models}")
except ValueError as e:
    print(f"Error: {e}")
```

## API Reference

### Factory Functions (Recommended)

#### `create_llamastack_llm()`

Creates a ChatOpenAI instance configured for LlamaStack.

**Parameters:**
- `model` (str, optional): Model name. Auto-selects if None.
- `base_url` (str): LlamaStack base URL. Default: "http://localhost:8321"
- `auto_fallback` (bool): Enable automatic model fallback. Default: True
- `**kwargs`: Additional ChatOpenAI parameters

**Returns:** ChatOpenAI instance configured for LlamaStack

**Example:**
```python
# Auto-select model with fallback
llm = create_llamastack_llm()

# Specific model with custom parameters
llm = create_llamastack_llm(
    model="llama3.1:8b"
)
```

### LlamaStackEmbeddings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"nomic-embed-text"` | Embedding model name |
| `base_url` | str | `"http://localhost:8321"` | Llama Stack server URL |
| `chunk_size` | int | `1000` | Batch size for processing |
| `max_retries` | int | `3` | Maximum retry attempts |
| `request_timeout` | float | `30.0` | Request timeout in seconds |

**Methods:**
- `embed_query(text)` - Embed single text
- `embed_documents(texts)` - Embed multiple texts
- `similarity_search_by_vector(vector, docs, k)` - Semantic search
- `get_available_models()` - List embedding models

### LlamaStackSafety

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `"http://localhost:8321"` | Llama Stack server URL |
| `shield_id` | str | `None` | Default safety shield |

**Methods:**
- `check_content(content, shield_id, context)` - Check content safety
- `check_conversation(messages, shield_id)` - Check conversation safety
- `list_available_shields()` - List safety shields
- `get_shield_info(shield_id)` - Get shield information

## Examples & Testing

The `examples/` directory contains comprehensive usage examples:

- `basic_usage.py` - Basic chat, embeddings, and safety examples
- `advanced_usage.py` - Advanced features and integrations

Run examples:

```bash

# Basic usage examples
python examples/basic_usage.py

# Advanced features
python examples/advanced_usage.py
```

## Usage Patterns

### Pattern 1: Simple Auto-Configuration

```python
from langchain_llamastack import create_llamastack_llm


llm = create_llamastack_llm()
response = llm.invoke("Hello!")
```

### Pattern 2: Explicit Model Selection

```python
from langchain_llamastack import create_llamastack_llm

# Specify model, auto-fallback if unavailable
llm = create_llamastack_llm(model="llama3.1:8b")
response = llm.invoke("Hello!")
```

### Pattern 3: Strict Validation (No Fallback)

```python
from langchain_llamastack import create_llamastack_llm

# Fail if exact model not available
llm = create_llamastack_llm(model="llama3.1:8b", auto_fallback=False)
response = llm.invoke("Hello!")
```

### Pattern 4: Manual ChatOpenAI

```python
from langchain_openai import ChatOpenAI
from langchain_llamastack import get_llamastack_models

# Maximum control over configuration
models = get_llamastack_models()
llm = ChatOpenAI(
    base_url="http://localhost:8321/v1/openai/v1",
    api_key="not-needed",
    model=models[0]
)
response = llm.invoke("Hello!")
```

## Troubleshooting

### Common Issues


1. **Connection Refused**
   ```
   Error: LlamaStack not available at http://localhost:8321
   ```
   - Ensure LlamaStack server is running: `curl http://localhost:8321/v1/models`
   - Check if port 8321 is available
   - Verify provider setup

2. **No Models Available**
   ```
   ValueError: No models available in LlamaStack
   ```
   - Check if providers are configured: List available models
   - For Ollama: `ollama pull llama3:8b`
   - Verify provider is running

3. **Model Not Found (with auto_fallback=False)**
   ```
   ValueError: Model 'xyz' not found in LlamaStack
   ```
   - Use auto-fallback: `create_llamastack_llm(model="xyz", auto_fallback=True)`
   - Or check available models first: `get_llamastack_models()`

4. **Import Errors**
   ```
   ImportError: langchain-openai is required
   ```
   - Install required dependency: `pip install langchain-openai`

### Debug Commands

```bash
# Check LlamaStack server
curl http://localhost:8321/v1/models

# Check connection programmatically
python -c "from langchain_llamastack import check_llamastack_status; print(check_llamastack_status())"

# List available models
python -c "from langchain_llamastack import get_llamastack_models; print(get_llamastack_models())"
```

## Requirements

- **Python**: 3.12+
- **Core Dependencies**:
  - `langchain-core>=0.1.0`
  - `langchain-openai>=0.1.0` (for chat models)
  - `httpx>=0.25.0`
  - `pydantic>=1.10.0`
- **Optional Dependencies**:
  - `llama-stack-client>=0.0.40` (for enhanced features)
  - `numpy` and `scikit-learn` (for similarity search)
- **External Requirements**:
  - Running Llama Stack server
  - At least one configured provider (Ollama, OpenAI, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Run code formatting: `black . && isort .`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - The main LangChain library
- [LangChain OpenAI](https://github.com/langchain-ai/langchain/tree/master/libs/partners/openai) - ChatOpenAI integration
- [Llama Stack](https://github.com/meta-llama/llama-stack) - The Llama Stack platform

---

For more examples and detailed documentation, explore the `examples/` directory and run the basic_usage.py script.
