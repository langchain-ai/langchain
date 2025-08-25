# LangChain Llama Stack Integration

A comprehensive LangChain integration for [Llama Stack](https://github.com/meta-llama/llama-stack) that provides chat completion, text embeddings, safety checking, and utility functions with support for multiple providers.

## 🚀 Features

- **💬 Chat Completion**: Full LangChain-compatible chat models with streaming support
- **🔤 Text Embeddings**: Vector embeddings for semantic search and RAG applications
- **🛡️ Safety Checking**: Content moderation using Llama Guard and other safety shields
- **🔍 Model Discovery**: Automatic detection and listing of available models
- **🔌 Multi-Provider Support**: Works with Ollama, OpenAI, Together AI, Fireworks, and more
- **⚡ Streaming**: Real-time streaming responses for chat completions
- **🔄 Provider Validation**: Built-in connection testing and environment validation
- **📊 Semantic Search**: Vector similarity search with embedding models

## 📦 Installation

```bash
# Install from PyPI (when published)
pip install langchain-llamastack

# Or install from source for development
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/partners/llamastack
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"
```

## 🏗️ Setup & Prerequisites

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

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Together AI
export TOGETHER_API_KEY="your-together-api-key"

# For Fireworks AI
export FIREWORKS_API_KEY="your-fireworks-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Start Llama Stack Server

```bash
# With Ollama provider
llama-stack-run --port 8321 --inference-provider remote::ollama

# With multiple providers
llama-stack-run --port 8321 \
  --inference-provider remote::ollama \
  --inference-provider remote::together \
  --embedding-provider remote::ollama \
  --safety-provider remote::ollama
```

### 4. Verify Setup

```bash
# Check server is running
curl http://localhost:8321/v1/models

# Or use the built-in utility
python -c "from langchain_llamastack import check_llamastack_connection; print(check_llamastack_connection())"
```

## 💬 Chat Completion Usage

### Basic Chat

```python
from langchain_llamastack import ChatLlamaStack

# Initialize the chat model
llm = ChatLlamaStack(
    model="ollama/llama3:8b",  # or "openai/gpt-4o-mini", "together/llama-3.1-8b", etc.
    base_url="http://localhost:8321",
)

# Simple completion
response = llm.invoke("What is artificial intelligence?")
print(response.content)
```

### Streaming Chat

```python
from langchain_llamastack import ChatLlamaStack

llm = ChatLlamaStack(
    model="ollama/llama3:8b",
    base_url="http://localhost:8321",
    streaming=True,
)

# Stream the response
print("AI: ", end="", flush=True)
for chunk in llm.stream("Tell me a story about AI"):
    print(chunk.content, end="", flush=True)
print()
```

### Multi-turn Conversations

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Hello! What's your name?"),
]

response = llm.invoke(messages)
print(response.content)
```

### Model Discovery

```python
# List available models
models = llm.get_available_models()
print(f"Available models: {models}")

# Get model information
model_info = llm.get_model_info("ollama/llama3:8b")
print(f"Model info: {model_info}")
```

## 🔤 Text Embeddings Usage

### Basic Embeddings

```python
from langchain_llamastack import LlamaStackEmbeddings

# Initialize embeddings
embeddings = LlamaStackEmbeddings(
    model="ollama/nomic-embed-text",  # or "openai/text-embedding-3-small"
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

## 🛡️ Safety Checking Usage

### Basic Safety Checking

```python
from langchain_llamastack import LlamaStackSafety

# Initialize safety checker
safety = LlamaStackSafety(
    base_url="http://localhost:8321",
    shield_id="ollama/shieldgemma:2b",  # or "meta-llama/Llama-Guard-3-8B"
)

# Check content safety
result = safety.check_content("Hello, how are you today?")
print(f"Is safe: {result.is_safe}")
print(f"Message: {result.message}")

if not result.is_safe:
    print(f"Violation type: {result.violation_type}")
    print(f"Confidence: {result.confidence_score}")
```

### Conversation Safety

```python
# Check a full conversation
messages = [
    {"role": "user", "content": "Hello there!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I'm looking for help with my research project."}
]

result = safety.check_conversation(messages)
print(f"Conversation safe: {result.is_safe}")
```

### Shield Management

```python
# List available safety shields
shields = safety.list_available_shields()
print(f"Available shields: {shields}")

# Get shield information
shield_info = safety.get_shield_info("ollama/shieldgemma:2b")
print(f"Shield info: {shield_info}")
```

## 🔗 Complete Safe AI Workflow

Combine all components for a comprehensive safe AI application:

```python
from langchain_llamastack import ChatLlamaStack, LlamaStackEmbeddings, LlamaStackSafety

class SafeAI:
    def __init__(self, base_url="http://localhost:8321"):
self.llm = ChatLlamaStack(
            model="ollama/llama3:8b",
            base_url=base_url,
        )

        self.embeddings = LlamaStackEmbeddings(
            model="ollama/nomic-embed-text",
            base_url=base_url,
        )

        self.safety = LlamaStackSafety(
            base_url=base_url,
            shield_id="ollama/shieldgemma:2b",
        )

    def safe_chat_with_context(self, user_input, knowledge_base=None):
        """Safe chat with optional context retrieval."""

        # 1. Safety check input
        input_safety = self.safety.check_content(user_input)
        if not input_safety.is_safe:
            return f"⚠️ Input rejected: {input_safety.message}"

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
            return f"⚠️ Response filtered: {output_safety.message}"

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

## 🔧 Utilities & Helpers

### Connection Testing

```python
from langchain_llamastack import check_llamastack_connection, list_available_models

# Check connection status
status = check_llamastack_connection("http://localhost:8321")
if status['connected']:
    print(f"✅ Connected! {status['models_count']} models available")
else:
    print(f"❌ Connection failed: {status['error']}")

# List all available models
try:
    models = list_available_models("http://localhost:8321")
    print(f"Available models: {models}")
except ValueError as e:
    print(f"Error: {e}")
```

### Provider Environment Checking

```python
# Check if required environment variables are set
from langchain_llamastack.check_provider_env import check_provider_environment

status = check_provider_environment()
print(f"Environment status: {status}")
```

## 📋 API Reference

### ChatLlamaStack

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"ollama/llama3:8b"` | Model identifier |
| `base_url` | str | `"http://localhost:8321"` | Llama Stack server URL |
| `streaming` | bool | `False` | Enable streaming |

**Methods:**
- `invoke(messages)` - Generate completion
- `stream(messages)` - Stream completion
- `get_available_models()` - List available models
- `get_model_info(model_id)` - Get model information

### LlamaStackEmbeddings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"ollama/nomic-embed-text"` | Embedding model name |
| `base_url` | str | `"http://localhost:8321"` | Llama Stack server URL |
| `chunk_size` | int | `1000` | Batch size for processing |
| `max_retries` | int | `3` | Maximum retry attempts |
| `request_timeout` | float | `30.0` | Request timeout in seconds |

**Methods:**
- `embed_query(text)` - Embed single text
- `embed_documents(texts)` - Embed multiple texts
- `similarity_search_by_vector(vector, docs, k)` - Semantic search
- `get_available_models()` - List embedding models
- `get_model_info(model_id)` - Get model information
- `get_embedding_dimension()` - Get embedding dimension

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

### SafetyResult

**Properties:**
- `is_safe: bool` - Whether content is safe
- `violation_type: str` - Type of violation (if any)
- `confidence_score: float` - Confidence score
- `message: str` - Human-readable message
- `shield_id: str` - Shield used for checking
- `raw_response: dict` - Raw API response

## 🧪 Examples & Testing

The `examples/` directory contains comprehensive usage examples:

- `getting_started.py` - Complete setup and usage guide
- `basic_usage.py` - Basic chat, embeddings, and safety examples
- `advanced_usage.py` - Advanced features and integrations
- `provider_examples.py` - Provider-specific configuration examples

Run examples:

```bash
# Complete getting started guide
python examples/getting_started.py

# Basic usage examples
python examples/basic_usage.py

# Advanced features
python examples/advanced_usage.py
```

Run tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=langchain_llamastack

# Run specific test
pytest tests/test_llamastack.py::test_basic_completion
```

## 🔧 Configuration

### Environment Variables

```bash
# LlamaStack server configuration
export LLAMASTACK_BASE_URL="http://localhost:8321"

# Provider API keys (choose what you need)
export OPENAI_API_KEY="your-openai-api-key"
export TOGETHER_API_KEY="your-together-api-key"
export FIREWORKS_API_KEY="your-fireworks-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GROQ_API_KEY="your-groq-api-key"

# Ollama configuration
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Error Handling & Logging

```python
import logging
from langchain_llamastack import ChatLlamaStack

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

try:
    llm = ChatLlamaStack(
        model="ollama/llama3:8b",
        base_url="http://localhost:8321",
    )
    response = llm.invoke("Hello")
    print(response.content)
except Exception as e:
    logging.error(f"Chat completion failed: {e}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: Failed to connect to LlamaStack
   ```
   - Ensure LlamaStack server is running: `curl http://localhost:8321/v1/models`
   - Check if port 8321 is available
   - Verify provider setup

2. **Model Not Found**
   ```
   Error: Model 'xyz' not found
   ```
   - List available models: `curl http://localhost:8321/v1/models`
   - For Ollama: `ollama pull llama3:8b`
   - Check model identifier format

3. **No Shields Available**
   ```
   No safety shields found
   ```
   - Install shield models: `ollama pull shieldgemma:2b`
   - Verify safety provider is configured
   - Check shield list: `curl http://localhost:8321/v1/shields`

4. **Import Errors**
   ```
   ImportError: llama-stack-client is required
   ```
   - Install dependencies: `pip install llama-stack-client`
   - Or install with all dependencies: `pip install -e ".[all]"`

5. **Embedding Errors**
   ```
   No embedding models available
   ```
   - Install embedding models: `ollama pull nomic-embed-text`
   - Verify embedding provider configuration
   - Check model availability

### Debug Commands

```bash
# Check LlamaStack server
curl http://localhost:8321/v1/models

# Check Ollama server
curl http://localhost:11434/api/tags

# List Ollama models
ollama list

# Test connection programmatically
python -c "from langchain_llamastack import check_llamastack_connection; print(check_llamastack_connection())"
```

## 📚 Requirements

- **Python**: 3.8+
- **Core Dependencies**:
  - `langchain-core>=0.1.0`
  - `httpx>=0.25.0`
  - `pydantic>=1.10.0`
- **Optional Dependencies**:
  - `llama-stack-client>=0.0.40` (for Llama Stack integration)
  - `numpy` and `scikit-learn` (for similarity search)
- **External Requirements**:
  - Running Llama Stack server
  - At least one configured provider (Ollama, OpenAI, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Run code formatting: `black . && isort .`
6. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - The main LangChain library
- [Llama Stack](https://github.com/meta-llama/llama-stack) - The Llama Stack platform
- [Llama Stack Client](https://github.com/meta-llama/llama-stack-client-python) - Python client for Llama Stack

---

For more examples and detailed documentation, explore the `examples/` directory and run the getting started guide.
