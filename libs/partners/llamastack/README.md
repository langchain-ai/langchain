# LangChain Llama Stack Integration

This package provides LangChain-compatible integrations for [Llama Stack](https://github.com/meta-llama/llama-stack), including chat completion and safety checking capabilities.

## 🚀 Installation

```bash
# Install from PyPI
pip install -U langchain-llamastack

# Or install with development dependencies
pip install -U "langchain-llamastack[dev]"

# Or install from source (for development)
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/partners/llamastack
pip install -e .
```

## 📋 Prerequisites

1. **Llama Stack Server**: You need a running Llama Stack server with inference and safety capabilities.
2. **Models**: Ensure you have appropriate models loaded in your Llama Stack setup.

### Setting up Llama Stack Server

```bash
# Example: Start Llama Stack server
llama stack run your-config.yaml --port 8321
```

## 🤖 Chat Completion Usage

The `ChatLlamaStack` class provides a LangChain-compatible interface for Llama Stack's chat completion endpoint.

### Basic Usage

```python
from langchain_llamastack import ChatLlamaStack

# Initialize the chat model
llm = ChatLlamaStack(
    model="meta-llama/Llama-3.1-8B-Instruct",  # Model identifier
    base_url="http://localhost:8321",          # Llama Stack server URL
    temperature=0.7,                           # Temperature for sampling
    max_tokens=1000,                          # Maximum tokens to generate
)

# Simple text completion
response = llm.invoke("Tell me a joke about programming")
print(response.content)
```

### Advanced Configuration

```python
from langchain_llamastack import ChatLlamaStack

# Advanced configuration
llm = ChatLlamaStack(
    model="meta-llama/Llama-3.1-70B-Instruct",
    base_url="http://localhost:8321",
    temperature=0.1,              # Lower temperature for more focused responses
    max_tokens=2000,             # Longer responses
    top_p=0.9,                   # Top-p sampling
    top_k=50,                    # Top-k sampling
    repetition_penalty=1.1,      # Repetition penalty
    llamastack_api_key="your-api-key",  # Optional API key
)

# Check available models
models = llm.get_available_models()
print("Available models:", models)

# Get model information
model_info = llm.get_model_info()
print("Model info:", model_info)
```

### Streaming Responses

```python
from langchain_llamastack import ChatLlamaStack

llm = ChatLlamaStack(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8321",
    streaming=True,
)

# Stream the response
print("AI: ", end="", flush=True)
for chunk in llm.stream("Write a short story about AI"):
    print(chunk.content, end="", flush=True)
print()
```

### Using with LangChain Messages

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_llamastack import ChatLlamaStack

llm = ChatLlamaStack(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8321",
)

# Using LangChain message types
messages = [
    SystemMessage(content="You are a helpful assistant that speaks like a pirate."),
    HumanMessage(content="Tell me about the weather today."),
]

response = llm.invoke(messages)
print(response.content)
```

### Integration with LangChain Chains

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_llamastack import ChatLlamaStack

# Create the chat model
llm = ChatLlamaStack(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8321",
)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed explanation about {topic} in simple terms."
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(topic="quantum computing")
print(result)
```

## 🛡️ Safety Checking Usage

The `LlamaStackSafety` class provides safety checking capabilities using Llama Stack's safety shields.

### Basic Safety Checking

```python
from langchain_llamastack import LlamaStackSafety

# Initialize the safety checker
safety = LlamaStackSafety(
    base_url="http://localhost:8321",
    # default_shield_id="meta-llama/Llama-Guard-3-8B",  # Optional
)

# Check content safety
result = safety.check_content("Hello, how are you today?")
print(f"Is safe: {result.is_safe}")
print(f"Message: {result.message}")

if not result.is_safe:
    print(f"Violation type: {result.violation_type}")
    print(f"Confidence: {result.confidence_score}")
```

### Conversation Safety Checking

```python
from langchain_llamastack import LlamaStackSafety

safety = LlamaStackSafety(base_url="http://localhost:8321")

# Check a conversation
messages = [
    {"role": "user", "content": "Hello there!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I'm looking for help with my research project."}
]

result = safety.check_conversation(messages)
print(f"Conversation safe: {result.is_safe}")
print(f"Result: {result.message}")
```

### Working with Safety Shields

```python
from langchain_llamastack import LlamaStackSafety

safety = LlamaStackSafety(base_url="http://localhost:8321")

# List available shields
shields = safety.get_available_shields()
print("Available shields:", shields)

# Use a specific shield
result = safety.check_content(
    "Check this content",
    shield_id="meta-llama/Llama-Guard-3-8B"
)

# Get shield information
shield_info = safety.get_shield_info("meta-llama/Llama-Guard-3-8B")
print("Shield info:", shield_info)

# Set default shield
success = safety.set_default_shield("meta-llama/Llama-Guard-3-8B")
print(f"Default shield set: {success}")
```

## 🔗 Combining Chat and Safety

You can combine both chat completion and safety checking for safe AI applications:

```python
from langchain_llamastack import ChatLlamaStack, LlamaStackSafety

# Initialize both components
llm = ChatLlamaStack(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8321",
)

safety = LlamaStackSafety(base_url="http://localhost:8321")

def safe_chat(user_input: str) -> str:
    """Safe chat function with content checking."""

    # First, check if the input is safe
    input_safety = safety.check_content(user_input)
    if not input_safety.is_safe:
        return f"⚠️ Input rejected: {input_safety.message}"

    # Generate response
    response = llm.invoke(user_input)

    # Check if the response is safe
    output_safety = safety.check_content(response.content)
    if not output_safety.is_safe:
        return f"⚠️ Response filtered: {output_safety.message}"

    return response.content

# Use the safe chat function
user_message = "Tell me about artificial intelligence"
safe_response = safe_chat(user_message)
print(safe_response)
```

## 🏗️ Advanced Integration Example

Here's a complete example that demonstrates a safe conversational AI system:

```python
from typing import List, Dict
from langchain_llamastack import ChatLlamaStack, LlamaStackSafety
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class SafeConversationalAI:
    def __init__(self, base_url: str = "http://localhost:8321"):
        self.llm = ChatLlamaStack(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url=base_url,
            temperature=0.7,
        )
        self.safety = LlamaStackSafety(base_url=base_url)
        self.conversation_history = []

    def add_system_message(self, content: str):
        """Add a system message to set the AI's behavior."""
        self.conversation_history.append(SystemMessage(content=content))

    def chat(self, user_input: str) -> Dict[str, str]:
        """Safe chat with conversation history."""

        # Check input safety
        input_safety = self.safety.check_content(user_input)
        if not input_safety.is_safe:
            return {
                "response": "I can't process that request due to safety concerns.",
                "status": "input_rejected",
                "reason": input_safety.message
            }

        # Add user message to history
        self.conversation_history.append(HumanMessage(content=user_input))

        # Generate response
        try:
            response = self.llm.invoke(self.conversation_history)

            # Check output safety
            output_safety = self.safety.check_content(response.content)
            if not output_safety.is_safe:
                return {
                    "response": "I need to filter my response for safety reasons.",
                    "status": "output_filtered",
                    "reason": output_safety.message
                }

            # Add AI response to history
            self.conversation_history.append(AIMessage(content=response.content))

            return {
                "response": response.content,
                "status": "success",
                "reason": "Response generated successfully"
            }

        except Exception as e:
            return {
                "response": "I encountered an error processing your request.",
                "status": "error",
                "reason": str(e)
            }

    def get_conversation_safety_score(self) -> Dict[str, any]:
        """Check the safety of the entire conversation."""
        # Convert conversation to safety check format
        messages = []
        for msg in self.conversation_history:
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                messages.append({"role": role, "content": msg.content})

        if messages:
            result = self.safety.check_conversation(messages)
            return {
                "is_safe": result.is_safe,
                "message": result.message,
                "confidence": result.confidence_score
            }
        return {"is_safe": True, "message": "No conversation to check"}

# Example usage
ai = SafeConversationalAI()
ai.add_system_message("You are a helpful, harmless, and honest assistant.")

# Chat with the AI
result = ai.chat("Hello! Can you help me learn about machine learning?")
print(f"AI: {result['response']}")
print(f"Status: {result['status']}")

# Check conversation safety
safety_score = ai.get_conversation_safety_score()
print(f"Conversation safety: {safety_score}")
```

## 🔧 Configuration

### Environment Variables

You can configure the integration using environment variables:

```bash
export LLAMASTACK_API_KEY="your-api-key"
export LLAMASTACK_BASE_URL="http://localhost:8321"
```

Then use them in your code:

```python
import os
from langchain_llamastack import ChatLlamaStack, LlamaStackSafety

# These will automatically use environment variables
llm = ChatLlamaStack(model="meta-llama/Llama-3.1-8B-Instruct")
safety = LlamaStackSafety()
```

### Error Handling

```python
from langchain_llamastack import ChatLlamaStack, LlamaStackSafety
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

try:
    llm = ChatLlamaStack(
        model="non-existent-model",
        base_url="http://localhost:8321",
    )
    response = llm.invoke("Hello")
except Exception as e:
    print(f"Error: {e}")

try:
    safety = LlamaStackSafety(base_url="http://invalid-url:9999")
    result = safety.check_content("Hello")
except Exception as e:
    print(f"Safety check error: {e}")
```

## 📚 API Reference

### ChatLlamaStack

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"meta-llama/Llama-3.1-8B-Instruct"` | Model identifier |
| `base_url` | str | `"http://localhost:8321"` | Llama Stack server URL |
| `temperature` | float | `0.7` | Sampling temperature |
| `max_tokens` | int | `None` | Maximum tokens to generate |
| `top_p` | float | `None` | Top-p sampling |
| `top_k` | int | `None` | Top-k sampling |
| `repetition_penalty` | float | `None` | Repetition penalty |
| `streaming` | bool | `False` | Enable streaming |
| `llamastack_api_key` | str | `None` | API key (optional) |

**Methods:**
- `invoke(messages)` - Generate completion
- `stream(messages)` - Stream completion
- `get_available_models()` - List available models
- `get_model_info(model_id)` - Get model information

### LlamaStackSafety

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | str | `"http://localhost:8321"` | Llama Stack server URL |
| `default_shield_id` | str | `None` | Default safety shield |
| `llamastack_api_key` | str | `None` | API key (optional) |

**Methods:**
- `check_content(content, shield_id, context)` - Check content safety
- `check_conversation(messages, shield_id)` - Check conversation safety
- `get_available_shields()` - List available shields
- `get_shield_info(shield_id)` - Get shield information
- `set_default_shield(shield_id)` - Set default shield

### SafetyResult

**Properties:**
- `is_safe: bool` - Whether content is safe
- `violation_type: str` - Type of violation (if any)
- `confidence_score: float` - Confidence score
- `message: str` - Human-readable message
- `shield_id: str` - Shield used for checking
- `raw_response: dict` - Raw API response

**Methods:**
- `to_dict()` - Convert to dictionary

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=langchain_llamastack

# Run specific test
pytest tests/test_chat_models.py::test_basic_completion
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📋 Requirements

- Python 3.8+
- `langchain-core>=0.1.0`
- `llama-stack-client>=0.1.0`
- A running Llama Stack server

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```
   Error: Failed to initialize Llama Stack client: Connection refused
   ```
   - Ensure Llama Stack server is running
   - Check the `base_url` parameter
   - Verify network connectivity

2. **Model Not Found**
   ```
   Error: Model 'model-name' not found
   ```
   - Check available models with `llm.get_available_models()`
   - Ensure the model is loaded in your Llama Stack configuration

3. **No Shields Available**
   ```
   SafetyResult(is_safe=False, violation_type=None)
   ```
   - Verify safety shields are configured in Llama Stack
   - Check `safety.get_available_shields()`

4. **Import Errors**
   ```
   ImportError: llama-stack-client is required
   ```
   - Install the required dependency: `pip install llama-stack-client`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all requests will show debug information
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - The main LangChain library
- [Llama Stack](https://github.com/meta-llama/llama-stack) - The Llama Stack platform
- [Llama Stack Client](https://github.com/meta-llama/llama-stack-client-python) - Python client for Llama Stack

---

For more examples and detailed documentation, check the `examples/` directory.
