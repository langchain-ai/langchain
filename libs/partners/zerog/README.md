# langchain-zerog

This package contains the LangChain integration for [0G Compute Network](https://0g.ai/), a decentralized AI inference platform that provides verified computations running in Trusted Execution Environments (TEE).

## Installation

```bash
pip install langchain-zerog
```

## Quick Start

To use the 0G Compute Network with LangChain, you need:

1. An Ethereum private key for wallet authentication
2. OG tokens to fund your account for inference payments

### Environment Setup

```bash
export ZEROG_PRIVATE_KEY="your-ethereum-private-key"
export ZEROG_RPC_URL="https://evmrpc-testnet.0g.ai"  # Optional, defaults to testnet
```

### Basic Usage

```python
import asyncio
from langchain_zerog import ChatZeroG

async def main():
    # Initialize the model
    llm = ChatZeroG(
        model="llama-3.3-70b-instruct",  # or "deepseek-r1-70b"
        temperature=0.7,
        max_tokens=1000,
    )

    # Fund your account (required before first use)
    await llm.fund_account("0.1")  # Add 0.1 OG tokens

    # Check balance
    balance = await llm.get_balance()
    print(f"Available balance: {balance['available']} OG tokens")

    # Chat with the model
    messages = [
        ("system", "You are a helpful AI assistant."),
        ("human", "What is the capital of France?"),
    ]

    response = await llm.ainvoke(messages)
    print(response.content)

asyncio.run(main())
```

## Features

### ✅ Supported Features

* **Chat Models**: Full LangChain chat model interface
* **Streaming**: Real-time token-level streaming responses
* **Tool Calling**: Function calling with JSON schema validation
* **Structured Output**: Pydantic model-based response formatting
* **Async Support**: Native async/await support for all operations
* **Account Management**: Balance checking, funding, and refunds
* **TEE Verification**: Verified computations in Trusted Execution Environments
* **Multiple Models**: Support for Llama and DeepSeek models

### 🚧 Planned Features

* **Embeddings**: Text embedding models
* **Fine-tuning**: Custom model training
* **Multimodal**: Image and audio input support
* **JSON Mode**: Structured JSON responses

## Supported Models

| Model | Provider Address | Description | Verification |
|-------|------------------|-------------|--------------|
| `llama-3.3-70b-instruct` | `0xf07240Efa67755B5311bc75784a061eDB47165Dd` | 70B parameter model for general AI tasks | TEE (TeeML) |
| `deepseek-r1-70b` | `0x3feE5a4dd5FDb8a32dDA97Bed899830605dBD9D3` | Advanced reasoning model | TEE (TeeML) |

## Advanced Usage

### Streaming Responses

```python
async for chunk in llm.astream(messages):
    print(chunk.content, end="", flush=True)
```

### Tool Calling

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    """Get weather information."""
    location: str = Field(description="City and state")
    unit: str = Field(default="celsius", description="Temperature unit")

llm_with_tools = llm.bind_tools([GetWeather])
response = await llm_with_tools.ainvoke([
    ("human", "What's the weather in San Francisco?")
])

if response.tool_calls:
    print(f"Tool: {response.tool_calls[0]['name']}")
    print(f"Args: {response.tool_calls[0]['args']}")
```

### Structured Output

```python
class Joke(BaseModel):
    """A joke structure."""
    setup: str = Field(description="The setup")
    punchline: str = Field(description="The punchline")
    rating: int = Field(description="Funny rating 1-10")

structured_llm = llm.with_structured_output(Joke)
joke = await structured_llm.ainvoke("Tell me a joke about AI")

print(f"Setup: {joke.setup}")
print(f"Punchline: {joke.punchline}")
print(f"Rating: {joke.rating}/10")
```

### Account Management

```python
# Check balance
balance = await llm.get_balance()
print(f"Balance: {balance['balance']} OG")
print(f"Available: {balance['available']} OG")
print(f"Locked: {balance['locked']} OG")

# Add funds
await llm.fund_account("0.5")

# Request refund
await llm.request_refund("0.1")
```

## Configuration

### Model Parameters

* `model`: Model name (`"llama-3.3-70b-instruct"` or `"deepseek-r1-70b"`)
* `provider_address`: Specific provider address (optional, uses official providers by default)
* `temperature`: Sampling temperature (0.0 to 2.0, default: 0.7)
* `max_tokens`: Maximum tokens to generate (default: None)
* `top_p`: Nucleus sampling parameter (default: 1.0)
* `frequency_penalty`: Frequency penalty (default: 0.0)
* `presence_penalty`: Presence penalty (default: 0.0)

### Network Configuration

* `private_key`: Ethereum private key (from `ZEROG_PRIVATE_KEY` env var)
* `rpc_url`: 0G Network RPC URL (default: testnet)
* `broker_url`: 0G broker service URL (default: official broker)

## Examples

See the `examples/` directory for comprehensive usage examples:

* `comprehensive_example.py`: Full feature demonstration
* `basic_usage.py`: Simple chat interaction
* `model_comparison.py`: Compare different models

## Error Handling

```python
try:
    response = await llm.ainvoke(messages)
except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Network error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[test]"

# Run unit tests
pytest tests/unit_tests/

# Run integration tests (requires ZEROG_PRIVATE_KEY)
pytest tests/integration_tests/
```

### Linting and Formatting

```bash
# Install lint dependencies
pip install -e ".[lint]"

# Run linting
ruff check .
ruff format .

# Type checking
mypy .
```

## Contributing

Contributions are welcome! Please see the [LangChain contributing guide](https://github.com/langchain-ai/langchain/blob/master/CONTRIBUTING.md) for development setup and guidelines.

## License

This package is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Links

* **0G Compute Network**: https://0g.ai/
* **Documentation**: https://docs.0g.ai/
* **LangChain**: https://python.langchain.com/
* **GitHub**: https://github.com/langchain-ai/langchain/tree/master/libs/partners/zerog
* **PyPI**: https://pypi.org/project/langchain-zerog/

## Support

* **Discord**: [0G Community](https://discord.gg/0g)
* **Issues**: [GitHub Issues](https://github.com/langchain-ai/langchain/issues)
* **Discussions**: [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)
