# langchain-sarvam

This package contains the LangChain integration with [Sarvam AI](https://www.sarvam.ai/).

Sarvam AI provides LLMs optimized for Indian languages understanding and efficiency, offering strong multilingual capabilities especially for Indic and low-resource languages.

## Installation

```bash
pip install -U langchain-sarvam
```

## Setup

To use the Sarvam AI models, you'll need to obtain an API key from Sarvam AI and set it as an environment variable:

```bash
export SARVAM_API_KEY="your-api-key-here"
```

## Chat Models

### Basic Usage

```python
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the model
chat = ChatSarvam(
    model="sarvam-1",
    temperature=0.7,
    max_tokens=1024,
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is the capital of India?"),
]

# Get response
response = chat.invoke(messages)
print(response.content)
```

### Streaming

```python
from langchain_sarvam import ChatSarvam

chat = ChatSarvam(model="sarvam-1")

for chunk in chat.stream("Tell me a short story about India"):
    print(chunk.content, end="", flush=True)
```

### Async Usage

```python
import asyncio
from langchain_sarvam import ChatSarvam
from langchain_core.messages import HumanMessage

async def main():
    chat = ChatSarvam(model="sarvam-1")
    response = await chat.ainvoke([HumanMessage(content="Hello!")])
    print(response.content)

asyncio.run(main())
```

### Configuration Options

The `ChatSarvam` class supports various configuration parameters:

- `model`: Model identifier (default: "sarvam-1")
- `temperature`: Controls randomness (0.0 to 2.0, default: 0.7)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Nucleus sampling parameter
- `frequency_penalty`: Penalize token frequency (-2.0 to 2.0)
- `presence_penalty`: Penalize token presence (-2.0 to 2.0)
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum retry attempts (default: 2)

```python
chat = ChatSarvam(
    model="sarvam-1",
    temperature=0.9,
    max_tokens=2048,
    top_p=0.95,
    timeout=30.0,
)
```

## Use Cases

Sarvam AI models are particularly well-suited for:

- **Multilingual Applications**: Strong support for Indian languages
- **RAG Pipelines**: Efficient retrieval-augmented generation
- **Conversational AI**: Building chatbots and assistants for Indian markets
- **Low-Resource Languages**: Working with languages that have limited training data

## Integration with LangChain Components

### Using with Chains

```python
from langchain_sarvam import ChatSarvam
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatSarvam(model="sarvam-1")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | chat | StrOutputParser()
result = chain.invoke({"input": "What is AI?"})
print(result)
```

### Using with Agents

```python
from langchain_sarvam import ChatSarvam
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool

chat = ChatSarvam(model="sarvam-1", temperature=0)

# Define your tools and create an agent
# ... (agent setup code)
```

## API Reference

For detailed API documentation, please refer to the [ChatSarvam API reference](https://python.langchain.com/api_reference/sarvam/chat_models.html).

## Support

For issues, questions, or contributions related to this integration:
- **LangChain Issues**: [GitHub Issues](https://github.com/langchain-ai/langchain/issues)
- **Sarvam AI Support**: Visit [Sarvam AI website](https://www.sarvam.ai/)

## License

This integration is released under the MIT License.
