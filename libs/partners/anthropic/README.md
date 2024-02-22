# langchain-anthropic

This package contains the LangChain integration for Anthropic's generative models.

## Installation

`pip install -U langchain-anthropic`

## Chat Models

| API Model Name     | Model Family   |
| ------------------ | -------------- |
| claude-instant-1.2 | Claude Instant |
| claude-2.1         | Claude         |
| claude-2.0         | Claude         |

To use, you should have an Anthropic API key configured. Initialize the model as:

```
from langchain_anthropic import ChatAnthropicMessages
from langchain_core.messages  import AIMessage, HumanMessage
model = ChatAnthropicMessages(model="claude-2.1", temperature=0, max_tokens=1024)
```

# Define the input message

`input_message = HumanMessage(content="What is the capital of France?")`

# Generate a response using the model

`response = model.invoke(input_message)`

# Using Messages

You can also use a list of messages to generate a response. This is useful for multi-turn conversations.

```
model = ChatAnthropicMessages(
    model="claude-2.1",
    temperature=0,
    max_tokens=1024
    )
```

# Define the input messages

```
messages = [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "What is the capital of France?"},
]

```

### Generate responses using the model

`responses = model.generate(input_messages)`

# Embeddings

Anthropic does not offer its own embedding model. However their documentation offers solutions using Voyage AI.
[Anthropic Embeddings Documentation](https://docs.anthropic.com/claude/docs/embeddings)

## Multimodal inputs

Currently, Anthropic models do not support direct multimodal inputs through this package.
