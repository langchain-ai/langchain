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
from langchain_core.messages import AIMessage, HumanMessage

model = ChatAnthropicMessages(model="claude-2.1", temperature=0, max_tokens=1024)
```

### Define the input message

`message = HumanMessage(content="What is the capital of France?")`

### Generate a response using the model

`response = model.invoke([message])`

For a more detailed walkthrough see [here](https://python.langchain.com/docs/integrations/chat/anthropic).
