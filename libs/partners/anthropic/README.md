# langchain-anthropic

This package contains the LangChain integration for Anthropic's generative models.

## Installation

`pip install -U langchain-anthropic`

## Chat Models

Anthropic recommends using their chat models over text completions.

You can see their recommended models [here](https://docs.anthropic.com/claude/docs/models-overview#model-recommendations).

To use, you should have an Anthropic API key configured. Initialize the model as:

```
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0, max_tokens=1024)
```

### Define the input message

`message = HumanMessage(content="What is the capital of France?")`

### Generate a response using the model

`response = model.invoke([message])`

For a more detailed walkthrough see [here](https://python.langchain.com/docs/integrations/chat/anthropic).

## LLMs (Legacy)

You can use the Claude 2 models for text completions.

```python
from langchain_anthropic import AnthropicLLM

model = AnthropicLLM(model="claude-2.1", temperature=0, max_tokens=1024)
response = model.invoke("The best restaurant in San Francisco is: ")
```