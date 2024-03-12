# langchain-unify

This package contains the LangChain integrations for [Unify](https://unify.ai/).

## Installation

```bash
pip install -U langchain-unify
```

## Chat Models

This package contains the `ChatUnify` class, which is the recommended way to interface with endpoints supported by Unify.

To use, install the requirements, and configure your environment.

```bash
export UNIFY_API_KEY=<YOUR API KEY>
```

Then initialize

```python
from langchain_core.messages import HumanMessage
from langchain_unify.chat_models import ChatUnify

chat = ChatUnify(model="llama-2-70b-chat@lowest-input-cost")
messages = [HumanMessage(content="Hello unify!")]
chat.invoke(messages)
```

`ChatUnify` also supports async and streaming functionality:

```python
# For async...
await chat.ainvoke(messages)

# For streaming...
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```