# LangServe 🦜️🔗 

## Overview

`LangServe` is a library that allows developers to host their Langchain runnables / 
call into them remotely from a runnable interface.

## Examples

For more examples, see the [examples](./examples) directory.

### Server

```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes
from typing_extensions import TypedDict


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)


# Serve Open AI and Anthropic models
LLMInput = Union[List[Union[SystemMessage, HumanMessage, str]], str]

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
    input_type=LLMInput,
    config_keys=[],
)
add_routes(
    app,
    ChatAnthropic(),
    path="/anthropic",
    input_type=LLMInput,
    config_keys=[],
)

# Serve a joke chain
class ChainInput(TypedDict):
    """The input to the chain."""

    topic: str
    """The topic of the joke."""

model = ChatAnthropic()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
add_routes(app, prompt | model, path="/chain", input_type=ChainInput)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```


### Client

```python

from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

openai = RemoteRunnable("http://localhost:8000/openai/")
anthropic = RemoteRunnable("http://localhost:8000/anthropic/")
joke_chain = RemoteRunnable("http://localhost:8000/chain/")

joke_chain.invoke({"topic": "parrots"})

# or async
await joke_chain.ainvoke({"topic": "parrots"})

prompt = [
    SystemMessage(content='Act like either a cat or a parrot.'), 
    HumanMessage(content='Hello!')
]

# Supports astream
async for msg in anthropic.astream(prompt):
    print(msg, end="", flush=True)
    
prompt = ChatPromptTemplate.from_messages(
    [("system", "Tell me a long story about {topic}")]
)
    
# Can define custom chains
chain = prompt | RunnableMap({
    "openai": openai,
    "anthropic": anthropic,
})

chain.batch([{ "topic": "parrots" }, { "topic": "cats" }])
```

## Installation

```bash
# pip install langserve[all] -- has not been published to pypi yet
```

or use `client` extra for client code, and `server` extra for server code.

## Features

- Deploy runnables with FastAPI
- Client can use remote runnables almost as if they were local
    - Supports async
    - Supports batch
    - Supports stream

### Limitations

- Chain callbacks cannot be passed from the client to the server
