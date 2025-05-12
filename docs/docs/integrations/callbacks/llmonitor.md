# LLMonitor

>[LLMonitor](https://llmonitor.com?utm_source=langchain&utm_medium=py&utm_campaign=docs) is an open-source observability platform that provides cost and usage analytics, user tracking, tracing and evaluation tools.

<video controls width='100%' >
  <source src='https://llmonitor.com/videos/demo-annotated.mp4'/>
</video>

## Setup

Create an account on [llmonitor.com](https://llmonitor.com?utm_source=langchain&utm_medium=py&utm_campaign=docs), then copy your new app's `tracking id`.

Once you have it, set it as an environment variable by running:

```bash
export LLMONITOR_APP_ID="..."
```

If you'd prefer not to set an environment variable, you can pass the key directly when initializing the callback handler:

```python
from langchain_community.callbacks.llmonitor_callback import LLMonitorCallbackHandler

handler = LLMonitorCallbackHandler(app_id="...")
```

## Usage with LLM/Chat models

```python
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI

handler = LLMonitorCallbackHandler()

llm = OpenAI(
    callbacks=[handler],
)

chat = ChatOpenAI(callbacks=[handler])

llm("Tell me a joke")

```

## Usage with chains and agents

Make sure to pass the callback handler to the `run` method so that all related chains and llm calls are correctly tracked.

It is also recommended to pass `agent_name` in the metadata to be able to distinguish between agents in the dashboard.

Example:

```python
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.llmonitor_callback import LLMonitorCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor, tool

llm = ChatOpenAI(temperature=0)

handler = LLMonitorCallbackHandler()

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=SystemMessage(
        content="You are very powerful assistant, but bad at calculating lengths of words."
    )
)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt, verbose=True)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, metadata={"agent_name": "WordCount"}  # <- recommended, assign a custom name
)
agent_executor.run("how many letters in the word educa?", callbacks=[handler])
```

Another example:

```python
import os

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.callbacks.llmonitor_callback import LLMonitorCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

os.environ["LLMONITOR_APP_ID"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

handler = LLMonitorCallbackHandler()
llm = ChatOpenAI(temperature=0, callbacks=[handler])
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = create_react_agent("openai:gpt-4.1-mini", tools)

input_message = {
    "role": "user",
    "content": "What's the weather in SF?",
}

agent.invoke({"messages": [input_message]})
```

## User Tracking
User tracking allows you to identify your users, track their cost, conversations and more.

```python
from langchain_community.callbacks.llmonitor_callback import LLMonitorCallbackHandler, identify

with identify("user-123"):
    llm.invoke("Tell me a joke")

with identify("user-456", user_props={"email": "user456@test.com"}):
    agent.invoke(...)
```
## Support

For any question or issue with integration you can reach out to the LLMonitor team on [Discord](http://discord.com/invite/8PafSG58kK) or via [email](mailto:vince@llmonitor.com).
