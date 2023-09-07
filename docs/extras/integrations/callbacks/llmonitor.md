# LLMonitor

[LLMonitor](https://llmonitor.com) is an open-source observability platform that provides cost tracking, user tracking and powerful agent tracing.

<video controls width='100%' >
  <source src='https://llmonitor.com/videos/demo-annotated.mp4'/>
</video>

## Setup
Create an account on [llmonitor.com](https://llmonitor.com), create an `App`, and then copy the associated `tracking id`.
Once you have it, set it as an environment variable by running:
```bash
export LLMONITOR_APP_ID="..."
```

If you'd prefer not to set an environment variable, you can pass the key directly when initializing the callback handler:
```python
from langchain.callbacks import LLMonitorCallbackHandler

handler = LLMonitorCallbackHandler(app_id="...")
```

## Usage with LLM/Chat models
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import LLMonitorCallbackHandler

handler = LLMonitorCallbackHandler(app_id="...")

llm = OpenAI(
    callbacks=[handler],
)

chat = ChatOpenAI(
    callbacks=[handler],
    metadata={"userId": "123"},  # you can assign user ids to models in the metadata
)
```


## Usage with agents
```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.callbacks import LLMonitorCallbackHandler

handler = LLMonitorCallbackHandler(app_id="...")

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
agent.run(
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
    callbacks=[handler],
    metadata={
        "agentName": "Leo DiCaprio's girlfriend",  # you can assign a custom agent in the metadata
    },
)
```

## Support
For any question or issue with integration you can reach out to the LLMonitor team on [Discord](http://discord.com/invite/8PafSG58kK) or via [email](mailto:vince@llmonitor.com).
