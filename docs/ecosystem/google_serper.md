# Google Serper Wrapper

This page covers how to use the [Serper](https://serper.dev) Google Search API within LangChain. Serper is a low-cost Google Search API that can be used to add answer box, knowledge graph, and organic results data from Google Search. 
It is broken into two parts: setup, and then references to the specific Google Serper wrapper.

## Setup
- Go to [serper.dev](https://serper.dev) to sign up for a free account
- Get the api key and set it as an environment variable (`SERPER_API_KEY`)

## Wrappers

### Utility

There exists a GoogleSerperAPIWrapper utility which wraps this API. To import this utility:

```python
from langchain.utilities import GoogleSerperAPIWrapper
```

You can use it as part of a Self Ask chain:

```python
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

import os

os.environ["SERPER_API_KEY"] = ""
os.environ['OPENAI_API_KEY'] = ""

llm = OpenAI(temperature=0)
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search"
    )
]

self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
self_ask_with_search.run("What is the hometown of the reigning men's U.S. Open champion?")
```

#### Output
```
Entering new AgentExecutor chain...
 Yes.
Follow up: Who is the reigning men's U.S. Open champion?
Intermediate answer: Current champions Carlos Alcaraz, 2022 men's singles champion.
Follow up: Where is Carlos Alcaraz from?
Intermediate answer: El Palmar, Spain
So the final answer is: El Palmar, Spain

> Finished chain.

'El Palmar, Spain'
```

For a more detailed walkthrough of this wrapper, see [this notebook](../modules/agents/tools/examples/google_serper.ipynb).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
```python
from langchain.agents import load_tools
tools = load_tools(["google-serper"])
```

For more information on this, see [this page](../modules/agents/tools/getting_started.md)
