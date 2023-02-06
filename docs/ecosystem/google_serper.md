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
question = "What is the hometown of the reigning men's U.S. Open champion?"
    chain = SelfAskWithSearchChain(
        llm=OpenAI(temperature=0),
        search_chain=GoogleSerperAPIWrapper(),
        input_key="q",
        output_key="a",
    )
    answer = chain.run(question)
    final_answer = answer.split("\n")[-1]
    assert final_answer == "El Palmar, Spain"
```

For a more detailed walkthrough of this wrapper, see [this notebook](../modules/utils/examples/google_serper.ipynb).

### Tool

You can also easily load this wrapper as a Tool (to use with an Agent).
You can do this with:
```python
from langchain.agents import load_tools
tools = load_tools(["google-serper"])
```

For more information on this, see [this page](../modules/agents/tools.md)
