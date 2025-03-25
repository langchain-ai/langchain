# Streamlit

> **[Streamlit](https://streamlit.io/) is a faster way to build and share data apps.**
> Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front‑end experience required.
> See more examples at [streamlit.io/generative-ai](https://streamlit.io/generative-ai).

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/streamlit-agent?quickstart=1)

In this guide we will demonstrate how to use `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an
interactive Streamlit app. Try it out with the running app below using the MRKL agent:

<iframe loading="lazy" src="https://langchain-mrkl.streamlit.app/?embed=true&embed_options=light_theme"
    style={{ width: 100 + '%', border: 'none', marginBottom: 1 + 'rem', height: 600 }}
    allow="camera;clipboard-read;clipboard-write;"
></iframe>

## Installation and Setup

```bash
pip install langchain streamlit
```

You can run `streamlit hello` to load a sample app and validate your install succeeded. See full instructions in Streamlit's
[Getting started documentation](https://docs.streamlit.io/library/get-started).

## Display thoughts and actions

To create a `StreamlitCallbackHandler`, you just need to provide a parent container to render the output.

```python
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import streamlit as st

st_callback = StreamlitCallbackHandler(st.container())
```

Additional keyword arguments to customize the display behavior are described in the
[API reference](https://python.langchain.com/api_reference/langchain/callbacks/langchain.callbacks.streamlit.streamlit_callback_handler.StreamlitCallbackHandler.html).

### Scenario 1: Using an Agent with Tools

The primary supported use case today is visualizing the actions of an Agent with Tools (or Agent Executor). You can create an
agent in your Streamlit app and simply pass the `StreamlitCallbackHandler` to `agent.run()` in order to visualize the
thoughts and actions live in your app.

```python
import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI

llm = OpenAI(temperature=0, streaming=True)
tools = load_tools(["ddg-search"])
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
```

**Note:** You will need to set `OPENAI_API_KEY` for the above app code to run successfully.
The easiest way to do this is via [Streamlit secrets.toml](https://docs.streamlit.io/library/advanced-features/secrets-management),
or any other local ENV management tool.

### Additional scenarios

Currently `StreamlitCallbackHandler` is geared towards use with a LangChain Agent Executor. Support for additional agent types,
use directly with Chains, etc will be added in the future.

You may also be interested in using
[StreamlitChatMessageHistory](/docs/integrations/memory/streamlit_chat_message_history) for LangChain.
