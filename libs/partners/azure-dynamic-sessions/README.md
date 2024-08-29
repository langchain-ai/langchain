# langchain-azure-dynamic-sessions

This package contains the LangChain integration for Azure Container Apps dynamic sessions. You can use it to add a secure and scalable code interpreter to your agents.

## Installation

```bash
pip install -U langchain-azure-dynamic-sessions
```

## Usage

You first need to create an Azure Container Apps session pool and obtain its management endpoint. Then you can use the `SessionsPythonREPLTool` tool to give your agent the ability to execute Python code.

```python
from langchain_azure_dynamic_sessions import SessionsPythonREPLTool


# get the management endpoint from the session pool in the Azure portal
tool = SessionsPythonREPLTool(pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT)

prompt = hub.pull("hwchase17/react")
tools=[tool]
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

react_agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

react_agent_executor.invoke({"input": "What is the current time in Vancouver, Canada?"})
```

By default, the tool uses `DefaultAzureCredential` to authenticate with Azure. If you're using a user-assigned managed identity, you must set the `AZURE_CLIENT_ID` environment variable to the ID of the managed identity.

