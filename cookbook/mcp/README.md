# langchain-mcp-adapters-example

MCP + GigaChat usage example

# Components
	1.	math_server.py – MCP server with two mathematical functions.
	2.	agent.py – MCP client for local interactions (automatically starts the MCP server).
	3.	agent_http.py – MCP client for HTTP interactions (requires the server to be started in SSE mode).

# Stdio mode (local)
## Configuration
1. Setup GigaChat credentials in .env 
2. Install requirements
```
pip install langchain-gigachat langchain_mcp_adapters langgraph rich
```

## Run agent and server (do not start server directly!)
```
python agent.py
```

Example output:
```
[HumanMessage] Сколько будет (3 + 5) x 12? 
[AIMessage]  [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': '99f7f6c7-baac-4e61-9577-03903e83f3a7', 'type': 'tool_call'}]
[ToolMessage] 8.0 
[AIMessage]  [{'name': 'multiply', 'args': {'a': 8, 'b': 12}, 'id': 'c923315e-0888-47c3-a380-2f91d95c3177', 'type': 'tool_call'}]
[ToolMessage] 96.0 
[AIMessage] Результат выражения $(3+5)\times12$ равен $96$. []
[HumanMessage] Найди сколько лет Джону Доу? 
[AIMessage]  [{'name': 'find_preson', 'args': {'name': {'query': 'Джон Доу'}}, 'id': 'fa2ecddc-c446-477b-adc7-7d4f09281953', 'type': 'tool_call'}]
[ToolMessage] {"name": "John Doe", "age": 30} 
[AIMessage] Джону Доу 30 лет. []
```

# HTTP mode (SSE)
## Start server
```
python math_server.py sse
```

## Run agent
```
python agent_http.py
```
