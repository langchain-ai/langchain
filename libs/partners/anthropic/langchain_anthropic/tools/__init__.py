"""Claude tools for [`ChatAnthropic`][langchain_anthropic.chat_models.ChatAnthropic].

These factories contain tools that support both:

- **Server-side execution**: Anthropic executes the tool on their infrastructure
- **Client-side execution**: You provide an `execute` callback function to handle tool
    calls locally

## Server tools

- [**Code Execution**][langchain_anthropic.tools.code_execution_20250825]: Run code in a
    sandboxed environment
- [**Remote MCP Toolset**][langchain_anthropic.tools.mcp_toolset]: Connect to remote MCP
    servers
- [**Web Fetch**][langchain_anthropic.tools.web_fetch_20250910]: Fetch content from web
    pages and PDFs
- [**Web Search**][langchain_anthropic.tools.web_search_20250305]: Real-time web search
    with citations
- **Tool Search**: Dynamic tool discovery, with two implementations:
    - [BM25-based search][langchain_anthropic.tools.tool_search_bm25_20251119]
    - [Regex-based search][langchain_anthropic.tools.tool_search_regex_20251119]


## Client tools

- [**Bash**][langchain_anthropic.tools.bash_20250124]: Shell command execution
- **Computer Use**: Desktop interaction. Two versions available:
    - [`2025-01-24`][langchain_anthropic.tools.computer_20250124]
    - [`2025-11-24`][langchain_anthropic.tools.computer_20251124]
- **Text Editor**: File viewing and modification. Three versions available:
    - [`2025-01-24`][langchain_anthropic.tools.text_editor.text_editor_20250124]
    - [`2025-04-29`][langchain_anthropic.tools.text_editor.text_editor_20250429]
    - [`2025-07-28`][langchain_anthropic.tools.text_editor.text_editor_20250728]
- [**Memory**][langchain_anthropic.tools.memory_20250818]: Persistent storage across
    conversations

Example:
    Server-side execution (Anthropic runs the tool):

    ```python title="Web search"
    from langchain_anthropic import ChatAnthropic, tools

    model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    model_with_search = model.bind_tools([tools.web_search_20250305(max_uses=5)])

    response = model_with_search.invoke("What are today's top news stories?")
    ```

    Client-executable tools (you provide the execution logic):

    ```python title="Bash tool"
    import subprocess

    from langchain_anthropic import ChatAnthropic, tools
    from langchain.messages import HumanMessage, ToolMessage


    def execute_bash(*, command: str | None = None, restart: bool = False, **kw):
        if restart:
            return "Bash session restarted"
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr


    model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    bash_tool = tools.bash_20250124(execute=execute_bash)
    model_with_bash = model.bind_tools([bash_tool])

    query = HumanMessage(content="List files in the current directory")

    # Initial request
    response = model_with_bash.invoke([query])

    # Process tool calls in a loop until no more tool calls
    messages = [
        query,
        response,
    ]

    while response.tool_calls:
        # Execute each tool call
        for tool_call in response.tool_calls:
            # Invoke the tool with the args from the model
            result = bash_tool.invoke(tool_call["args"])

            # Add the tool result to messages
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        # Get the next response
        response = model_with_bash.invoke(messages)
        messages.append(response)

    # Final response with the answer
    print(response.content)
    ```

    Using with [`create_agent`][langchain.agents.create_agent]:

    ```python title="Automatic tool execution"
    import subprocess

    from langchain.agents import create_agent
    from langchain_anthropic import ChatAnthropic, tools


    def execute_bash(*, command: str | None = None, restart: bool = False, **kw):
        if restart:
            return "Bash session restarted"
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr


    agent = create_agent(
        model=ChatAnthropic(model="claude-sonnet-4-5-20250929"),
        tools=[tools.bash_20250124(execute=execute_bash)],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "List files"}]})

    for message in result["messages"]:
        message.pretty_print()
    ```
"""

from langchain_anthropic.tools.bash import bash_20250124
from langchain_anthropic.tools.code_execution import code_execution_20250825
from langchain_anthropic.tools.computer import computer_20250124, computer_20251124
from langchain_anthropic.tools.mcp import (
    MCPDefaultConfig,
    MCPToolConfig,
    mcp_toolset,
)
from langchain_anthropic.tools.memory import memory_20250818
from langchain_anthropic.tools.text_editor import (
    text_editor_20250124,
    text_editor_20250429,
    text_editor_20250728,
)
from langchain_anthropic.tools.tool_search import (
    tool_search_bm25_20251119,
    tool_search_regex_20251119,
)
from langchain_anthropic.tools.web_fetch import CitationsConfig, web_fetch_20250910
from langchain_anthropic.tools.web_search import web_search_20250305

# Custom sort for reference docs ordering
__all__ = [  # noqa: RUF022
    "bash_20250124",
    "code_execution_20250825",
    "computer_20251124",
    "computer_20250124",
    "mcp_toolset",
    "MCPDefaultConfig",
    "MCPToolConfig",
    "text_editor_20250728",
    "text_editor_20250429",
    "text_editor_20250124",
    "web_fetch_20250910",
    "CitationsConfig",
    "web_search_20250305",
    "memory_20250818",
    "tool_search_bm25_20251119",
    "tool_search_regex_20251119",
]
