"""Claude tools for `ChatAnthropic`.

Factory functions for creating typed tool definitions. These tools support both:

- **Server-side execution**: Anthropic executes the tool on their infrastructure
- **Client-side execution**: You provide an `execute` callback to handle tool calls
    locally

Tools include:

- **Bash** (`bash_20250124`): Shell command execution
- **Code Execution** (`code_execution_20250825`): Run code in a sandboxed environment
- **Computer Use** (`computer_20251124`, `computer_20250124`): Desktop interaction
- **Remote MCP Toolset** (`mcp_toolset`): Connect to remote MCP servers
- **Text Editor** (`text_editor_20250728`, etc.): File viewing and modification
- **Web Fetch** (`web_fetch_20250910`): Fetch content from web pages and PDFs
- **Web Search** (`web_search_20250305`): Real-time web search with citations
- **Memory** (`memory_20250818`): Persistent storage across conversations
- **Tool Search** (`tool_search_regex_20251119`, `tool_search_bm25_20251119`):
    Dynamic tool discovery

Example:
    Server-side execution (Anthropic runs the tool):

    ```python title="Web search example"
    from langchain_anthropic import ChatAnthropic, tools

    model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    model_with_search = model.bind_tools([tools.web_search_20250305(max_uses=5)])

    response = model_with_search.invoke("What are today's top news stories?")
    ```

    Client-side execution (you run the tool):

    ```python title="Bash tool example"
    import subprocess

    from langchain_anthropic import ChatAnthropic, tools
    from langchain_core.messages import HumanMessage, ToolMessage


    def execute_bash(args):
        if args.get("restart"):
            return "Bash session restarted"
        result = subprocess.run(
            args["command"],
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.stdout + result.stderr


    model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    bash_tool = tools.bash_20250124(execute=execute_bash)
    model_with_bash = model.bind_tools([bash_tool])

    # Initial request
    response = model_with_bash.invoke("List Python files in the current directory")

    # Process tool calls in a loop until no more tool calls
    messages = [
        HumanMessage(content="List Python files in the current directory"),
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

    Binding multiple tools:

    ```python
    from langchain_anthropic import ChatAnthropic, tools

    model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    model_with_tools = model.bind_tools(
        [
            tools.web_search_20250305(max_uses=3),
            tools.web_fetch_20250910(citations={"enabled": True}),
            tools.code_execution_20250825(),
        ]
    )

    response = model_with_tools.invoke(
        "Search for Python tutorials, fetch the best one, and summarize it"
    )
    ```
"""

from langchain_anthropic.tools.bash import bash_20250124
from langchain_anthropic.tools.code_execution import code_execution_20250825
from langchain_anthropic.tools.computer import computer_20250124, computer_20251124
from langchain_anthropic.tools.mcp_toolset import (
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
from langchain_anthropic.tools.types import (
    BashCommand,
    BashExecuteCommand,
    BashRestartCommand,
    ComputerAction20250124,
    ComputerAction20251124,
    ComputerDoubleClickAction,
    ComputerHoldKeyAction,
    ComputerKeyAction,
    ComputerLeftClickAction,
    ComputerLeftClickDragAction,
    ComputerLeftMouseDownAction,
    ComputerLeftMouseUpAction,
    ComputerMiddleClickAction,
    ComputerMouseMoveAction,
    ComputerRightClickAction,
    ComputerScreenshotAction,
    ComputerScrollAction,
    ComputerTripleClickAction,
    ComputerTypeAction,
    ComputerWaitAction,
    ComputerZoomAction,
    MemoryCommand,
    MemoryCreateCommand,
    MemoryDeleteCommand,
    MemoryInsertCommand,
    MemoryRenameCommand,
    MemoryStrReplaceCommand,
    MemoryViewCommand,
    TextEditorCommand,
    TextEditorCreateCommand,
    TextEditorInsertCommand,
    TextEditorStrReplaceCommand,
    TextEditorViewCommand,
)
from langchain_anthropic.tools.web_fetch import CitationsConfig, web_fetch_20250910
from langchain_anthropic.tools.web_search import web_search_20250305

__all__ = [
    "BashCommand",
    "BashExecuteCommand",
    "BashRestartCommand",
    "CitationsConfig",
    "ComputerAction20250124",
    "ComputerAction20251124",
    "ComputerDoubleClickAction",
    "ComputerHoldKeyAction",
    "ComputerKeyAction",
    "ComputerLeftClickAction",
    "ComputerLeftClickDragAction",
    "ComputerLeftMouseDownAction",
    "ComputerLeftMouseUpAction",
    "ComputerMiddleClickAction",
    "ComputerMouseMoveAction",
    "ComputerRightClickAction",
    "ComputerScreenshotAction",
    "ComputerScrollAction",
    "ComputerTripleClickAction",
    "ComputerTypeAction",
    "ComputerWaitAction",
    "ComputerZoomAction",
    "MCPDefaultConfig",
    "MCPToolConfig",
    "MemoryCommand",
    "MemoryCreateCommand",
    "MemoryDeleteCommand",
    "MemoryInsertCommand",
    "MemoryRenameCommand",
    "MemoryStrReplaceCommand",
    "MemoryViewCommand",
    "TextEditorCommand",
    "TextEditorCreateCommand",
    "TextEditorInsertCommand",
    "TextEditorStrReplaceCommand",
    "TextEditorViewCommand",
    "bash_20250124",
    "code_execution_20250825",
    "computer_20250124",
    "computer_20251124",
    "mcp_toolset",
    "memory_20250818",
    "text_editor_20250124",
    "text_editor_20250429",
    "text_editor_20250728",
    "tool_search_bm25_20251119",
    "tool_search_regex_20251119",
    "web_fetch_20250910",
    "web_search_20250305",
]
