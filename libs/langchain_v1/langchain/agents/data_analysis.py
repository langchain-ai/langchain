from typing import Optional, Sequence, Literal, List

from langgraph.prebuilt import create_react_agent
from langgraph.pregel import Pregel

from langchain.sandboxes import SandboxManager
from langchain_core.tools import BaseTool


def create_data_analysis_agent(
    model: str,
    sandbox_manager: SandboxManager,
    tool_selection: Optional[
        Sequence[
            Literal["run_code", "list_files", "upload_file", "download_file", "exec"]
        ]
    ] = None,
    *,
    prompt: Optional[str] = None,
    **agent_kwargs,
) -> Pregel:
    """
    Create a data analysis Pregel agent using Daytona sandbox.

    Args:
        model: LLM model ID to use (e.g., "claude-sonnet-4").
        sandbox_manager: An initialized SandboxManager instance.
        files: Local file paths to upload into the sandbox.
        tool_selection: Optional list of tools to include.
        prompt: Optional system prompt string.
        **agent_kwargs: Additional keyword args passed to create_react_agent.

    Returns:
        Pregel: A LangGraph Pregel object, ready to `.invoke(...)`.
    """
    # This is problematic potentially... I can't pass the ID since I don't know
    # the thread ID at this point.
    # May need a life cycle event? We can use pre-model hook for now, but
    # not super efficient to call it on every invocation of the agent.
    sandbox = sandbox_manager.create()
    sandbox.upload_files(files)

    for file_path in files:
        filename = os.path.basename(file_path)
        remote_path = f"tmp/{filename}"
        with open(file_path, "rb") as f:
            data = f.read()
        toolkit.upload_file(data, remote_path)

    # 5. Create tools
    tools: List[BaseTool] = create_tools(toolkit, tool_selection)

    # 6. Default system prompt
    if prompt is None:
        prompt = (
            "You are a data analysis agent. Files are located in './tmp'. "
            "Use Python code to explore and analyze the data. "
            "Each code execution must be a complete self-contained script."
        )

    # 7. Create and return Pregel agent
    graph: Pregel = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        **agent_kwargs,
    )

    return graph
