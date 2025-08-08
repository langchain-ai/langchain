from typing import Optional, Sequence, Literal, List

from langgraph.prebuilt import create_react_agent
from langgraph.pregel import Pregel

from langchain.sandboxes import create_tools, DaytonaSandboxToolkit, FileUpload
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.tools import BaseTool


def create_data_analysis_agent(
    model: BaseLanguageModel,
    sandbox: DaytonaSandboxToolkit,
    tools: Sequence[
        Literal["run_code", "list_files", "upload_file", "download_file", "exec"]
    ],
    *,
    prompt: Optional[str] = None,
    files: Optional[list[FileUpload]] = None,
    name: str = "Data Analysis Agent",
) -> Pregel:
    """Create a data analysis Pregel agent using Daytona sandbox."""
    if files:
        sandbox.upload_files(files)

    tools_: List[BaseTool] = create_tools(sandbox, tools)

    if prompt is None:
        files_section = ""
        if files:
            files_list = "\n".join([f"- {file['destination']}" for file in files])
            files_section = f"\n\nThe user has uploaded the following files to ./data/:\n{files_list}"

        prompt = f"""
You are a data analysis agent designed to write and execute Python code to answer questions.

You have access to:
- A Python environment where you can run complete scripts
- Root access to install any packages you need using pip
- A filesystem with data files located in {files_section}

Guidelines:
- Write complete, self-contained Python scripts for each execution
- Use proper imports and handle errors gracefully
- Install required packages if needed (e.g., pandas, numpy, matplotlib)
- Always run code to verify your analysis, even if you think you know the answer
- If you encounter errors, debug and fix your code before trying again
- Only use the output of your executed code to answer questions
- If the question cannot be answered with code or with the data, respond with "I don't know"

Answer the user's question using the tools available to you.
"""
    return create_react_agent(model=model, tools=tools_, prompt=prompt, name=name)
