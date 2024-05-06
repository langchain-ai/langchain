import os
import re
import subprocess  # nosec
import tempfile

from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import Tool
from langchain.pydantic_v1 import BaseModel, Field, ValidationError, validator
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField, Runnable


def strip_python_markdown_tags(text: str) -> str:
    pat = re.compile(r"```python\n(.*)```", re.DOTALL)
    code = pat.match(text)
    if code:
        return code.group(1)
    else:
        return text


def format_black(filepath: str):
    """Format a file with black."""
    subprocess.run(  # nosec
        f"black {filepath}",
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        timeout=3,
        check=False,
    )


def format_ruff(filepath: str):
    """Run ruff format on a file."""
    subprocess.run(  # nosec
        f"ruff check --fix {filepath}",
        shell=True,
        text=True,
        timeout=3,
        universal_newlines=True,
        check=False,
    )

    subprocess.run(  # nosec
        f"ruff format {filepath}",
        stderr=subprocess.STDOUT,
        shell=True,
        timeout=3,
        text=True,
        check=False,
    )


def check_ruff(filepath: str):
    """Run ruff check on a file."""
    subprocess.check_output(  # nosec
        f"ruff check {filepath}",
        stderr=subprocess.STDOUT,
        shell=True,
        timeout=3,
        text=True,
    )


def check_mypy(filepath: str, strict: bool = True, follow_imports: str = "skip"):
    """Run mypy on a file."""
    cmd = (
        f"mypy {'--strict' if strict else ''} "
        f"--follow-imports={follow_imports} {filepath}"
    )

    subprocess.check_output(  # nosec
        cmd,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        timeout=3,
    )


class PythonCode(BaseModel):
    code: str = Field(
        description="Python code conforming to "
        "ruff, black, and *strict* mypy standards.",
    )

    @validator("code")
    @classmethod
    def check_code(cls, v: str) -> str:
        v = strip_python_markdown_tags(v).strip()
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_file.write(v)
                temp_file_path = temp_file.name

            try:
                # format with black and ruff
                format_black(temp_file_path)
                format_ruff(temp_file_path)
            except subprocess.CalledProcessError:
                pass

            # update `v` with formatted code
            with open(temp_file_path, "r") as temp_file:
                v = temp_file.read()

            # check
            complaints = dict(ruff=None, mypy=None)

            try:
                check_ruff(temp_file_path)
            except subprocess.CalledProcessError as e:
                complaints["ruff"] = e.output

            try:
                check_mypy(temp_file_path)
            except subprocess.CalledProcessError as e:
                complaints["mypy"] = e.output

            # raise ValueError if ruff or mypy had complaints
            if any(complaints.values()):
                code_str = f"```{temp_file_path}\n{v}```"
                error_messages = [
                    f"```{key}\n{value}```"
                    for key, value in complaints.items()
                    if value
                ]
                raise ValueError("\n\n".join([code_str] + error_messages))

        finally:
            os.remove(temp_file_path)
        return v


def check_code(code: str) -> str:
    try:
        code_obj = PythonCode(code=code)
        return (
            f"# LGTM\n"
            f"# use the `submit` tool to submit this code:\n\n"
            f"```python\n{code_obj.code}\n```"
        )
    except ValidationError as e:
        return e.errors()[0]["msg"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a world class Python coder who uses "
            "black, ruff, and *strict* mypy for all of your code. "
            "Provide complete, end-to-end Python code "
            "to meet the user's description/requirements. "
            "Always `check` your code. When you're done, "
            "you must ALWAYS use the `submit` tool.",
        ),
        (
            "human",
            ": {input}",
        ),
    ],
)

check_code_tool = Tool.from_function(
    check_code,
    name="check-code",
    description="Always check your code before submitting it!",
)

submit_code_tool = Tool.from_function(
    strip_python_markdown_tags,
    name="submit-code",
    description="THIS TOOL is the most important. "
    "use it to submit your code to the user who requested it... "
    "but be sure to `check` it first!",
    return_direct=True,
)

tools = [check_code_tool, submit_code_tool]


def get_agent_executor(
    llm: BaseLLM,
    agent_type: AgentType = AgentType.OPENAI_FUNCTIONS,
) -> Runnable:
    _agent_executor = initialize_agent(
        tools,
        llm,
        agent=agent_type,
        verbose=True,
        handle_parsing_errors=True,
        prompt=prompt,
    )
    return _agent_executor | (lambda output: output["output"])


class Instruction(BaseModel):
    __root__: str


agent_executor = (
    get_agent_executor(ChatOpenAI(model="gpt-4-1106-preview", temperature=0.0))
    .configurable_alternatives(
        ConfigurableField("model_name"),
        default_key="gpt4turbo",
        gpt4=get_agent_executor(ChatOpenAI(model="gpt-4", temperature=0.0)),
        gpt35t=get_agent_executor(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0),
        ),
    )
    .with_types(input_type=Instruction, output_type=str)
)
