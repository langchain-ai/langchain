from __future__ import annotations

import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManager,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool, Tool
from langchain.tools.e2b_data_analysis.unparse import Unparser

if TYPE_CHECKING:
    from e2b import EnvVars
    from e2b.templates.data_analysis import Artifact

base_description = """Evaluates python code in a sandbox environment. \
The environment is long running and exists across multiple executions. \
You must send the whole script every time and print your outputs. \
Script should be pure python code that can be evaluated. \
It should be in python format NOT markdown. \
The code should NOT be wrapped in backticks. \
All python packages including requests, matplotlib, scipy, numpy, pandas, \
etc are available. Create and display chart using `plt.show()`."""


def _unparse(tree: ast.AST) -> str:
    """Unparse the AST."""
    if version_info.minor < 9:
        s = StringIO()
        Unparser(tree, file=s)
        source_code = s.getvalue()
        s.close()
    else:
        source_code = ast.unparse(tree)  # type: ignore[attr-defined]
    return source_code


def add_last_line_print(code: str) -> str:
    """Add print statement to the last line if it's missing.

    Sometimes, the LLM-generated code doesn't have `print(variable_name)`, instead the
        LLM tries to print the variable only by writing `variable_name` (as you would in
        REPL, for example).

    This methods checks the AST of the generated Python code and adds the print
        statement to the last line if it's missing.
    """
    tree = ast.parse(code)
    node = tree.body[-1]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
            return _unparse(tree)

    if isinstance(node, ast.Expr):
        tree.body[-1] = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[node.value],
                keywords=[],
            )
        )

    return _unparse(tree)


class UploadedFile(BaseModel):
    """Description of the uploaded path with its remote path."""

    name: str
    remote_path: str
    description: str


class E2BDataAnalysisToolArguments(BaseModel):
    """Arguments for the E2BDataAnalysisTool."""

    python_code: str = Field(
        ...,
        example="print('Hello World')",
        description=(
            "The python script to be evaluated. "
            "The contents will be in main.py. "
            "It should not be in markdown format."
        ),
    )


class E2BDataAnalysisTool(BaseTool):
    """Tool for running python code in a sandboxed environment for data analysis."""

    name = "e2b_data_analysis"
    args_schema: Type[BaseModel] = E2BDataAnalysisToolArguments
    session: Any
    _uploaded_files: List[UploadedFile] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        api_key: Optional[str] = None,
        cwd: Optional[str] = None,
        env_vars: Optional[EnvVars] = None,
        on_stdout: Optional[Callable[[str], Any]] = None,
        on_stderr: Optional[Callable[[str], Any]] = None,
        on_artifact: Optional[Callable[[Artifact], Any]] = None,
        on_exit: Optional[Callable[[int], Any]] = None,
        **kwargs: Any,
    ):
        try:
            from e2b import DataAnalysis
        except ImportError as e:
            raise ImportError(
                "Unable to import e2b, please install with `pip install e2b`."
            ) from e

        # If no API key is provided, E2B will try to read it from the environment
        # variable E2B_API_KEY
        super().__init__(description=base_description, **kwargs)
        self.session = DataAnalysis(
            api_key=api_key,
            cwd=cwd,
            env_vars=env_vars,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_exit=on_exit,
            on_artifact=on_artifact,
        )

    def close(self) -> None:
        """Close the cloud sandbox."""
        self._uploaded_files = []
        self.session.close()

    @property
    def uploaded_files_description(self) -> str:
        if len(self._uploaded_files) == 0:
            return ""
        lines = ["The following files available in the sandbox:"]

        for f in self._uploaded_files:
            if f.description == "":
                lines.append(f"- path: `{f.remote_path}`")
            else:
                lines.append(
                    f"- path: `{f.remote_path}` \n description: `{f.description}`"
                )
        return "\n".join(lines)

    def _run(
        self,
        python_code: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        callbacks: Optional[CallbackManager] = None,
    ) -> str:
        python_code = add_last_line_print(python_code)

        if callbacks is not None:
            on_artifact = getattr(callbacks.metadata, "on_artifact", None)
        else:
            on_artifact = None

        stdout, stderr, artifacts = self.session.run_python(
            python_code, on_artifact=on_artifact
        )

        out = {
            "stdout": stdout,
            "stderr": stderr,
            "artifacts": list(map(lambda artifact: artifact.name, artifacts)),
        }
        return json.dumps(out)

    async def _arun(
        self,
        python_code: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        raise NotImplementedError("e2b_data_analysis does not support async")

    def run_command(
        self,
        cmd: str,
    ) -> dict:
        """Run shell command in the sandbox."""
        proc = self.session.process.start(cmd)
        output = proc.wait()
        return {
            "stdout": output.stdout,
            "stderr": output.stderr,
            "exit_code": output.exit_code,
        }

    def install_python_packages(self, package_names: str | List[str]) -> None:
        """Install python packages in the sandbox."""
        self.session.install_python_packages(package_names)

    def install_system_packages(self, package_names: str | List[str]) -> None:
        """Install system packages (via apt) in the sandbox."""
        self.session.install_system_packages(package_names)

    def download_file(self, remote_path: str) -> bytes:
        """Download file from the sandbox."""
        return self.session.download_file(remote_path)

    def upload_file(self, file: IO, description: str) -> UploadedFile:
        """Upload file to the sandbox.

        The file is uploaded to the '/home/user/<filename>' path."""
        remote_path = self.session.upload_file(file)

        f = UploadedFile(
            name=os.path.basename(file.name),
            remote_path=remote_path,
            description=description,
        )
        self._uploaded_files.append(f)
        self.description = self.description + "\n" + self.uploaded_files_description
        return f

    def remove_uploaded_file(self, uploaded_file: UploadedFile) -> None:
        """Remove uploaded file from the sandbox."""
        self.session.filesystem.remove(uploaded_file.remote_path)
        self._uploaded_files = [
            f
            for f in self._uploaded_files
            if f.remote_path != uploaded_file.remote_path
        ]
        self.description = self.description + "\n" + self.uploaded_files_description

    def as_tool(self) -> Tool:
        return Tool.from_function(
            func=self._run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
