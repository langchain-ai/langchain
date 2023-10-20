import ast
from typing import IO, Any, Callable, List, Optional, Type

from e2b import DataAnalysis, EnvVars
from e2b.templates.data_analysis import Artifact

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool

base_description = """Evaluates python code in a sandbox environment. \
The environment is long running and exists accross multiple executions. \
You must send the whole script every time and print your outputs. \
Script should be pure python code that can be evaluated. \
It should be in python format NOT markdown. \
The code should NOT be wrapped in backticks. \
All python packages including requests, matplotlib, scipy, numpy, pandas, \
etc are available. \
If you have any files outputted write them to "/home/user" directory \
path."""


def add_last_line_print(self, code: str):
    """Add print statement to the last line if it's missing.

    Sometimes, the LLM-generated code doesn't have `print(variable_name)`, instead the LLM tries to print the variable only by writing `variable_name` (as you would in REPL, for example).
    This methods checks the AST of the generated Python code and adds the print statement to the last line if it's missing.
    """
    tree = ast.parse(code)
    node = tree.body[-1]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
            return tree
    tree.body[-1] = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[node.value],
            keywords=[],
        )
    )
    return ast.unparse(tree)


class UploadedFile(BaseModel):
    """Description of the uploaded path with its remote path."""

    name: str
    description: str
    remote_path: str


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


class E2BDataAnalysisTool:
    """Tool for running python code in a sandboxed environment for data analysis over data files."""

    name = "e2b_data_analysis"
    args_schema: Type[BaseModel] = E2BDataAnalysisToolArguments
    _session: DataAnalysis
    _uploaded_files: List[str] = []

    def __init__(
        self,
        api_key: Optional[str] = None,
        cwd: Optional[str] = None,
        env_vars: Optional[EnvVars] = None,
        on_stdout: Optional[Callable[[str], Any]] = None,
        on_stderr: Optional[Callable[[str], Any]] = None,
        on_artifact: Optional[Callable[[Artifact], Any]] = None,
        on_exit: Optional[Callable[[int], Any]] = None,
    ):
        # If no API key is provided, E2B will try to read it from the environment variable E2B_API_KEY
        self._session = DataAnalysis(
            api_key=api_key,
            cwd=cwd,
            env_vars=env_vars,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
            on_exit=on_exit,
            on_artifact=on_artifact,
        )

    def close(self):
        """Close the cloud sandbox."""
        self._session.close()

    @property
    def description(self) -> str:
        return (base_description + "\n\n" + self.uploaded_files_description).strip()

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

    def _run(self, python_code: str) -> dict:
        # TODO: Alternatively, we could stream outputs and artifacts
        # self._session.run_python(
        #     code=python_code,
        #     on_stderr=print,
        #     on_stdout=print,
        #     on_artifact=lambda artifact: print(f"New chart file: {artifact.name}"),
        # )

        # Artifacts are charts created by mytplotlib when `plt.show()` is called
        stdout, stderr, artifacts = self._session.run_python(python_code)

        # Matplotlib charts created by `plt.show()`
        # We return them as `bytes` and leave it up to the user to display them (on frontend, for example)
        charts: List[dict] = []
        for artifact in artifacts:
            charts.push(
                {
                    "name": artifact.name,
                    "bytes": artifact.download(),
                }
            )

        return {
            "stdout": stdout,
            "stderr": stderr,
            "charts": charts,
        }

    async def _arun(self, python_code: str) -> str:
        raise NotImplementedError("e2b_data_analysis does not support async")

    def run_command(
        self,
        cmd: str,
    ) -> dict:
        """Run shell command in the sandbox."""
        proc = self._session.process.start(cmd)
        output = proc.wait()
        return {
            "stdout": output.stdout,
            "stderr": output.stderr,
            "exit_code": output.exit_code,
        }

    def install_python_packages(self, package_names: str | List[str]) -> None:
        """Install python packages in the sandbox."""
        self._session.install_python_packages(package_names)

    def install_system_packages(self, package_names: str | List[str]) -> None:
        """Install system packages (via apt) in the sandbox."""
        self._session.install_system_packages(package_names)

    def download_file(self, remote_path: str) -> bytes:
        """Download file from the sandbox."""
        return self._session.download_file(remote_path)

    def upload_file(self, file: IO, description: str) -> UploadedFile:
        """Upload file to the sandbox. The file is uploaded to the '/home/user/<filename>' path."""
        remote_path = self._session.upload_file(file)

        f = UploadedFile(
            name=file.name, description=description, remote_path=remote_path
        )
        self._uploaded_files.append(f)
        return f

    def remove_uploaded_file(self, uploaded_file: UploadedFile) -> None:
        """Remove uploaded file from the sandbox."""
        self._session.filesystem.remove(uploaded_file.remote_path)
        self._uploaded_files.filter(
            lambda f: f.remote_path != uploaded_file.remote_path
        )

    def as_tool(self) -> Tool:
        return Tool.from_function(
            func=self._run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
