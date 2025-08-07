"""Temporary wrapper for sandbox integrations."""

from typing import List, Optional, Literal, Sequence
from typing import TypedDict, NotRequired

from daytona import Sandbox

from langchain_core.tools import BaseTool, StructuredTool


class FileInfo(TypedDict):
    """Metadata of a file in the sandbox."""

    name: str
    is_dir: bool
    size: NotRequired[float]
    mod_time: NotRequired[str]
    mode: NotRequired[str]
    permissions: NotRequired[str]
    owner: NotRequired[str]
    group: NotRequired[str]


class SandboxCapabilities(TypedDict):
    """Capabilities of the sandbox."""

    can_upload: bool
    can_download: bool
    can_list_files: bool
    can_run_code: bool
    supported_languages: List[str]


class ExecuteResponse(TypedDict):
    """Result of code execution."""

    result: str
    """The output of the executed code.

    This will usually be the standard output of the command executed.
    """
    exit_code: NotRequired[int]
    """The exit code of the executed code, if applicable."""
    session_id: NotRequired[str]
    """The ID of the session in which the code was executed."""


class DaytonaSandboxToolkit:
    """A toolkit for interacting with sandboxes."""

    def __init__(self, sandbox: Sandbox, *, default_language: str | None) -> None:
        """Initialize the SandboxToolkit with a DaytonSandbox instance."""
        self.sandbox = sandbox
        self._default_language = default_language

    def list_files(self, path: str) -> List[FileInfo]:
        """List files in the specified path."""
        return self.sandbox.fs.list_files(path)

    def move_files(self, source: str, destination: str) -> None:
        """Move a file from source to destination."""
        return self.sandbox.fs.move_files(source, destination)

    def upload_file(
        self, file: bytes, remote_path: str, timeout: int = 30 * 60
    ) -> None:
        """Upload a file to the sandbox."""
        return self.sandbox.fs.upload_file(file, remote_path, timeout=timeout)

    def upload_files(
        self, files: List[str | bytes], remote_path: str, timeout: int = 30 * 60
    ):
        """Upload one or more files to the sandbox."""
        return self.sandbox.fs.upload_files(files, remote_path, timeout=timeout)

    def download_file(self, remote_path: str, timeout: int = 30 * 60) -> bytes:
        """Download a file from the sandbox."""
        return self.sandbox.fs.download_file(remote_path, timeout=timeout)

    def run_code(self, code: str, timeout: int = 30 * 60) -> ExecuteResponse:
        """Run code in the sandbox."""
        execute_response = self.sandbox.process.code_run(code, timeout=timeout)
        return ExecuteResponse(
            result=execute_response.result,
            exit_code=execute_response.exit_code,
            session_id=execute_response.session_id,
        )

    def repl(self, code: str, timeout: int = 30 * 60) -> ExecuteResponse:
        """Support for REPL execution in the sandbox."""
        raise NotImplementedError()

    def exec(
        self, command: str, cwd: Optional[str] = None, timeout: int = 30 * 60
    ) -> ExecuteResponse:
        """Execute a command in the sandbox."""
        execute_response = self.sandbox.process.exec(command, cwd=cwd, timeout=timeout)
        return ExecuteResponse(
            result=execute_response.result,
            exit_code=execute_response.exit_code,
            session_id=execute_response.session_id,
        )

    @property
    def default_language(self) -> str | None:
        """Get the default language for code execution."""
        return self._default_language

    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox."""
        return {
            "can_upload": True,
            "can_download": True,
            "can_list_files": True,
            "can_run_code": True,
            "supported_languages": ["python"],
        }


class Adapter:
    """An adapter to integrate responses from a sandbox toolkit to an LLM."""

    @staticmethod
    def format_response(response: ExecuteResponse) -> str:
        """Format the response for code execution."""
        return f"Result: {response['result']}\nExit Code: {response.get('exit_code', 'N/A')}"

    @staticmethod
    def format_list_files(files: List[FileInfo]) -> str:
        """Format the response for a list of files."""
        if not files:
            return "No files found."
        return "\n".join(
            f"{'[DIR]' if f['is_dir'] else '[FILE]'} {f['name']}" for f in files
        )

    @staticmethod
    def format_upload_file(_: None) -> str:
        """Format the response for an uploaded file."""
        return "File uploaded successfully."

    @staticmethod
    def format_download_file(_: bytes) -> str:
        """Format the response for a downloaded file."""
        return "File downloaded successfully. (Binary content omitted)"

    @staticmethod
    def format_default(obj: object) -> str:
        return str(obj)


def _wrap_tool(func, formatter):
    def wrapped(*args, **kwargs):
        raw = func(*args, **kwargs)
        return formatter(raw)

    return wrapped


def create_tools(
    toolkit: DaytonaSandboxToolkit,
    tool_selection: Sequence[
        Literal["run_code", "list_files", "upload_file", "download_file"]
    ],
) -> List[BaseTool]:
    """Create tools for the given sandbox.

    Args:
        toolkit: The SandboxToolkit to use for creating tools.
        tool_selection: A sequence of tool names to create.
    """
    requested_tools = set(tool_selection)
    capabilities = toolkit.get_capabilities()

    # Let's create each tool now
    tools: list[BaseTool] = []
    for tool_name in requested_tools:
        if tool_name == "run_code":
            if not capabilities["can_run_code"]:
                raise ValueError("Sandbox does not support running code.")
            tools.append(
                StructuredTool(
                    description="Run code in the sandbox.",
                    name="run_code",
                    func=toolkit.run_code,
                )
            )
        elif tool_name == "list_files":
            if not capabilities["can_list_files"]:
                raise ValueError("Sandbox does not support listing files.")
            tools.append(
                StructuredTool(
                    description="List files in the sandbox.",
                    name="list_files",
                    func=toolkit.list_files,
                )
            )
        elif tool_name == "upload_file":
            if not capabilities["can_upload"]:
                raise ValueError("Sandbox does not support uploading files.")
            tools.append(
                StructuredTool(
                    description="Upload a file to the sandbox.",
                    name="upload_file",
                    func=toolkit.upload_file,
                )
            )
        elif tool_name == "download_file":
            if not capabilities["can_download"]:
                raise ValueError("Sandbox does not support downloading files.")
            tools.append(
                StructuredTool(
                    description="Download a file from the sandbox.",
                    name="download_file",
                    func=toolkit.download_file,
                )
            )
        elif tool_name == "repl":
            if not capabilities["can_run_code"]:
                raise ValueError("Sandbox does not support REPL execution.")
            tools.append(
                StructuredTool(
                    description="Run code in a REPL environment in the sandbox.",
                    name="repl",
                    func=toolkit.repl,
                )
            )
        elif tool_name == "exec":
            if not capabilities["can_run_code"]:
                raise ValueError("Sandbox does not support executing commands.")
            tools.append(
                StructuredTool(
                    description="Execute a command in the sandbox.",
                    name="exec",
                    func=toolkit.exec,
                )
            )
        else:
            known_tools = {
                "run_code",
                "list_files",
                "upload_file",
                "download_file",
            }
            msg = (
                f"Unsupported tool: {tool_name}. "
                f"Supported tools are: {', '.join(known_tools)}."
            )
            raise NotImplementedError(msg)

    return tools
