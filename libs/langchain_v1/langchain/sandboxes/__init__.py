"""Temporary wrapper for sandbox integrations."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
)

from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from daytona import Sandbox


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
    supported_languages: list[str]


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

    def list_files(self, path: str) -> list[FileInfo]:
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
        self, files: list[str | bytes], remote_path: str, timeout: int = 30 * 60
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
        raise NotImplementedError

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
        result = f"<execute_result>\n<output>{response['result']}</output>"

        if "exit_code" in response and response["exit_code"] is not None:
            result += f"\n<exit_code>{response['exit_code']}</exit_code>"

        result += "\n</execute_result>"
        return result

    @staticmethod
    def format_list_files(files: list[FileInfo]) -> str:
        """Format the response for a list of files."""
        if not files:
            return "<file_list>\n<message>No files found</message>\n</file_list>"

        result = "<file_list>\n"
        for file_info in files:
            file_type = "directory" if file_info["is_dir"] else "file"
            result += f'<item type="{file_type}">\n<name>{file_info["name"]}</name>\n'

            if "size" in file_info:
                result += f"<size>{file_info['size']}</size>\n"
            if "mod_time" in file_info:
                result += f"<modified>{file_info['mod_time']}</modified>\n"
            if "permissions" in file_info:
                result += f"<permissions>{file_info['permissions']}</permissions>\n"
            if "owner" in file_info:
                result += f"<owner>{file_info['owner']}</owner>\n"
            if "group" in file_info:
                result += f"<group>{file_info['group']}</group>\n"

            result += "</item>\n"
        result += "</file_list>"
        return result

    @staticmethod
    def format_upload_file() -> str:
        """Format the response for an uploaded file."""
        return (
            "<upload_result>\n"
            "<status>success</status>\n"
            "<message>File uploaded successfully</message>\n"
            "</upload_result>"
        )

    @staticmethod
    def format_download_file(size: int) -> str:
        """Format the response for a downloaded file."""
        return (
            "<download_result>\n"
            "<status>success</status>\n"
            "<message>File downloaded successfully</message>\n"
            f"<size_bytes>{size}</size_bytes>\n"
            "<note>Binary content omitted from display</note>\n"
            "</download_result>"
        )

    @staticmethod
    def format_default(obj: object) -> str:
        """Default formatter for objects."""
        return f"<result>\n<content>{obj!s}</content>\n</result>"


def _wrap_tool(func, formatter, *, no_return: bool = False) -> Callable:
    def wrapped(*args, **kwargs):
        raw = func(*args, **kwargs)
        if no_return:
            return formatter(None)
        return formatter(raw)

    return wrapped


def create_tools(
    toolkit: DaytonaSandboxToolkit,
    tool_selection: Sequence[
        Literal["run_code", "list_files", "upload_file", "download_file"]
    ],
) -> list[BaseTool]:
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
                msg = "Sandbox does not support running code."
                raise ValueError(msg)
            tools.append(
                StructuredTool(
                    description="Run code in the sandbox.",
                    name="run_code",
                    func=_wrap_tool(toolkit.run_code, Adapter.format_response),
                )
            )
        elif tool_name == "list_files":
            if not capabilities["can_list_files"]:
                msg = "Sandbox does not support listing files."
                raise ValueError(msg)
            tools.append(
                StructuredTool(
                    description="List files in the sandbox.",
                    name="list_files",
                    func=_wrap_tool(toolkit.list_files, Adapter.format_list_files),
                )
            )
        elif tool_name == "upload_file":
            if not capabilities["can_upload"]:
                msg = "Sandbox does not support uploading files."
                raise ValueError(msg)
            tools.append(
                StructuredTool(
                    description="Upload a file to the sandbox.",
                    name="upload_file",
                    func=_wrap_tool(
                        toolkit.upload_file, Adapter.format_upload_file, no_return=True
                    ),
                )
            )
        elif tool_name == "download_file":
            if not capabilities["can_download"]:
                msg = "Sandbox does not support downloading files."
                raise ValueError(msg)
            tools.append(
                StructuredTool(
                    description="Download a file from the sandbox.",
                    name="download_file",
                    func=_wrap_tool(
                        toolkit.download_file, Adapter.format_download_file
                    ),
                )
            )
        elif tool_name == "repl":
            if not capabilities["can_run_code"]:
                msg = "Sandbox does not support REPL execution."
                raise ValueError(msg)
            tools.append(
                StructuredTool(
                    description="Run code in a REPL environment in the sandbox.",
                    name="repl",
                    func=_wrap_tool(toolkit.repl, Adapter.format_response),
                )
            )
        elif tool_name == "exec":
            if not capabilities["can_run_code"]:
                msg = "Sandbox does not support executing commands."
                raise ValueError(msg)
            tools.append(
                StructuredTool(
                    description="Execute a command in the sandbox.",
                    name="exec",
                    func=_wrap_tool(toolkit.exec, Adapter.format_response),
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
