"""Temporary wrapper for sandbox integrations."""

from typing import List, Optional, Literal, Sequence
from typing import TypedDict, NotRequired

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


CODE_TOOLS = Literal["run_code", "list_files", "upload_file", "download_file"]


class DaytonaSandboxToolkit:
    """A toolkit for interacting with sandboxes."""

    def __init__(self, sandbox: Sandbox) -> None:
        """Initialize the SandboxToolkit with a DaytonSandbox instance."""
        self.sandbox = sandbox

    def get_tools(self) -> List[BaseException]:
        """Get the tools for the given sandbox."""

    def list_files(self, path: str) -> List[FileInfo]:
        """List files in the specified path."""
        return self.sandbox.fs.list_files(path)

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

    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox."""
        return {
            "can_upload": True,
            "can_download": True,
            "can_list_files": True,
            "can_run_code": True,
            "supported_languages": ["python"],
        }


def create_tools(
    sandbox: Sandbox,
    tool_selection: Sequence[
        Literal["run_code", "list_files", "upload_file", "download_file"]
    ],
) -> List[BaseException]:
    """Create tools for the given sandbox."""
    toolkit = DaytonaSandboxToolkit(sandbox)
    capabilities = toolkit.get_capabilities()
    return toolkit.get_tools()
