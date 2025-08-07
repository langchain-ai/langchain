"""Temporary wrapper for sandbox integrations."""

import os
from typing import List
from typing import TypedDict, NotRequired

from daytona import Daytona, DaytonaConfig, Sandbox


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


class SandboxToolkit:
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

    def run_code(self, code: str, timeout: int = 30 * 60) -> str:
        """Run code in the sandbox."""
        return self.sandbox.run_code(code, timeout=timeout)

    def get_capabilities(self) -> SandboxCapabilities:
        """Get the capabilities of the sandbox."""
        return {
        }


class DaytonSandbox:
    """A sandbox for testing Daytona API interactions."""

    def __init__(self, *, api_key: str | None) -> None:
        # Define the configuration
        if api_key is None:
            api_key = os.environ.get("DAYTONA_API_KEY")
            if api_key is None:
                raise ValueError(
                    "API key must be provided either as an argument "
                    "or through the DAYTONA_API_KEY environment variable."
                )

        api_key = os.environ.get("DAYTONA_API_KEY", api_key)
        config = DaytonaConfig(api_key=api_key)
        # Initialize the Daytona client
        daytona = Daytona(config)
        # Create the Sandbox instance
        sandbox = daytona.create()
        self.sandbox = sandbox
