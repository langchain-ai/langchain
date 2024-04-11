from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BufferedReader, BytesIO
import os
import re
from typing import Any, Callable, Optional, Annotated
from uuid import uuid4
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import Field
from azure.core.credentials import AccessToken
from langchain_core.runnables.config import run_in_executor
from azure.identity import DefaultAzureCredential
import requests


def _access_token_provider_factory() -> Callable[[], Optional[str]]:
    """Factory function for creating an access token provider function.

    Returns:
        Callable[[], Optional[str]]: The access token provider function
    """
    
    access_token: AccessToken = None

    def access_token_provider() -> Optional[str]:
        nonlocal access_token
        if access_token is None or datetime.fromtimestamp(access_token.expires_on, timezone.utc) < datetime.now(timezone.utc) + timedelta(minutes=5):
            credential = DefaultAzureCredential()
            access_token = credential.get_token("https://acasessions.io/.default")
        return access_token.token
    
    return access_token_provider


def _sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


@dataclass
class RemoteFileMetadata:
    """Metadata for a file in the session."""

    filename: str
    """The filename relative to `/mnt/data`."""

    size_in_bytes: int
    """The size of the file in bytes."""

    @property
    def full_path(self) -> str:
        """Get the full path of the file."""
        return f"/mnt/data/{self.filename}"
    
    @staticmethod
    def from_dict(data: dict) -> "RemoteFileMetadata":
        """Create a RemoteFileMetadata object from a dictionary."""
        return RemoteFileMetadata(
            filename=data["filename"],
            size_in_bytes=data["bytes"]
        )


class SessionsPythonREPLTool(BaseTool):
    """A tool for running Python code in an Azure Container Apps dynamic sessions code interpreter."""

    name: str = "Sessions_Python_REPL"
    description: str = (
        "A Python shell. Use this to execute python commands "
        "when you need to perform calculations or computations. "
        "Input should be a valid python command. "
        "Returns the result, stdout, and stderr. "
    )

    sanitize_input: bool = True
    """Whether to sanitize input to the python REPL."""

    pool_management_endpoint: str
    """The management endpoint of the session pool. Should end with a '/'."""

    access_token_provider: Callable[[], Optional[str]] = _access_token_provider_factory()
    """A function that returns the access token to use for the session pool."""

    session_id: str = str(uuid4())
    """The session ID to use for the session pool. Defaults to a random UUID."""

    def _build_url(self, path) -> str:
        pool_management_endpoint = self.pool_management_endpoint
        if not pool_management_endpoint:
            raise ValueError("pool_management_endpoint is not set")
        if not pool_management_endpoint.endswith("/"):
            pool_management_endpoint += "/"
        return pool_management_endpoint + path

    def _run(self, python_code: str) -> Any:
        if self.sanitize_input:
            python_code = _sanitize_input(python_code)

        access_token = self.access_token_provider()
        api_url = self._build_url("python/execute")
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        body = {
            "properties": {
                "identifier": self.session_id,
                "codeInputType": "inline",
                "executionType": "synchronous",
                "pythonCode": python_code,
            }
        }

        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        response_json = response.json()

        return f"result:\n{response_json['result']}\n\nstdout:\n{response_json['stdout']}\n\nstderr:\n{response_json['stderr']}"

    async def _arun(self, python_code: str) -> Any:
        """Use the tool asynchronously."""
        if self.sanitize_input:
            python_code = _sanitize_input(python_code)

        return await run_in_executor(None, self.run, python_code)

    def upload_file(self, data: BufferedReader = None, remote_file_path: str = None, local_file_path: str = None) -> RemoteFileMetadata:
        """Upload a file to the session pool.

        Args:
            data: The data to upload.
            remote_file_path: The path to upload the file to, relative to `/mnt/data`. If local_file_path is provided, this is defaulted to its filename.
            local_file_path: The path to the local file to upload.

        Returns:
            RemoteFileMetadata: The metadata for the uploaded file
        """

        if data and local_file_path:
            raise ValueError("data and local_file_path cannot be provided together")

        if local_file_path:
            if not remote_file_path:
                remote_file_path = os.path.basename(local_file_path)
            data = open(local_file_path, "rb")

        access_token = self.access_token_provider()
        api_url = self._build_url(f"python/uploadFile?identifier={self.session_id}")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }
        payload = {}
        files=[
            ('file',(remote_file_path, data, 'application/octet-stream'))
        ]

        response = requests.request("POST", api_url, headers=headers, data=payload, files=files)
        response.raise_for_status()

        response_json = response.json()
        return RemoteFileMetadata.from_dict(response_json)
    
    def download_file(self, *, remote_filename: str) -> BufferedReader:
        """Download a file from the session pool.

        Args:
            remote_filename: The path to download the file from, relative to `/mnt/data`

        Returns:
            BufferedReader: The data of the downloaded file
        """
        access_token = self.access_token_provider()
        api_url = self._build_url(f"python/downloadFile?identifier={self.session_id}&filename={remote_filename}")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        return BytesIO(response.content)

    def list_files(self) -> list[RemoteFileMetadata]:
        """List the files in the session pool.

        Returns:
            list[RemoteFileMetadata]: The metadata for the files in the session pool
        """
        access_token = self.access_token_provider()
        api_url = self._build_url(f"python/files?identifier={self.session_id}")
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        response_json = response.json()
        return [RemoteFileMetadata.from_dict(entry) for entry in response_json["$values"]]
