"""Push and pull to the LangChain Hub."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain.load.dump import dumps
from langchain.load.load import loads

if TYPE_CHECKING:
    from langchainhub import Client


def _get_client(api_url: Optional[str] = None, api_key: Optional[str] = None) -> Client:
    try:
        from langchainhub import Client
    except ImportError as e:
        raise ImportError(
            "Could not import langchainhub, please install with `pip install "
            "langchainhub`."
        ) from e

    # Client logic will also attempt to load URL/key from environment variables
    return Client(api_url, api_key=api_key)


def push(
    repo_full_name: str,
    object: Any,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    parent_commit_hash: Optional[str] = "latest",
    new_repo_is_public: bool = False,
    new_repo_description: str = "",
) -> str:
    """
    Pushes an object to the hub and returns the URL.
    """
    client = _get_client(api_url=api_url, api_key=api_key)
    manifest_json = dumps(object)
    resp = client.push(
        repo_full_name,
        manifest_json,
        parent_commit_hash=parent_commit_hash,
        new_repo_is_public=new_repo_is_public,
        new_repo_description=new_repo_description,
    )
    commit_hash: str = resp["commit"]["commit_hash"]
    return commit_hash


def pull(
    owner_repo_commit: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Pulls an object from the hub and returns it.
    """
    client = _get_client(api_url=api_url, api_key=api_key)
    resp: str = client.pull(owner_repo_commit)
    return loads(resp)
