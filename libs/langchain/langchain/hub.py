"""Interface with the LangChain Hub."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.prompts import BasePromptTemplate

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
    new_repo_is_public: bool = True,
    new_repo_description: str = "",
) -> str:
    """
    Pushes an object to the hub and returns the URL it can be viewed at in a browser.

    :param repo_full_name: The full name of the repo to push to in the format of
        `owner/repo`.
    :param object: The LangChain to serialize and push to the hub.
    :param api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
        if you have an api key set, or a localhost instance if not.
    :param api_key: The API key to use to authenticate with the LangChain Hub API.
    :param parent_commit_hash: The commit hash of the parent commit to push to. Defaults
        to the latest commit automatically.
    :param new_repo_is_public: Whether the repo should be public. Defaults to
        True (Public by default).
    :param new_repo_description: The description of the repo. Defaults to an empty
        string.
    """
    client = _get_client(api_url=api_url, api_key=api_key)
    manifest_json = dumps(object)
    message = client.push(
        repo_full_name,
        manifest_json,
        parent_commit_hash=parent_commit_hash,
        new_repo_is_public=new_repo_is_public,
        new_repo_description=new_repo_description,
    )
    return message


def pull(
    owner_repo_commit: str,
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Pulls an object from the hub and returns it as a LangChain object.

    :param owner_repo_commit: The full name of the repo to pull from in the format of
        `owner/repo:commit_hash`.
    :param api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
        if you have an api key set, or a localhost instance if not.
    :param api_key: The API key to use to authenticate with the LangChain Hub API.
    """
    client = _get_client(api_url=api_url, api_key=api_key)

    if hasattr(client, "pull_repo"):
        # >= 0.1.15
        res_dict = client.pull_repo(owner_repo_commit)
        obj = loads(json.dumps(res_dict["manifest"]))
        if isinstance(obj, BasePromptTemplate):
            if obj.metadata is None:
                obj.metadata = {}
            obj.metadata["lc_hub_owner"] = res_dict["owner"]
            obj.metadata["lc_hub_repo"] = res_dict["repo"]
            obj.metadata["lc_hub_commit_hash"] = res_dict["commit_hash"]
        return obj

    # Then it's < 0.1.15
    resp: str = client.pull(owner_repo_commit)
    return loads(resp)
