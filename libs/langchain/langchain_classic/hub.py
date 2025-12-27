"""Interface with the [LangChain Hub](https://smith.langchain.com/hub)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.prompts import BasePromptTemplate


def _get_client(
    api_key: str | None = None,
    api_url: str | None = None,
) -> Any:
    """Get a client for interacting with the LangChain Hub.

    Attempts to use LangSmith client if available, otherwise falls back to
    the legacy `langchainhub` client.

    Args:
        api_key: API key to authenticate with the LangChain Hub API.
        api_url: URL of the LangChain Hub API.

    Returns:
        Client instance for interacting with the hub.

    Raises:
        ImportError: If neither `langsmith` nor `langchainhub` can be imported.
    """
    try:
        from langsmith import Client as LangSmithClient

        ls_client = LangSmithClient(api_url, api_key=api_key)
        if hasattr(ls_client, "push_prompt") and hasattr(ls_client, "pull_prompt"):
            return ls_client
        from langchainhub import Client as LangChainHubClient

        return LangChainHubClient(api_url, api_key=api_key)
    except ImportError:
        try:
            from langchainhub import Client as LangChainHubClient

            return LangChainHubClient(api_url, api_key=api_key)
        except ImportError as e:
            msg = (
                "Could not import langsmith or langchainhub (deprecated),"
                "please install with `pip install langsmith`."
            )
            raise ImportError(msg) from e


def push(
    repo_full_name: str,
    object: Any,  # noqa: A002
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    parent_commit_hash: str | None = None,
    new_repo_is_public: bool = False,
    new_repo_description: str | None = None,
    readme: str | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    """Push an object to the hub and returns the URL it can be viewed at in a browser.

    Args:
        repo_full_name: The full name of the prompt to push to in the format of
            `owner/prompt_name` or `prompt_name`.
        object: The LangChain object to serialize and push to the hub.
        api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
            if you have an API key set, or a localhost instance if not.
        api_key: The API key to use to authenticate with the LangChain Hub API.
        parent_commit_hash: The commit hash of the parent commit to push to. Defaults
            to the latest commit automatically.
        new_repo_is_public: Whether the prompt should be public.
        new_repo_description: The description of the prompt.
        readme: README content for the repository.
        tags: Tags to associate with the prompt.

    Returns:
        URL where the pushed object can be viewed in a browser.
    """
    client = _get_client(api_key=api_key, api_url=api_url)

    # Then it's langsmith
    if hasattr(client, "push_prompt"):
        return client.push_prompt(
            repo_full_name,
            object=object,
            parent_commit_hash=parent_commit_hash,
            is_public=new_repo_is_public,
            description=new_repo_description,
            readme=readme,
            tags=tags,
        )

    # Then it's langchainhub
    manifest_json = dumps(object)
    return client.push(
        repo_full_name,
        manifest_json,
        parent_commit_hash=parent_commit_hash,
        new_repo_is_public=new_repo_is_public,
        new_repo_description=new_repo_description,
    )


def pull(
    owner_repo_commit: str,
    *,
    include_model: bool | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Pull an object from the hub and returns it as a LangChain object.

    Args:
        owner_repo_commit: The full name of the prompt to pull from in the format of
            `owner/prompt_name:commit_hash` or `owner/prompt_name`
            or just `prompt_name` if it's your own prompt.
        include_model: Whether to include the model configuration in the pulled prompt.
        api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
            if you have an API key set, or a localhost instance if not.
        api_key: The API key to use to authenticate with the LangChain Hub API.

    Returns:
        The pulled LangChain object.
    """
    client = _get_client(api_key=api_key, api_url=api_url)

    # Then it's langsmith
    if hasattr(client, "pull_prompt"):
        return client.pull_prompt(owner_repo_commit, include_model=include_model)

    # Then it's langchainhub
    if hasattr(client, "pull_repo"):
        # >= 0.1.15
        res_dict = client.pull_repo(owner_repo_commit)
        allowed_objects: Literal["all", "core"] = "all" if include_model else "core"
        obj = loads(json.dumps(res_dict["manifest"]), allowed_objects=allowed_objects)
        if isinstance(obj, BasePromptTemplate):
            if obj.metadata is None:
                obj.metadata = {}
            obj.metadata["lc_hub_owner"] = res_dict["owner"]
            obj.metadata["lc_hub_repo"] = res_dict["repo"]
            obj.metadata["lc_hub_commit_hash"] = res_dict["commit_hash"]
        return obj

    # Then it's < 0.1.15 langchainhub
    resp: str = client.pull(owner_repo_commit)
    return loads(resp)
