"""Interface with the [LangChain Hub](https://smith.langchain.com/hub)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.prompts import BasePromptTemplate


def _get_client(
    api_key: str | None = None,
    api_url: str | None = None,
) -> Any:
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

    :param repo_full_name: The full name of the prompt to push to in the format of
        `owner/prompt_name` or `prompt_name`.
    :param object: The LangChain to serialize and push to the hub.
    :param api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
        if you have an api key set, or a localhost instance if not.
    :param api_key: The API key to use to authenticate with the LangChain Hub API.
    :param parent_commit_hash: The commit hash of the parent commit to push to. Defaults
        to the latest commit automatically.
    :param new_repo_is_public: Whether the prompt should be public. Defaults to
        False (Private by default).
    :param new_repo_description: The description of the prompt. Defaults to an empty
        string.
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

    :param owner_repo_commit: The full name of the prompt to pull from in the format of
        `owner/prompt_name:commit_hash` or `owner/prompt_name`
        or just `prompt_name` if it's your own prompt.
    :param api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
        if you have an api key set, or a localhost instance if not.
    :param api_key: The API key to use to authenticate with the LangChain Hub API.
    """
    client = _get_client(api_key=api_key, api_url=api_url)

    # Then it's langsmith
    if hasattr(client, "pull_prompt"):
        return client.pull_prompt(owner_repo_commit, include_model=include_model)

    # Then it's langchainhub
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

    # Then it's < 0.1.15 langchainhub
    resp: str = client.pull(owner_repo_commit)
    return loads(resp)
