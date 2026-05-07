"""Interface with the `LangChain Hub <https://smith.langchain.com/hub>`__."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from langchain_core._api.deprecation import deprecated
from langsmith import Client as LangSmithClient


@deprecated(
    since="0.3.30",
    message="langchain.hub.push is deprecated. Use the LangSmith SDK instead.",
    pending=True,
)
def push(
    repo_full_name: str,
    object: Any,  # noqa: A002
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    parent_commit_hash: str = "latest",
    new_repo_is_public: bool = False,
    new_repo_description: Optional[str] = None,
    readme: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
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
    client = LangSmithClient(api_url, api_key=api_key)
    return client.push_prompt(
        repo_full_name,
        object=object,
        parent_commit_hash=parent_commit_hash,
        is_public=new_repo_is_public,
        description=new_repo_description,
        readme=readme,
        tags=tags,
    )


@deprecated(
    since="0.3.30",
    message="langchain.hub.pull is deprecated. Use the LangSmith SDK instead.",
    pending=True,
)
def pull(
    owner_repo_commit: str,
    *,
    include_model: Optional[bool] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Pull an object from the hub and returns it as a LangChain object.

    .. danger::

        Hub manifests are untrusted input. Treat every prompt pulled from the
        hub as untrusted, regardless of the owner — public prompts authored by
        other users are obviously external content, but prompts from your own
        account or your organization's account are also unsafe if that account,
        a teammate's account, or the upstream prompt has been compromised.

        ``pull()`` deserializes the manifest via ``load()``, so the
        ``langchain_core.load.load`` threat model applies — a manifest can
        intentionally configure a model with a custom base URL, headers, model
        name, or other constructor arguments. These are supported features, but
        they also mean the prompt contents are executable configuration rather
        than plain text: a compromised prompt can redirect API traffic, inject
        headers, or trigger arbitrary code paths in the classes it instantiates.

        Prefer the LangSmith SDK directly. If you must use ``pull()``, pin the
        commit hash and audit the manifest before deserializing.

    :param owner_repo_commit: The full name of the prompt to pull from in the format of
        `owner/prompt_name:commit_hash` or `owner/prompt_name`
        or just `prompt_name` if it's your own prompt.
    :param api_url: The URL of the LangChain Hub API. Defaults to the hosted API service
        if you have an api key set, or a localhost instance if not.
    :param api_key: The API key to use to authenticate with the LangChain Hub API.
    """
    client = LangSmithClient(api_url, api_key=api_key)
    return client.pull_prompt(owner_repo_commit, include_model=include_model)
