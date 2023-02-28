"""Utilities for loading configurations from langchain-hub and Hugging Face Hub."""

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Set, TypeVar, Union
from urllib.parse import urljoin

import requests

DEFAULT_REF = os.environ.get("LANGCHAIN_HUB_DEFAULT_REF", "master")
URL_BASE = os.environ.get(
    "LANGCHAIN_HUB_URL_BASE",
    "https://raw.githubusercontent.com/hwchase17/langchain-hub/{ref}/",
)
HUB_PATH_RE = re.compile(r"(hf|lc)(?P<ref>@[^:]+)?://(?P<path>.*)")

T = TypeVar("T")


def try_load_from_hf_hub(
    path: Path,
    loader: Callable[[str], T],
    **kwargs: Any,
) -> Optional[T]:
    """Load configuration from the Hugging Face Hub.

    The Hugging Face Hub automatically has version control, simple sharing mechanism,
    local caching, and social features (such as likes and Discussions).

    Example:
        .. code-block:: python

            from langchain.prompts import load_prompt
            load_prompt("hf://prompts/LLM_Bash/prompt.json")
    """
    from huggingface_hub import hf_hub_download

    if len(path.parts) != 3:
        raise ValueError(
            "Invalid path. Make sure the path is in the format of "
            "hf://<suffix>/<repo_id>/<filename>"
        )
    prefix, repo_id, filename = path.parts
    if prefix == "prompts":
        org_id = "LangChainHub-Prompts"
    elif prefix == "agents":
        org_id = "LangChainHub-Agents"
    elif prefix == "chains":
        org_id = "LangChainHub-Chain"
    else:
        raise ValueError(f"Unsupported prefix: '{prefix}'. Must be one of 'agents', 'chains' or 'prompts'")

    downloaded_file = hf_hub_download(
        repo_id=f"{org_id}/{repo_id}", filename=filename, repo_type="dataset"
    )
    return loader(downloaded_file, **kwargs)


def try_load_from_hub(
    path: Union[str, Path],
    loader: Callable[[str], T],
    valid_prefix: str,
    valid_suffixes: Set[str],
    **kwargs: Any,
) -> Optional[T]:
    """Load configuration from hub.  Returns None if path is not a hub path."""
    if not isinstance(path, str) or not (match := HUB_PATH_RE.match(path)):
        return None
    source, ref, remote_path_str = match.groups()
    remote_path = Path(remote_path_str)
    if remote_path.parts[0] != valid_prefix:
        return None
    if remote_path.suffix[1:] not in valid_suffixes:
        raise ValueError("Unsupported file type.")
    if source == "lc":
        ref = ref[1:] if ref else DEFAULT_REF
        full_url = urljoin(URL_BASE.format(ref=ref), str(remote_path))
        r = requests.get(full_url, timeout=5)
        if r.status_code != 200:
            raise ValueError(f"Could not find file at {full_url}")
        with tempfile.TemporaryDirectory() as tmpdirname:
            file = Path(tmpdirname) / remote_path.name
            with open(file, "wb") as f:
                f.write(r.content)
            return loader(str(file), **kwargs)
    else:
        return try_load_from_hf_hub(remote_path, loader, **kwargs)
