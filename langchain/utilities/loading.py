"""Utilities for loading configurations from langchain-hub."""

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
HUB_PATH_RE = re.compile(r"lc(?P<ref>@[^:]+)?://(?P<path>.*)")


T = TypeVar("T")


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
    ref, remote_path_str = match.groups()
    ref = ref[1:] if ref else DEFAULT_REF
    remote_path = Path(remote_path_str)
    if remote_path.parts[0] != valid_prefix:
        return None
    if remote_path.suffix[1:] not in valid_suffixes:
        raise ValueError("Unsupported file type.")

    full_url = urljoin(URL_BASE.format(ref=ref), str(remote_path))
    r = requests.get(full_url, timeout=5)
    if r.status_code != 200:
        raise ValueError(f"Could not find file at {full_url}")
    with tempfile.TemporaryDirectory() as tmpdirname:
        file = Path(tmpdirname) / remote_path.name
        with open(file, "wb") as f:
            f.write(r.content)
        return loader(str(file), **kwargs)
