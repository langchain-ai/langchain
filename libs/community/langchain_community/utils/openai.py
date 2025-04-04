from __future__ import annotations

import functools
from importlib.metadata import version

from packaging.version import parse


@functools.cache
def is_openai_v1() -> bool:
    """Return whether OpenAI API is v1 or more."""
    _version = parse(version("openai"))
    return _version.major >= 1
