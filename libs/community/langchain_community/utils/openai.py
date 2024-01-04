from __future__ import annotations

from importlib.metadata import version

from packaging.version import parse


def is_openai_v1() -> bool:
    _version = parse(version("openai"))
    return _version.major >= 1
