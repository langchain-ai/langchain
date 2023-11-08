from __future__ import annotations

from importlib.metadata import version

from packaging.version import Version, parse


def is_openai_v1() -> bool:
    _version = parse(version("openai"))
    return _version >= Version("1.0.0")
