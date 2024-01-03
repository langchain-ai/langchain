from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING

from packaging.version import parse

if TYPE_CHECKING:
    import openai


def is_openai_v1() -> bool:
    """Return whether OpenAI API is v1 or more."""
    _version = parse(version("openai"))
    return _version.major >= 1


def get_openai_chat_completion() -> "openai.ChatCompletion":
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "Could not import openai python package. "
            "Please install it with `pip install openai`.",
        ) from e

    try:
        return openai.ChatCompletion
    except AttributeError as e:
        raise ValueError(
            "`openai` has no `ChatCompletion` attribute, this is likely "
            "due to an old version of the openai package. Try upgrading it "
            "with `pip install --upgrade openai`.",
        ) from e
