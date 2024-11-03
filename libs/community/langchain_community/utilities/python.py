import logging
from typing import Any

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    if name in "PythonREPL":
        raise AssertionError(
            "PythonREPL has been deprecated from langchain_community due to being "
            "flagged by security scanners. See: "
            "https://github.com/langchain-ai/langchain/issues/14345 "
            "If you need to use it, please use the version "
            "from langchain_experimental. "
            "from langchain_experimental.utilities.python import PythonREPL."
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
