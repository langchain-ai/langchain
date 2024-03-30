"""Tools for interacting with the user."""


import warnings
from typing import Any

from langchain_community.tools.human.tool import HumanInputRun


def StdInInquireTool(*args: Any, **kwargs: Any) -> HumanInputRun:
    """Tool for asking the user for input."""
    warnings.warn(
        "StdInInquireTool will be deprecated in the future. "
        "Please use HumanInputRun instead.",
        DeprecationWarning,
    )
    return HumanInputRun(*args, **kwargs)
