"""Test base tool child implementations."""


import inspect
import re
from typing import List, Type

import pytest

from langchain.tools.amadeus.base import AmadeusBaseTool
from langchain.tools.base import BaseTool
from langchain.tools.gmail.base import GmailBaseTool
from langchain.tools.office365.base import O365BaseTool
from langchain.tools.playwright.base import BaseBrowserTool


def get_non_abstract_subclasses(cls: Type[BaseTool]) -> List[Type[BaseTool]]:
    to_skip = {
        AmadeusBaseTool,
        BaseBrowserTool,
        GmailBaseTool,
        O365BaseTool,
    }  # Abstract but not recognized
    subclasses = []
    for subclass in cls.__subclasses__():
        if (
            not getattr(subclass, "__abstract__", None)
            and not subclass.__name__.startswith("_")
            and subclass not in to_skip
        ):
            subclasses.append(subclass)
        sc = get_non_abstract_subclasses(subclass)
        subclasses.extend(sc)
    return subclasses


@pytest.mark.parametrize("cls", get_non_abstract_subclasses(BaseTool))  # type: ignore
def test_all_subclasses_accept_run_manager(cls: Type[BaseTool]) -> None:
    """Test that tools defined in this repo accept a run manager argument."""
    # This wouldn't be necessary if the BaseTool had a strict API.
    if cls._run is not BaseTool._arun:
        run_func = cls._run
        params = inspect.signature(run_func).parameters
        assert "run_manager" in params
        pattern = re.compile(r"(?!Async)CallbackManagerForToolRun")
        assert bool(re.search(pattern, str(params["run_manager"].annotation)))
        assert params["run_manager"].default is None

    if cls._arun is not BaseTool._arun:
        run_func = cls._arun
        params = inspect.signature(run_func).parameters
        assert "run_manager" in params
        assert "AsyncCallbackManagerForToolRun" in str(params["run_manager"].annotation)
        assert params["run_manager"].default is None
