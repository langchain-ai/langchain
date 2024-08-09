from typing import List, Type

from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.pydantic import get_fields

import langchain_community.tools
from langchain_community.tools import _DEPRECATED_TOOLS
from langchain_community.tools import __all__ as tools_all

_EXCLUDE = {
    BaseTool,
    StructuredTool,
}


def _get_tool_classes(skip_tools_without_default_names: bool) -> List[Type[BaseTool]]:
    results = []
    for tool_class_name in tools_all:
        if tool_class_name in _DEPRECATED_TOOLS:
            continue
        # Resolve the str to the class
        tool_class = getattr(langchain_community.tools, tool_class_name)
        if isinstance(tool_class, type) and issubclass(tool_class, BaseTool):
            if tool_class in _EXCLUDE:
                continue
            if skip_tools_without_default_names and get_fields(tool_class)[
                "name"
            ].default in [  # type: ignore
                None,
                "",
            ]:
                continue
            results.append(tool_class)
    return results


def test_tool_names_unique() -> None:
    """Test that the default names for our core tools are unique."""
    tool_classes = _get_tool_classes(skip_tools_without_default_names=True)
    names = sorted([get_fields(tool_cls)["name"].default for tool_cls in tool_classes])
    duplicated_names = [name for name in names if names.count(name) > 1]
    assert not duplicated_names
