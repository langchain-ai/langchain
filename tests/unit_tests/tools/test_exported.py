from typing import List, Type

import langchain.tools
from langchain.tools import __all__ as tools_all
from langchain.tools.base import BaseTool, StructuredTool

_EXCLUDE = {
    BaseTool,
    StructuredTool,
}


def _get_tool_classes(skip_tools_without_default_names: bool) -> List[Type[BaseTool]]:
    results = []
    for tool_class_name in tools_all:
        # Resolve the str to the class
        tool_class = getattr(langchain.tools, tool_class_name)
        if isinstance(tool_class, type) and issubclass(tool_class, BaseTool):
            if tool_class in _EXCLUDE:
                continue
            if (
                skip_tools_without_default_names
                and tool_class.__fields__["name"].default is None
            ):
                continue
            results.append(tool_class)
    return results


def test_tool_names_unique() -> None:
    """Test that the default names for our core tools are unique."""
    tool_classes = _get_tool_classes(skip_tools_without_default_names=True)
    names = sorted([tool_cls.__fields__["name"].default for tool_cls in tool_classes])
    duplicated_names = [name for name in names if names.count(name) > 1]
    assert not duplicated_names
