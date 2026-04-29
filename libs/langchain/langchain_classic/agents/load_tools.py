from typing import Any

from langchain_core.tools import BaseTool

from langchain_classic._api import create_importer

_importer = create_importer(
    __package__,
    fallback_module="langchain_community.agent_toolkits.load_tools",
)


def _get_perplexity_search(**kwargs: Any) -> BaseTool:
    """Construct a `PerplexitySearchResults` tool.

    Requires the `langchain-perplexity` package.
    """
    try:
        from langchain_perplexity import PerplexitySearchResults
    except ImportError as e:
        msg = (
            "Could not import langchain_perplexity python package. "
            "Please install it with `pip install -U langchain-perplexity`."
        )
        raise ImportError(msg) from e
    return PerplexitySearchResults(**kwargs)


_EXTRA_OPTIONAL_TOOLS: dict[str, Any] = {
    "perplexity_search": (_get_perplexity_search, []),
}


def load_tools(tool_names: list[str], **kwargs: Any) -> list[BaseTool]:
    """Load tools by name.

    Extends `langchain_community.agent_toolkits.load_tools.load_tools` with
    partner-package tools registered locally (e.g. `perplexity_search`).
    """
    local_names = [n for n in tool_names if n in _EXTRA_OPTIONAL_TOOLS]
    remaining = [n for n in tool_names if n not in _EXTRA_OPTIONAL_TOOLS]

    tools: list[BaseTool] = []
    for name in local_names:
        get_tool_func, extra_keys = _EXTRA_OPTIONAL_TOOLS[name]
        sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}
        tools.append(get_tool_func(**sub_kwargs))

    if remaining:
        community_load_tools = _importer("load_tools")
        tools.extend(community_load_tools(remaining, **kwargs))

    return tools


def get_all_tool_names() -> list[str]:
    """Return all tool names known to `load_tools`, including local extensions."""
    community_get_all_tool_names = _importer("get_all_tool_names")
    return list(_EXTRA_OPTIONAL_TOOLS) + community_get_all_tool_names()


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically, falling back to langchain_community."""
    return _importer(name)
