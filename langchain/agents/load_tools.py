# flake8: noqa
"""Load tools."""
from typing import List, Optional

from langchain.agents.tools import Tool
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.pal.base import PALChain
from langchain.llms.base import LLM
from langchain.python import PythonREPL
from langchain.requests import RequestsWrapper
from langchain.serpapi import SerpAPIWrapper
from langchain.utilities.bash import BashProcess


def _get_python_repl() -> Tool:
    return Tool(
        "Python REPL",
        PythonREPL().run,
        "A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out.",
    )


def _get_serpapi() -> Tool:
    return Tool(
        "Search",
        SerpAPIWrapper().run,
        "A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
    )


def _get_requests() -> Tool:
    return Tool(
        "Requests",
        RequestsWrapper().run,
        "A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific url, and the output will be all the text on that page.",
    )


def _get_terminal() -> Tool:
    return Tool(
        "Terminal",
        BashProcess().run,
        "Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command.",
    )


_BASE_TOOLS = {
    "python_repl": _get_python_repl,
    "serpapi": _get_serpapi,
    "requests": _get_requests,
    "terminal": _get_terminal,
}


def _get_pal_math(llm: LLM) -> Tool:
    return Tool(
        "PAL-MATH",
        PALChain.from_math_prompt(llm).run,
        "A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.",
    )


def _get_pal_colored_objects(llm: LLM) -> Tool:
    return Tool(
        "PAL-COLOR-OBJ",
        PALChain.from_colored_object_prompt(llm).run,
        "A language model that is really good at reasoning about position and the color attributes of objects. Input should be a fully worded hard reasoning problem. Make sure to include all information about the objects AND the final question you want to answer.",
    )


def _get_llm_math(llm: LLM) -> Tool:
    return Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
        "Useful for when you need to answer questions about math.",
    )


_LLM_TOOLS = {
    "pal-math": _get_pal_math,
    "pal-colored-objects": _get_pal_colored_objects,
    "llm-math": _get_llm_math,
}


def load_tools(tool_names: List[str], llm: Optional[LLM] = None) -> List[Tool]:
    """Load tools based on their name.

    Args:
        tool_names: name of tools to load.
        llm: Optional language model, may be needed to initialize certain tools.

    Returns:
        List of tools.
    """
    tools = []
    for name in tool_names:
        if name in _BASE_TOOLS:
            tools.append(_BASE_TOOLS[name]())
        elif name in _LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            tools.append(_LLM_TOOLS[name](llm))
        else:
            raise ValueError(f"Got unknown tool {name}")
    return tools
