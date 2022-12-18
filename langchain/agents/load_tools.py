# flake8: noqa
"""Load tools."""
from typing import Any, List, Optional

from langchain.agents.tools import Tool
from langchain.chains.api import news_docs, open_meteo_docs, tmdb_docs
from langchain.chains.api.base import APIChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.pal.base import PALChain
from langchain.llms.base import BaseLLM
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


def _get_pal_math(llm: BaseLLM) -> Tool:
    return Tool(
        "PAL-MATH",
        PALChain.from_math_prompt(llm).run,
        "A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.",
    )


def _get_pal_colored_objects(llm: BaseLLM) -> Tool:
    return Tool(
        "PAL-COLOR-OBJ",
        PALChain.from_colored_object_prompt(llm).run,
        "A language model that is really good at reasoning about position and the color attributes of objects. Input should be a fully worded hard reasoning problem. Make sure to include all information about the objects AND the final question you want to answer.",
    )


def _get_llm_math(llm: BaseLLM) -> Tool:
    return Tool(
        "Calculator",
        LLMMathChain(llm=llm).run,
        "Useful for when you need to answer questions about math.",
    )


def _get_open_meteo_api(llm: BaseLLM) -> Tool:
    chain = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS)
    return Tool(
        "Open Meteo API",
        chain.run,
        "Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
    )


_LLM_TOOLS = {
    "pal-math": _get_pal_math,
    "pal-colored-objects": _get_pal_colored_objects,
    "llm-math": _get_llm_math,
    "open-meteo-api": _get_open_meteo_api,
}


def _get_news_api(llm: BaseLLM, **kwargs: Any) -> Tool:
    news_api_key = kwargs["news_api_key"]
    chain = APIChain.from_llm_and_api_docs(
        llm, news_docs.NEWS_DOCS, headers={"X-Api-Key": news_api_key}
    )
    return Tool(
        "News API",
        chain.run,
        "Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
    )


def _get_tmdb_api(llm: BaseLLM, **kwargs: Any) -> Tool:
    tmdb_bearer_token = kwargs["tmdb_bearer_token"]
    chain = APIChain.from_llm_and_api_docs(
        llm,
        tmdb_docs.TMDB_DOCS,
        headers={"Authorization": f"Bearer {tmdb_bearer_token}"},
    )
    return Tool(
        "TMDB API",
        chain.run,
        "Useful for when you want to get information from The Movie Database. The input should be a question in natural language that this API can answer.",
    )


_EXTRA_TOOLS = {
    "news-api": (_get_news_api, ["news_api_key"]),
    "tmdb-api": (_get_tmdb_api, ["tmdb_bearer_token"]),
}


def load_tools(
    tool_names: List[str], llm: Optional[BaseLLM] = None, **kwargs: Any
) -> List[Tool]:
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
        elif name in _EXTRA_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            _get_tool_func, extra_keys = _EXTRA_TOOLS[name]
            missing_keys = set(extra_keys).difference(kwargs)
            if missing_keys:
                raise ValueError(
                    f"Tool {name} requires some parameters that were not "
                    f"provided: {missing_keys}"
                )
            sub_kwargs = {k: kwargs[k] for k in extra_keys}
            tools.append(_get_tool_func(llm=llm, **sub_kwargs))
        else:
            raise ValueError(f"Got unknown tool {name}")
    return tools


def get_all_tool_names() -> List[str]:
    """Get a list of all possible tool names."""
    return list(_BASE_TOOLS) + list(_EXTRA_TOOLS) + list(_LLM_TOOLS)
