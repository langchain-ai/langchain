# flake8: noqa
"""Load tools."""
from typing import Any, List, Optional

from langchain.agents.tools import (
    Tool,
    register_tool,
    register_llm_tool,
    _TOOLS,
    _LLM_TOOLS,
)
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.api import news_docs, open_meteo_docs, tmdb_docs
from langchain.chains.api.base import APIChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.chains.pal.base import PALChain
from langchain.llms.base import BaseLLM
from langchain.requests import RequestsWrapper
from langchain.tools.base import BaseTool
from langchain.tools.bing_search.tool import BingSearchRun
from langchain.tools.google_search.tool import GoogleSearchResults, GoogleSearchRun
from langchain.tools.python.tool import PythonREPLTool
from langchain.tools.requests.tool import RequestsGetTool
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities.bash import BashProcess
from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.utilities.searx_search import SearxSearchWrapper
from langchain.utilities.serpapi import SerpAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper


@register_tool("python_repl")
def _get_python_repl() -> BaseTool:
    return PythonREPLTool()


@register_tool("requests")
def _get_requests() -> BaseTool:
    return RequestsGetTool(requests_wrapper=RequestsWrapper())


@register_tool("terminal")
def _get_terminal() -> BaseTool:
    return Tool(
        name="Terminal",
        description="Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command.",
        func=BashProcess().run,
    )


@register_llm_tool("pal-math")
def _get_pal_math(llm: BaseLLM) -> BaseTool:
    return Tool(
        name="PAL-MATH",
        description="A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.",
        func=PALChain.from_math_prompt(llm).run,
    )


@register_llm_tool("pal-colored-objects")
def _get_pal_colored_objects(llm: BaseLLM) -> BaseTool:
    return Tool(
        name="PAL-COLOR-OBJ",
        description="A language model that is really good at reasoning about position and the color attributes of objects. Input should be a fully worded hard reasoning problem. Make sure to include all information about the objects AND the final question you want to answer.",
        func=PALChain.from_colored_object_prompt(llm).run,
    )


@register_llm_tool("llm-math")
def _get_llm_math(llm: BaseLLM) -> BaseTool:
    return Tool(
        name="Calculator",
        description="Useful for when you need to answer questions about math.",
        func=LLMMathChain(llm=llm, callback_manager=llm.callback_manager).run,
        coroutine=LLMMathChain(llm=llm, callback_manager=llm.callback_manager).arun,
    )


@register_llm_tool("open-meteo-api")
def _get_open_meteo_api(llm: BaseLLM) -> BaseTool:
    chain = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS)
    return Tool(
        name="Open Meteo API",
        description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


@register_tool("news-api", ["news_api_key"])
def _get_news_api(llm: BaseLLM, **kwargs: Any) -> BaseTool:
    news_api_key = kwargs["news_api_key"]
    chain = APIChain.from_llm_and_api_docs(
        llm, news_docs.NEWS_DOCS, headers={"X-Api-Key": news_api_key}
    )
    return Tool(
        name="News API",
        description="Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


@register_tool("tmdb-api", ["tmdb_bearer_token"])
def _get_tmdb_api(llm: BaseLLM, **kwargs: Any) -> BaseTool:
    tmdb_bearer_token = kwargs["tmdb_bearer_token"]
    chain = APIChain.from_llm_and_api_docs(
        llm,
        tmdb_docs.TMDB_DOCS,
        headers={"Authorization": f"Bearer {tmdb_bearer_token}"},
    )
    return Tool(
        name="TMDB API",
        description="Useful for when you want to get information from The Movie Database. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


@register_tool("wolfram-alpha", ["wolfram_alpha_appid"])
def _get_wolfram_alpha(**kwargs: Any) -> BaseTool:
    return WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper(**kwargs))


@register_tool("google-search", ["google_api_key", "google_cse_id"])
def _get_google_search(**kwargs: Any) -> BaseTool:
    return GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(**kwargs))


@register_tool("wikipedia", ["top_k_results"])
def _get_wikipedia(**kwargs: Any) -> BaseTool:
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(**kwargs))


@register_tool("google-search-results-json", ["serper_api_key"])
def _get_google_serper(**kwargs: Any) -> BaseTool:
    return Tool(
        name="Serper Search",
        func=GoogleSerperAPIWrapper(**kwargs).run,
        description="A low-cost Google Search API. Useful for when you need to answer questions about current events. Input should be a search query.",
    )


@register_tool(
    "google-search-results-json", ["google_api_key", "google_cse_id", "num_results"]
)
def _get_google_search_results_json(**kwargs: Any) -> BaseTool:
    return GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(**kwargs))


@register_tool("serpapi", ["serpapi_api_key", "aiosession"])
def _get_serpapi(**kwargs: Any) -> BaseTool:
    return Tool(
        name="Search",
        description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
        func=SerpAPIWrapper(**kwargs).run,
        coroutine=SerpAPIWrapper(**kwargs).arun,
    )


@register_tool("searx-search", ["searx_host"])
def _get_searx_search(**kwargs: Any) -> BaseTool:
    return Tool(
        name="SearX Search",
        description="A meta search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
        func=SearxSearchWrapper(**kwargs).run,
    )


@register_tool("bing-search", ["bing_subscription_key", "bing_search_url"])
def _get_bing_search(**kwargs: Any) -> BaseTool:
    return BingSearchRun(api_wrapper=BingSearchAPIWrapper(**kwargs))


def load_tools(
    tool_names: List[str],
    llm: Optional[BaseLLM] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> List[BaseTool]:
    """Load tools based on their name.

    Args:
        tool_names: name of tools to load.
        llm: Optional language model, may be needed to initialize certain tools.
        callback_manager: Optional callback manager. If not provided, default global callback manager will be used.

    Returns:
        List of tools.
    """
    tools = []
    for name in tool_names:
        if name in _TOOLS:
            _get_tool_func, extra_keys = _TOOLS[name]
        elif name in _LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            _get_tool_func, extra_keys = _LLM_TOOLS[name]
            extra_keys += ["llm"]
        else:
            raise ValueError(f"Got unknown tool {name}")

        missing_keys = set(extra_keys).difference(kwargs)
        if missing_keys:
            raise ValueError(
                f"Tool {name} requires some parameters that were not "
                f"provided: {missing_keys}"
            )
        sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}

        tool = _get_tool_func(**sub_kwargs)
        if callback_manager is not None:
            tool.callback_manager = callback_manager
        tools.append(tool)

    return tools


def get_all_tool_names() -> List[str]:
    """Get a list of all possible tool names."""
    return list(_TOOLS) + list(_LLM_TOOLS)
