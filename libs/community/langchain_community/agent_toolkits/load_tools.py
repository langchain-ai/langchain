# flake8: noqa
"""Tools provide access to various resources and services.

LangChain has a large ecosystem of integrations with various external resources
like local and remote file systems, APIs and databases.

These integrations allow developers to create versatile applications that combine the
power of LLMs with the ability to access, interact with and manipulate external
resources.

When developing an application, developers should inspect the capabilities and
permissions of the tools that underlie the given agent toolkit, and determine
whether permissions of the given toolkit are appropriate for the application.

See [Security](https://python.langchain.com/docs/security) for more information.
"""

import warnings
from typing import Any, Dict, List, Optional, Callable, Tuple

from mypy_extensions import Arg, KwArg

from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.bing_search.tool import BingSearchRun
from langchain_community.tools.dataforseo_api_search import DataForSeoAPISearchResults
from langchain_community.tools.dataforseo_api_search import DataForSeoAPISearchRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
from langchain_community.tools.file_management import ReadFileTool
from langchain_community.tools.golden_query.tool import GoldenQueryRun
from langchain_community.tools.google_cloud.texttospeech import (
    GoogleCloudTextToSpeechTool,
)
from langchain_community.tools.google_finance.tool import GoogleFinanceQueryRun
from langchain_community.tools.google_jobs.tool import GoogleJobsQueryRun
from langchain_community.tools.google_lens.tool import GoogleLensQueryRun
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.tools.google_search.tool import (
    GoogleSearchResults,
    GoogleSearchRun,
)
from langchain_community.tools.google_serper.tool import (
    GoogleSerperResults,
    GoogleSerperRun,
)
from langchain_community.tools.google_trends.tool import GoogleTrendsQueryRun
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from langchain_community.tools.human.tool import HumanInputRun
from langchain_community.tools.memorize.tool import Memorize
from langchain_community.tools.merriam_webster.tool import MerriamWebsterQueryRun
from langchain_community.tools.metaphor_search.tool import MetaphorSearchResults
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
)
from langchain_community.tools.scenexplain.tool import SceneXplainTool
from langchain_community.tools.searchapi.tool import SearchAPIResults, SearchAPIRun
from langchain_community.tools.searx_search.tool import (
    SearxSearchResults,
    SearxSearchRun,
)
from langchain_community.tools.shell.tool import ShellTool
from langchain_community.tools.sleep.tool import SleepTool
from langchain_community.tools.stackexchange.tool import StackExchangeTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.awslambda import LambdaWrapper
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.utilities.dataforseo_api_search import DataForSeoAPIWrapper
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper
from langchain_community.utilities.google_books import GoogleBooksAPIWrapper
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_community.utilities.graphql import GraphQLAPIWrapper
from langchain_community.utilities.merriam_webster import MerriamWebsterAPIWrapper
from langchain_community.utilities.metaphor_search import MetaphorSearchAPIWrapper
from langchain_community.utilities.openweathermap import OpenWeatherMapAPIWrapper
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.utilities.searchapi import SearchApiAPIWrapper
from langchain_community.utilities.searx_search import SearxSearchWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.utilities.stackexchange import StackExchangeAPIWrapper
from langchain_community.utilities.twilio import TwilioAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool, Tool


def _get_tools_requests_get() -> BaseTool:
    # Dangerous requests are allowed here, because there's another flag that the user
    # has to provide in order to actually opt in.
    # This is a private function and should not be used directly.
    return RequestsGetTool(
        requests_wrapper=TextRequestsWrapper(), allow_dangerous_requests=True
    )


def _get_tools_requests_post() -> BaseTool:
    # Dangerous requests are allowed here, because there's another flag that the user
    # has to provide in order to actually opt in.
    # This is a private function and should not be used directly.
    return RequestsPostTool(
        requests_wrapper=TextRequestsWrapper(), allow_dangerous_requests=True
    )


def _get_tools_requests_patch() -> BaseTool:
    # Dangerous requests are allowed here, because there's another flag that the user
    # has to provide in order to actually opt in.
    # This is a private function and should not be used directly.
    return RequestsPatchTool(
        requests_wrapper=TextRequestsWrapper(), allow_dangerous_requests=True
    )


def _get_tools_requests_put() -> BaseTool:
    # Dangerous requests are allowed here, because there's another flag that the user
    # has to provide in order to actually opt in.
    # This is a private function and should not be used directly.
    return RequestsPutTool(
        requests_wrapper=TextRequestsWrapper(), allow_dangerous_requests=True
    )


def _get_tools_requests_delete() -> BaseTool:
    # Dangerous requests are allowed here, because there's another flag that the user
    # has to provide in order to actually opt in.
    # This is a private function and should not be used directly.
    return RequestsDeleteTool(
        requests_wrapper=TextRequestsWrapper(), allow_dangerous_requests=True
    )


def _get_terminal() -> BaseTool:
    return ShellTool()


def _get_sleep() -> BaseTool:
    return SleepTool()


_BASE_TOOLS: Dict[str, Callable[[], BaseTool]] = {
    "sleep": _get_sleep,
}

DANGEROUS_TOOLS = {
    # Tools that contain some level of risk.
    # Please use with caution and read the documentation of these tools
    # to understand the risks and how to mitigate them.
    # Refer to https://python.langchain.com/docs/security
    # for more information.
    "requests": _get_tools_requests_get,  # preserved for backwards compatibility
    "requests_get": _get_tools_requests_get,
    "requests_post": _get_tools_requests_post,
    "requests_patch": _get_tools_requests_patch,
    "requests_put": _get_tools_requests_put,
    "requests_delete": _get_tools_requests_delete,
    "terminal": _get_terminal,
}


def _get_llm_math(llm: BaseLanguageModel) -> BaseTool:
    try:
        from langchain.chains.llm_math.base import LLMMathChain
    except ImportError:
        raise ImportError(
            "LLM Math tools require the library `langchain` to be installed."
            " Please install it with `pip install langchain`."
        )
    return Tool(
        name="Calculator",
        description="Useful for when you need to answer questions about math.",
        func=LLMMathChain.from_llm(llm=llm).run,
        coroutine=LLMMathChain.from_llm(llm=llm).arun,
    )


def _get_open_meteo_api(llm: BaseLanguageModel) -> BaseTool:
    try:
        from langchain.chains.api.base import APIChain
        from langchain.chains.api import (
            open_meteo_docs,
        )
    except ImportError:
        raise ImportError(
            "API tools require the library `langchain` to be installed."
            " Please install it with `pip install langchain`."
        )
    chain = APIChain.from_llm_and_api_docs(
        llm,
        open_meteo_docs.OPEN_METEO_DOCS,
        limit_to_domains=["https://api.open-meteo.com/"],
    )
    return Tool(
        name="Open-Meteo-API",
        description="Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


_LLM_TOOLS: Dict[str, Callable[[BaseLanguageModel], BaseTool]] = {
    "llm-math": _get_llm_math,
    "open-meteo-api": _get_open_meteo_api,
}


def _get_news_api(llm: BaseLanguageModel, **kwargs: Any) -> BaseTool:
    news_api_key = kwargs["news_api_key"]
    try:
        from langchain.chains.api.base import APIChain
        from langchain.chains.api import (
            news_docs,
        )
    except ImportError:
        raise ImportError(
            "API tools require the library `langchain` to be installed."
            " Please install it with `pip install langchain`."
        )
    chain = APIChain.from_llm_and_api_docs(
        llm,
        news_docs.NEWS_DOCS,
        headers={"X-Api-Key": news_api_key},
        limit_to_domains=["https://newsapi.org/"],
    )
    return Tool(
        name="News-API",
        description="Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


def _get_tmdb_api(llm: BaseLanguageModel, **kwargs: Any) -> BaseTool:
    tmdb_bearer_token = kwargs["tmdb_bearer_token"]
    try:
        from langchain.chains.api.base import APIChain
        from langchain.chains.api import (
            tmdb_docs,
        )
    except ImportError:
        raise ImportError(
            "API tools require the library `langchain` to be installed."
            " Please install it with `pip install langchain`."
        )
    chain = APIChain.from_llm_and_api_docs(
        llm,
        tmdb_docs.TMDB_DOCS,
        headers={"Authorization": f"Bearer {tmdb_bearer_token}"},
        limit_to_domains=["https://api.themoviedb.org/"],
    )
    return Tool(
        name="TMDB-API",
        description="Useful for when you want to get information from The Movie Database. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


def _get_podcast_api(llm: BaseLanguageModel, **kwargs: Any) -> BaseTool:
    listen_api_key = kwargs["listen_api_key"]
    try:
        from langchain.chains.api.base import APIChain
        from langchain.chains.api import (
            podcast_docs,
        )
    except ImportError:
        raise ImportError(
            "API tools require the library `langchain` to be installed."
            " Please install it with `pip install langchain`."
        )
    chain = APIChain.from_llm_and_api_docs(
        llm,
        podcast_docs.PODCAST_DOCS,
        headers={"X-ListenAPI-Key": listen_api_key},
        limit_to_domains=["https://listen-api.listennotes.com/"],
    )
    return Tool(
        name="Podcast-API",
        description="Use the Listen Notes Podcast API to search all podcasts or episodes. The input should be a question in natural language that this API can answer.",
        func=chain.run,
    )


def _get_lambda_api(**kwargs: Any) -> BaseTool:
    return Tool(
        name=kwargs["awslambda_tool_name"],
        description=kwargs["awslambda_tool_description"],
        func=LambdaWrapper(**kwargs).run,
    )


def _get_wolfram_alpha(**kwargs: Any) -> BaseTool:
    return WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper(**kwargs))


def _get_google_search(**kwargs: Any) -> BaseTool:
    return GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper(**kwargs))


def _get_merriam_webster(**kwargs: Any) -> BaseTool:
    return MerriamWebsterQueryRun(api_wrapper=MerriamWebsterAPIWrapper(**kwargs))


def _get_wikipedia(**kwargs: Any) -> BaseTool:
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(**kwargs))


def _get_arxiv(**kwargs: Any) -> BaseTool:
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(**kwargs))


def _get_golden_query(**kwargs: Any) -> BaseTool:
    return GoldenQueryRun(api_wrapper=GoldenQueryAPIWrapper(**kwargs))


def _get_pubmed(**kwargs: Any) -> BaseTool:
    return PubmedQueryRun(api_wrapper=PubMedAPIWrapper(**kwargs))


def _get_google_books(**kwargs: Any) -> BaseTool:
    from langchain_community.tools.google_books import GoogleBooksQueryRun

    return GoogleBooksQueryRun(api_wrapper=GoogleBooksAPIWrapper(**kwargs))


def _get_google_jobs(**kwargs: Any) -> BaseTool:
    return GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper(**kwargs))


def _get_google_lens(**kwargs: Any) -> BaseTool:
    return GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper(**kwargs))


def _get_google_serper(**kwargs: Any) -> BaseTool:
    return GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper(**kwargs))


def _get_google_scholar(**kwargs: Any) -> BaseTool:
    return GoogleScholarQueryRun(api_wrapper=GoogleScholarAPIWrapper(**kwargs))


def _get_google_finance(**kwargs: Any) -> BaseTool:
    return GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper(**kwargs))


def _get_google_trends(**kwargs: Any) -> BaseTool:
    return GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper(**kwargs))


def _get_google_serper_results_json(**kwargs: Any) -> BaseTool:
    return GoogleSerperResults(api_wrapper=GoogleSerperAPIWrapper(**kwargs))


def _get_google_search_results_json(**kwargs: Any) -> BaseTool:
    return GoogleSearchResults(api_wrapper=GoogleSearchAPIWrapper(**kwargs))


def _get_searchapi(**kwargs: Any) -> BaseTool:
    return SearchAPIRun(api_wrapper=SearchApiAPIWrapper(**kwargs))


def _get_searchapi_results_json(**kwargs: Any) -> BaseTool:
    return SearchAPIResults(api_wrapper=SearchApiAPIWrapper(**kwargs))


def _get_serpapi(**kwargs: Any) -> BaseTool:
    return Tool(
        name="Search",
        description="A search engine. Useful for when you need to answer questions about current events. Input should be a search query.",
        func=SerpAPIWrapper(**kwargs).run,
        coroutine=SerpAPIWrapper(**kwargs).arun,
    )


def _get_stackexchange(**kwargs: Any) -> BaseTool:
    return StackExchangeTool(api_wrapper=StackExchangeAPIWrapper(**kwargs))


def _get_dalle_image_generator(**kwargs: Any) -> Tool:
    return Tool(
        "Dall-E-Image-Generator",
        DallEAPIWrapper(**kwargs).run,
        "A wrapper around OpenAI DALL-E API. Useful for when you need to generate images from a text description. Input should be an image description.",
    )


def _get_twilio(**kwargs: Any) -> BaseTool:
    return Tool(
        name="Text-Message",
        description="Useful for when you need to send a text message to a provided phone number.",
        func=TwilioAPIWrapper(**kwargs).run,
    )


def _get_searx_search(**kwargs: Any) -> BaseTool:
    return SearxSearchRun(wrapper=SearxSearchWrapper(**kwargs))


def _get_searx_search_results_json(**kwargs: Any) -> BaseTool:
    wrapper_kwargs = {k: v for k, v in kwargs.items() if k != "num_results"}
    return SearxSearchResults(wrapper=SearxSearchWrapper(**wrapper_kwargs), **kwargs)


def _get_bing_search(**kwargs: Any) -> BaseTool:
    return BingSearchRun(api_wrapper=BingSearchAPIWrapper(**kwargs))


def _get_metaphor_search(**kwargs: Any) -> BaseTool:
    return MetaphorSearchResults(api_wrapper=MetaphorSearchAPIWrapper(**kwargs))


def _get_ddg_search(**kwargs: Any) -> BaseTool:
    return DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(**kwargs))


def _get_human_tool(**kwargs: Any) -> BaseTool:
    return HumanInputRun(**kwargs)


def _get_scenexplain(**kwargs: Any) -> BaseTool:
    return SceneXplainTool(**kwargs)


def _get_graphql_tool(**kwargs: Any) -> BaseTool:
    return BaseGraphQLTool(graphql_wrapper=GraphQLAPIWrapper(**kwargs))


def _get_openweathermap(**kwargs: Any) -> BaseTool:
    return OpenWeatherMapQueryRun(api_wrapper=OpenWeatherMapAPIWrapper(**kwargs))


def _get_dataforseo_api_search(**kwargs: Any) -> BaseTool:
    return DataForSeoAPISearchRun(api_wrapper=DataForSeoAPIWrapper(**kwargs))


def _get_dataforseo_api_search_json(**kwargs: Any) -> BaseTool:
    return DataForSeoAPISearchResults(api_wrapper=DataForSeoAPIWrapper(**kwargs))


def _get_eleven_labs_text2speech(**kwargs: Any) -> BaseTool:
    return ElevenLabsText2SpeechTool(**kwargs)


def _get_memorize(llm: BaseLanguageModel, **kwargs: Any) -> BaseTool:
    return Memorize(llm=llm)  # type: ignore[arg-type]


def _get_google_cloud_texttospeech(**kwargs: Any) -> BaseTool:
    return GoogleCloudTextToSpeechTool(**kwargs)


def _get_file_management_tool(**kwargs: Any) -> BaseTool:
    return ReadFileTool(**kwargs)


def _get_reddit_search(**kwargs: Any) -> BaseTool:
    return RedditSearchRun(api_wrapper=RedditSearchAPIWrapper(**kwargs))


_EXTRA_LLM_TOOLS: Dict[
    str,
    Tuple[Callable[[Arg(BaseLanguageModel, "llm"), KwArg(Any)], BaseTool], List[str]],
] = {
    "news-api": (_get_news_api, ["news_api_key"]),
    "tmdb-api": (_get_tmdb_api, ["tmdb_bearer_token"]),
    "podcast-api": (_get_podcast_api, ["listen_api_key"]),
    "memorize": (_get_memorize, []),
}
_EXTRA_OPTIONAL_TOOLS: Dict[str, Tuple[Callable[[KwArg(Any)], BaseTool], List[str]]] = {
    "wolfram-alpha": (_get_wolfram_alpha, ["wolfram_alpha_appid"]),
    "google-search": (_get_google_search, ["google_api_key", "google_cse_id"]),
    "google-search-results-json": (
        _get_google_search_results_json,
        ["google_api_key", "google_cse_id", "num_results"],
    ),
    "searx-search-results-json": (
        _get_searx_search_results_json,
        ["searx_host", "engines", "num_results", "aiosession"],
    ),
    "bing-search": (_get_bing_search, ["bing_subscription_key", "bing_search_url"]),
    "metaphor-search": (_get_metaphor_search, ["metaphor_api_key"]),
    "ddg-search": (_get_ddg_search, []),
    "google-books": (_get_google_books, ["google_books_api_key"]),
    "google-lens": (_get_google_lens, ["serp_api_key"]),
    "google-serper": (_get_google_serper, ["serper_api_key", "aiosession"]),
    "google-scholar": (
        _get_google_scholar,
        ["top_k_results", "hl", "lr", "serp_api_key"],
    ),
    "google-finance": (
        _get_google_finance,
        ["serp_api_key"],
    ),
    "google-trends": (
        _get_google_trends,
        ["serp_api_key"],
    ),
    "google-jobs": (
        _get_google_jobs,
        ["serp_api_key"],
    ),
    "google-serper-results-json": (
        _get_google_serper_results_json,
        ["serper_api_key", "aiosession"],
    ),
    "searchapi": (_get_searchapi, ["searchapi_api_key", "aiosession"]),
    "searchapi-results-json": (
        _get_searchapi_results_json,
        ["searchapi_api_key", "aiosession"],
    ),
    "serpapi": (_get_serpapi, ["serpapi_api_key", "aiosession"]),
    "dalle-image-generator": (_get_dalle_image_generator, ["openai_api_key"]),
    "twilio": (_get_twilio, ["account_sid", "auth_token", "from_number"]),
    "searx-search": (_get_searx_search, ["searx_host", "engines", "aiosession"]),
    "merriam-webster": (_get_merriam_webster, ["merriam_webster_api_key"]),
    "wikipedia": (_get_wikipedia, ["top_k_results", "lang"]),
    "arxiv": (
        _get_arxiv,
        ["top_k_results", "load_max_docs", "load_all_available_meta"],
    ),
    "golden-query": (_get_golden_query, ["golden_api_key"]),
    "pubmed": (_get_pubmed, ["top_k_results"]),
    "human": (_get_human_tool, ["prompt_func", "input_func"]),
    "awslambda": (
        _get_lambda_api,
        ["awslambda_tool_name", "awslambda_tool_description", "function_name"],
    ),
    "stackexchange": (_get_stackexchange, []),
    "sceneXplain": (_get_scenexplain, []),
    "graphql": (
        _get_graphql_tool,
        ["graphql_endpoint", "custom_headers", "fetch_schema_from_transport"],
    ),
    "openweathermap-api": (_get_openweathermap, ["openweathermap_api_key"]),
    "dataforseo-api-search": (
        _get_dataforseo_api_search,
        ["api_login", "api_password", "aiosession"],
    ),
    "dataforseo-api-search-json": (
        _get_dataforseo_api_search_json,
        ["api_login", "api_password", "aiosession"],
    ),
    "eleven_labs_text2speech": (_get_eleven_labs_text2speech, ["eleven_api_key"]),
    "google_cloud_texttospeech": (_get_google_cloud_texttospeech, []),
    "read_file": (_get_file_management_tool, []),
    "reddit_search": (
        _get_reddit_search,
        ["reddit_client_id", "reddit_client_secret", "reddit_user_agent"],
    ),
}


def _handle_callbacks(
    callback_manager: Optional[BaseCallbackManager], callbacks: Callbacks
) -> Callbacks:
    if callback_manager is not None:
        warnings.warn(
            "callback_manager is deprecated. Please use callbacks instead.",
            DeprecationWarning,
        )
        if callbacks is not None:
            raise ValueError(
                "Cannot specify both callback_manager and callbacks arguments."
            )
        return callback_manager
    return callbacks


def load_huggingface_tool(
    task_or_repo_id: str,
    model_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    remote: bool = False,
    **kwargs: Any,
) -> BaseTool:
    """Loads a tool from the HuggingFace Hub.

    Args:
        task_or_repo_id: Task or model repo id.
        model_repo_id: Optional model repo id. Defaults to None.
        token: Optional token. Defaults to None.
        remote: Optional remote. Defaults to False.
        kwargs: Additional keyword arguments.

    Returns:
        A tool.

    Raises:
        ImportError: If the required libraries are not installed.
        NotImplementedError: If multimodal outputs or inputs are not supported.
    """
    try:
        from transformers import load_tool
    except ImportError:
        raise ImportError(
            "HuggingFace tools require the libraries `transformers>=4.29.0`"
            " and `huggingface_hub>=0.14.1` to be installed."
            " Please install it with"
            " `pip install --upgrade transformers huggingface_hub`."
        )
    hf_tool = load_tool(
        task_or_repo_id,
        model_repo_id=model_repo_id,
        token=token,
        remote=remote,
        **kwargs,
    )
    outputs = hf_tool.outputs
    if set(outputs) != {"text"}:
        raise NotImplementedError("Multimodal outputs not supported yet.")
    inputs = hf_tool.inputs
    if set(inputs) != {"text"}:
        raise NotImplementedError("Multimodal inputs not supported yet.")
    return Tool.from_function(
        hf_tool.__call__, name=hf_tool.name, description=hf_tool.description
    )


def raise_dangerous_tools_exception(name: str) -> None:
    raise ValueError(
        f"{name} is a dangerous tool. You cannot use it without opting in "
        "by setting allow_dangerous_tools to True. "
        "Most tools have some inherit risk to them merely because they are "
        'allowed to interact with the "real world".'
        "Please refer to LangChain security guidelines "
        "to https://python.langchain.com/docs/security."
        "Some tools have been designated as dangerous because they pose "
        "risk that is not intuitively obvious. For example, a tool that "
        "allows an agent to make requests to the web, can also be used "
        "to make requests to a server that is only accessible from the "
        "server hosting the code."
        "Again, all tools carry some risk, and it's your responsibility to "
        "understand which tools you're using and the risks associated with "
        "them."
    )


def load_tools(
    tool_names: List[str],
    llm: Optional[BaseLanguageModel] = None,
    callbacks: Callbacks = None,
    allow_dangerous_tools: bool = False,
    **kwargs: Any,
) -> List[BaseTool]:
    """Load tools based on their name.

    Tools allow agents to interact with various resources and services like
    APIs, databases, file systems, etc.

    Please scope the permissions of each tools to the minimum required for the
    application.

    For example, if an application only needs to read from a database,
    the database tool should not be given write permissions. Moreover
    consider scoping the permissions to only allow accessing specific
    tables and impose user-level quota for limiting resource usage.

    Please read the APIs of the individual tools to determine which configuration
    they support.

    See [Security](https://python.langchain.com/docs/security) for more information.

    Args:
        tool_names: name of tools to load.
        llm: An optional language model may be needed to initialize certain tools.
            Defaults to None.
        callbacks: Optional callback manager or list of callback handlers.
            If not provided, default global callback manager will be used.
        allow_dangerous_tools: Optional flag to allow dangerous tools.
            Tools that contain some level of risk.
            Please use with caution and read the documentation of these tools
            to understand the risks and how to mitigate them.
            Refer to https://python.langchain.com/docs/security
            for more information.
            Please note that this list may not be fully exhaustive.
            It is your responsibility to understand which tools
            you're using and the risks associated with them.
            Defaults to False.
        kwargs: Additional keyword arguments.

    Returns:
        List of tools.

    Raises:
        ValueError: If the tool name is unknown.
        ValueError: If the tool requires an LLM to be provided.
        ValueError: If the tool requires some parameters that were not provided.
        ValueError: If the tool is a dangerous tool and allow_dangerous_tools is False.
    """
    tools = []
    callbacks = _handle_callbacks(
        callback_manager=kwargs.get("callback_manager"), callbacks=callbacks
    )
    for name in tool_names:
        if name in DANGEROUS_TOOLS and not allow_dangerous_tools:
            raise_dangerous_tools_exception(name)

        if name in {"requests"}:
            warnings.warn(
                "tool name `requests` is deprecated - "
                "please use `requests_all` or specify the requests method"
            )
        if name == "requests_all":
            # expand requests into various methods
            if not allow_dangerous_tools:
                raise_dangerous_tools_exception(name)
            requests_method_tools = [
                _tool for _tool in DANGEROUS_TOOLS if _tool.startswith("requests_")
            ]
            tool_names.extend(requests_method_tools)
        elif name in _BASE_TOOLS:
            tools.append(_BASE_TOOLS[name]())
        elif name in DANGEROUS_TOOLS:
            tools.append(DANGEROUS_TOOLS[name]())
        elif name in _LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            tool = _LLM_TOOLS[name](llm)
            tools.append(tool)
        elif name in _EXTRA_LLM_TOOLS:
            if llm is None:
                raise ValueError(f"Tool {name} requires an LLM to be provided")
            _get_llm_tool_func, extra_keys = _EXTRA_LLM_TOOLS[name]
            missing_keys = set(extra_keys).difference(kwargs)
            if missing_keys:
                raise ValueError(
                    f"Tool {name} requires some parameters that were not "
                    f"provided: {missing_keys}"
                )
            sub_kwargs = {k: kwargs[k] for k in extra_keys}
            tool = _get_llm_tool_func(llm=llm, **sub_kwargs)
            tools.append(tool)
        elif name in _EXTRA_OPTIONAL_TOOLS:
            _get_tool_func, extra_keys = _EXTRA_OPTIONAL_TOOLS[name]
            sub_kwargs = {k: kwargs[k] for k in extra_keys if k in kwargs}
            tool = _get_tool_func(**sub_kwargs)
            tools.append(tool)
        else:
            raise ValueError(f"Got unknown tool {name}")
    if callbacks is not None:
        for tool in tools:
            tool.callbacks = callbacks
    return tools


def get_all_tool_names() -> List[str]:
    """Get a list of all possible tool names."""
    return (
        list(_BASE_TOOLS)
        + list(_EXTRA_OPTIONAL_TOOLS)
        + list(_EXTRA_LLM_TOOLS)
        + list(_LLM_TOOLS)
        + list(DANGEROUS_TOOLS)
    )
