"""**Utilities** are the integrations with third-part systems and packages.

Other LangChain classes use **Utilities** to interact with third-part systems
and packages.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.utilities.alpha_vantage import (
        AlphaVantageAPIWrapper,
    )
    from langchain_community.utilities.apify import (
        ApifyWrapper,
    )
    from langchain_community.utilities.arcee import (
        ArceeWrapper,
    )
    from langchain_community.utilities.arxiv import (
        ArxivAPIWrapper,
    )
    from langchain_community.utilities.asknews import (
        AskNewsAPIWrapper,
    )
    from langchain_community.utilities.awslambda import (
        LambdaWrapper,
    )
    from langchain_community.utilities.bibtex import (
        BibtexparserWrapper,
    )
    from langchain_community.utilities.bing_search import (
        BingSearchAPIWrapper,
    )
    from langchain_community.utilities.brave_search import (
        BraveSearchWrapper,
    )
    from langchain_community.utilities.dataherald import DataheraldAPIWrapper
    from langchain_community.utilities.dria_index import (
        DriaAPIWrapper,
    )
    from langchain_community.utilities.duckduckgo_search import (
        DuckDuckGoSearchAPIWrapper,
    )
    from langchain_community.utilities.golden_query import (
        GoldenQueryAPIWrapper,
    )
    from langchain_community.utilities.google_books import (
        GoogleBooksAPIWrapper,
    )
    from langchain_community.utilities.google_finance import (
        GoogleFinanceAPIWrapper,
    )
    from langchain_community.utilities.google_jobs import (
        GoogleJobsAPIWrapper,
    )
    from langchain_community.utilities.google_lens import (
        GoogleLensAPIWrapper,
    )
    from langchain_community.utilities.google_places_api import (
        GooglePlacesAPIWrapper,
    )
    from langchain_community.utilities.google_scholar import (
        GoogleScholarAPIWrapper,
    )
    from langchain_community.utilities.google_search import (
        GoogleSearchAPIWrapper,
    )
    from langchain_community.utilities.google_serper import (
        GoogleSerperAPIWrapper,
    )
    from langchain_community.utilities.google_trends import (
        GoogleTrendsAPIWrapper,
    )
    from langchain_community.utilities.graphql import (
        GraphQLAPIWrapper,
    )
    from langchain_community.utilities.infobip import (
        InfobipAPIWrapper,
    )
    from langchain_community.utilities.jira import (
        JiraAPIWrapper,
    )
    from langchain_community.utilities.max_compute import (
        MaxComputeAPIWrapper,
    )
    from langchain_community.utilities.merriam_webster import (
        MerriamWebsterAPIWrapper,
    )
    from langchain_community.utilities.metaphor_search import (
        MetaphorSearchAPIWrapper,
    )
    from langchain_community.utilities.mindsdb import AIDataMindWrapper
    from langchain_community.utilities.mojeek_search import (
        MojeekSearchAPIWrapper,
    )
    from langchain_community.utilities.nasa import (
        NasaAPIWrapper,
    )
    from langchain_community.utilities.nvidia_riva import (
        AudioStream,
        NVIDIARivaASR,
        NVIDIARivaStream,
        NVIDIARivaTTS,
        RivaASR,
        RivaTTS,
    )
    from langchain_community.utilities.openweathermap import (
        OpenWeatherMapAPIWrapper,
    )
    from langchain_community.utilities.oracleai import (
        OracleSummary,
    )
    from langchain_community.utilities.outline import (
        OutlineAPIWrapper,
    )
    from langchain_community.utilities.passio_nutrition_ai import (
        NutritionAIAPI,
    )
    from langchain_community.utilities.portkey import (
        Portkey,
    )
    from langchain_community.utilities.powerbi import (
        PowerBIDataset,
    )
    from langchain_community.utilities.pubmed import (
        PubMedAPIWrapper,
    )
    from langchain_community.utilities.rememberizer import RememberizerAPIWrapper
    from langchain_community.utilities.requests import (
        Requests,
        RequestsWrapper,
        TextRequestsWrapper,
    )
    from langchain_community.utilities.scenexplain import (
        SceneXplainAPIWrapper,
    )
    from langchain_community.utilities.searchapi import (
        SearchApiAPIWrapper,
    )
    from langchain_community.utilities.searx_search import (
        SearxSearchWrapper,
    )
    from langchain_community.utilities.serpapi import (
        SerpAPIWrapper,
    )
    from langchain_community.utilities.spark_sql import (
        SparkSQL,
    )
    from langchain_community.utilities.sql_database import (
        SQLDatabase,
    )
    from langchain_community.utilities.stackexchange import (
        StackExchangeAPIWrapper,
    )
    from langchain_community.utilities.steam import (
        SteamWebAPIWrapper,
    )
    from langchain_community.utilities.tensorflow_datasets import (
        TensorflowDatasets,
    )
    from langchain_community.utilities.twilio import (
        TwilioAPIWrapper,
    )
    from langchain_community.utilities.wikipedia import (
        WikipediaAPIWrapper,
    )
    from langchain_community.utilities.wolfram_alpha import (
        WolframAlphaAPIWrapper,
    )
    from langchain_community.utilities.you import (
        YouSearchAPIWrapper,
    )
    from langchain_community.utilities.zapier import (
        ZapierNLAWrapper,
    )

__all__ = [
    "AIDataMindWrapper",
    "AlphaVantageAPIWrapper",
    "ApifyWrapper",
    "ArceeWrapper",
    "ArxivAPIWrapper",
    "AskNewsAPIWrapper",
    "AudioStream",
    "BibtexparserWrapper",
    "BingSearchAPIWrapper",
    "BraveSearchWrapper",
    "DataheraldAPIWrapper",
    "DriaAPIWrapper",
    "DuckDuckGoSearchAPIWrapper",
    "GoldenQueryAPIWrapper",
    "GoogleBooksAPIWrapper",
    "GoogleFinanceAPIWrapper",
    "GoogleJobsAPIWrapper",
    "GoogleLensAPIWrapper",
    "GooglePlacesAPIWrapper",
    "GoogleScholarAPIWrapper",
    "GoogleSearchAPIWrapper",
    "GoogleSerperAPIWrapper",
    "GoogleTrendsAPIWrapper",
    "GraphQLAPIWrapper",
    "InfobipAPIWrapper",
    "JiraAPIWrapper",
    "LambdaWrapper",
    "MaxComputeAPIWrapper",
    "MerriamWebsterAPIWrapper",
    "MetaphorSearchAPIWrapper",
    "MojeekSearchAPIWrapper",
    "NVIDIARivaASR",
    "NVIDIARivaStream",
    "NVIDIARivaTTS",
    "NasaAPIWrapper",
    "NutritionAIAPI",
    "OpenWeatherMapAPIWrapper",
    "OracleSummary",
    "OutlineAPIWrapper",
    "Portkey",
    "PowerBIDataset",
    "PubMedAPIWrapper",
    "RememberizerAPIWrapper",
    "Requests",
    "RequestsWrapper",
    "RivaASR",
    "RivaTTS",
    "SceneXplainAPIWrapper",
    "SearchApiAPIWrapper",
    "SQLDatabase",
    "SearxSearchWrapper",
    "SerpAPIWrapper",
    "SparkSQL",
    "StackExchangeAPIWrapper",
    "SteamWebAPIWrapper",
    "TensorflowDatasets",
    "TextRequestsWrapper",
    "TwilioAPIWrapper",
    "WikipediaAPIWrapper",
    "WolframAlphaAPIWrapper",
    "YouSearchAPIWrapper",
    "ZapierNLAWrapper",
]

_module_lookup = {
    "AIDataMindWrapper": (
        "langchain_community.utilities.mindsdb.ai_data_mind.ai_data_mind_wrapper"
    ),
    "AlphaVantageAPIWrapper": "langchain_community.utilities.alpha_vantage",
    "ApifyWrapper": "langchain_community.utilities.apify",
    "ArceeWrapper": "langchain_community.utilities.arcee",
    "ArxivAPIWrapper": "langchain_community.utilities.arxiv",
    "AskNewsAPIWrapper": "langchain_community.utilities.asknews",
    "AudioStream": "langchain_community.utilities.nvidia_riva",
    "BibtexparserWrapper": "langchain_community.utilities.bibtex",
    "BingSearchAPIWrapper": "langchain_community.utilities.bing_search",
    "BraveSearchWrapper": "langchain_community.utilities.brave_search",
    "DataheraldAPIWrapper": "langchain_community.utilities.dataherald",
    "DriaAPIWrapper": "langchain_community.utilities.dria_index",
    "DuckDuckGoSearchAPIWrapper": "langchain_community.utilities.duckduckgo_search",
    "GoldenQueryAPIWrapper": "langchain_community.utilities.golden_query",
    "GoogleBooksAPIWrapper": "langchain_community.utilities.google_books",
    "GoogleFinanceAPIWrapper": "langchain_community.utilities.google_finance",
    "GoogleJobsAPIWrapper": "langchain_community.utilities.google_jobs",
    "GoogleLensAPIWrapper": "langchain_community.utilities.google_lens",
    "GooglePlacesAPIWrapper": "langchain_community.utilities.google_places_api",
    "GoogleScholarAPIWrapper": "langchain_community.utilities.google_scholar",
    "GoogleSearchAPIWrapper": "langchain_community.utilities.google_search",
    "GoogleSerperAPIWrapper": "langchain_community.utilities.google_serper",
    "GoogleTrendsAPIWrapper": "langchain_community.utilities.google_trends",
    "GraphQLAPIWrapper": "langchain_community.utilities.graphql",
    "InfobipAPIWrapper": "langchain_community.utilities.infobip",
    "JiraAPIWrapper": "langchain_community.utilities.jira",
    "LambdaWrapper": "langchain_community.utilities.awslambda",
    "MaxComputeAPIWrapper": "langchain_community.utilities.max_compute",
    "MerriamWebsterAPIWrapper": "langchain_community.utilities.merriam_webster",
    "MetaphorSearchAPIWrapper": "langchain_community.utilities.metaphor_search",
    "MojeekSearchAPIWrapper": "langchain_community.utilities.mojeek_search",
    "NVIDIARivaASR": "langchain_community.utilities.nvidia_riva",
    "NVIDIARivaStream": "langchain_community.utilities.nvidia_riva",
    "NVIDIARivaTTS": "langchain_community.utilities.nvidia_riva",
    "NasaAPIWrapper": "langchain_community.utilities.nasa",
    "NutritionAIAPI": "langchain_community.utilities.passio_nutrition_ai",
    "OpenWeatherMapAPIWrapper": "langchain_community.utilities.openweathermap",
    "OracleSummary": "langchain_community.utilities.oracleai",
    "OutlineAPIWrapper": "langchain_community.utilities.outline",
    "Portkey": "langchain_community.utilities.portkey",
    "PowerBIDataset": "langchain_community.utilities.powerbi",
    "PubMedAPIWrapper": "langchain_community.utilities.pubmed",
    "RememberizerAPIWrapper": "langchain_community.utilities.rememberizer",
    "Requests": "langchain_community.utilities.requests",
    "RequestsWrapper": "langchain_community.utilities.requests",
    "RivaASR": "langchain_community.utilities.nvidia_riva",
    "RivaTTS": "langchain_community.utilities.nvidia_riva",
    "SQLDatabase": "langchain_community.utilities.sql_database",
    "SceneXplainAPIWrapper": "langchain_community.utilities.scenexplain",
    "SearchApiAPIWrapper": "langchain_community.utilities.searchapi",
    "SearxSearchWrapper": "langchain_community.utilities.searx_search",
    "SerpAPIWrapper": "langchain_community.utilities.serpapi",
    "SparkSQL": "langchain_community.utilities.spark_sql",
    "StackExchangeAPIWrapper": "langchain_community.utilities.stackexchange",
    "SteamWebAPIWrapper": "langchain_community.utilities.steam",
    "TensorflowDatasets": "langchain_community.utilities.tensorflow_datasets",
    "TextRequestsWrapper": "langchain_community.utilities.requests",
    "TwilioAPIWrapper": "langchain_community.utilities.twilio",
    "WikipediaAPIWrapper": "langchain_community.utilities.wikipedia",
    "WolframAlphaAPIWrapper": "langchain_community.utilities.wolfram_alpha",
    "YouSearchAPIWrapper": "langchain_community.utilities.you",
    "ZapierNLAWrapper": "langchain_community.utilities.zapier",
}

REMOVED = {
    "PythonREPL": (
        "PythonREPL has been deprecated from langchain_community "
        "due to being flagged by security scanners. See: "
        "https://github.com/langchain-ai/langchain/issues/14345 "
        "If you need to use it, please use the version "
        "from langchain_experimental. "
        "from langchain_experimental.utilities.python import PythonREPL."
    )
}


def __getattr__(name: str) -> Any:
    if name in REMOVED:
        raise AssertionError(REMOVED[name])
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
