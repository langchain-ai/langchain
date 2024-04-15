"""**Utilities** are the integrations with third-part systems and packages.

Other LangChain classes use **Utilities** to interact with third-part systems
and packages.
"""
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.utilities.alpha_vantage import (
        AlphaVantageAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.apify import (
        ApifyWrapper,  # noqa: F401
    )
    from langchain_community.utilities.arcee import (
        ArceeWrapper,  # noqa: F401
    )
    from langchain_community.utilities.arxiv import (
        ArxivAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.awslambda import (
        LambdaWrapper,  # noqa: F401
    )
    from langchain_community.utilities.bibtex import (
        BibtexparserWrapper,  # noqa: F401
    )
    from langchain_community.utilities.bing_search import (
        BingSearchAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.brave_search import (
        BraveSearchWrapper,  # noqa: F401
    )
    from langchain_community.utilities.dria_index import (
        DriaAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.duckduckgo_search import (
        DuckDuckGoSearchAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.golden_query import (
        GoldenQueryAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_finance import (
        GoogleFinanceAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_jobs import (
        GoogleJobsAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_lens import (
        GoogleLensAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_places_api import (
        GooglePlacesAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_scholar import (
        GoogleScholarAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_search import (
        GoogleSearchAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_serper import (
        GoogleSerperAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.google_trends import (
        GoogleTrendsAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.graphql import (
        GraphQLAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.infobip import (
        InfobipAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.jira import (
        JiraAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.max_compute import (
        MaxComputeAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.merriam_webster import (
        MerriamWebsterAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.metaphor_search import (
        MetaphorSearchAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.nasa import (
        NasaAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.nvidia_riva import (
        AudioStream,  # noqa: F401
        NVIDIARivaASR,  # noqa: F401
        NVIDIARivaStream,  # noqa: F401
        NVIDIARivaTTS,  # noqa: F401
        RivaASR,  # noqa: F401
        RivaTTS,  # noqa: F401
    )
    from langchain_community.utilities.openweathermap import (
        OpenWeatherMapAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.outline import (
        OutlineAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.passio_nutrition_ai import (
        NutritionAIAPI,  # noqa: F401
    )
    from langchain_community.utilities.portkey import (
        Portkey,  # noqa: F401
    )
    from langchain_community.utilities.powerbi import (
        PowerBIDataset,  # noqa: F401
    )
    from langchain_community.utilities.pubmed import (
        PubMedAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.python import (
        PythonREPL,  # noqa: F401
    )
    from langchain_community.utilities.requests import (
        Requests,  # noqa: F401
        RequestsWrapper,  # noqa: F401
        TextRequestsWrapper,  # noqa: F401
    )
    from langchain_community.utilities.scenexplain import (
        SceneXplainAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.searchapi import (
        SearchApiAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.searx_search import (
        SearxSearchWrapper,  # noqa: F401
    )
    from langchain_community.utilities.serpapi import (
        SerpAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.spark_sql import (
        SparkSQL,  # noqa: F401
    )
    from langchain_community.utilities.sql_database import (
        SQLDatabase,  # noqa: F401
    )
    from langchain_community.utilities.stackexchange import (
        StackExchangeAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.steam import (
        SteamWebAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.tensorflow_datasets import (
        TensorflowDatasets,  # noqa: F401
    )
    from langchain_community.utilities.twilio import (
        TwilioAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.wikipedia import (
        WikipediaAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.wolfram_alpha import (
        WolframAlphaAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.you import (
        YouSearchAPIWrapper,  # noqa: F401
    )
    from langchain_community.utilities.zapier import (
        ZapierNLAWrapper,  # noqa: F401
    )

__all__ = [
    "AlphaVantageAPIWrapper",
    "ApifyWrapper",
    "ArceeWrapper",
    "ArxivAPIWrapper",
    "AudioStream",
    "BibtexparserWrapper",
    "BingSearchAPIWrapper",
    "BraveSearchWrapper",
    "DriaAPIWrapper",
    "DuckDuckGoSearchAPIWrapper",
    "GoldenQueryAPIWrapper",
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
    "NVIDIARivaASR",
    "NVIDIARivaStream",
    "NVIDIARivaTTS",
    "NasaAPIWrapper",
    "NutritionAIAPI",
    "OpenWeatherMapAPIWrapper",
    "OutlineAPIWrapper",
    "Portkey",
    "PowerBIDataset",
    "PubMedAPIWrapper",
    "PythonREPL",
    "Requests",
    "RequestsWrapper",
    "RivaASR",
    "RivaTTS",
    "SQLDatabase",
    "SceneXplainAPIWrapper",
    "SearchApiAPIWrapper",
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
    "AlphaVantageAPIWrapper": "langchain_community.utilities.alpha_vantage",
    "ApifyWrapper": "langchain_community.utilities.apify",
    "ArceeWrapper": "langchain_community.utilities.arcee",
    "ArxivAPIWrapper": "langchain_community.utilities.arxiv",
    "AudioStream": "langchain_community.utilities.nvidia_riva",
    "BibtexparserWrapper": "langchain_community.utilities.bibtex",
    "BingSearchAPIWrapper": "langchain_community.utilities.bing_search",
    "BraveSearchWrapper": "langchain_community.utilities.brave_search",
    "DataheraldAPIWrapper": "langchain_community.utilities.dataherald",
    "DriaAPIWrapper": "langchain_community.utilities.dria_index",
    "DuckDuckGoSearchAPIWrapper": "langchain_community.utilities.duckduckgo_search",
    "GoldenQueryAPIWrapper": "langchain_community.utilities.golden_query",
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
    "NVIDIARivaASR": "langchain_community.utilities.nvidia_riva",
    "NVIDIARivaStream": "langchain_community.utilities.nvidia_riva",
    "NVIDIARivaTTS": "langchain_community.utilities.nvidia_riva",
    "NasaAPIWrapper": "langchain_community.utilities.nasa",
    "NutritionAIAPI": "langchain_community.utilities.passio_nutrition_ai",
    "OpenWeatherMapAPIWrapper": "langchain_community.utilities.openweathermap",
    "OutlineAPIWrapper": "langchain_community.utilities.outline",
    "Portkey": "langchain_community.utilities.portkey",
    "PowerBIDataset": "langchain_community.utilities.powerbi",
    "PubMedAPIWrapper": "langchain_community.utilities.pubmed",
    "PythonREPL": "langchain_community.utilities.python",
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


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
