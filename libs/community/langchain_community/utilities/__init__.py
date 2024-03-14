"""**Utilities** are the integrations with third-part systems and packages.

Other LangChain classes use **Utilities** to interact with third-part systems
and packages.
"""
import importlib
from typing import Any

_module_lookup = {
    "AlphaVantageAPIWrapper": "langchain_community.utilities.alpha_vantage",
    "ApifyWrapper": "langchain_community.utilities.apify",
    "ArceeWrapper": "langchain_community.utilities.arcee",
    "ArxivAPIWrapper": "langchain_community.utilities.arxiv",
    "AudioStream": "langchain_community.utilities.nvidia_riva",
    "BibtexparserWrapper": "langchain_community.utilities.bibtex",
    "BingSearchAPIWrapper": "langchain_community.utilities.bing_search",
    "BraveSearchWrapper": "langchain_community.utilities.brave_search",
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
