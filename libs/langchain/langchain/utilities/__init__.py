"""**Utilities** are the integrations with third-part systems and packages.

Other LangChain classes use **Utilities** to interact with third-part systems
and packages.
"""
from typing import Any

from langchain.utilities.requests import Requests, RequestsWrapper, TextRequestsWrapper


def _import_alpha_vantage() -> Any:
    from langchain.utilities.alpha_vantage import AlphaVantageAPIWrapper

    return AlphaVantageAPIWrapper


def _import_apify() -> Any:
    from langchain.utilities.apify import ApifyWrapper

    return ApifyWrapper


def _import_arcee() -> Any:
    from langchain.utilities.arcee import ArceeWrapper

    return ArceeWrapper


def _import_arxiv() -> Any:
    from langchain.utilities.arxiv import ArxivAPIWrapper

    return ArxivAPIWrapper


def _import_awslambda() -> Any:
    from langchain.utilities.awslambda import LambdaWrapper

    return LambdaWrapper


def _import_bibtex() -> Any:
    from langchain.utilities.bibtex import BibtexparserWrapper

    return BibtexparserWrapper


def _import_bing_search() -> Any:
    from langchain.utilities.bing_search import BingSearchAPIWrapper

    return BingSearchAPIWrapper


def _import_brave_search() -> Any:
    from langchain.utilities.brave_search import BraveSearchWrapper

    return BraveSearchWrapper


def _import_duckduckgo_search() -> Any:
    from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

    return DuckDuckGoSearchAPIWrapper


def _import_golden_query() -> Any:
    from langchain.utilities.golden_query import GoldenQueryAPIWrapper

    return GoldenQueryAPIWrapper


def _import_google_lens() -> Any:
    from langchain.utilities.google_lens import GoogleLensAPIWrapper

    return GoogleLensAPIWrapper


def _import_google_places_api() -> Any:
    from langchain.utilities.google_places_api import GooglePlacesAPIWrapper

    return GooglePlacesAPIWrapper


def _import_google_jobs() -> Any:
    from langchain.utilities.google_jobs import GoogleJobsAPIWrapper

    return GoogleJobsAPIWrapper


def _import_google_scholar() -> Any:
    from langchain.utilities.google_scholar import GoogleScholarAPIWrapper

    return GoogleScholarAPIWrapper


def _import_google_trends() -> Any:
    from langchain.utilities.google_trends import GoogleTrendsAPIWrapper

    return GoogleTrendsAPIWrapper


def _import_google_finance() -> Any:
    from langchain.utilities.google_finance import GoogleFinanceAPIWrapper

    return GoogleFinanceAPIWrapper


def _import_google_search() -> Any:
    from langchain.utilities.google_search import GoogleSearchAPIWrapper

    return GoogleSearchAPIWrapper


def _import_google_serper() -> Any:
    from langchain.utilities.google_serper import GoogleSerperAPIWrapper

    return GoogleSerperAPIWrapper


def _import_graphql() -> Any:
    from langchain.utilities.graphql import GraphQLAPIWrapper

    return GraphQLAPIWrapper


def _import_jira() -> Any:
    from langchain.utilities.jira import JiraAPIWrapper

    return JiraAPIWrapper


def _import_max_compute() -> Any:
    from langchain.utilities.max_compute import MaxComputeAPIWrapper

    return MaxComputeAPIWrapper


def _import_merriam_webster() -> Any:
    from langchain.utilities.merriam_webster import MerriamWebsterAPIWrapper

    return MerriamWebsterAPIWrapper


def _import_metaphor_search() -> Any:
    from langchain.utilities.metaphor_search import MetaphorSearchAPIWrapper

    return MetaphorSearchAPIWrapper


def _import_openweathermap() -> Any:
    from langchain.utilities.openweathermap import OpenWeatherMapAPIWrapper

    return OpenWeatherMapAPIWrapper


def _import_outline() -> Any:
    from langchain.utilities.outline import OutlineAPIWrapper

    return OutlineAPIWrapper


def _import_portkey() -> Any:
    from langchain.utilities.portkey import Portkey

    return Portkey


def _import_powerbi() -> Any:
    from langchain.utilities.powerbi import PowerBIDataset

    return PowerBIDataset


def _import_pubmed() -> Any:
    from langchain.utilities.pubmed import PubMedAPIWrapper

    return PubMedAPIWrapper


def _import_python() -> Any:
    from langchain.utilities.python import PythonREPL

    return PythonREPL


def _import_scenexplain() -> Any:
    from langchain.utilities.scenexplain import SceneXplainAPIWrapper

    return SceneXplainAPIWrapper


def _import_searchapi() -> Any:
    from langchain.utilities.searchapi import SearchApiAPIWrapper

    return SearchApiAPIWrapper


def _import_searx_search() -> Any:
    from langchain.utilities.searx_search import SearxSearchWrapper

    return SearxSearchWrapper


def _import_serpapi() -> Any:
    from langchain.utilities.serpapi import SerpAPIWrapper

    return SerpAPIWrapper


def _import_spark_sql() -> Any:
    from langchain.utilities.spark_sql import SparkSQL

    return SparkSQL


def _import_sql_database() -> Any:
    from langchain.utilities.sql_database import SQLDatabase

    return SQLDatabase


def _import_stackexchange() -> Any:
    from langchain.utilities.stackexchange import StackExchangeAPIWrapper

    return StackExchangeAPIWrapper


def _import_tensorflow_datasets() -> Any:
    from langchain.utilities.tensorflow_datasets import TensorflowDatasets

    return TensorflowDatasets


def _import_twilio() -> Any:
    from langchain.utilities.twilio import TwilioAPIWrapper

    return TwilioAPIWrapper


def _import_wikipedia() -> Any:
    from langchain.utilities.wikipedia import WikipediaAPIWrapper

    return WikipediaAPIWrapper


def _import_wolfram_alpha() -> Any:
    from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

    return WolframAlphaAPIWrapper


def _import_zapier() -> Any:
    from langchain.utilities.zapier import ZapierNLAWrapper

    return ZapierNLAWrapper


def __getattr__(name: str) -> Any:
    if name == "AlphaVantageAPIWrapper":
        return _import_alpha_vantage()
    elif name == "ApifyWrapper":
        return _import_apify()
    elif name == "ArceeWrapper":
        return _import_arcee()
    elif name == "ArxivAPIWrapper":
        return _import_arxiv()
    elif name == "LambdaWrapper":
        return _import_awslambda()
    elif name == "BibtexparserWrapper":
        return _import_bibtex()
    elif name == "BingSearchAPIWrapper":
        return _import_bing_search()
    elif name == "BraveSearchWrapper":
        return _import_brave_search()
    elif name == "DuckDuckGoSearchAPIWrapper":
        return _import_duckduckgo_search()
    elif name == "GoogleLensAPIWrapper":
        return _import_google_lens()
    elif name == "GoldenQueryAPIWrapper":
        return _import_golden_query()
    elif name == "GoogleJobsAPIWrapper":
        return _import_google_jobs()
    elif name == "GoogleScholarAPIWrapper":
        return _import_google_scholar()
    elif name == "GoogleFinanceAPIWrapper":
        return _import_google_finance()
    elif name == "GoogleTrendsAPIWrapper":
        return _import_google_trends()
    elif name == "GooglePlacesAPIWrapper":
        return _import_google_places_api()
    elif name == "GoogleSearchAPIWrapper":
        return _import_google_search()
    elif name == "GoogleSerperAPIWrapper":
        return _import_google_serper()
    elif name == "GraphQLAPIWrapper":
        return _import_graphql()
    elif name == "JiraAPIWrapper":
        return _import_jira()
    elif name == "MaxComputeAPIWrapper":
        return _import_max_compute()
    elif name == "MerriamWebsterAPIWrapper":
        return _import_merriam_webster()
    elif name == "MetaphorSearchAPIWrapper":
        return _import_metaphor_search()
    elif name == "OpenWeatherMapAPIWrapper":
        return _import_openweathermap()
    elif name == "OutlineAPIWrapper":
        return _import_outline()
    elif name == "Portkey":
        return _import_portkey()
    elif name == "PowerBIDataset":
        return _import_powerbi()
    elif name == "PubMedAPIWrapper":
        return _import_pubmed()
    elif name == "PythonREPL":
        return _import_python()
    elif name == "SceneXplainAPIWrapper":
        return _import_scenexplain()
    elif name == "SearchApiAPIWrapper":
        return _import_searchapi()
    elif name == "SearxSearchWrapper":
        return _import_searx_search()
    elif name == "SerpAPIWrapper":
        return _import_serpapi()
    elif name == "SparkSQL":
        return _import_spark_sql()
    elif name == "StackExchangeAPIWrapper":
        return _import_stackexchange()
    elif name == "SQLDatabase":
        return _import_sql_database()
    elif name == "TensorflowDatasets":
        return _import_tensorflow_datasets()
    elif name == "TwilioAPIWrapper":
        return _import_twilio()
    elif name == "WikipediaAPIWrapper":
        return _import_wikipedia()
    elif name == "WolframAlphaAPIWrapper":
        return _import_wolfram_alpha()
    elif name == "ZapierNLAWrapper":
        return _import_zapier()
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "AlphaVantageAPIWrapper",
    "ApifyWrapper",
    "ArceeWrapper",
    "ArxivAPIWrapper",
    "BibtexparserWrapper",
    "BingSearchAPIWrapper",
    "BraveSearchWrapper",
    "DuckDuckGoSearchAPIWrapper",
    "GoldenQueryAPIWrapper",
    "GoogleFinanceAPIWrapper",
    "GoogleLensAPIWrapper",
    "GoogleJobsAPIWrapper",
    "GooglePlacesAPIWrapper",
    "GoogleScholarAPIWrapper",
    "GoogleTrendsAPIWrapper",
    "GoogleSearchAPIWrapper",
    "GoogleSerperAPIWrapper",
    "GraphQLAPIWrapper",
    "JiraAPIWrapper",
    "LambdaWrapper",
    "MaxComputeAPIWrapper",
    "MerriamWebsterAPIWrapper",
    "MetaphorSearchAPIWrapper",
    "OpenWeatherMapAPIWrapper",
    "OutlineAPIWrapper",
    "Portkey",
    "PowerBIDataset",
    "PubMedAPIWrapper",
    "PythonREPL",
    "Requests",
    "RequestsWrapper",
    "SQLDatabase",
    "SceneXplainAPIWrapper",
    "SearchApiAPIWrapper",
    "SearxSearchWrapper",
    "SerpAPIWrapper",
    "SparkSQL",
    "StackExchangeAPIWrapper",
    "TensorflowDatasets",
    "TextRequestsWrapper",
    "TwilioAPIWrapper",
    "WikipediaAPIWrapper",
    "WolframAlphaAPIWrapper",
    "ZapierNLAWrapper",
]
