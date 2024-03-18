"""DEPRECATED: This module has been moved to the langchain-community package.

**Utilities** are the integrations with third-part systems and packages.
"""
from typing import Any

from langchain_community.utilities.requests import (
    Requests,
    RequestsWrapper,
    TextRequestsWrapper,
)

DEPRECATED_IMPORTS = [
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
    "NasaAPIWrapper",
    "OpenWeatherMapAPIWrapper",
    "OutlineAPIWrapper",
    "Portkey",
    "PowerBIDataset",
    "PubMedAPIWrapper",
    "PythonREPL",
    "SteamWebAPIWrapper",
    "SQLDatabase",
    "SceneXplainAPIWrapper",
    "SearchApiAPIWrapper",
    "SearxSearchWrapper",
    "SerpAPIWrapper",
    "SparkSQL",
    "StackExchangeAPIWrapper",
    "TensorflowDatasets",
    "TwilioAPIWrapper",
    "WikipediaAPIWrapper",
    "WolframAlphaAPIWrapper",
    "ZapierNLAWrapper",
]


def __getattr__(name: str) -> Any:
    if name in DEPRECATED_IMPORTS:
        raise ImportError(
            f"{name} has been moved to the langchain-community package. "
            f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
            f"information.\n\nTo use it install langchain-community:\n\n"
            f"`pip install -U langchain-community`\n\n"
            f"then import with:\n\n"
            f"`from langchain_community.utilities import {name}`"
        )

    raise AttributeError()


__all__ = [
    "Requests",
    "RequestsWrapper",
    "TextRequestsWrapper",
]
