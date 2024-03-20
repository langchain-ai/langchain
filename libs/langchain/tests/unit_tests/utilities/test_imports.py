import pytest

from langchain import utilities

EXPECTED_DEPRECATED_IMPORT = [
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
    "GoogleJobsAPIWrapper",
    "GoogleLensAPIWrapper",
    "GooglePlacesAPIWrapper",
    "GoogleScholarAPIWrapper",
    "GoogleSearchAPIWrapper",
    "GoogleSerperAPIWrapper",
    "GoogleTrendsAPIWrapper",
    "GraphQLAPIWrapper",
    "JiraAPIWrapper",
    "LambdaWrapper",
    "MaxComputeAPIWrapper",
    "MetaphorSearchAPIWrapper",
    "NasaAPIWrapper",
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
    "SteamWebAPIWrapper",
    "TensorflowDatasets",
    "TextRequestsWrapper",
    "TwilioAPIWrapper",
    "WikipediaAPIWrapper",
    "WolframAlphaAPIWrapper",
    "ZapierNLAWrapper",
    "MerriamWebsterAPIWrapper",
]


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(utilities, import_)
            assert "langchain_community" in e, f"{import_=} didn't error"
    with pytest.raises(AttributeError):
        getattr(utilities, "foo")
