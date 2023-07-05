"""Integration test for Dataforseo API Wrapper."""
import pytest

from langchain.utilities.dataforseo_api_search import DataForSeoAPIWrapper


def test_search_call() -> None:
    search = DataForSeoAPIWrapper()
    output = search.run("pi value")
    assert "3.14159" in output


def test_news_call() -> None:
    search = DataForSeoAPIWrapper(
        params={"se_type": "news"}, json_result_fields=["title", "snippet"]
    )
    output = search.results("iphone")
    assert any("Apple" in d["title"] or "Apple" in d["snippet"] for d in output)


def test_loc_call() -> None:
    search = DataForSeoAPIWrapper(
        params={"location_name": "Spain", "language_code": "es"}
    )
    output = search.results("iphone")
    assert "/es/" in output[0]["url"]


def test_maps_call() -> None:
    search = DataForSeoAPIWrapper(
        params={"location_name": "Spain", "language_code": "es", "se_type": "maps"}
    )
    output = search.results("coffee")
    assert all(i["address_info"]["country_code"] == "ES" for i in output)


def test_events_call() -> None:
    search = DataForSeoAPIWrapper(
        params={"location_name": "Spain", "language_code": "es", "se_type": "events"}
    )
    output = search.results("concerts")
    assert any(
        "Madrid" in ((i["location_info"] or dict())["address"] or "") for i in output
    )


@pytest.mark.asyncio
async def test_async_call() -> None:
    search = DataForSeoAPIWrapper()
    output = await search.arun("pi value")
    assert "3.14159" in output


@pytest.mark.asyncio
async def test_async_results() -> None:
    search = DataForSeoAPIWrapper(json_result_types=["answer_box"])
    output = await search.aresults("New York timezone")
    assert "Eastern Daylight Time" in output[0]["text"]
