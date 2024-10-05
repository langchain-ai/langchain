import pytest
import responses

from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper


@pytest.fixture
def wrapper():
    """Fixture to set up YandexSearchAPIWrapper instance."""
    return YandexSearchAPIWrapper(
        api_key="fake_api_key", yandex_folder_id="fake_folder_id"
    )


@responses.activate
def test_yandex_search_results_success(wrapper):
    """Success API call"""
    # Мок ответа API
    responses.add(
        responses.POST,
        "https://yandex.ru/search/xml",
        body="""<?xml version="1.0" encoding="UTF-8"?>
                <yandexsearch version="1.0">
                    <request>
                        <query>Query Example </query>
                        <page>1</page>
                        <sortby order="descending" priority="no">tm</sortby>
                        <maxpassages>2</maxpassages>
                        <groupings>
                            <groupby attr="d" mode="deep" groups-on-page="10" 
                                docs-in-group="3" curcateg="-1"/>
                        </groupings>
                    </request>
                    <response>
                        <group>
                            <doc>
                                <url>http://example.com</url>
                                <title>Example Title</title>
                                <domain>example.com</domain>
                                <passage>Example Snippet</passage>
                            </doc>
                            <doc>
                                <url>http://example2.com</url>
                                <title>Example Title 2</title>
                                <domain>example2.com</domain>
                                <passage>Another Example Snippet</passage>
                            </doc>
                        </group>
                    </response>
                </yandexsearch>""",
        status=200,
        content_type="text/xml",
    )

    results = wrapper._yandex_search_results("Query Example", num_results=2)
    assert len(results) == 2
    assert results[0]["url"] == "http://example.com"
    assert results[0]["title"] == "Example Title"
    assert results[0]["snippet"] == "Example Snippet"
    assert results[1]["url"] == "http://example2.com"
    assert results[1]["title"] == "Example Title 2"
    assert results[1]["snippet"] == "Another Example Snippet"


@responses.activate
def test_yandex_search_results_failure(wrapper):
    """API error."""
    responses.add(
        responses.POST,
        "https://yandex.ru/search/xml",
        body="""<?xml version="1.0" encoding="UTF-8"?>
                <yandexsearch version="1.0">
                    <request>
                        <query>Query Example </query>
                        <page>1</page>
                        <sortby order="descending" priority="no">tm</sortby>
                        <maxpassages>2</maxpassages>
                        <groupings>
                            <groupby attr="d" mode="deep" groups-on-page="10" 
                                docs-in-group="3" curcateg="-1"/>
                        </groupings>
                    </request>
                    <response>
                        <error>Unauthorized</error>
                    </response>
                </yandexsearch>""",
        status=401,
        content_type="text/xml",
    )

    results = wrapper._yandex_search_results("test query")
    assert len(results) == 0


@responses.activate
def test_yandex_search_empty_query(wrapper):
    """Empty request"""
    responses.add(
        responses.POST,
        "https://yandex.ru/search/xml",
        body="""<?xml version="1.0" encoding="utf-8"?>
                <yandexsearch version="1.0">
                    <request>
                        <query></query>
                        <page>1</page>
                        <sortby order="descending" priority="no">rlv</sortby>
                        <maxpassages>1</maxpassages>
                        <groupings>
                            <groupby attr="d" mode="deep" groups-on-page="10" 
                                docs-in-group="3" curcateg="-1"/>
                        </groupings>
                    </request>
                    <response date="20240502T061612">
                        <error code="2">Empty request</error>
                        <reqid>
                            1714630572494676-9557161538559543159-
                            balancer-l7leveler-kubr-yp-sas-144-BAL
                        </reqid>
                    </response>
                </yandexsearch>""",
        status=400,
        content_type="text/xml",
    )

    results = wrapper._yandex_search_results("")
    assert len(results) == 0
