import unittest
from unittest.mock import patch
import responses
from langchain_community.utilities.yandex_search import YandexSearchAPIWrapper

class TestYandexSearchAPIWrapper(unittest.TestCase):

    def setUp(self):
        """Settings."""
        self.wrapper = YandexSearchAPIWrapper(
            yandex_api_key="fake_api_key",
            yandex_folder_id="fake_folder_id"
        )

    @responses.activate
    def test_yandex_search_results_success(self):
        """Success API call"""
        # Мок ответа API
        responses.add(
            responses.POST,
            "https://yandex.ru/search/xml",
            body='''<?xml version="1.0" encoding="UTF-8"?>
                    <yandexsearch version="1.0">
                        <request>
                            <query>Query Example </query>
                            <page>1</page>
                            <sortby order="descending" priority="no">tm</sortby>
                            <maxpassages>2</maxpassages>
                            <groupings>
                                <groupby attr="d" mode="deep" groups-on-page="10" docs-in-group="3" curcateg="-1"/>
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
                    </yandexsearch>''',
            status=200,
            content_type='text/xml'
        )

        results = self.wrapper._yandex_search_results("Query Example", num_results=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['url'], 'http://example.com')
        self.assertEqual(results[0]['title'], 'Example Title')
        self.assertEqual(results[0]['snippet'], 'Example Snippet')
        self.assertEqual(results[1]['url'], 'http://example2.com')
        self.assertEqual(results[1]['title'], 'Example Title 2')
        self.assertEqual(results[1]['snippet'], 'Another Example Snippet')

    @responses.activate
    def test_yandex_search_results_failure(self):
        """API error."""
        responses.add(
            responses.POST,
            "https://yandex.ru/search/xml",
            body='''<?xml version="1.0" encoding="UTF-8"?>
                    <yandexsearch version="1.0">
                        <request>
                            <query>Query Example </query>
                            <page>1</page>
                            <sortby order="descending" priority="no">tm</sortby>
                            <maxpassages>2</maxpassages>
                            <groupings>
                                <groupby attr="d" mode="deep" groups-on-page="10" docs-in-group="3" curcateg="-1"/>
                            </groupings>
                        </request>
                        <response>
                            <error>Unauthorized</error>
                        </response>
                    </yandexsearch>''',
            status=401,
            content_type='text/xml'
        )

        results = self.wrapper._yandex_search_results("test query")
        self.assertEqual(len(results), 0)

    @responses.activate
    def test_yandex_search_empty_query(self):
        """Empty request"""
        responses.add(
            responses.POST,
            "https://yandex.ru/search/xml",
            body='''<?xml version="1.0" encoding="utf-8"?>
                    <yandexsearch version="1.0">
                        <request>
                            <query></query>
                            <page>1</page>
                            <sortby order="descending" priority="no">rlv</sortby>
                            <maxpassages>1</maxpassages>
                            <groupings>
                                <groupby attr="d" mode="deep" groups-on-page="10" docs-in-group="3" curcateg="-1"/>
                            </groupings>
                        </request>
                        <response date="20240502T061612">
                            <error code="2">Empty request</error>
                            <reqid>1714630572494676-9557161538559543159-balancer-l7leveler-kubr-yp-sas-144-BAL</reqid>
                        </response>
                    </yandexsearch>''',
            status=400,
            content_type='text/xml'
        )

        results = self.wrapper._yandex_search_results("")
        self.assertEqual(len(results), 0)
        
if __name__ == '__main__':
    unittest.main()
