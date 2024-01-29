"""Unit test for Google Trends API Wrapper."""
import os
from unittest.mock import patch

from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper


@patch("serpapi.SerpApiClient.get_json")
def test_unexpected_response(mocked_serpapiclient):
    os.environ["SERPAPI_API_KEY"] = "123abcd"
    resp = {
        "search_metadata": {
            "id": "659f32ec36e6a9107b46b5b4",
            "status": "Error",
            "json_endpoint": "https://serpapi.com/searches/.../659f32ec36e6a9107b46b5b4.json",
            "created_at": "2024-01-11 00:14:36 UTC",
            "processed_at": "2024-01-11 00:14:36 UTC",
            "google_trends_url": "https://trends.google.com/trends/api/explore?tz=420&req=%7B%22comparisonItem%22%3A%5B%7B%22keyword%22%3A%22Lego+building+trends+2022%22%2C%22geo%22%3A%22%22%2C%22time%22%3A%22today+12-m%22%7D%5D%2C%22category%22%3A0%2C%22property%22%3A%22%22%2C%22userConfig%22%3A%22%7BuserType%3A+%5C%22USER_TYPE_LEGIT_USER%5C%22%7D%22%7D",
            "prettify_html_file": "https://serpapi.com/searches/.../659f32ec36e6a9107b46b5b4.prettify",
            "total_time_taken": 90.14,
        },
        "search_parameters": {
            "engine": "google_trends",
            "q": "Lego building trends 2022",
            "date": "today 12-m",
            "tz": "420",
            "data_type": "TIMESERIES",
        },
        "error": "We couldn't get valid ... Please try again later.",
    }
    mocked_serpapiclient.return_value = resp
    tool = GoogleTrendsAPIWrapper()
    tool.run("does not matter")
