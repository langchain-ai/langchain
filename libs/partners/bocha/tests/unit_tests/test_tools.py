from unittest.mock import MagicMock, patch
from langchain_bocha import BochaSearchRun, BochaSearchResults


def test_bocha_search_run():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "webPages": {
            "value": [
                {"name": "Result 1", "snippet": "Some snippet about weather"},
                {"name": "Result 2", "snippet": "Another snippet"},
            ]
        }
    }
    with patch("langchain_bocha.tools.requests.get", return_value=mock_resp):
        tool = BochaSearchRun(api_key="test-key")
        output = tool._run("Beijing weather")
        assert "Result 1" in output
        assert "Some snippet about weather" in output


def test_bocha_search_results():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "webPages": {
            "value": [{"name": "Result 1", "snippet": "Some snippet"}]
        }
    }
    with patch("langchain_bocha.tools.requests.get", return_value=mock_resp):
        tool = BochaSearchResults(api_key="test-key")
        output = tool._run("Beijing weather")
        assert "webPages" in output


def test_bocha_search_uses_env_key():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"webPages": {"value": []}}
    with patch("langchain_bocha.tools.requests.get", return_value=mock_resp) as mock_get:
        with patch.dict("os.environ", {"BOCHA_API_KEY": "env-test-key"}):
            tool = BochaSearchRun()
            tool._run("test query")
            call_kwargs = mock_get.call_args
            assert "Authorization" in call_kwargs.kwargs["headers"]
