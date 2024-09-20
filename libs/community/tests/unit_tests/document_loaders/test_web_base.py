from unittest.mock import MagicMock, patch

import pytest as pytest

from langchain_community.document_loaders.web_base import WebBaseLoader


class TestWebBaseLoader:
    @pytest.mark.requires("bs4")
    def test_respect_user_specified_user_agent(self) -> None:
        user_specified_user_agent = "user_specified_user_agent"
        header_template = {"User-Agent": user_specified_user_agent}
        url = "https://www.example.com"
        loader = WebBaseLoader(url, header_template=header_template)
        assert loader.session.headers["User-Agent"] == user_specified_user_agent

    def test_web_path_parameter(self) -> None:
        web_base_loader = WebBaseLoader(web_paths=["https://www.example.com"])
        assert web_base_loader.web_paths == ["https://www.example.com"]
        web_base_loader = WebBaseLoader(web_path=["https://www.example.com"])
        assert web_base_loader.web_paths == ["https://www.example.com"]
        web_base_loader = WebBaseLoader(web_path="https://www.example.com")
        assert web_base_loader.web_paths == ["https://www.example.com"]


@pytest.mark.requires("bs4")
@patch("langchain_community.document_loaders.web_base.requests.Session.get")
def test_lazy_load(mock_get):
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Test content</p></body></html>"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    loader = WebBaseLoader(web_paths=["https://www.example.com"])
    results = list(loader.lazy_load())
    mock_get.assert_called_with("https://www.example.com")
    assert len(results) == 1
    assert results[0].page_content == "Test content"
