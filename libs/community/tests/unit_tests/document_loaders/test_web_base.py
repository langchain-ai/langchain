from textwrap import dedent
from typing import Any
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
def test_lazy_load(mock_get: Any) -> None:
    import bs4

    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Test content</p></body></html>"
    mock_get.return_value = mock_response

    loader = WebBaseLoader(web_paths=["https://www.example.com"])
    results = list(loader.lazy_load())
    mock_get.assert_called_with("https://www.example.com")
    assert len(results) == 1
    assert results[0].page_content == "Test content"

    # Test bs4 kwargs
    mock_html = dedent("""
        <html>
        <body>
            <p>Test content</p>
            <div class="special-class">This is a div with a special class</div>
        </body>
        </html>
        """)
    mock_response = MagicMock()
    mock_response.text = mock_html
    mock_get.return_value = mock_response

    loader = WebBaseLoader(
        web_paths=["https://www.example.com"],
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_="special-class")},
    )
    results = list(loader.lazy_load())
    assert len(results) == 1
    assert results[0].page_content == "This is a div with a special class"


@pytest.mark.requires("bs4")
@patch("aiohttp.ClientSession.get")
async def test_aload(mock_get: Any) -> None:
    async def mock_text() -> str:
        return "<html><body><p>Test content</p></body></html>"

    mock_response = MagicMock()
    mock_response.text = mock_text
    mock_get.return_value.__aenter__.return_value = mock_response

    loader = WebBaseLoader(
        web_paths=["https://www.example.com"],
        header_template={"User-Agent": "test-user-agent"},
    )
    results = await loader.aload()
    assert len(results) == 1
    assert results[0].page_content == "Test content"
    mock_get.assert_called_with(
        "https://www.example.com", headers={"User-Agent": "test-user-agent"}, cookies={}
    )
