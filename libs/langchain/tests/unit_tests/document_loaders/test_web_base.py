import pytest as pytest

from langchain.document_loaders.web_base import WebBaseLoader


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
