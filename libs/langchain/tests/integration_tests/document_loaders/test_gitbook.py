from typing import Optional

import pytest

from langchain.document_loaders.gitbook import GitbookLoader


class TestGitbookLoader:
    @pytest.mark.parametrize(
        "web_page, load_all_paths, base_url, expected_web_path",
        [
            ("https://example.com/page1", False, None, "https://example.com/page1"),
            (
                "https://example.com/",
                True,
                "https://example.com",
                "https://example.com/sitemap.xml",
            ),
        ],
    )
    def test_init(
        self,
        web_page: str,
        load_all_paths: bool,
        base_url: Optional[str],
        expected_web_path: str,
    ) -> None:
        loader = GitbookLoader(
            web_page, load_all_paths=load_all_paths, base_url=base_url
        )
        print(loader.__dict__)
        assert (
            loader.base_url == (base_url or web_page)[:-1]
            if (base_url or web_page).endswith("/")
            else (base_url or web_page)
        )
        assert loader.web_path == expected_web_path
        assert loader.load_all_paths == load_all_paths

    @pytest.mark.parametrize(
        "web_page, expected_number_results",
        [("https://platform-docs.opentargets.org/getting-started", 1)],
    )
    def test_load_single_page(
        self, web_page: str, expected_number_results: int
    ) -> None:
        loader = GitbookLoader(web_page)
        result = loader.load()
        assert len(result) == expected_number_results

    @pytest.mark.parametrize("web_page", [("https://platform-docs.opentargets.org/")])
    def test_load_multiple_pages(self, web_page: str) -> None:
        loader = GitbookLoader(web_page, load_all_paths=True)
        result = loader.load()
        print(len(result))
        assert len(result) > 10
