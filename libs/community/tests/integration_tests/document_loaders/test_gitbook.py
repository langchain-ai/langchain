from typing import Optional

import pytest

from langchain_community.document_loaders.gitbook import GitbookLoader


class TestGitbookLoader:
    @pytest.mark.parametrize(
        "web_page, load_all_paths, base_url, expected_start_url",
        [
            # Single page load
            ("https://example.com/page1", False, None, "https://example.com/page1"),
            # Load all paths, default sitemap
            (
                "https://example.com/",
                True,
                "https://example.com",
                "https://example.com/sitemap.xml",
            ),
            # Load all paths, default sitemap with base_url
            (
                "https://docs.example.com/product",
                True,
                "https://docs.example.com/",  # Base URL specified
                "https://docs.example.com/sitemap.xml",
            ),
        ],
    )
    def test_init(
        self,
        web_page: str,
        load_all_paths: bool,
        base_url: Optional[str],
        expected_start_url: str,
    ) -> None:
        loader = GitbookLoader(
            web_page, load_all_paths=load_all_paths, base_url=base_url
        )
        print(loader.__dict__)  # noqa: T201

        # Check base_url handling
        expected_base = base_url or web_page
        if expected_base.endswith("/"):
            expected_base = expected_base[:-1]
        assert loader.base_url == expected_base

        # Check the determined start_url
        assert loader.start_url == expected_start_url
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

    @pytest.mark.requires("lxml")
    @pytest.mark.parametrize("web_page", [("https://platform-docs.opentargets.org/")])
    def test_load_multiple_pages(self, web_page: str) -> None:
        pytest.importorskip("lxml")  # Skip if lxml is not available
        loader = GitbookLoader(web_page, load_all_paths=True)
        result = loader.load()
        print(len(result))  # noqa: T201
        assert len(result) > 10

    @pytest.mark.parametrize(
        "web_page, sitemap_url, expected_start_url",
        [
            (
                "https://example.com/",
                "https://example.com/custom-sitemap.xml",
                "https://example.com/custom-sitemap.xml",
            ),
        ],
    )
    def test_init_with_custom_sitemap(
        self,
        web_page: str,
        sitemap_url: str,
        expected_start_url: str,
    ) -> None:
        """Test that the custom sitemap URL is correctly used when provided."""
        loader = GitbookLoader(web_page, load_all_paths=True, sitemap_url=sitemap_url)
        assert loader.start_url == expected_start_url
        assert loader.load_all_paths
