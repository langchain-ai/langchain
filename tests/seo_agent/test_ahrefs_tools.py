"""Tests for Ahrefs API tools — all tests use mock mode."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Force mock mode for all tests
os.environ["AHREFS_MOCK"] = "true"
os.environ["GSC_MOCK"] = "true"
os.environ["SUPABASE_MOCK"] = "true"
os.environ["TAVILY_MOCK"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"

from agents.seo_agent.tools.ahrefs_tools import (
    AhrefsClient,
    get_backlinks,
    get_broken_backlinks,
    get_competing_domains,
    get_content_gap,
    get_keyword_ideas,
    get_keyword_overview,
    get_rank_tracking,
    search_content_explorer,
)


class TestGetKeywordOverview:
    def test_returns_list(self) -> None:
        result = get_keyword_overview.invoke(
            {"keywords": ["kitchen makers UK"], "country": "gb"}
        )
        assert isinstance(result, list)

    def test_unknown_keyword_returns_default(self) -> None:
        result = get_keyword_overview.invoke(
            {"keywords": ["zzznonexistent"], "country": "gb"}
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0]["keyword"] == "zzznonexistent"

    def test_known_keyword_returns_fixture(self) -> None:
        result = get_keyword_overview.invoke(
            {"keywords": ["bespoke kitchen manufacturers UK"]}
        )
        assert any(kw["keyword"] == "bespoke kitchen manufacturers UK" for kw in result)


class TestGetKeywordIdeas:
    def test_returns_list(self) -> None:
        result = get_keyword_ideas.invoke({"seed_keyword": "kitchen"})
        assert isinstance(result, list)
        assert len(result) > 0

    def test_filters_by_seed(self) -> None:
        result = get_keyword_ideas.invoke({"seed_keyword": "bespoke"})
        assert isinstance(result, list)
        # Should return keywords matching the seed
        assert any("bespoke" in kw.get("keyword", "") for kw in result)


class TestGetCompetingDomains:
    def test_returns_competitors(self) -> None:
        result = get_competing_domains.invoke(
            {"target": "kitchensdirectory.co.uk"}
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert "domain" in result[0]

    def test_has_expected_fields(self) -> None:
        result = get_competing_domains.invoke(
            {"target": "kitchensdirectory.co.uk"}
        )
        for comp in result:
            assert "domain" in comp
            assert "common_keywords" in comp
            assert "traffic" in comp


class TestGetContentGap:
    def test_returns_gaps(self) -> None:
        result = get_content_gap.invoke(
            {
                "target": "kitchensdirectory.co.uk",
                "competitors": ["houzz.com", "checkatrade.com"],
            }
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_gap_has_keyword_and_volume(self) -> None:
        result = get_content_gap.invoke(
            {
                "target": "kitchensdirectory.co.uk",
                "competitors": ["houzz.com"],
            }
        )
        for gap in result:
            assert "keyword" in gap
            assert "volume" in gap


class TestGetBacklinks:
    def test_returns_backlinks(self) -> None:
        result = get_backlinks.invoke(
            {"target": "kitchensdirectory.co.uk"}
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_filters_by_dr(self) -> None:
        result = get_backlinks.invoke(
            {"target": "kitchensdirectory.co.uk", "dr_min": 70}
        )
        for bl in result:
            assert bl["dr"] >= 70

    def test_backlink_has_required_fields(self) -> None:
        result = get_backlinks.invoke(
            {"target": "kitchensdirectory.co.uk"}
        )
        for bl in result:
            assert "referring_domain" in bl
            assert "page_url" in bl
            assert "dr" in bl


class TestGetBrokenBacklinks:
    def test_returns_broken_links(self) -> None:
        result = get_broken_backlinks.invoke(
            {"target": "competitor.com"}
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_has_dead_url(self) -> None:
        result = get_broken_backlinks.invoke({"target": "competitor.com"})
        for bl in result:
            assert "dead_url" in bl
            assert "referring_page" in bl


class TestGetRankTracking:
    def test_returns_tracking_data(self) -> None:
        result = get_rank_tracking.invoke(
            {"target": "kitchensdirectory.co.uk"}
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_has_position_data(self) -> None:
        result = get_rank_tracking.invoke(
            {"target": "kitchensdirectory.co.uk"}
        )
        for entry in result:
            assert "keyword" in entry
            assert "position" in entry


class TestSearchContentExplorer:
    def test_returns_results(self) -> None:
        result = search_content_explorer.invoke(
            {"query": "best kitchen companies UK"}
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_has_url_and_title(self) -> None:
        result = search_content_explorer.invoke(
            {"query": "room planning tools"}
        )
        for item in result:
            assert "url" in item
            assert "title" in item


class TestAhrefsClient:
    def test_client_instantiates(self) -> None:
        client = AhrefsClient()
        assert client.BASE_URL == "https://api.ahrefs.com/v4"
        client.close()
