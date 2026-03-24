"""Tests for the content pipeline — keyword research, briefs, writer, and tools."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["AHREFS_MOCK"] = "true"
os.environ["GSC_MOCK"] = "true"
os.environ["SUPABASE_MOCK"] = "true"
os.environ["TAVILY_MOCK"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"

from agents.seo_agent.agent import build_graph, create_initial_state
from agents.seo_agent.config import SITE_PROFILES, TOKEN_BUDGETS
from agents.seo_agent.tools.file_tools import (
    OUTPUT_ROOT,
    list_output_files,
    read_brief,
    write_brief,
    write_draft,
    write_report,
)
from agents.seo_agent.tools.gsc_tools import get_top_pages, get_top_queries
from agents.seo_agent.tools.llm_router import (
    calculate_cost,
    get_budget_remaining,
    get_model,
)
from agents.seo_agent.tools.supabase_tools import (
    _mock_store,
    ensure_tables,
    get_weekly_spend,
    insert_record,
    log_llm_cost,
    query_table,
)


# ---------------------------------------------------------------------------
# LLM Router tests
# ---------------------------------------------------------------------------


class TestLLMRouter:
    def test_get_model_default(self) -> None:
        model = get_model("write_blog_post")
        assert model == "claude-sonnet-4-6"

    def test_get_model_haiku_task(self) -> None:
        model = get_model("classify_prospect")
        assert model == "claude-haiku-4-5-20251001"

    def test_get_model_opus_task(self) -> None:
        model = get_model("write_tier1_email")
        assert model == "claude-opus-4-6"

    def test_get_model_unknown_defaults_to_sonnet(self) -> None:
        model = get_model("unknown_task_xyz")
        assert model == "claude-sonnet-4-6"

    def test_budget_downgrade_sonnet_to_haiku(self) -> None:
        model = get_model("write_blog_post", budget_remaining=0.10)
        assert model == "claude-haiku-4-5-20251001"

    def test_budget_downgrade_opus_to_sonnet(self) -> None:
        model = get_model("write_tier1_email", budget_remaining=0.10)
        assert model == "claude-sonnet-4-6"

    def test_no_downgrade_when_budget_ok(self) -> None:
        model = get_model("write_blog_post", budget_remaining=0.50)
        assert model == "claude-sonnet-4-6"

    def test_calculate_cost_basic(self) -> None:
        cost = calculate_cost("claude-sonnet-4-6", 1000, 500)
        expected = (1000 / 1_000_000 * 3.00) + (500 / 1_000_000 * 15.00)
        assert abs(cost - expected) < 0.000001

    def test_calculate_cost_with_cache(self) -> None:
        cost = calculate_cost("claude-sonnet-4-6", 1000, 500, cached_tokens=800)
        uncached = 200
        expected = (uncached / 1_000_000 * 3.00)
        expected += (800 / 1_000_000 * 3.00 * 0.10)
        expected += (500 / 1_000_000 * 15.00)
        assert abs(cost - round(expected, 6)) < 0.000001

    def test_calculate_cost_haiku_cheaper(self) -> None:
        haiku_cost = calculate_cost("claude-haiku-4-5-20251001", 1000, 500)
        sonnet_cost = calculate_cost("claude-sonnet-4-6", 1000, 500)
        assert haiku_cost < sonnet_cost

    def test_get_budget_remaining(self) -> None:
        assert get_budget_remaining(0.0) == 1.0
        assert get_budget_remaining(25.0) == 0.5
        assert get_budget_remaining(50.0) == 0.0
        assert get_budget_remaining(100.0) == 0.0  # capped at 0


# ---------------------------------------------------------------------------
# Supabase tools tests
# ---------------------------------------------------------------------------


class TestSupabaseTools:
    def setup_method(self) -> None:
        _mock_store.clear()
        ensure_tables()

    def test_ensure_tables_creates_all(self) -> None:
        assert "seo_keyword_opportunities" in _mock_store
        assert "llm_cost_log" in _mock_store
        assert "seo_backlink_prospects" in _mock_store

    def test_insert_and_query(self) -> None:
        insert_record("seo_keyword_opportunities", {
            "keyword": "test keyword",
            "volume": 500,
            "target_site": "kitchensdirectory",
        })
        rows = query_table(
            "seo_keyword_opportunities",
            filters={"target_site": "kitchensdirectory"},
        )
        assert len(rows) == 1
        assert rows[0]["keyword"] == "test keyword"

    def test_log_llm_cost(self) -> None:
        log_llm_cost(
            task_type="test_task",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            site="kitchensdirectory",
        )
        rows = query_table("llm_cost_log")
        assert len(rows) == 1
        assert rows[0]["cost_usd"] == 0.001

    def test_get_weekly_spend(self) -> None:
        log_llm_cost(
            task_type="test",
            model="claude-sonnet-4-6",
            input_tokens=100,
            output_tokens=50,
            cost_usd=1.50,
            site="test",
        )
        spend = get_weekly_spend()
        assert spend >= 1.50


# ---------------------------------------------------------------------------
# File tools tests
# ---------------------------------------------------------------------------


class TestFileTools:
    def test_write_and_read_brief(self, tmp_path: object) -> None:
        path = write_brief("test keyword here", "# Test Brief\n\nContent here.")
        assert os.path.exists(path)
        content = read_brief("test keyword here")
        assert content is not None
        assert "Test Brief" in content

    def test_write_draft(self) -> None:
        path = write_draft("another test", "# Draft\n\nDraft content.")
        assert os.path.exists(path)

    def test_write_report(self) -> None:
        path = write_report("seo-report-2026-03-24", "# Report\n\nWeekly report.")
        assert os.path.exists(path)

    def test_list_output_files(self) -> None:
        write_brief("list test", "content")
        files = list_output_files("briefs")
        assert "list-test.md" in files


# ---------------------------------------------------------------------------
# GSC tools tests
# ---------------------------------------------------------------------------


class TestGSCTools:
    def test_get_top_queries_returns_list(self) -> None:
        result = get_top_queries("https://www.kitchensdirectory.co.uk")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_top_pages_returns_list(self) -> None:
        result = get_top_pages("https://www.kitchensdirectory.co.uk")
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_site_profiles_has_all_sites(self) -> None:
        assert "kitchensdirectory" in SITE_PROFILES
        assert "freeroomplanner" in SITE_PROFILES
        assert "kitchen_estimator" in SITE_PROFILES

    def test_site_profile_has_required_keys(self) -> None:
        for site_key, profile in SITE_PROFILES.items():
            assert "domain" in profile, f"{site_key} missing domain"
            assert "primary_topic" in profile
            assert "seed_keywords" in profile
            assert isinstance(profile["seed_keywords"], list)

    def test_token_budgets_all_positive(self) -> None:
        for task, budget in TOKEN_BUDGETS.items():
            assert budget > 0, f"{task} has non-positive budget"


# ---------------------------------------------------------------------------
# Graph integration tests (mock mode)
# ---------------------------------------------------------------------------


class TestGraphIntegration:
    def setup_method(self) -> None:
        _mock_store.clear()
        ensure_tables()

    def test_keyword_research_end_to_end(self) -> None:
        graph = build_graph()
        state = create_initial_state(
            task_type="keyword_research",
            target_site="kitchensdirectory",
            seed_keyword="bespoke kitchens",
        )
        result = graph.invoke(state)
        assert len(result.get("keyword_opportunities", [])) > 0
        assert len(result.get("errors", [])) == 0

    def test_content_gap_end_to_end(self) -> None:
        graph = build_graph()
        state = create_initial_state(
            task_type="content_gap",
            target_site="kitchensdirectory",
        )
        result = graph.invoke(state)
        assert len(result.get("content_gaps", [])) > 0

    def test_rank_report_end_to_end(self) -> None:
        graph = build_graph()
        state = create_initial_state(
            task_type="rank_report",
            target_site="kitchensdirectory",
        )
        result = graph.invoke(state)
        assert len(result.get("rank_data", [])) > 0

    def test_discover_prospects_end_to_end(self) -> None:
        graph = build_graph()
        state = create_initial_state(
            task_type="discover_prospects",
            target_site="kitchensdirectory",
        )
        result = graph.invoke(state)
        assert len(result.get("backlink_prospects", [])) > 0
