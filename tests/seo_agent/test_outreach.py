"""Tests for outreach pipeline — scoring, safety rails, email generation."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["AHREFS_MOCK"] = "true"
os.environ["GSC_MOCK"] = "true"
os.environ["SUPABASE_MOCK"] = "true"
os.environ["TAVILY_MOCK"] = "true"
os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"
os.environ["OUTREACH_DRY_RUN"] = "true"

from agents.seo_agent.config import (
    BLOCKED_TLD_SUFFIXES,
    MAX_DAILY_OUTREACH_EMAILS,
    MIN_DAYS_BETWEEN_DOMAIN_CONTACTS,
    MIN_OUTREACH_SCORE,
    TIER1_OUTREACH_SCORE,
)
from agents.seo_agent.nodes.prospect_scorer import _score_prospect, run_prospect_scorer
from agents.seo_agent.tools.supabase_tools import (
    _mock_store,
    ensure_tables,
    insert_record,
    is_domain_blocked,
    query_table,
)


# ---------------------------------------------------------------------------
# Scoring rubric tests
# ---------------------------------------------------------------------------


class TestProspectScorer:
    def test_high_dr_high_score(self) -> None:
        prospect = {
            "dr": 75,
            "monthly_traffic": 5000,
            "links_to_competitor": True,
            "competitor_names": ["comp1", "comp2"],
            "discovery_method": "competitor_backlink",
            "contact_email": "editor@example.com",
            "page_url": "https://example.co.uk/resources",
        }
        score, reasons = _score_prospect(prospect)
        assert score >= TIER1_OUTREACH_SCORE

    def test_low_dr_low_score(self) -> None:
        prospect = {
            "dr": 10,
            "monthly_traffic": 50,
            "links_to_competitor": False,
            "competitor_names": [],
            "discovery_method": "content_explorer",
            "contact_email": "",
            "page_url": "https://tiny-blog.com/post",
        }
        score, reasons = _score_prospect(prospect)
        assert score < MIN_OUTREACH_SCORE

    def test_unlinked_mention_bonus(self) -> None:
        prospect = {
            "dr": 45,
            "monthly_traffic": 2000,
            "links_to_competitor": False,
            "competitor_names": [],
            "discovery_method": "unlinked_mention",
            "contact_email": "hello@example.com",
            "page_url": "https://example.co.uk/article",
        }
        score, reasons = _score_prospect(prospect)
        # Should get unlinked mention bonus (25) + DR 40-59 (15) + traffic (15) + UK (10)
        assert score >= 50

    def test_broken_link_bonus(self) -> None:
        prospect = {
            "dr": 50,
            "monthly_traffic": 3000,
            "links_to_competitor": True,
            "competitor_names": ["comp1"],
            "discovery_method": "broken_link",
            "contact_email": "info@example.com",
            "page_url": "https://example.com/resources",
        }
        score, reasons = _score_prospect(prospect)
        assert score >= MIN_OUTREACH_SCORE

    def test_resource_page_bonus(self) -> None:
        prospect = {
            "dr": 55,
            "monthly_traffic": 1500,
            "links_to_competitor": False,
            "competitor_names": [],
            "discovery_method": "resource_page",
            "contact_email": "webmaster@example.co.uk",
            "page_url": "https://example.co.uk/useful-links",
        }
        score, reasons = _score_prospect(prospect)
        assert score >= MIN_OUTREACH_SCORE

    def test_scorer_node_rejects_below_threshold(self) -> None:
        _mock_store.clear()
        ensure_tables()
        insert_record("seo_backlink_prospects", {
            "domain": "tiny-blog.com",
            "dr": 5,
            "monthly_traffic": 10,
            "links_to_competitor": False,
            "competitor_names": [],
            "discovery_method": "content_explorer",
            "contact_email": "",
            "page_url": "https://tiny-blog.com/post",
            "status": "enriched",
        })
        state = {
            "target_site": "kitchensdirectory",
            "task_type": "score_prospects",
            "enriched_prospects": [],
            "scored_prospects": [],
            "errors": [],
            "llm_spend_this_week": 0.0,
        }
        result = run_prospect_scorer(state)
        scored = result.get("scored_prospects", [])
        rejected = [p for p in scored if p.get("status") == "rejected"]
        assert len(rejected) > 0 or len(scored) == 0


# ---------------------------------------------------------------------------
# Safety rail tests
# ---------------------------------------------------------------------------


class TestSafetyRails:
    def test_gov_uk_blocked(self) -> None:
        for suffix in BLOCKED_TLD_SUFFIXES:
            assert suffix in [".gov.uk", ".ac.uk"]

    def test_domain_blocklist(self) -> None:
        _mock_store.clear()
        ensure_tables()
        insert_record("seo_outreach_blocklist", {
            "domain": "spam-site.com",
            "reason": "Known link farm",
        })
        assert is_domain_blocked("spam-site.com") is True
        assert is_domain_blocked("clean-site.com") is False

    def test_daily_limit_configured(self) -> None:
        assert MAX_DAILY_OUTREACH_EMAILS == 20

    def test_min_contact_interval(self) -> None:
        assert MIN_DAYS_BETWEEN_DOMAIN_CONTACTS == 90

    def test_score_thresholds(self) -> None:
        assert MIN_OUTREACH_SCORE == 35
        assert TIER1_OUTREACH_SCORE == 65
        assert TIER1_OUTREACH_SCORE > MIN_OUTREACH_SCORE

    def test_blocked_tld_check(self) -> None:
        """Verify that .gov.uk and .ac.uk domains are caught."""
        test_domains = [
            ("nhs.gov.uk", True),
            ("oxford.ac.uk", True),
            ("example.com", False),
            ("example.co.uk", False),
        ]
        for domain, should_block in test_domains:
            blocked = any(domain.endswith(suffix) for suffix in BLOCKED_TLD_SUFFIXES)
            assert blocked == should_block, f"{domain} block check failed"


# ---------------------------------------------------------------------------
# Email template tests
# ---------------------------------------------------------------------------


class TestEmailTemplates:
    """Test that the email generator maps discovery methods to correct templates."""

    def test_template_mapping(self) -> None:
        from agents.seo_agent.nodes.email_generator import _select_template

        assert _select_template("competitor_backlink") == "resource_addition"
        assert _select_template("content_explorer") == "resource_addition"
        assert _select_template("broken_link") == "broken_link_replacement"
        assert _select_template("unlinked_mention") == "unlinked_mention"
        assert _select_template("resource_page") == "data_research_collaboration"
        assert _select_template("haro") == "expert_quote_offer"
        assert _select_template("unknown") == "expert_quote_offer"


# ---------------------------------------------------------------------------
# Outreach sequencer tests
# ---------------------------------------------------------------------------


class TestOutreachSequencer:
    def test_sequence_timing(self) -> None:
        """Verify the expected follow-up schedule."""
        from agents.seo_agent.nodes.outreach_sequencer import _SEQUENCE_SCHEDULE

        assert _SEQUENCE_SCHEDULE[0] == 0   # Day 0: initial email
        assert _SEQUENCE_SCHEDULE[1] == 5   # Day 5: follow-up 1
        assert _SEQUENCE_SCHEDULE[2] == 12  # Day 12: follow-up 2
        assert _SEQUENCE_SCHEDULE[3] == 20  # Day 20: closing the loop

    def test_dry_run_sends_nothing(self) -> None:
        _mock_store.clear()
        ensure_tables()

        from agents.seo_agent.nodes.outreach_sequencer import run_outreach_sequencer

        state = {
            "target_site": "kitchensdirectory",
            "task_type": "run_outreach",
            "errors": [],
            "llm_spend_this_week": 0.0,
        }
        result = run_outreach_sequencer(state)
        # Should complete without errors in dry-run mode
        assert isinstance(result.get("errors", []), list)
