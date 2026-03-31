"""Tests for the promote_to_crm worker skill.

Covers the null-website crash fix and prospect status updates.
All tests use SUPABASE_MOCK=true so no network calls are needed.
"""

from __future__ import annotations

import os

import pytest

os.environ["SUPABASE_MOCK"] = "true"


@pytest.fixture(autouse=True)
def _clear_mock_store():
    """Reset the in-memory mock store before each test."""
    from agents.seo_agent.tools.supabase_tools import _mock_store

    _mock_store.clear()
    yield
    _mock_store.clear()


def _seed_prospect(domain: str, *, status: str = "scored", **extra):
    """Insert a fake prospect into the mock store."""
    import uuid

    from agents.seo_agent.tools.supabase_tools import _mock_store

    record = {
        "id": str(uuid.uuid4()),
        "domain": domain,
        "page_title": f"Title for {domain}",
        "status": status,
        "segment": extra.pop("segment", "kitchen_provider"),
        "discovery_method": extra.pop("discovery_method", "competitor_analysis"),
        "dr": extra.pop("dr", 30),
        **extra,
    }
    _mock_store.setdefault("seo_backlink_prospects", []).append(record)
    return record


def _seed_crm_contact(company: str, *, website=None, category="kitchen_company"):
    """Insert a fake CRM contact into the mock store."""
    import uuid

    from agents.seo_agent.tools.supabase_tools import _mock_store

    record = {
        "id": str(uuid.uuid4()),
        "company_name": company,
        "website": website,
        "category": category,
        "outreach_status": "not_contacted",
        "outreach_segment": "",
    }
    _mock_store.setdefault("crm_contacts", []).append(record)
    return record


class TestPromoteToCrm:
    """Tests for _execute_promote_to_crm."""

    def test_does_not_crash_on_null_website(self):
        """Existing contacts with website=None must not crash the function."""
        _seed_crm_contact("Existing Co", website=None)
        _seed_prospect("newdomain.com")

        from agents.seo_agent.worker import _execute_promote_to_crm

        result = _execute_promote_to_crm()

        assert result["promoted"] == 1

    def test_promotes_scored_prospects(self):
        """Scored prospects should be promoted to CRM contacts."""
        _seed_prospect("example.com")

        from agents.seo_agent.worker import _execute_promote_to_crm

        result = _execute_promote_to_crm()

        assert result["promoted"] == 1

        from agents.seo_agent.tools.supabase_tools import _mock_store

        contacts = _mock_store.get("crm_contacts", [])
        assert len(contacts) == 1
        assert contacts[0]["company_name"] == "Title for example.com"

    def test_skips_existing_domains(self):
        """Prospects whose domain already exists in CRM should be skipped."""
        _seed_crm_contact("Existing Co", website="https://example.com")
        _seed_prospect("example.com")

        from agents.seo_agent.worker import _execute_promote_to_crm

        result = _execute_promote_to_crm()

        assert result["promoted"] == 0

    def test_skips_empty_domain(self):
        """Prospects with no domain should be skipped."""
        _seed_prospect("", status="scored")

        from agents.seo_agent.worker import _execute_promote_to_crm

        result = _execute_promote_to_crm()

        assert result["promoted"] == 0

    def test_updates_prospect_status_after_promotion(self):
        """After promotion, the prospect status should change to 'promoted'."""
        prospect = _seed_prospect("example.com")

        from agents.seo_agent.worker import _execute_promote_to_crm

        _execute_promote_to_crm()

        from agents.seo_agent.tools.supabase_tools import _mock_store

        prospects = _mock_store.get("seo_backlink_prospects", [])
        updated = [p for p in prospects if p["id"] == prospect["id"]]
        assert len(updated) == 1
        assert updated[0]["status"] == "promoted"

    def test_blogger_category_for_unknown_segment(self):
        """Prospects with unrecognised segments get category 'blogger'."""
        _seed_prospect("blog.example.com", segment="random_blog")

        from agents.seo_agent.worker import _execute_promote_to_crm

        _execute_promote_to_crm()

        from agents.seo_agent.tools.supabase_tools import _mock_store

        contacts = _mock_store.get("crm_contacts", [])
        assert len(contacts) == 1
        assert contacts[0]["category"] == "blogger"

    def test_kitchen_category_for_kitchen_segment(self):
        """Prospects with 'kitchen' in segment get kitchen_company category."""
        _seed_prospect("kitchen.example.com", segment="kitchen_supplier")

        from agents.seo_agent.worker import _execute_promote_to_crm

        _execute_promote_to_crm()

        from agents.seo_agent.tools.supabase_tools import _mock_store

        contacts = _mock_store.get("crm_contacts", [])
        assert contacts[0]["category"] == "kitchen_company"


class TestCrmCategories:
    """Tests for CRM_CATEGORIES consistency."""

    def test_blogger_in_categories(self):
        """The 'blogger' category must be in CRM_CATEGORIES."""
        from agents.seo_agent.tools.crm_tools import CRM_CATEGORIES

        assert "blogger" in CRM_CATEGORIES

    def test_all_categories_have_subcategories(self):
        """Every category in CRM_CATEGORIES should have a subcategory entry."""
        from agents.seo_agent.tools.crm_tools import (
            CRM_CATEGORIES,
            CRM_SUBCATEGORIES,
        )

        for cat in CRM_CATEGORIES:
            assert cat in CRM_SUBCATEGORIES, f"Missing subcategories for {cat}"
