"""CRM client — add companies and check for duplicates."""

import logging
from typing import Any
from urllib.parse import urlparse

from agents.scraper_agent.tools.supabase_client import insert_record, query_table

logger = logging.getLogger(__name__)


def get_existing_domains(limit: int = 5000) -> set[str]:
    """Get all domains already in the CRM to avoid duplicates."""
    contacts = query_table("crm_contacts", limit=limit)
    domains = set()
    for c in contacts:
        website = c.get("website", "")
        if website:
            domain = urlparse(website).netloc.lower().removeprefix("www.")
            if domain:
                domains.add(domain)
    return domains


def add_company(
    company_name: str,
    website: str,
    email: str = "",
    phone: str = "",
    city: str = "",
    region: str = "",
    country: str = "GB",
    category: str = "kitchen_company",
    description: str = "",
) -> dict | None:
    """Add a company to the CRM. Returns the record or None on failure."""
    try:
        return insert_record("crm_contacts", {
            "company_name": company_name[:200],
            "website": website,
            "email": email[:200] if email else "",
            "phone": phone[:50] if phone else "",
            "city": city[:100] if city else "",
            "region": region[:100] if region else "",
            "country": country,
            "category": category,
            "source": "firecrawl_scraper",
            "outreach_status": "not_contacted",
            "outreach_segment": "kitchen_bathroom_providers",
            "notes": description[:500] if description else "",
        })
    except Exception as e:
        logger.error("Failed to add %s to CRM: %s", company_name, e, exc_info=True)
        return None


def get_crm_stats() -> dict[str, Any]:
    """Get CRM stats by country, category, and source."""
    contacts = query_table("crm_contacts", limit=10000)
    by_country: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for c in contacts:
        co = c.get("country", "?")
        by_country[co] = by_country.get(co, 0) + 1
        cat = c.get("category", "?")
        by_category[cat] = by_category.get(cat, 0) + 1
        src = c.get("source", "?")
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "total": len(contacts),
        "by_country": by_country,
        "by_category": by_category,
        "by_source": by_source,
    }
