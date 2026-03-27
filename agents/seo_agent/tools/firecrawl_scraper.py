"""Firecrawl-powered scraper for finding kitchen & bathroom companies.

Uses the Firecrawl Search API to discover company websites and the Extract API
to pull structured contact data, then feeds results into the CRM.

Requires ``FIRECRAWL_API_KEY`` environment variable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

FIRECRAWL_BASE = "https://api.firecrawl.dev/v2"


def _get_api_key() -> str:
    key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not key:
        raise ValueError("FIRECRAWL_API_KEY not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_companies(query: str, country: str = "US", limit: int = 20) -> list[dict]:
    """Search for companies using Firecrawl Search API.

    Returns list of {url, title, description} dicts.
    """
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{FIRECRAWL_BASE}/search",
            headers=_headers(),
            json={
                "query": query,
                "limit": min(limit, 100),
                "country": country,
                "sources": [{"type": "web"}],
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("data", {}).get("web", []):
        url = item.get("url", "")
        if not url:
            continue
        results.append({
            "url": url,
            "title": item.get("title", ""),
            "description": item.get("description", ""),
        })

    logger.info("Firecrawl search '%s' (country=%s): %d results", query, country, len(results))
    return results


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string", "description": "The company or business name"},
        "email": {"type": "string", "description": "Contact email address"},
        "phone": {"type": "string", "description": "Phone number"},
        "city": {"type": "string", "description": "City where the company is based"},
        "region": {"type": "string", "description": "State, county, or region"},
        "description": {"type": "string", "description": "Brief description of what the company does"},
    },
    "required": ["company_name"],
}


def extract_company_data(urls: list[str]) -> list[dict]:
    """Extract structured company data from URLs using Firecrawl Extract API.

    Sends URLs to Firecrawl extract endpoint with a schema for company_name,
    email, phone, city, region, description.  Since extract is async, polls for
    results.

    Returns list of extracted company dicts.
    """
    if not urls:
        return []

    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{FIRECRAWL_BASE}/extract",
            headers=_headers(),
            json={
                "urls": urls,
                "prompt": (
                    "Extract the company name, contact email address, phone number, "
                    "city, and region from this business website. Focus on the contact "
                    "page, about page, and footer."
                ),
                "schema": _EXTRACT_SCHEMA,
                "enableWebSearch": False,
            },
        )
        resp.raise_for_status()
        job = resp.json()

        if not job.get("success"):
            logger.warning("Firecrawl extract failed: %s", job)
            return []

        job_id = job.get("id")
        if not job_id:
            # Synchronous response
            return job.get("data", [])

        # Poll for results (async job) — max ~5 minutes
        for _attempt in range(30):
            time.sleep(10)
            poll_resp = client.get(
                f"{FIRECRAWL_BASE}/extract/{job_id}",
                headers=_headers(),
            )
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            status = poll_data.get("status", "")
            if status == "completed":
                return poll_data.get("data", [])
            if status == "failed":
                logger.error("Firecrawl extract job failed: %s", poll_data)
                return []
            # still processing, continue polling

    logger.warning("Firecrawl extract timed out for job %s", job_id)
    return []


# ---------------------------------------------------------------------------
# Query sets by country and category
# ---------------------------------------------------------------------------

SEARCH_QUERIES: dict[str, dict[str, list[str]]] = {
    "UK": {
        "kitchen_company": [
            "kitchen showroom UK",
            "fitted kitchen company UK",
            "bespoke kitchen maker UK",
            "kitchen installation company",
            "kitchen design studio",
            "kitchen company London",
            "kitchen showroom Manchester",
            "kitchen fitter Birmingham",
            "kitchen company Leeds",
            "kitchen maker Bristol",
            "kitchen showroom Glasgow",
            "kitchen company Edinburgh",
            "fitted kitchen Liverpool",
            "kitchen designer Newcastle",
            "kitchen company Sheffield",
            "kitchen showroom Nottingham",
            "kitchen fitter Cardiff",
            "kitchen company Brighton",
            "kitchen showroom Southampton",
            "kitchen maker York",
        ],
        "bathroom_company": [
            "bathroom showroom UK",
            "bathroom fitting company UK",
            "bathroom renovation company UK",
            "bathroom design specialist",
            "bathroom installer UK",
            "bathroom company London",
            "bathroom showroom Manchester",
            "bathroom fitter Birmingham",
            "bathroom company Leeds",
            "bathroom showroom Bristol",
        ],
    },
    "US": {
        "kitchen_company": [
            "kitchen remodeling company",
            "kitchen cabinet company",
            "kitchen design firm",
            "kitchen renovation contractor",
            "custom kitchen company",
            "kitchen remodeling New York",
            "kitchen company Los Angeles",
            "kitchen renovation Chicago",
            "kitchen remodeling Houston",
            "kitchen company Phoenix",
            "kitchen design San Francisco",
            "kitchen renovation Miami",
            "kitchen company Dallas",
            "kitchen remodeling Seattle",
            "kitchen company Denver",
        ],
        "bathroom_company": [
            "bathroom remodeling company",
            "bathroom renovation contractor",
            "bathroom design company",
            "bathroom remodeling New York",
            "bathroom company Los Angeles",
            "bathroom renovation Chicago",
            "bathroom company Houston",
            "bathroom remodeling Miami",
        ],
    },
    "CA": {
        "kitchen_company": [
            "kitchen renovation company Canada",
            "kitchen cabinet maker Toronto",
            "kitchen company Vancouver",
            "kitchen renovation Calgary",
            "kitchen fitter Montreal",
            "kitchen company Ottawa",
            "kitchen renovation Edmonton",
            "kitchen company Winnipeg",
        ],
        "bathroom_company": [
            "bathroom renovation company Canada",
            "bathroom company Toronto",
            "bathroom renovation Vancouver",
            "bathroom company Calgary",
            "bathroom fitter Montreal",
        ],
    },
}


# ---------------------------------------------------------------------------
# Skip list — domains to ignore
# ---------------------------------------------------------------------------

SKIP_DOMAINS = {
    # Major retailers
    "bq.co.uk", "diy.com", "ikea.com", "wickes.co.uk", "howdens.com",
    "magnet.co.uk", "wren.co.uk", "homedepot.com", "lowes.com",
    # Directories and marketplaces
    "checkatrade.com", "mybuilder.com", "trustatrader.com", "bark.com",
    "yell.com", "yelp.com", "houzz.com", "houzz.co.uk", "thumbtack.com",
    "homeadvisor.com", "angi.com", "angieslist.com",
    # Social / generic
    "facebook.com", "instagram.com", "pinterest.com", "twitter.com",
    "linkedin.com", "youtube.com", "google.com", "amazon.com",
    "amazon.co.uk", "ebay.com", "ebay.co.uk",
    # Our own sites
    "freeroomplanner.com", "kitchencostestimator.com", "kitchensdirectory.co.uk",
    "ralfseo.com",
}


def _should_skip(url: str) -> bool:
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    return any(domain.endswith(d) for d in SKIP_DOMAINS)


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------


def run_scraper(
    country: str = "UK",
    category: str = "kitchen_company",
    max_queries: int = 10,
    results_per_query: int = 20,
    extract_batch_size: int = 5,
) -> dict[str, Any]:
    """Run a scraping session: search for companies, extract data, add to CRM.

    Args:
        country: ISO country code (UK, US, CA).
        category: kitchen_company or bathroom_company.
        max_queries: Maximum number of search queries to run.
        results_per_query: Results per search query.
        extract_batch_size: Number of URLs to extract at once.

    Returns:
        Summary dict with counts.
    """
    from agents.seo_agent.tools.crm_tools import add_crm_contact, get_crm_contacts

    queries = SEARCH_QUERIES.get(country, {}).get(category, [])[:max_queries]
    if not queries:
        return {"error": f"No queries for country={country}, category={category}"}

    # Get existing CRM domains to avoid duplicates
    existing = get_crm_contacts(limit=5000)
    existing_domains: set[str] = set()
    for c in existing:
        w = c.get("website", "")
        if w:
            existing_domains.add(urlparse(w).netloc.lower().removeprefix("www."))

    logger.info(
        "Starting scraper: country=%s, category=%s, %d queries, %d existing CRM contacts",
        country, category, len(queries), len(existing_domains),
    )

    # Phase 1: Search
    all_urls: list[dict] = []
    seen_domains: set[str] = set()

    for query in queries:
        try:
            results = search_companies(query, country=country, limit=results_per_query)
            for r in results:
                url = r["url"]
                domain = urlparse(url).netloc.lower().removeprefix("www.")

                if _should_skip(url):
                    continue
                if domain in seen_domains or domain in existing_domains:
                    continue

                seen_domains.add(domain)
                all_urls.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "description": r.get("description", ""),
                    "domain": domain,
                })
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query, e)

    logger.info("Search phase complete: %d unique company URLs found", len(all_urls))

    # Phase 2: Extract contact data in batches
    extracted: list[dict] = []
    for i in range(0, len(all_urls), extract_batch_size):
        batch = all_urls[i : i + extract_batch_size]
        batch_urls = [item["url"] for item in batch]

        try:
            batch_data = extract_company_data(batch_urls)

            for j, item in enumerate(batch):
                company_data = batch_data[j] if j < len(batch_data) else {}
                if isinstance(company_data, dict):
                    item.update(company_data)
                extracted.append(item)
        except Exception as e:
            logger.warning("Extract failed for batch %d: %s", i, e)
            # Still add with basic info from search
            for item in batch:
                extracted.append(item)

    # Phase 3: Add to CRM
    added = 0
    skipped = 0
    country_map = {"UK": "GB", "US": "US", "CA": "CA"}

    for item in extracted:
        company_name = item.get("company_name") or item.get("title", "")
        if not company_name or len(company_name) < 3:
            skipped += 1
            continue

        domain = item.get("domain", "")
        if domain in existing_domains:
            skipped += 1
            continue

        try:
            add_crm_contact(
                company_name=company_name[:200],
                category=category,
                website=item.get("url", f"https://{domain}"),
                email=item.get("email", ""),
                phone=item.get("phone", ""),
                city=item.get("city", ""),
                region=item.get("region", ""),
                country=country_map.get(country, country),
                notes=item.get("description", "")[:500],
                source="firecrawl_scraper",
                outreach_segment="kitchen_bathroom_providers",
            )
            existing_domains.add(domain)
            added += 1
        except Exception as e:
            logger.warning("Failed to add %s to CRM: %s", company_name, e)
            skipped += 1

    summary = {
        "country": country,
        "category": category,
        "queries_run": len(queries),
        "urls_found": len(all_urls),
        "extracted": len(extracted),
        "added_to_crm": added,
        "skipped": skipped,
    }
    logger.info("Scraper complete: %s", summary)
    return summary


def run_full_scrape(max_queries_per_country: int = 5) -> dict[str, Any]:
    """Run scraper across all countries and categories.

    Returns combined summary.
    """
    results: dict[str, Any] = {}
    for country in ["UK", "US", "CA"]:
        for category in ["kitchen_company", "bathroom_company"]:
            key = f"{country}_{category}"
            try:
                results[key] = run_scraper(
                    country=country,
                    category=category,
                    max_queries=max_queries_per_country,
                )
            except Exception as e:
                results[key] = {"error": str(e)[:200]}

    total_added = sum(r.get("added_to_crm", 0) for r in results.values())
    results["total_added"] = total_added
    return results
