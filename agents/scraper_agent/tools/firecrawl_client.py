"""Firecrawl-powered scraper for finding kitchen & bathroom companies.

Uses the Firecrawl Search API to discover company websites and the Extract API
to pull structured contact data, then feeds results into the shared CRM.

Requires ``FIRECRAWL_API_KEY`` environment variable.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from agents.scraper_agent.config import FIRECRAWL_API_KEY, SEARCH_QUERIES, SKIP_DOMAINS

logger = logging.getLogger(__name__)

FIRECRAWL_BASE = "https://api.firecrawl.dev/v2"


def _get_api_key() -> str:
    key = FIRECRAWL_API_KEY
    if not key:
        raise ValueError("FIRECRAWL_API_KEY not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _should_skip(url: str) -> bool:
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    return any(domain.endswith(d) for d in SKIP_DOMAINS)


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
    from agents.scraper_agent.tools.crm_client import get_existing_domains, add_company

    queries = SEARCH_QUERIES.get(country, {}).get(category, [])[:max_queries]
    if not queries:
        return {"error": f"No queries for country={country}, category={category}"}

    # Get existing CRM domains to avoid duplicates
    existing_domains = get_existing_domains()

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

        result = add_company(
            company_name=company_name,
            website=item.get("url", f"https://{domain}"),
            email=item.get("email", ""),
            phone=item.get("phone", ""),
            city=item.get("city", ""),
            region=item.get("region", ""),
            country=country_map.get(country, country),
            category=category,
            description=item.get("description", ""),
        )
        if result:
            existing_domains.add(domain)
            added += 1
        else:
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


def run_daily_batch(daily_target: int = 50) -> dict[str, Any]:
    """Run a daily batch targeting a specific number of new companies.

    Rotates through countries and categories to build a diverse pipeline.
    Stops once daily_target is reached or all queries exhausted.
    """
    from agents.scraper_agent.tools.crm_client import get_existing_domains, add_company

    existing = get_existing_domains()
    added_total = 0
    results = {}

    # Rotate: UK kitchen → UK bathroom → US kitchen → US bathroom → CA kitchen → CA bathroom
    rotation = []
    for country in ["UK", "US", "CA"]:
        for category in ["kitchen_company", "bathroom_company"]:
            queries = SEARCH_QUERIES.get(country, {}).get(category, [])
            if queries:
                rotation.append((country, category, queries))

    query_index = 0  # Track which query to use next across rotations

    while added_total < daily_target and rotation:
        for country, category, queries in rotation:
            if added_total >= daily_target:
                break

            # Pick the next unused query for this country/category
            qi = query_index % len(queries)
            query = queries[qi]

            try:
                search_results = search_companies(query, country=country, limit=15)

                batch_urls = []
                for r in search_results:
                    url = r["url"]
                    domain = urlparse(url).netloc.lower().removeprefix("www.")
                    if _should_skip(url) or domain in existing:
                        continue
                    batch_urls.append(r)
                    existing.add(domain)

                if batch_urls:
                    urls_to_extract = [b["url"] for b in batch_urls[:5]]
                    try:
                        extracted = extract_company_data(urls_to_extract)
                    except Exception:
                        extracted = []

                    country_map = {"UK": "GB", "US": "US", "CA": "CA"}
                    for i, item in enumerate(batch_urls):
                        if added_total >= daily_target:
                            break

                        ext_data = extracted[i] if i < len(extracted) and isinstance(extracted[i], dict) else {}
                        company_name = ext_data.get("company_name") or item.get("title", "")
                        if not company_name or len(company_name) < 3:
                            continue

                        result = add_company(
                            company_name=company_name,
                            website=item["url"],
                            email=ext_data.get("email", ""),
                            phone=ext_data.get("phone", ""),
                            city=ext_data.get("city", ""),
                            region=ext_data.get("region", ""),
                            country=country_map.get(country, country),
                            category=category,
                            description=ext_data.get("description", item.get("description", "")),
                        )
                        if result:
                            added_total += 1

                key = f"{country}_{category}"
                results[key] = results.get(key, 0) + len(batch_urls)

            except Exception as e:
                logger.warning("Batch query failed '%s': %s", query, e)

        query_index += 1
        # Safety: if we've cycled through all queries, stop
        max_queries = max(len(q) for _, _, q in rotation) if rotation else 0
        if query_index >= max_queries:
            break

    return {
        "daily_target": daily_target,
        "added": added_total,
        "by_segment": results,
    }
