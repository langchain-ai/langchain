"""Backlink prospector node — discovers link-building opportunities.

Runs seven discovery methods (competitor backlink mining, content explorer,
unlinked mentions, resource pages, broken links, HARO requests, and niche
blog search via Tavily) and persists all prospects to Supabase with a
``discovery_method`` tag.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from agents.seo_agent.config import SITE_PROFILES
from agents.seo_agent.state import SEOAgentState
from agents.seo_agent.tools import ahrefs_tools, supabase_tools, web_search_tools

logger = logging.getLogger(__name__)

# Domains that look like link farms or PBNs — reject on sight
_LINK_FARM_SIGNALS = frozenset({
    "blogspot.com",
    "wordpress.com",
    "weebly.com",
    "wixsite.com",
    "tumblr.com",
    "medium.com",
})


def _extract_domain(url: str) -> str:
    """Extract the root domain from a URL.

    Args:
        url: A fully-qualified URL string.

    Returns:
        The netloc portion of the URL, lowercased.
    """
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_link_farm(domain: str) -> bool:
    """Check whether a domain belongs to a known link farm platform.

    Args:
        domain: The domain to check.

    Returns:
        True if the domain matches a known link farm host.
    """
    return any(domain.endswith(sig) for sig in _LINK_FARM_SIGNALS)


def _deduplicate_prospects(
    prospects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate prospects by page URL.

    Args:
        prospects: Raw list of prospect dicts.

    Returns:
        Deduplicated list preserving first occurrence.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for p in prospects:
        url = p.get("page_url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(p)
    return unique


def _mine_competitor_backlinks(
    competitors: list[str],
) -> list[dict[str, Any]]:
    """Run competitor backlink mining across all competitor domains.

    Args:
        competitors: List of competitor domain strings.

    Returns:
        List of prospect dicts tagged with ``discovery_method``.
    """
    prospects: list[dict[str, Any]] = []
    for competitor in competitors:
        try:
            backlinks = ahrefs_tools.get_backlinks.invoke(competitor)
        except Exception:
            logger.warning(
                "Failed to get backlinks for competitor %s",
                competitor,
                exc_info=True,
            )
            continue

        for bl in backlinks:
            domain = _extract_domain(bl.get("page_url", ""))
            if _is_link_farm(domain):
                logger.debug("Filtered link farm domain: %s", domain)
                continue

            prospects.append({
                "domain": domain or bl.get("referring_domain", ""),
                "page_url": bl.get("page_url", ""),
                "page_title": "",
                "dr": bl.get("dr", 0),
                "monthly_traffic": bl.get("traffic", 0),
                "discovery_method": "competitor_backlink",
                "links_to_competitor": True,
                "competitor_names": [competitor],
                "dofollow": bl.get("dofollow", False),
            })

    return prospects


def _explore_content(target_site: str) -> list[dict[str, Any]]:
    """Search Ahrefs Content Explorer for site-relevant pages.

    Args:
        target_site: The target site key used to look up the profile.

    Returns:
        List of prospect dicts from content explorer results.
    """
    profile = SITE_PROFILES.get(target_site, {})
    queries = [
        profile.get("primary_topic", ""),
        *profile.get("seed_keywords", [])[:3],
    ]
    queries = [q for q in queries if q]

    prospects: list[dict[str, Any]] = []
    for query in queries:
        try:
            results = ahrefs_tools.search_content_explorer.invoke(query)
        except Exception:
            logger.warning(
                "Content explorer failed for query '%s'",
                query,
                exc_info=True,
            )
            continue

        for r in results:
            domain = _extract_domain(r.get("url", ""))
            prospects.append({
                "domain": domain,
                "page_url": r.get("url", ""),
                "page_title": r.get("title", ""),
                "dr": r.get("dr", 0),
                "monthly_traffic": r.get("traffic", 0),
                "discovery_method": "content_explorer",
            })

    return prospects


def _find_unlinked_mentions(domain: str) -> list[dict[str, Any]]:
    """Search for unlinked brand mentions across the web.

    Args:
        domain: The target domain to find mentions of.

    Returns:
        List of prospect dicts for pages mentioning the domain without linking.
    """
    try:
        results = web_search_tools.find_unlinked_mentions(domain)
    except Exception:
        logger.warning(
            "Unlinked mentions search failed for %s", domain, exc_info=True
        )
        return []

    prospects: list[dict[str, Any]] = []
    for r in results:
        source_domain = _extract_domain(r.get("url", ""))
        prospects.append({
            "domain": source_domain,
            "page_url": r.get("url", ""),
            "page_title": r.get("title", ""),
            "discovery_method": "unlinked_mention",
            "has_link": r.get("has_link", False),
        })

    return prospects


def _find_resource_pages(niche: str) -> list[dict[str, Any]]:
    """Search for resource and links pages in the target niche.

    Args:
        niche: The niche topic to search for resource pages in.

    Returns:
        List of prospect dicts from resource page discovery.
    """
    try:
        results = web_search_tools.search_resource_pages(niche)
    except Exception:
        logger.warning(
            "Resource page search failed for niche '%s'", niche, exc_info=True
        )
        return []

    prospects: list[dict[str, Any]] = []
    for r in results:
        domain = _extract_domain(r.get("url", ""))
        prospects.append({
            "domain": domain,
            "page_url": r.get("url", ""),
            "page_title": r.get("title", ""),
            "discovery_method": "resource_page",
        })

    return prospects


def _find_broken_links(competitors: list[str]) -> list[dict[str, Any]]:
    """Find broken backlinks pointing to competitor domains.

    Args:
        competitors: List of competitor domain strings.

    Returns:
        List of prospect dicts for broken link replacement opportunities.
    """
    prospects: list[dict[str, Any]] = []
    for competitor in competitors:
        try:
            broken = ahrefs_tools.get_broken_backlinks.invoke(competitor)
        except Exception:
            logger.warning(
                "Broken backlinks check failed for %s",
                competitor,
                exc_info=True,
            )
            continue

        for bl in broken:
            domain = _extract_domain(bl.get("referring_page", ""))
            prospects.append({
                "domain": domain,
                "page_url": bl.get("referring_page", ""),
                "page_title": "",
                "dr": bl.get("dr", 0),
                "monthly_traffic": bl.get("traffic", 0),
                "discovery_method": "broken_link",
                "dead_url": bl.get("dead_url", ""),
                "anchor": bl.get("anchor", ""),
            })

    return prospects


def _search_haro() -> list[dict[str, Any]]:
    """Search for HARO (Help A Reporter Out) journalist requests.

    Returns:
        List of prospect dicts for HARO opportunities.
    """
    try:
        results = web_search_tools.search_haro_requests()
    except Exception:
        logger.warning("HARO search failed", exc_info=True)
        return []

    prospects: list[dict[str, Any]] = []
    for r in results:
        prospects.append({
            "domain": _extract_domain(r.get("url", "")),
            "page_url": r.get("url", ""),
            "page_title": r.get("title", ""),
            "discovery_method": "haro",
            "topic": r.get("topic", ""),
            "deadline": r.get("deadline", ""),
        })

    return prospects


def _search_niche_blogs(target_site: str) -> list[dict[str, Any]]:
    """Search for niche blogs relevant to our sites using web search.

    Uses Tavily (cheap) to discover blogs, then fetches DR from Ahrefs
    only for promising domains.
    """
    profile = SITE_PROFILES.get(target_site, {})

    # Build search queries targeting blogs and content sites in our niche
    domain = profile.get("domain", "")
    queries = [
        "best home renovation blogs UK",
        "kitchen design blog UK",
        "bathroom planning tips blog",
        "interior design bloggers UK home",
        "room planning tips blog homeowner",
        "property renovation blog UK",
        "home improvement advice blog",
    ]

    prospects: list[dict[str, Any]] = []
    seen_domains: set[str] = set()

    try:
        from agents.seo_agent.tools.web_search_tools import search
        for query in queries[:4]:  # Limit to 4 queries to manage API spend
            results = search(query, max_results=5)
            for r in results:
                result_domain = _extract_domain(r.get("url", ""))
                # Skip our own sites, link farms, and duplicates
                if result_domain in seen_domains:
                    continue
                if result_domain == domain or _is_link_farm(result_domain):
                    continue
                seen_domains.add(result_domain)

                prospects.append({
                    "domain": result_domain,
                    "page_url": r.get("url", ""),
                    "page_title": r.get("title", ""),
                    "dr": 0,  # Will be fetched during enrichment
                    "monthly_traffic": 0,
                    "discovery_method": "niche_blog_search",
                })
    except Exception:
        logger.warning("Niche blog search failed", exc_info=True)

    # Fetch DR for the top prospects (limit Ahrefs calls)
    from agents.seo_agent.tools.ahrefs_tools import get_domain_rating
    for p in prospects[:10]:  # Only check DR for first 10
        try:
            dr_data = get_domain_rating(p["domain"])
            if isinstance(dr_data, dict):
                p["dr"] = dr_data.get("domain_rating", 0)
            elif isinstance(dr_data, (int, float)):
                p["dr"] = dr_data
        except Exception:
            pass  # DR stays at 0, will be retried during enrichment

    logger.info("Niche blog search found %d prospects", len(prospects))
    return prospects


def run_backlink_prospector(state: SEOAgentState) -> dict[str, Any]:
    """Discover backlink prospects using seven complementary methods.

    Runs competitor backlink mining, content explorer, unlinked mentions,
    resource pages, broken links, HARO searches, and niche blog search.
    Deduplicates results, filters link farms, and saves all prospects to
    Supabase.

    Args:
        state: The current SEO agent state.

    Returns:
        State update with `backlink_prospects`, `errors`, and `next_node`.
    """
    errors: list[str] = list(state.get("errors", []))
    target_site = state["target_site"]

    profile = SITE_PROFILES.get(target_site)
    if profile is None:
        msg = f"No site profile found for '{target_site}'"
        logger.error(msg)
        errors.append(msg)
        return {
            "backlink_prospects": [],
            "errors": errors,
            "next_node": "END",
        }

    domain = profile.get("domain", "")
    competitors = profile.get("competitors", [])
    primary_topic = profile.get("primary_topic", "kitchen")
    # Extract a simple niche word from the primary topic for resource search
    niche = primary_topic.split()[0] if primary_topic else "kitchen"

    all_prospects: list[dict[str, Any]] = []

    # 1. Competitor Backlink Mining
    logger.info("Step 1/7: Mining competitor backlinks for %s", target_site)
    try:
        competitor_prospects = _mine_competitor_backlinks(competitors)
        all_prospects.extend(competitor_prospects)
        logger.info("Found %d competitor backlink prospects", len(competitor_prospects))
    except Exception as exc:
        msg = f"Competitor backlink mining failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # 2. Content Explorer
    logger.info("Step 2/7: Searching content explorer for %s", target_site)
    try:
        content_prospects = _explore_content(target_site)
        all_prospects.extend(content_prospects)
        logger.info("Found %d content explorer prospects", len(content_prospects))
    except Exception as exc:
        msg = f"Content explorer search failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # 3. Unlinked Mentions
    logger.info("Step 3/7: Finding unlinked mentions for %s", domain)
    try:
        mention_prospects = _find_unlinked_mentions(domain)
        all_prospects.extend(mention_prospects)
        logger.info("Found %d unlinked mention prospects", len(mention_prospects))
    except Exception as exc:
        msg = f"Unlinked mentions search failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # 4. Resource Pages
    logger.info("Step 4/7: Searching resource pages for niche '%s'", niche)
    try:
        resource_prospects = _find_resource_pages(niche)
        all_prospects.extend(resource_prospects)
        logger.info("Found %d resource page prospects", len(resource_prospects))
    except Exception as exc:
        msg = f"Resource page search failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # 5. Broken Links
    logger.info("Step 5/7: Checking broken links for competitors")
    try:
        broken_prospects = _find_broken_links(competitors)
        all_prospects.extend(broken_prospects)
        logger.info("Found %d broken link prospects", len(broken_prospects))
    except Exception as exc:
        msg = f"Broken link search failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # 6. HARO Requests
    logger.info("Step 6/7: Searching HARO requests")
    try:
        haro_prospects = _search_haro()
        all_prospects.extend(haro_prospects)
        logger.info("Found %d HARO prospects", len(haro_prospects))
    except Exception as exc:
        msg = f"HARO search failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # 7. Niche Blog Search (Tavily + Ahrefs DR)
    logger.info("Step 7/7: Searching for niche blogs relevant to %s", target_site)
    try:
        blog_prospects = _search_niche_blogs(target_site)
        all_prospects.extend(blog_prospects)
        logger.info("Found %d niche blog prospects", len(blog_prospects))
    except Exception as exc:
        msg = f"Niche blog search failed: {exc}"
        logger.error(msg, exc_info=True)
        errors.append(msg)

    # Deduplicate across all methods
    unique_prospects = _deduplicate_prospects(all_prospects)
    logger.info(
        "Total prospects after deduplication: %d (from %d raw)",
        len(unique_prospects),
        len(all_prospects),
    )

    # Save each prospect to Supabase
    saved_prospects: list[dict[str, Any]] = []
    for prospect in unique_prospects:
        record = {
            "domain": prospect.get("domain", ""),
            "page_url": prospect.get("page_url", ""),
            "page_title": prospect.get("page_title", ""),
            "dr": prospect.get("dr", 0),
            "monthly_traffic": prospect.get("monthly_traffic", 0),
            "discovery_method": prospect.get("discovery_method", "unknown"),
            "links_to_competitor": prospect.get("links_to_competitor", False),
            "competitor_names": prospect.get("competitor_names", []),
            "status": "new",
            "target_site": target_site,
        }
        try:
            saved = supabase_tools.insert_record(
                "seo_backlink_prospects", record
            )
            saved_prospects.append(saved)
        except Exception:
            msg = (
                f"Failed to save prospect '{prospect.get('page_url', '')}' "
                f"to Supabase"
            )
            logger.warning(msg, exc_info=True)
            errors.append(msg)

    # Save HARO requests with pending_review status
    for prospect in unique_prospects:
        if prospect.get("discovery_method") == "haro":
            try:
                supabase_tools.insert_record(
                    "haro_responses",
                    {
                        "request_topic": prospect.get("topic", ""),
                        "target_publication": prospect.get("page_title", ""),
                        "status": "pending_review",
                    },
                )
            except Exception:
                msg = f"Failed to save HARO response for '{prospect.get('topic', '')}'"
                logger.warning(msg, exc_info=True)
                errors.append(msg)

    logger.info(
        "Backlink prospecting complete for %s: %d prospects saved",
        target_site,
        len(saved_prospects),
    )

    return {
        "backlink_prospects": saved_prospects,
        "errors": errors,
        "next_node": "END",
    }
