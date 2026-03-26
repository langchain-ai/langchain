"""SEO audit tool — consistent framework for scoring and tracking site health.

Audits a website across 8 categories, scores each 0-100, and persists
results to Supabase for tracking improvements over time.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from agents.seo_agent.tools.supabase_tools import insert_record, query_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audit categories and weights
# ---------------------------------------------------------------------------

AUDIT_CATEGORIES = {
    "meta_tags": {"weight": 15, "description": "Title tags, meta descriptions, canonical URLs"},
    "headings": {"weight": 10, "description": "H1/H2/H3 hierarchy, keyword usage in headings"},
    "content_quality": {"weight": 20, "description": "Word count, readability, keyword density, E-E-A-T signals"},
    "technical": {"weight": 15, "description": "HTTPS, mobile viewport, page speed indicators, robots.txt"},
    "structured_data": {"weight": 15, "description": "JSON-LD schema markup, FAQ/HowTo/Article schema"},
    "internal_linking": {"weight": 10, "description": "Internal links, anchor text variety, orphan pages"},
    "images": {"weight": 5, "description": "Alt text, image compression indicators, lazy loading"},
    "aeo_readiness": {"weight": 10, "description": "AI engine optimisation: answer-first content, fact density, speakable markup"},
}


# ---------------------------------------------------------------------------
# Page fetcher
# ---------------------------------------------------------------------------

def _fetch_page(url: str) -> dict[str, Any]:
    """Fetch a page and extract raw HTML + headers."""
    try:
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "RalfSEOBot/1.0"})
            return {
                "url": str(resp.url),
                "status": resp.status_code,
                "html": resp.text,
                "headers": dict(resp.headers),
                "content_type": resp.headers.get("content-type", ""),
                "redirect_chain": [str(r.url) for r in resp.history],
            }
    except Exception as e:
        return {"url": url, "status": 0, "html": "", "error": str(e)}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_meta_tags(html: str, url: str) -> dict[str, Any]:
    """Check title, description, canonical, OG tags."""
    issues = []
    score = 100

    # Title
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    if not title:
        issues.append("Missing <title> tag")
        score -= 30
    elif len(title) > 60:
        issues.append(f"Title too long ({len(title)} chars, aim for <60)")
        score -= 10
    elif len(title) < 20:
        issues.append(f"Title too short ({len(title)} chars)")
        score -= 10

    # Meta description
    desc_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', html, re.IGNORECASE)
    desc = desc_match.group(1) if desc_match else ""
    if not desc:
        issues.append("Missing meta description")
        score -= 25
    elif len(desc) > 160:
        issues.append(f"Meta description too long ({len(desc)} chars, aim for <160)")
        score -= 5
    elif len(desc) < 70:
        issues.append(f"Meta description too short ({len(desc)} chars)")
        score -= 5

    # Canonical
    canonical = re.search(r'<link\s+rel=["\']canonical["\']\s+href=["\'](.*?)["\']', html, re.IGNORECASE)
    if not canonical:
        issues.append("Missing canonical URL")
        score -= 15

    # OG tags
    og_title = re.search(r'<meta\s+property=["\']og:title["\']\s+content=', html, re.IGNORECASE)
    og_desc = re.search(r'<meta\s+property=["\']og:description["\']\s+content=', html, re.IGNORECASE)
    if not og_title:
        issues.append("Missing og:title")
        score -= 10
    if not og_desc:
        issues.append("Missing og:description")
        score -= 5

    return {"score": max(0, score), "issues": issues, "title": title, "description": desc}


def _check_headings(html: str) -> dict[str, Any]:
    """Check heading hierarchy."""
    issues = []
    score = 100

    h1s = re.findall(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL | re.IGNORECASE)
    h2s = re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.DOTALL | re.IGNORECASE)
    h3s = re.findall(r'<h3[^>]*>(.*?)</h3>', html, re.DOTALL | re.IGNORECASE)

    if len(h1s) == 0:
        issues.append("No H1 tag found")
        score -= 40
    elif len(h1s) > 1:
        issues.append(f"Multiple H1 tags ({len(h1s)}) — should have exactly 1")
        score -= 20

    if len(h2s) == 0:
        issues.append("No H2 subheadings — add section headings for scannability")
        score -= 20

    if len(h2s) > 0 and len(h3s) == 0 and len(h2s) > 3:
        issues.append("No H3 tags — consider adding sub-sections under H2s")
        score -= 10

    return {
        "score": max(0, score),
        "issues": issues,
        "h1_count": len(h1s),
        "h2_count": len(h2s),
        "h3_count": len(h3s),
        "h1_text": [re.sub(r'<[^>]+>', '', h).strip() for h in h1s],
    }


def _check_content_quality(html: str) -> dict[str, Any]:
    """Check content depth and quality signals."""
    issues = []
    score = 100

    # Strip HTML tags for word count
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    word_count = len(text.split())

    if word_count < 300:
        issues.append(f"Thin content ({word_count} words — aim for 1000+ for blog posts)")
        score -= 40
    elif word_count < 800:
        issues.append(f"Light content ({word_count} words — 1500+ recommended for ranking)")
        score -= 20

    # Check for lists (good for readability)
    lists = len(re.findall(r'<(ul|ol)[^>]*>', html, re.IGNORECASE))
    if lists == 0 and word_count > 500:
        issues.append("No bullet/numbered lists — add lists for scannability")
        score -= 10

    # Check for tables (good for AEO)
    tables = len(re.findall(r'<table[^>]*>', html, re.IGNORECASE))

    # Check for FAQ signals
    faq_signals = bool(re.search(r'(FAQ|frequently asked|common questions)', html, re.IGNORECASE))

    return {
        "score": max(0, score),
        "issues": issues,
        "word_count": word_count,
        "has_lists": lists > 0,
        "has_tables": tables > 0,
        "has_faq": faq_signals,
    }


def _check_technical(html: str, page_data: dict) -> dict[str, Any]:
    """Check technical SEO fundamentals."""
    issues = []
    score = 100

    url = page_data.get("url", "")

    # HTTPS
    if not url.startswith("https://"):
        issues.append("Not using HTTPS")
        score -= 30

    # Viewport meta
    viewport = re.search(r'<meta\s+name=["\']viewport["\']', html, re.IGNORECASE)
    if not viewport:
        issues.append("Missing viewport meta tag (mobile-friendliness)")
        score -= 25

    # Robots meta
    noindex = re.search(r'<meta\s+name=["\']robots["\']\s+content=["\'][^"\']*noindex', html, re.IGNORECASE)
    if noindex:
        issues.append("Page has noindex — won't appear in search results")
        score -= 40

    # Language
    lang = re.search(r'<html[^>]*\slang=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if not lang:
        issues.append("Missing lang attribute on <html>")
        score -= 10

    # Charset
    charset = re.search(r'<meta\s+charset=|<meta\s+http-equiv=["\']Content-Type["\']', html, re.IGNORECASE)
    if not charset:
        issues.append("Missing charset declaration")
        score -= 5

    return {"score": max(0, score), "issues": issues, "is_https": url.startswith("https://")}


def _check_structured_data(html: str) -> dict[str, Any]:
    """Check for JSON-LD structured data."""
    issues = []
    score = 100

    # Find JSON-LD blocks
    ld_blocks = re.findall(r'<script\s+type=["\']application/ld\+json["\']\s*>(.*?)</script>', html, re.DOTALL | re.IGNORECASE)

    if not ld_blocks:
        issues.append("No JSON-LD structured data found — add Article, FAQPage, or Organization schema")
        score -= 50
    else:
        schema_types = []
        for block in ld_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict):
                    schema_types.append(data.get("@type", "unknown"))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            schema_types.append(item.get("@type", "unknown"))
            except json.JSONDecodeError:
                issues.append("Malformed JSON-LD block")
                score -= 10

        if "FAQPage" not in schema_types and "Question" not in str(schema_types):
            issues.append("No FAQ schema — add FAQPage for AI search visibility")
            score -= 15

    return {"score": max(0, score), "issues": issues, "schema_types": schema_types if ld_blocks else []}


def _check_internal_linking(html: str, base_url: str) -> dict[str, Any]:
    """Check internal linking structure."""
    issues = []
    score = 100

    domain = urlparse(base_url).netloc
    all_links = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE)

    internal = [l for l in all_links if domain in l or l.startswith("/")]
    external = [l for l in all_links if l.startswith("http") and domain not in l]

    if len(internal) < 3:
        issues.append(f"Only {len(internal)} internal links — aim for 5+ per page")
        score -= 25

    if len(external) == 0 and len(re.sub(r'<[^>]+>', '', html).split()) > 500:
        issues.append("No external links — linking to authoritative sources builds trust")
        score -= 10

    return {
        "score": max(0, score),
        "issues": issues,
        "internal_links": len(internal),
        "external_links": len(external),
    }


def _check_images(html: str) -> dict[str, Any]:
    """Check image optimisation."""
    issues = []
    score = 100

    images = re.findall(r'<img\s+[^>]*>', html, re.IGNORECASE)
    if not images:
        return {"score": 100, "issues": [], "image_count": 0, "missing_alt": 0}

    missing_alt = 0
    for img in images:
        if not re.search(r'alt=["\'][^"\']+["\']', img, re.IGNORECASE):
            missing_alt += 1

    if missing_alt > 0:
        issues.append(f"{missing_alt}/{len(images)} images missing alt text")
        score -= min(30, missing_alt * 10)

    # Check for lazy loading
    lazy = sum(1 for img in images if 'loading="lazy"' in img or 'loading=\'lazy\'' in img)

    return {
        "score": max(0, score),
        "issues": issues,
        "image_count": len(images),
        "missing_alt": missing_alt,
        "lazy_loaded": lazy,
    }


def _check_aeo_readiness(html: str) -> dict[str, Any]:
    """Check AI Engine Optimisation signals."""
    issues = []
    score = 100

    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    word_count = len(text.split())

    # Answer-first structure (check if first paragraph is substantial)
    first_p = re.search(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
    if first_p:
        first_p_text = re.sub(r'<[^>]+>', '', first_p.group(1)).strip()
        if len(first_p_text.split()) < 20:
            issues.append("First paragraph too short — AI engines prefer immediate, substantive answers")
            score -= 15

    # Fact density (numbers, percentages, dates)
    facts = len(re.findall(r'£[\d,]+|\$[\d,]+|\d+%|\d{4}|\d+\s*(million|billion|thousand)', text, re.IGNORECASE))
    if word_count > 500 and facts < 3:
        issues.append(f"Low fact density ({facts} data points) — AI engines cite fact-rich content (aim for 9+)")
        score -= 20

    # FAQ presence (strongly preferred by AI)
    has_faq = bool(re.search(r'(FAQ|frequently asked|common questions)', html, re.IGNORECASE))
    if not has_faq and word_count > 800:
        issues.append("No FAQ section — adding FAQs significantly improves AI citation rates")
        score -= 15

    # JSON-LD presence (for AI parsing)
    has_schema = bool(re.search(r'application/ld\+json', html, re.IGNORECASE))
    if not has_schema:
        issues.append("No structured data — AI engines parse schema markup 94% of the time vs 23% for JS-rendered content")
        score -= 20

    # Speakable markup
    has_speakable = bool(re.search(r'"speakable"', html, re.IGNORECASE))

    return {
        "score": max(0, score),
        "issues": issues,
        "fact_count": facts,
        "has_faq": has_faq,
        "has_schema": has_schema,
        "has_speakable": has_speakable,
    }


# ---------------------------------------------------------------------------
# Full audit
# ---------------------------------------------------------------------------

def audit_page(url: str) -> dict[str, Any]:
    """Run a full SEO audit on a single page."""
    page_data = _fetch_page(url)
    if page_data.get("error") or page_data.get("status", 0) >= 400:
        return {"url": url, "error": page_data.get("error", f"HTTP {page_data.get('status')}"), "overall_score": 0}

    html = page_data["html"]
    base_url = page_data["url"]

    results = {
        "meta_tags": _check_meta_tags(html, base_url),
        "headings": _check_headings(html),
        "content_quality": _check_content_quality(html),
        "technical": _check_technical(html, page_data),
        "structured_data": _check_structured_data(html),
        "internal_linking": _check_internal_linking(html, base_url),
        "images": _check_images(html),
        "aeo_readiness": _check_aeo_readiness(html),
    }

    # Calculate weighted overall score
    overall = sum(
        results[cat]["score"] * (AUDIT_CATEGORIES[cat]["weight"] / 100)
        for cat in AUDIT_CATEGORIES
    )

    all_issues = []
    for cat, data in results.items():
        for issue in data.get("issues", []):
            all_issues.append(f"[{cat}] {issue}")

    return {
        "url": base_url,
        "overall_score": round(overall),
        "categories": results,
        "all_issues": all_issues,
        "audited_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def audit_site(domain: str, pages: list[str] | None = None) -> dict[str, Any]:
    """Audit multiple pages on a site and produce a site-wide score.

    If no pages specified, audits the homepage and common pages.
    """
    if not pages:
        base = f"https://{domain}"
        pages = [
            base,
            f"{base}/blog",
        ]

    page_results = []
    for url in pages:
        logger.info("Auditing: %s", url)
        result = audit_page(url)
        page_results.append(result)

    # Site-wide score is average of page scores
    valid_scores = [r["overall_score"] for r in page_results if r.get("overall_score", 0) > 0]
    site_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0

    # Collect all unique issues
    all_issues = []
    seen = set()
    for r in page_results:
        for issue in r.get("all_issues", []):
            if issue not in seen:
                seen.add(issue)
                all_issues.append(issue)

    return {
        "domain": domain,
        "site_score": site_score,
        "pages_audited": len(page_results),
        "page_results": page_results,
        "all_issues": all_issues,
        "audited_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def save_audit(target_site: str, audit_result: dict) -> None:
    """Save audit results to Supabase for tracking over time."""
    try:
        insert_record("seo_content_performance", {
            "url": f"audit:{audit_result.get('domain', target_site)}",
            "title": f"SEO Audit — {audit_result.get('domain', target_site)}",
            "target_site": target_site,
            "target_keyword": "site_audit",
            "organic_traffic": audit_result.get("site_score", 0),  # Repurpose for score
            "keywords_ranking": audit_result.get("pages_audited", 0),
            "snapshot_date": date.today().isoformat(),
        })
    except Exception:
        logger.warning("Failed to save audit", exc_info=True)


def format_audit_report(audit_result: dict) -> str:
    """Format audit results into a readable report."""
    if audit_result.get("error"):
        return f"Audit failed for {audit_result.get('url', 'unknown')}: {audit_result['error']}"

    lines = []

    if "site_score" in audit_result:
        # Site-wide report
        lines.append(f"SEO Audit: {audit_result['domain']}")
        lines.append(f"Overall score: {audit_result['site_score']}/100")
        lines.append(f"Pages audited: {audit_result['pages_audited']}")
        lines.append("")

        for page in audit_result.get("page_results", []):
            lines.append(f"  {page.get('url', 'N/A')}: {page.get('overall_score', 0)}/100")

        lines.append("")
        lines.append(f"Issues found ({len(audit_result.get('all_issues', []))}):")
        for issue in audit_result.get("all_issues", [])[:15]:
            lines.append(f"  - {issue}")
        if len(audit_result.get("all_issues", [])) > 15:
            lines.append(f"  ...and {len(audit_result['all_issues']) - 15} more")

    else:
        # Single page report
        lines.append(f"Page audit: {audit_result.get('url', 'N/A')}")
        lines.append(f"Score: {audit_result.get('overall_score', 0)}/100")
        lines.append("")

        for cat, data in audit_result.get("categories", {}).items():
            weight = AUDIT_CATEGORIES.get(cat, {}).get("weight", 0)
            lines.append(f"  {cat}: {data.get('score', 0)}/100 (weight: {weight}%)")
            for issue in data.get("issues", []):
                lines.append(f"    - {issue}")

    return "\n".join(lines)
