"""GitHub tools — commit files to repos for blog publishing.

Requires ``GITHUB_TOKEN`` environment variable with repo write access.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Repo mapping: site key → GitHub repo and blog path
SITE_REPOS: dict[str, dict[str, str]] = {
    "freeroomplanner": {
        "repo": "benshevlane/roomsketch",
        "blog_path": "client/public/blog",
        "file_ext": ".html",
        "branch": "main",
    },
    "kitchensdirectory": {
        "repo": "",  # No repo yet — will need to be added
        "blog_path": "",
        "file_ext": ".html",
        "branch": "main",
    },
    "kitchen_estimator": {
        "repo": "benshevlane/KitchenCostEstimator",
        "blog_path": "src/content/blog",
        "file_ext": ".mdx",
        "branch": "main",
    },
}


def _get_token() -> str:
    """Get the GitHub token, stripping any whitespace."""
    return "".join(os.getenv("GITHUB_TOKEN", "").split())


def _get_client() -> httpx.Client:
    """Return an authenticated GitHub API client."""
    token = _get_token()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set")
    return httpx.Client(
        base_url="https://api.github.com",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
    )


def slugify(title: str) -> str:
    """Convert a title to a URL-friendly slug.

    Args:
        title: The blog post title.

    Returns:
        A lowercase, hyphenated slug.
    """
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug.strip("-")[:80]


def publish_blog_post(
    site: str,
    title: str,
    content: str,
    meta_description: str = "",
    author: str = "Ralf SEO Agent",
) -> dict[str, Any]:
    """Publish a blog post by committing it to the site's GitHub repo.

    Args:
        site: Site key (freeroomplanner, kitchensdirectory, kitchen_estimator).
        title: Blog post title.
        content: Blog post content (HTML or MDX depending on site).
        meta_description: SEO meta description.
        author: Author name.

    Returns:
        Dict with commit_url, file_path, slug, and published_url.
    """
    repo_config = SITE_REPOS.get(site)
    if not repo_config or not repo_config.get("repo"):
        raise ValueError(f"No GitHub repo configured for site '{site}'")

    repo = repo_config["repo"]
    blog_path = repo_config["blog_path"]
    ext = repo_config["file_ext"]
    branch = repo_config["branch"]
    slug = slugify(title)
    file_path = f"{blog_path}/{slug}{ext}"

    now = datetime.now(tz=timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # Build the file content based on site type
    if ext == ".html":
        file_content = _build_html_blog_post(title, content, meta_description, slug, site, date_str)
    elif ext == ".mdx":
        file_content = _build_mdx_blog_post(title, content, meta_description, slug, date_str, author)
    else:
        file_content = content

    # Commit to GitHub
    client = _get_client()
    encoded_content = base64.b64encode(file_content.encode()).decode()

    # Check if file already exists (to get SHA for update)
    sha = None
    try:
        resp = client.get(f"/repos/{repo}/contents/{quote(file_path, safe='/')}?ref={branch}")
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass

    body: dict[str, Any] = {
        "message": f"blog: {title}\n\nPublished by Ralf SEO Agent",
        "content": encoded_content,
        "branch": branch,
    }
    if sha:
        body["sha"] = sha

    resp = client.put(
        f"/repos/{repo}/contents/{quote(file_path, safe='/')}",
        json=body,
    )
    resp.raise_for_status()
    result = resp.json()

    # Determine the published URL
    domain_map = {
        "freeroomplanner": f"https://freeroomplanner.com/blog/{slug}",
        "kitchen_estimator": f"https://kitchencostestimator.com/blog/{slug}",
        "kitchensdirectory": f"https://kitchensdirectory.co.uk/blog/{slug}",
    }

    commit_url = result.get("commit", {}).get("html_url", "")
    logger.info("Published blog post: %s to %s (%s)", title, file_path, commit_url)

    return {
        "commit_url": commit_url,
        "file_path": file_path,
        "slug": slug,
        "published_url": domain_map.get(site, ""),
        "repo": repo,
    }


def list_blog_posts(site: str) -> list[dict[str, str]]:
    """List existing blog posts in a site's repo.

    Args:
        site: Site key.

    Returns:
        List of dicts with name, path, and url.
    """
    repo_config = SITE_REPOS.get(site)
    if not repo_config or not repo_config.get("repo"):
        return []

    client = _get_client()
    repo = repo_config["repo"]
    blog_path = repo_config["blog_path"]

    try:
        resp = client.get(f"/repos/{repo}/contents/{quote(blog_path, safe='/')}")
        if resp.status_code != 200:
            return []
        files = resp.json()
        return [
            {"name": f["name"], "path": f["path"], "url": f.get("html_url", "")}
            for f in files
            if f["type"] == "file" and f["name"] != "index.html"
        ]
    except Exception:
        logger.warning("Failed to list blog posts for %s", site, exc_info=True)
        return []


def _build_html_blog_post(
    title: str, content: str, meta_description: str, slug: str, site: str, date_str: str
) -> str:
    """Build a full HTML blog post page for static sites."""
    domain = {
        "freeroomplanner": "freeroomplanner.com",
        "kitchensdirectory": "kitchensdirectory.co.uk",
    }.get(site, "example.com")

    site_name = {
        "freeroomplanner": "Free Room Planner",
        "kitchensdirectory": "Kitchens Directory",
    }.get(site, "Blog")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | {site_name} Blog</title>
<meta name="description" content="{meta_description}">
<link rel="canonical" href="https://{domain}/blog/{slug}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="{site_name}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{meta_description}">
<meta property="og:url" content="https://{domain}/blog/{slug}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{meta_description}">
<meta name="author" content="Ralf SEO Agent">
<meta name="date" content="{date_str}">
<link rel="preconnect" href="https://api.fontshare.com">
<link href="https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap" rel="stylesheet">
<style>
  :root {{ --accent: #0d9488; --text: #1a1a2e; --muted: #64748b; --bg: #faf9f6; --surface: #fff; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'General Sans', system-ui, sans-serif; color: var(--text); background: var(--bg); line-height: 1.7; }}
  .container {{ max-width: 720px; margin: 0 auto; padding: 2rem 1.5rem; }}
  header {{ padding: 1rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 2rem; }}
  header a {{ color: var(--accent); text-decoration: none; font-weight: 600; }}
  h1 {{ font-size: 2rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.5rem; }}
  .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
  article h2 {{ font-size: 1.4rem; font-weight: 600; margin: 2rem 0 0.75rem; }}
  article h3 {{ font-size: 1.15rem; font-weight: 600; margin: 1.5rem 0 0.5rem; }}
  article p {{ margin-bottom: 1rem; }}
  article ul, article ol {{ margin: 0 0 1rem 1.5rem; }}
  article li {{ margin-bottom: 0.4rem; }}
  article a {{ color: var(--accent); }}
  article table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
  article th, article td {{ padding: 0.6rem; border: 1px solid #e2e8f0; text-align: left; }}
  article th {{ background: #f8fafc; font-weight: 600; }}
  .cta {{ background: var(--accent); color: #fff; padding: 1.5rem; border-radius: 8px; text-align: center; margin: 2rem 0; }}
  .cta a {{ color: #fff; font-weight: 600; text-decoration: underline; }}
  footer {{ margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; color: var(--muted); font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <a href="/">{site_name}</a> &rsaquo; <a href="/blog">Blog</a>
  </header>
  <h1>{title}</h1>
  <div class="meta">Published {date_str} &middot; by Ralf SEO Agent</div>
  <article>
{content}
  </article>
  <div class="cta">
    <p>Ready to plan your room? <a href="https://freeroomplanner.com">Start planning for free</a> — no sign-up required.</p>
  </div>
  <footer>
    <p>&copy; 2026 {site_name}. All rights reserved.</p>
    <p><a href="/">Home</a> &middot; <a href="/blog">Blog</a></p>
  </footer>
</div>
</body>
</html>"""


def _build_mdx_blog_post(
    title: str, content: str, meta_description: str, slug: str, date_str: str, author: str
) -> str:
    """Build an MDX blog post with frontmatter for Next.js sites."""
    return f"""---
title: "{title}"
description: "{meta_description}"
date: "{date_str}"
author: "{author}"
slug: "{slug}"
---

{content}
"""
