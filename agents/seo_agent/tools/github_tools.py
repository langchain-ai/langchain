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
        "repo": "benshevlane/kitchensdirectory.co.uk",
        "blog_path": "supabase:feature_articles",  # Uses Supabase, not file commits
        "file_ext": ".md",
        "branch": "main",
        "publish_via": "supabase",  # Flag to use Supabase instead of GitHub
    },
    "kitchen_estimator": {
        "repo": "benshevlane/KitchenCostEstimator",
        "blog_path": "src/content/blog",
        "file_ext": ".ts",
        "branch": "main",
    },
    "ralf_seo": {
        "repo": "benshevlane/ralf-seo",
        "blog_path": "posts",
        "file_ext": ".html",
        "branch": "master",
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
    author: str = "",
    **kwargs: Any,
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
    if not repo_config:
        raise ValueError(f"No repo configured for site '{site}'")

    slug = slugify(title)
    now = datetime.now(tz=timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # Kitchensdirectory publishes via Supabase, not GitHub
    if repo_config.get("publish_via") == "supabase":
        return _publish_to_supabase(
            site=site, slug=slug, title=title, content=content,
            meta_description=meta_description, author=author, date_str=date_str,
        )

    if not repo_config.get("repo"):
        raise ValueError(f"No GitHub repo configured for site '{site}'")

    repo = repo_config["repo"]
    blog_path = repo_config["blog_path"]
    ext = repo_config["file_ext"]
    branch = repo_config["branch"]
    file_path = f"{blog_path}/{slug}{ext}"

    # Sanitize content before publishing to prevent sensitive data leaks
    try:
        from agents.seo_agent.tools.reflection_engine import sanitize_content
        content = sanitize_content(content)
        title = sanitize_content(title)
        meta_description = sanitize_content(meta_description)
    except ImportError:
        pass

    # Build the file content based on site type
    if site == "ralf_seo":
        # Ralf's personal blog uses a different template
        category = kwargs.get("category", "Field Report")
        what_i_learned = kwargs.get("what_i_learned", [])
        file_content = _build_ralf_blog_post(title, content, meta_description, slug, date_str, category, what_i_learned)
    elif site == "kitchen_estimator":
        file_content = _build_ts_blog_post(title, content, meta_description, slug, date_str)
    elif ext == ".html":
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
        "message": f"blog: {title}",
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
        "ralf_seo": f"https://ralfseo.com/posts/{slug}",
    }

    commit_url = result.get("commit", {}).get("html_url", "")
    logger.info("Published blog post: %s to %s (%s)", title, file_path, commit_url)

    # Update the blog index page for freeroomplanner
    if site == "freeroomplanner":
        _update_blog_index(repo, branch, slug, title, meta_description)
    elif site == "ralf_seo":
        category = kwargs.get("category", "Field Report")
        _update_ralf_blog_index(repo, branch, slug, title, meta_description, date_str, category)
    elif site == "kitchen_estimator":
        _update_kce_blog_index(repo, branch, slug)

    published_url = domain_map.get(site, "")

    # Auto-track the target keyword for ranking monitoring
    try:
        from agents.seo_agent.tools.crm_tools import add_tracked_keyword
        add_tracked_keyword(site, title, published_url)
    except Exception:
        logger.debug("Failed to auto-track keyword after publish", exc_info=True)

    return {
        "commit_url": commit_url,
        "file_path": file_path,
        "slug": slug,
        "published_url": published_url,
        "repo": repo,
    }


def _categorize_post(title: str) -> str:
    """Determine the category tag for a blog post based on title keywords."""
    lower = title.lower()
    if "kitchen" in lower:
        return "Kitchen Planning"
    if "bathroom" in lower:
        return "Bathroom Planning"
    if "bedroom" in lower:
        return "Bedroom Planning"
    if "living room" in lower:
        return "Room Planning"
    if "extension" in lower:
        return "Extensions"
    if "floor plan" in lower:
        return "Room Planning"
    return "Home Renovation"


def _update_blog_index(
    repo: str, branch: str, slug: str, title: str, meta_description: str
) -> None:
    """Insert a card for a new post into the blog index page.

    This fetches client/public/blog/index.html, inserts a post card at the top
    of the "Latest articles" section, and commits the update. If anything fails,
    the error is logged but not raised — the blog post itself is already live.
    """
    index_path = "client/public/blog/index.html"
    try:
        client = _get_client()

        # Fetch current index.html
        resp = client.get(
            f"/repos/{repo}/contents/{quote(index_path, safe='/')}?ref={branch}"
        )
        resp.raise_for_status()
        data = resp.json()
        sha = data["sha"]
        current_content = base64.b64decode(data["content"]).decode()

        # Find the "Latest articles" section and the first <article tag after it
        marker = "Latest articles"
        marker_pos = current_content.find(marker)
        if marker_pos == -1:
            logger.warning("Could not find 'Latest articles' in blog index")
            return

        article_pos = current_content.find("<article", marker_pos)
        if article_pos == -1:
            logger.warning("Could not find <article> tag after 'Latest articles'")
            return

        # Skip if this slug already exists in the index (prevent duplicates)
        if f'/blog/{slug}"' in current_content or f'/blog/{slug}\'' in current_content:
            logger.info("Slug '%s' already in blog index, skipping update", slug)
            return

        # Build the new card HTML — hardcoded format matching the blog's card structure
        category_tag = _categorize_post(title)
        # HTML-escape the title and description for safe insertion
        safe_title = title.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        safe_desc = meta_description.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        new_card = (
            f'<article class="post-card fade-in" style="transition-delay:0ms" '
            f'itemscope itemtype="https://schema.org/BlogPosting">\n'
            f'  <div class="post-card__body">\n'
            f'    <div class="post-card__tag" itemprop="keywords">{category_tag}</div>\n'
            f'    <h2 class="post-card__title" itemprop="headline">'
            f'<a href="/blog/{slug}">{safe_title}</a></h2>\n'
            f'    <p class="post-card__excerpt" itemprop="description">'
            f'{safe_desc}</p>\n'
            f'    <div class="post-card__meta">5 min read</div>\n'
            f'  </div>\n'
            f'</article>'
        )

        # Insert the new card before the first existing article
        updated_content = current_content[:article_pos] + new_card + current_content[article_pos:]

        # Commit the updated index
        encoded = base64.b64encode(updated_content.encode()).decode()
        put_resp = client.put(
            f"/repos/{repo}/contents/{quote(index_path, safe='/')}",
            json={
                "message": f"blog index: add {title}",
                "content": encoded,
                "sha": sha,
                "branch": branch,
            },
        )
        put_resp.raise_for_status()
        logger.info("Updated blog index with new post: %s", title)
    except Exception:
        logger.error("Failed to update blog index for '%s'", title, exc_info=True)


def _build_ralf_blog_post(title: str, content: str, meta_description: str, slug: str, date_str: str, category: str, what_i_learned: list) -> str:
    """Build an HTML blog post for Ralf's personal blog (simplified template)."""
    safe_title = title.replace('"', '&quot;')
    safe_desc = meta_description.replace('"', '&quot;')
    learned_html = "\n".join(f"        <li>{item}</li>" for item in what_i_learned)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title} \u2014 Ralf</title>
<meta name="description" content="{safe_desc}">
<meta name="generator" content="Perplexity Computer">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="icon" type="image/svg+xml" href="../assets/favicon.svg">
<link rel="stylesheet" href="../base.css">
<link rel="stylesheet" href="../style.css">
</head>
<body>
<header>
  <div class="wrap">
    <a href="../" class="logo">ralf</a>
    <nav>
      <a href="../about">about</a>
      <button class="theme-btn" id="theme-toggle" aria-label="Toggle theme">\u263e</button>
    </nav>
  </div>
</header>
<main class="wrap">
  <div class="post-header">
    <div class="meta">{date_str} \u00b7 {category}</div>
    <h1>{safe_title}</h1>
  </div>
  <div class="post-body">
    {content}
    <div class="learned">
      <h2>what i learned</h2>
      <ul>
{learned_html}
      </ul>
    </div>
  </div>
  <a href="../" class="back">\u2190 Back to posts</a>
</main>
<footer>
  <div class="wrap">
    <span>ralf \u2014 autonomous seo agent</span>
    <a href="https://www.perplexity.ai/computer" target="_blank" rel="noopener">Built with Perplexity Computer</a>
  </div>
</footer>
<script>
(function(){{
  const t=document.getElementById('theme-toggle'),r=document.documentElement;
  let d=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
  r.setAttribute('data-theme',d);t.textContent=d==='dark'?'\u2600':'\u263e';
  t.addEventListener('click',()=>{{d=d==='dark'?'light':'dark';r.setAttribute('data-theme',d);t.textContent=d==='dark'?'\u2600':'\u263e';}});
}})();
</script>
</body>
</html>"""


def _update_ralf_blog_index(
    repo: str, branch: str, slug: str, title: str,
    meta_description: str, date_str: str, category: str,
) -> None:
    """Insert a card for a new post into Ralf's blog index page.

    Fetches the root index.html, inserts a post card after the
    <!-- POSTS_START --> marker, and commits the update.
    """
    index_path = "index.html"
    try:
        client = _get_client()

        # Fetch current index.html
        resp = client.get(
            f"/repos/{repo}/contents/{quote(index_path, safe='/')}?ref={branch}"
        )
        resp.raise_for_status()
        data = resp.json()
        sha = data["sha"]
        current_content = base64.b64decode(data["content"]).decode()

        # Find the POSTS_START marker or first post-card
        marker = "<!-- POSTS_START -->"
        marker_pos = current_content.find(marker)

        if marker_pos != -1:
            insert_pos = marker_pos + len(marker)
        else:
            # Fallback: find the first post-card article
            insert_pos = current_content.find('<article class="post-card"')
            if insert_pos == -1:
                logger.warning("Could not find insertion point in ralf-seo index.html")
                return

        # Skip if this slug already exists in the index
        if f'/posts/{slug}"' in current_content or f'/posts/{slug}\'' in current_content:
            logger.info("Slug '%s' already in ralf-seo blog index, skipping", slug)
            return

        # Build the new card HTML (simplified li > a format)
        safe_title = title.replace('&', '&amp;').replace('<', '&lt;')
        safe_desc = meta_description.replace('&', '&amp;').replace('<', '&lt;')
        new_card = (
            f'\n    <li>\n'
            f'      <a href="/posts/{slug}">\n'
            f'        <div class="meta">{date_str} <span class="tag">{category}</span></div>\n'
            f'        <h2>{safe_title}</h2>\n'
            f'        <p class="excerpt">{safe_desc}</p>\n'
            f'      </a>\n'
            f'    </li>'
        )

        # Insert the new card
        updated_content = current_content[:insert_pos] + new_card + current_content[insert_pos:]

        # Commit the updated index
        encoded = base64.b64encode(updated_content.encode()).decode()
        put_resp = client.put(
            f"/repos/{repo}/contents/{quote(index_path, safe='/')}",
            json={
                "message": f"blog index: add {title}",
                "content": encoded,
                "sha": sha,
                "branch": branch,
            },
        )
        put_resp.raise_for_status()
        logger.info("Updated ralf-seo blog index with new post: %s", title)
    except Exception:
        logger.error("Failed to update ralf-seo blog index for '%s'", title, exc_info=True)


def _update_kce_blog_index(repo: str, branch: str, slug: str) -> None:
    """Update KitchenCostEstimator's src/content/blog/index.ts to import and register a new post.

    Fetches the index.ts file, adds an import for the new post, and appends
    the export name to the allBlogPosts array. If anything fails, the error
    is logged but not raised — the blog post itself is already live.
    """
    index_path = "src/content/blog/index.ts"
    # Convert slug to camelCase export name (same logic as _build_ts_blog_post)
    parts = slug.replace('-', ' ').split()
    export_name = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])

    try:
        client = _get_client()

        # Fetch current index.ts
        resp = client.get(
            f"/repos/{repo}/contents/{quote(index_path, safe='/')}?ref={branch}"
        )
        resp.raise_for_status()
        data = resp.json()
        sha = data["sha"]
        current_content = base64.b64decode(data["content"]).decode()

        # Skip if this export is already imported (prevent duplicates)
        if export_name in current_content:
            logger.info("Export '%s' already in KCE blog index, skipping", export_name)
            return

        # Add import line after the last existing import
        import_line = f"import {{ {export_name} }} from './{slug}';"
        lines = current_content.split('\n')
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('import '):
                last_import_idx = i
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_line)
        else:
            # No imports found — add at the top
            lines.insert(0, import_line)

        updated_content = '\n'.join(lines)

        # Add the export name to the allBlogPosts array
        import re
        updated_content = re.sub(
            r'(const\s+allBlogPosts\s*:\s*BlogPost\[\]\s*=\s*\[)',
            rf'\1{export_name}, ',
            updated_content,
        )

        # Commit the updated index
        encoded = base64.b64encode(updated_content.encode()).decode()
        put_resp = client.put(
            f"/repos/{repo}/contents/{quote(index_path, safe='/')}",
            json={
                "message": f"blog index: register {slug}",
                "content": encoded,
                "sha": sha,
                "branch": branch,
            },
        )
        put_resp.raise_for_status()
        logger.info("Updated KCE blog index with new post: %s", slug)
    except Exception:
        logger.error("Failed to update KCE blog index for '%s'", slug, exc_info=True)


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
  <div class="meta">Published {date_str}</div>
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


def _build_ts_blog_post(title: str, content: str, meta_description: str, slug: str, date_str: str) -> str:
    """Build a TypeScript blog post file for KitchenCostEstimator."""
    # Convert slug to camelCase export name
    parts = slug.replace('-', ' ').split()
    export_name = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])

    # Estimate reading time (200 words per minute)
    word_count = len(content.split())
    reading_time = max(1, round(word_count / 200))

    # Extract potential tags from title
    tags = []
    title_lower = title.lower()
    if 'uk' in title_lower: tags.append('UK')
    if 'cost' in title_lower or 'price' in title_lower: tags.append('kitchen cost')
    if '2026' in title_lower: tags.append('2026')
    if 'renovation' in title_lower: tags.append('renovation')
    if not tags: tags = ['kitchen', 'guide']

    # The content should be markdown. If it contains HTML tags, convert basic ones.
    import re
    md_content = content
    md_content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', md_content)
    md_content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', md_content)
    md_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n', md_content)
    md_content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', md_content)
    md_content = re.sub(r'<em>(.*?)</em>', r'*\1*', md_content)
    md_content = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', md_content)
    md_content = re.sub(r'</?[uo]l[^>]*>', '', md_content)
    md_content = re.sub(r'<a href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', md_content)
    md_content = re.sub(r'</?[^>]+>', '', md_content)  # strip remaining HTML
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)  # collapse multiple newlines

    # Escape backticks in content for the template literal
    md_content = md_content.replace('`', '\\`').replace('${', '\\${')

    # Escape quotes in title and description for the TS string
    safe_title = title.replace("'", "\\'")
    safe_desc = meta_description.replace("'", "\\'")
    tags_str = ', '.join(f"'{t}'" for t in tags)

    return f"""import type {{ BlogPost }} from './types';

export const {export_name}: BlogPost = {{
  title: '{safe_title}',
  slug: '{slug}',
  description: '{safe_desc}',
  date: '{date_str}',
  author: {{
    name: 'Kitchen Cost Estimator',
    role: 'Editorial',
  }},
  tags: [{tags_str}],
  readingTime: {reading_time},
  content: `{md_content}`,
}};
"""


def _publish_to_supabase(
    site: str,
    slug: str,
    title: str,
    content: str,
    meta_description: str,
    author: str,
    date_str: str,
) -> dict[str, Any]:
    """Publish a blog post to kitchensdirectory's Supabase feature_articles table."""
    from agents.seo_agent.tools.supabase_tools import get_client

    client = get_client()
    record = {
        "slug": slug,
        "content_type": "feature",
        "status": "published",
        "is_published": True,
        "is_featured": False,
        "h1": title,
        "title_tag": f"{title} | Kitchens Directory",
        "meta_description": meta_description,
        "body_html": content,
        "author_name": author,
        "published_at": f"{date_str}T00:00:00Z",
    }

    resp = client.table("feature_articles").insert(record).execute()
    inserted = resp.data[0] if resp.data else record

    logger.info("Published to Supabase feature_articles: %s", slug)

    return {
        "commit_url": "",
        "file_path": f"supabase:feature_articles/{slug}",
        "slug": slug,
        "published_url": f"https://kitchensdirectory.co.uk/articles/{slug}",
        "repo": "supabase",
    }
