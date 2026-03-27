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

from agents.seo_agent.config import BLOG_CONFIG

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

    Produces short, keyword-focused slugs (2-5 words). Logs a warning if the
    raw slug exceeds 5 words and truncates intelligently.

    Args:
        title: The blog post title.

    Returns:
        A lowercase, hyphenated slug of 2-5 words.
    """
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "-", slug)
    slug = slug.strip("-")

    parts = slug.split("-")
    if len(parts) > 5:
        logger.warning(
            "Slug '%s' has %d words — truncating to 5 for SEO best practice",
            slug,
            len(parts),
        )
        # Keep the most keyword-rich first 5 words
        slug = "-".join(parts[:5])

    return slug[:80]


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

    # Auto-detect category
    category = kwargs.get("category", "") or _categorize_post(title)

    # Build the file content based on site type
    if site == "ralf_seo":
        what_i_learned = kwargs.get("what_i_learned", [])
        file_content = _build_ralf_blog_post(
            title, content, meta_description, slug, date_str, category, what_i_learned,
        )
    elif site == "kitchen_estimator":
        file_content = _build_ts_blog_post(title, content, meta_description, slug, date_str)
    elif ext == ".html":
        # Fetch related articles for the related section
        related_slugs = _fetch_related_slugs(repo, branch, blog_path, slug, category)
        related_articles = [
            {"slug": s, "title": s.replace("-", " ").title(), "category": category, "excerpt": ""}
            for s in related_slugs
        ]
        file_content = _build_html_blog_post(
            title, content, meta_description, slug, site, date_str,
            category=category,
            related_articles=related_articles,
        )
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

    # Update the blog index page
    if site == "freeroomplanner":
        _update_blog_index(repo, branch, slug, title, meta_description)
    elif site == "ralf_seo":
        _update_ralf_blog_index(repo, branch, slug, title, meta_description, date_str, category)
    elif site == "kitchen_estimator":
        _update_kce_blog_index(repo, branch, slug)

    # Update sitemap.xml with the new post URL
    blog_config = BLOG_CONFIG.get(site, {})
    sitemap_path = blog_config.get("sitemap_path", "")
    domain = blog_config.get("domain", "")
    if sitemap_path and domain:
        blog_url_prefix = "/posts" if site == "ralf_seo" else "/blog"
        _update_sitemap(repo, branch, sitemap_path, slug, domain, date_str, blog_url_prefix)

    # Insert backlinks in related existing posts
    if ext == ".html":
        related_for_backlinks = _fetch_related_slugs(repo, branch, blog_path, slug, category)
        _insert_backlinks(repo, branch, blog_path, related_for_backlinks, slug, title, ext)

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


def _build_ralf_blog_post(
    title: str,
    content: str,
    meta_description: str,
    slug: str,
    date_str: str,
    category: str,
    what_i_learned: list[str],
) -> str:
    """Build an HTML blog post for Ralf's personal blog.

    Includes proper SEO elements: OG tags, Twitter Card, JSON-LD structured
    data, breadcrumb navigation, and canonical URL alongside the existing
    minimal design.

    Args:
        title: Post title.
        content: Article body HTML.
        meta_description: SEO meta description.
        slug: URL slug.
        date_str: Publication date (YYYY-MM-DD).
        category: Post category (e.g. Field Report, SEO).
        what_i_learned: List of key takeaways.

    Returns:
        Complete HTML document string.
    """
    config = BLOG_CONFIG.get("ralf_seo", {})
    safe_title = title.replace('"', '&quot;')
    safe_desc = meta_description.replace('"', '&quot;')
    learned_html = "\n".join(f"        <li>{item}</li>" for item in what_i_learned)

    domain = config.get("domain", "ralfseo.com")
    og_image = config.get("og_image", f"https://{domain}/assets/og-image.png")
    og_w = config.get("og_image_width", 1200)
    og_h = config.get("og_image_height", 630)
    canonical = f"https://{domain}/posts/{slug}"

    json_ld = _build_json_ld(
        config, title, meta_description, slug,
        date_published=date_str,
        date_modified=date_str,
        category=category,
        breadcrumb_label=title if len(title) <= 40 else " ".join(title.split()[:5]) + "...",
        blog_url_prefix="/posts",
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta name="author" content="Ralf SEO">
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title} \u2014 Ralf</title>
<meta name="description" content="{safe_desc}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Ralf">
<meta property="og:title" content="{safe_title}">
<meta property="og:description" content="{safe_desc}">
<meta property="og:url" content="{canonical}">
<meta property="og:image" content="{og_image}">
<meta property="og:image:width" content="{og_w}">
<meta property="og:image:height" content="{og_h}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{safe_title}">
<meta name="twitter:description" content="{safe_desc}">
<meta name="twitter:image" content="{og_image}">
<link rel="icon" type="image/svg+xml" href="../assets/favicon.svg">
<meta name="theme-color" content="#1a1a2e">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
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
<nav class="breadcrumb wrap" aria-label="Breadcrumb">
  <a href="../">Home</a> &rsaquo; <span>{safe_title}</span>
</nav>
<main class="wrap">
  <article itemscope itemtype="https://schema.org/BlogPosting">
    <div class="post-header">
      <div class="meta">{date_str} \u00b7 {category}</div>
      <h1 itemprop="headline">{safe_title}</h1>
    </div>
    <div class="post-body" itemprop="articleBody">
      {content}
      <div class="learned">
        <h2>what i learned</h2>
        <ul>
{learned_html}
        </ul>
      </div>
    </div>
  </article>
  <a href="../" class="back">\u2190 Back to posts</a>
</main>
{json_ld}
<footer>
  <div class="wrap">
    <span>ralf \u2014 autonomous seo agent</span>
    <a href="https://www.perplexity.ai/computer" target="_blank" rel="noopener noreferrer">Built with Perplexity Computer</a>
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


def _update_sitemap(
    repo: str,
    branch: str,
    sitemap_path: str,
    slug: str,
    domain: str,
    date_str: str,
    blog_url_prefix: str = "/blog",
) -> None:
    """Add a new URL entry to the site's sitemap.xml.

    Fetches the sitemap, inserts a ``<url>`` entry for the new blog post
    with ``<changefreq>monthly</changefreq>`` and ``<priority>0.7</priority>``,
    and commits the update. Errors are logged but not raised.

    Args:
        repo: GitHub repo (owner/name).
        branch: Target branch.
        sitemap_path: Path to sitemap.xml in the repo.
        slug: Blog post URL slug.
        domain: Site domain (e.g. freeroomplanner.com).
        date_str: Publication date (YYYY-MM-DD) for ``<lastmod>``.
        blog_url_prefix: URL path prefix for blog posts.
    """
    if not sitemap_path:
        return

    try:
        client = _get_client()

        resp = client.get(
            f"/repos/{repo}/contents/{quote(sitemap_path, safe='/')}?ref={branch}"
        )
        if resp.status_code != 200:
            logger.warning("Sitemap not found at %s — skipping update", sitemap_path)
            return

        data = resp.json()
        sha = data["sha"]
        current_content = base64.b64decode(data["content"]).decode()

        blog_url = f"https://{domain}{blog_url_prefix}/{slug}"

        # Skip if URL already in sitemap
        if blog_url in current_content:
            logger.info("URL '%s' already in sitemap, skipping", blog_url)
            return

        new_entry = (
            f"  <url>\n"
            f"    <loc>{blog_url}</loc>\n"
            f"    <lastmod>{date_str}</lastmod>\n"
            f"    <changefreq>monthly</changefreq>\n"
            f"    <priority>0.7</priority>\n"
            f"  </url>\n"
        )

        # Insert before the closing </urlset> tag
        close_tag = "</urlset>"
        close_pos = current_content.rfind(close_tag)
        if close_pos == -1:
            logger.warning("No </urlset> found in sitemap — skipping")
            return

        updated = current_content[:close_pos] + new_entry + current_content[close_pos:]

        encoded = base64.b64encode(updated.encode()).decode()
        put_resp = client.put(
            f"/repos/{repo}/contents/{quote(sitemap_path, safe='/')}",
            json={
                "message": f"sitemap: add {slug}",
                "content": encoded,
                "sha": sha,
                "branch": branch,
            },
        )
        put_resp.raise_for_status()
        logger.info("Updated sitemap with: %s", blog_url)
    except Exception:
        logger.error("Failed to update sitemap for '%s'", slug, exc_info=True)


def _insert_backlinks(
    repo: str,
    branch: str,
    blog_path: str,
    related_slugs: list[str],
    new_slug: str,
    new_title: str,
    file_ext: str = ".html",
) -> None:
    """Insert a contextual backlink to the new post in related existing posts.

    For each related slug, fetches the blog post HTML, finds a safe insertion
    point within the article body, and adds an inline link. Commits each
    update individually. Errors are logged but not raised.

    Args:
        repo: GitHub repo (owner/name).
        branch: Target branch.
        blog_path: Directory path containing blog posts.
        related_slugs: Slugs of existing posts to add backlinks in (1-2).
        new_slug: The new post's slug.
        new_title: The new post's title (used as anchor text).
        file_ext: File extension for blog post files.
    """
    if not related_slugs:
        return

    try:
        client = _get_client()
    except RuntimeError:
        logger.warning("No GitHub token — skipping backlink insertion")
        return

    for related_slug in related_slugs[:2]:
        try:
            file_path = f"{blog_path}/{related_slug}{file_ext}"
            resp = client.get(
                f"/repos/{repo}/contents/{quote(file_path, safe='/')}?ref={branch}"
            )
            if resp.status_code != 200:
                logger.debug("Related post not found: %s", file_path)
                continue

            data = resp.json()
            sha = data["sha"]
            content = base64.b64decode(data["content"]).decode()

            # Skip if backlink already exists
            if f"/blog/{new_slug}" in content:
                logger.debug("Backlink to '%s' already in '%s'", new_slug, related_slug)
                continue

            # Find a safe insertion point: before the last </p> in the article body
            # Look for the closing article body marker or last paragraph
            article_body_end = content.rfind("</div>", 0, content.rfind("</article>"))
            if article_body_end == -1:
                # Fallback: find last </p> before </article>
                article_end = content.rfind("</article>")
                if article_end == -1:
                    logger.debug("No article tag in '%s', skipping backlink", related_slug)
                    continue
                article_body_end = content.rfind("</p>", 0, article_end)
                if article_body_end == -1:
                    continue

            safe_title = new_title.replace("&", "&amp;").replace("<", "&lt;")
            link_html = (
                f'\n<p>You might also find our guide on '
                f'<a href="/blog/{new_slug}">{safe_title}</a> useful.</p>\n'
            )

            updated = content[:article_body_end] + link_html + content[article_body_end:]

            encoded = base64.b64encode(updated.encode()).decode()
            put_resp = client.put(
                f"/repos/{repo}/contents/{quote(file_path, safe='/')}",
                json={
                    "message": f"blog: add backlink to {new_slug} in {related_slug}",
                    "content": encoded,
                    "sha": sha,
                    "branch": branch,
                },
            )
            put_resp.raise_for_status()
            logger.info("Added backlink to '%s' in '%s'", new_slug, related_slug)
        except Exception:
            logger.error(
                "Failed to insert backlink in '%s'", related_slug, exc_info=True
            )


def _fetch_related_slugs(
    repo: str,
    branch: str,
    blog_path: str,
    current_slug: str,
    category: str,
) -> list[str]:
    """Fetch slugs of related blog posts from the repo's blog directory.

    Lists files in the blog directory and returns up to 3 slugs that are
    not the current post, preferring posts from the same category cluster
    based on keyword matching.

    Args:
        repo: GitHub repo (owner/name).
        branch: Target branch.
        blog_path: Directory path containing blog posts.
        current_slug: Slug of the current post (to exclude).
        category: Category of the current post for relevance matching.

    Returns:
        List of up to 3 related blog post slugs.
    """
    try:
        client = _get_client()
        resp = client.get(
            f"/repos/{repo}/contents/{quote(blog_path, safe='/')}?ref={branch}"
        )
        if resp.status_code != 200:
            return []

        files = resp.json()
        slugs = [
            f["name"].rsplit(".", 1)[0]
            for f in files
            if f["type"] == "file"
            and f["name"] != "index.html"
            and f["name"].endswith(".html")
            and f["name"].rsplit(".", 1)[0] != current_slug
        ]

        # Simple relevance: prefer slugs that share words with the category
        cat_words = set(category.lower().split())
        scored = []
        for s in slugs:
            slug_words = set(s.split("-"))
            overlap = len(cat_words & slug_words)
            scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [s for _, s in scored[:3]]
    except Exception:
        logger.debug("Failed to fetch related slugs", exc_info=True)
        return []


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


def _estimate_read_time(content: str) -> int:
    """Estimate reading time in minutes from content.

    Args:
        content: The article HTML or text content.

    Returns:
        Reading time in minutes (minimum 1).
    """
    text = re.sub(r"<[^>]+>", "", content)
    word_count = len(text.split())
    return max(1, round(word_count / 200))


def _build_head_tags(
    config: dict[str, Any],
    title: str,
    meta_description: str,
    slug: str,
    blog_url_prefix: str = "/blog",
) -> str:
    """Build all required ``<head>`` tags from blog config.

    Produces: analytics script, meta author, charset, viewport, title with
    suffix, meta description, canonical URL, full OG tags (with image dims),
    full Twitter Card tags (with image), favicon links, theme-color, font
    preload (preconnect + lazy-load + noscript fallback), and CSS link.

    Args:
        config: The site's ``BLOG_CONFIG`` entry.
        title: Blog post title (without site suffix).
        meta_description: SEO meta description (150-160 chars).
        slug: URL slug for the post.
        blog_url_prefix: URL path prefix for blog posts.

    Returns:
        Complete ``<head>`` inner HTML string.
    """
    domain = config["domain"]
    site_name = config["site_name"]
    suffix = config["title_suffix"]
    og_image = config["og_image"]
    og_w = config["og_image_width"]
    og_h = config["og_image_height"]
    theme_color = config["theme_color"]
    css_path = config.get("css_path", "")
    analytics = config.get("analytics_script", "")
    author = config.get("meta_author", site_name)
    canonical = f"https://{domain}{blog_url_prefix}/{slug}"

    safe_title = title.replace('"', "&quot;")
    safe_desc = meta_description.replace('"', "&quot;")

    parts: list[str] = []

    # 1. Analytics
    if analytics:
        parts.append(analytics)

    # 2-4. Author, charset, viewport
    parts.append(f'<meta name="author" content="{author}">')
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')

    # 5. Title
    parts.append(f"<title>{safe_title} {suffix}</title>")

    # 6. Meta description
    parts.append(f'<meta name="description" content="{safe_desc}">')

    # 7. Canonical
    parts.append(f'<link rel="canonical" href="{canonical}">')

    # 8. Open Graph tags
    parts.append('<meta property="og:type" content="article">')
    parts.append(f'<meta property="og:site_name" content="{site_name}">')
    parts.append(f'<meta property="og:title" content="{safe_title}">')
    parts.append(f'<meta property="og:description" content="{safe_desc}">')
    parts.append(f'<meta property="og:url" content="{canonical}">')
    parts.append(f'<meta property="og:image" content="{og_image}">')
    parts.append(f'<meta property="og:image:width" content="{og_w}">')
    parts.append(f'<meta property="og:image:height" content="{og_h}">')

    # 9. Twitter Card tags
    parts.append('<meta name="twitter:card" content="summary_large_image">')
    parts.append(f'<meta name="twitter:title" content="{safe_title}">')
    parts.append(f'<meta name="twitter:description" content="{safe_desc}">')
    parts.append(f'<meta name="twitter:image" content="{og_image}">')

    # 10. Favicon links
    parts.append('<link rel="icon" href="/favicon.ico" sizes="any">')
    parts.append('<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">')
    parts.append('<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">')
    parts.append('<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">')
    parts.append('<link rel="manifest" href="/site.webmanifest">')

    # 11. Theme colour
    parts.append(f'<meta name="theme-color" content="{theme_color}">')

    # 12. Font preload (General Sans via Fontshare)
    parts.append('<link rel="preconnect" href="https://api.fontshare.com">')
    parts.append(
        '<link rel="preload" as="style" '
        'href="https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap">'
    )
    parts.append(
        '<link rel="stylesheet" '
        'href="https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap" '
        "media=\"print\" onload=\"this.media='all'\">"
    )
    parts.append("<noscript>")
    parts.append(
        '<link rel="stylesheet" '
        'href="https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap">'
    )
    parts.append("</noscript>")

    # 13. Main stylesheet
    if css_path:
        parts.append(f'<link rel="stylesheet" href="{css_path}">')

    return "\n".join(parts)


def _build_json_ld(
    config: dict[str, Any],
    title: str,
    meta_description: str,
    slug: str,
    date_published: str,
    date_modified: str,
    category: str,
    breadcrumb_label: str,
    blog_url_prefix: str = "/blog",
) -> str:
    """Build BlogPosting and BreadcrumbList JSON-LD structured data.

    Args:
        config: The site's ``BLOG_CONFIG`` entry.
        title: Post title (no site suffix).
        meta_description: Meta description text.
        slug: URL slug.
        date_published: ISO date string (YYYY-MM-DD).
        date_modified: ISO date string (YYYY-MM-DD).
        category: Category badge text.
        breadcrumb_label: Short label for the breadcrumb trail.
        blog_url_prefix: URL path prefix for blog posts.

    Returns:
        Two ``<script type="application/ld+json">`` blocks as a single string.
    """
    domain = config["domain"]
    site_name = config["site_name"]
    og_image = config["og_image"]
    canonical = f"https://{domain}{blog_url_prefix}/{slug}"

    blog_posting = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "description": meta_description,
        "image": og_image,
        "datePublished": date_published,
        "dateModified": date_modified,
        "keywords": category,
        "url": canonical,
        "author": {
            "@type": "Organization",
            "name": site_name,
        },
        "publisher": {
            "@type": "Organization",
            "name": site_name,
            "logo": {
                "@type": "ImageObject",
                "url": og_image,
            },
        },
    }

    breadcrumb_list = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": 1,
                "name": "Home",
                "item": f"https://{domain}/",
            },
            {
                "@type": "ListItem",
                "position": 2,
                "name": "Blog",
                "item": f"https://{domain}{blog_url_prefix}",
            },
            {
                "@type": "ListItem",
                "position": 3,
                "name": breadcrumb_label,
            },
        ],
    }

    bp_json = json.dumps(blog_posting, indent=2)
    bc_json = json.dumps(breadcrumb_list, indent=2)

    return (
        f'<script type="application/ld+json">\n{bp_json}\n</script>\n'
        f'<script type="application/ld+json">\n{bc_json}\n</script>'
    )


def _build_frp_header(config: dict[str, Any]) -> str:
    """Build the freeroomplanner header HTML.

    Includes SVG logo, navigation links, theme toggle with moon/sun SVGs,
    mobile nav toggle button, and CTA button.

    Args:
        config: The site's ``BLOG_CONFIG`` entry.

    Returns:
        Complete header HTML string.
    """
    nav_items = "".join(
        f'<a href="{link["href"]}">{link["label"]}</a>'
        for link in config.get("nav_links", [])
    )
    cta = config.get("cta_button") or {}
    cta_label = cta.get("label", "Start planning")
    cta_href = cta.get("href", "/app")

    return f"""<header class="header">
  <div class="container header__inner">
    <a href="/" class="header__logo" aria-label="Free Room Planner home">
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <rect width="32" height="32" rx="8" fill="#0d9488"/>
        <path d="M8 8h16v16H8z" fill="none" stroke="#fff" stroke-width="2"/>
        <path d="M8 16h8v8" fill="none" stroke="#fff" stroke-width="2"/>
        <path d="M20 8v8" fill="none" stroke="#fff" stroke-width="2"/>
      </svg>
      <span>Free Room Planner</span>
    </a>
    <nav class="header__nav" id="main-nav">
      {nav_items}
    </nav>
    <div class="header__actions">
      <button class="theme-toggle" id="theme-toggle" aria-label="Toggle dark mode">
        <svg class="icon-moon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
        <svg class="icon-sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      </button>
      <button class="nav-toggle" id="nav-toggle" aria-label="Toggle navigation" aria-expanded="false">
        <span></span><span></span><span></span>
      </button>
      <a href="{cta_href}" class="btn btn--primary header__cta">{cta_label}</a>
    </div>
  </div>
</header>
<style>
  .icon-sun {{ display: none; }}
  [data-theme="dark"] .icon-moon {{ display: none; }}
  [data-theme="dark"] .icon-sun {{ display: block; }}
</style>"""


def _build_frp_footer(config: dict[str, Any]) -> str:
    """Build the freeroomplanner footer HTML.

    Includes brand with SVG logo, Planners/Resources/Use cases columns,
    and copyright line.

    Args:
        config: The site's ``BLOG_CONFIG`` entry.

    Returns:
        Complete footer HTML string.
    """
    columns_html = ""
    for col_title, links in config.get("footer_columns", {}).items():
        items = "".join(
            f'<li><a href="{link["href"]}">{link["label"]}</a></li>'
            for link in links
        )
        columns_html += f"""    <div class="footer__col">
      <h4>{col_title}</h4>
      <ul>{items}</ul>
    </div>
"""

    brand_desc = config.get("footer_brand_description", "")

    return f"""<footer class="footer">
  <div class="container footer__inner">
    <div class="footer__brand">
      <a href="/" aria-label="Free Room Planner home">
        <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <rect width="32" height="32" rx="8" fill="#0d9488"/>
          <path d="M8 8h16v16H8z" fill="none" stroke="#fff" stroke-width="2"/>
          <path d="M8 16h8v8" fill="none" stroke="#fff" stroke-width="2"/>
          <path d="M20 8v8" fill="none" stroke="#fff" stroke-width="2"/>
        </svg>
      </a>
      <p>{brand_desc}</p>
    </div>
{columns_html}  </div>
  <div class="container footer__copy">
    <p>&copy; {datetime.now(tz=timezone.utc).year} Free Room Planner. All rights reserved.</p>
  </div>
</footer>"""


def _build_related_articles(
    related: list[dict[str, str]],
) -> str:
    """Build the related articles section with 3 card links.

    Args:
        related: List of dicts with ``slug``, ``title``, ``category``, and
            ``excerpt`` keys. Up to 3 articles.

    Returns:
        Related articles HTML section, or empty string if no articles.
    """
    if not related:
        return ""

    cards = ""
    for article in related[:3]:
        safe_title = article.get("title", "").replace("&", "&amp;").replace("<", "&lt;")
        category = article.get("category", "")
        excerpt = article.get("excerpt", "").replace("&", "&amp;").replace("<", "&lt;")
        slug = article.get("slug", "")
        cards += f"""  <a href="/blog/{slug}" class="card">
    <span class="badge">{category}</span>
    <h3>{safe_title}</h3>
    <p>{excerpt}</p>
  </a>
"""

    return f"""<section class="related-articles">
  <div class="container">
    <h2>Related articles</h2>
    <div class="grid-3">
{cards}    </div>
  </div>
</section>"""


def _build_html_blog_post(
    title: str,
    content: str,
    meta_description: str,
    slug: str,
    site: str,
    date_str: str,
    *,
    category: str = "",
    related_articles: list[dict[str, str]] | None = None,
) -> str:
    """Build a full HTML blog post page following SEO standards.

    Produces a complete HTML document with: analytics, full meta tags,
    OG + Twitter Card tags, favicons, font preloading, external CSS,
    schema.org BlogPosting article wrapper, breadcrumbs, hero section,
    CTA boxes, related articles, JSON-LD structured data, full
    header/footer, and external JS.

    Args:
        title: Blog post title (no site suffix).
        content: Article body HTML content.
        meta_description: SEO meta description (150-160 chars).
        slug: URL slug for the post.
        site: Site key from ``BLOG_CONFIG``.
        date_str: Publication date as YYYY-MM-DD.
        category: Post category badge text. Auto-detected if empty.
        related_articles: List of related article dicts for the related
            articles section. Each dict needs ``slug``, ``title``,
            ``category``, and ``excerpt`` keys.

    Returns:
        Complete HTML document string.
    """
    config = BLOG_CONFIG.get(site, BLOG_CONFIG.get("freeroomplanner", {}))

    if not category:
        category = _categorize_post(title)

    read_time = _estimate_read_time(content)
    breadcrumb_label = title if len(title) <= 40 else " ".join(title.split()[:5]) + "..."
    safe_title = title.replace('"', "&quot;").replace("&", "&amp;").replace("<", "&lt;")

    head_tags = _build_head_tags(config, title, meta_description, slug)
    json_ld = _build_json_ld(
        config, title, meta_description, slug,
        date_published=date_str,
        date_modified=date_str,
        category=category,
        breadcrumb_label=breadcrumb_label,
    )

    # Site-specific header/footer
    if site == "freeroomplanner":
        header_html = _build_frp_header(config)
        footer_html = _build_frp_footer(config)
    else:
        # Generic header/footer for other HTML sites
        site_name = config.get("site_name", "Blog")
        header_html = f"""<header class="header">
  <div class="container header__inner">
    <a href="/" class="header__logo">{site_name}</a>
    <nav class="header__nav"><a href="/blog">Blog</a></nav>
  </div>
</header>"""
        footer_html = f"""<footer class="footer">
  <div class="container footer__copy">
    <p>&copy; {datetime.now(tz=timezone.utc).year} {site_name}. All rights reserved.</p>
    <p><a href="/">Home</a> &middot; <a href="/blog">Blog</a></p>
  </div>
</footer>"""

    # Breadcrumb
    breadcrumb_safe = breadcrumb_label.replace("&", "&amp;").replace("<", "&lt;")
    breadcrumb_html = f"""<nav class="breadcrumb container" aria-label="Breadcrumb">
  <a href="/">Home</a> &rsaquo; <a href="/blog">Blog</a> &rsaquo; <span>{breadcrumb_safe}</span>
</nav>"""

    # CTA box after article
    cta = config.get("cta_button") or {}
    cta_href = cta.get("href", "/app")
    cta_box_html = f"""<div class="cta-box">
    <h2>Draw your own floor plan &mdash; free</h2>
    <p>Plan your room with accurate measurements. No sign-up required.</p>
    <a href="{cta_href}" class="btn btn--primary">Open Free Room Planner free</a>
  </div>"""

    # Related articles
    related_html = _build_related_articles(related_articles or [])

    # Final CTA section
    final_cta_html = f"""<section class="final-cta">
  <div class="container">
    <h2>Ready to plan your room?</h2>
    <p>Free. No account. Works in your browser.</p>
    <a href="{cta_href}" class="btn btn--primary">Start planning free</a>
  </div>
</section>"""

    js_path = config.get("js_path", "")
    js_tag = f'\n<script src="{js_path}" defer></script>' if js_path else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
{head_tags}
</head>
<body>
{header_html}
{breadcrumb_html}
<article itemscope itemtype="https://schema.org/BlogPosting">
  <section class="hero">
    <div class="container">
      <span class="badge">{category}</span>
      <h1 itemprop="headline">{safe_title}</h1>
      <div class="meta">{read_time} min read</div>
      <a href="{cta_href}" class="btn btn--primary">{cta.get("label", "Start planning")}</a>
    </div>
  </section>
  <div class="prose fade-in container" itemprop="articleBody">
{content}
  </div>
  {cta_box_html}
</article>
{related_html}
{final_cta_html}
{json_ld}
{footer_html}
{js_tag}
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


def _build_ts_blog_post(
    title: str, content: str, meta_description: str, slug: str, date_str: str
) -> str:
    """Build a TypeScript blog post file for KitchenCostEstimator.

    Includes SEO metadata fields (canonicalUrl, ogImage, jsonLd) for the
    React app to render into the page ``<head>``.

    Args:
        title: Post title.
        content: Article body (HTML or markdown).
        meta_description: SEO meta description.
        slug: URL slug.
        date_str: Publication date (YYYY-MM-DD).

    Returns:
        TypeScript source file content.
    """
    config = BLOG_CONFIG.get("kitchen_estimator", {})
    domain = config.get("domain", "kitchencostestimator.com")
    og_image = config.get("og_image", f"https://{domain}/og-image.png")

    # Convert slug to camelCase export name
    parts = slug.replace('-', ' ').split()
    export_name = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])

    # Estimate reading time (200 words per minute)
    word_count = len(content.split())
    reading_time = max(1, round(word_count / 200))

    # Extract potential tags from title
    tags: list[str] = []
    title_lower = title.lower()
    if 'uk' in title_lower:
        tags.append('UK')
    if 'cost' in title_lower or 'price' in title_lower:
        tags.append('kitchen cost')
    if '2026' in title_lower:
        tags.append('2026')
    if 'renovation' in title_lower:
        tags.append('renovation')
    if not tags:
        tags = ['kitchen', 'guide']

    # Convert HTML content to markdown
    md_content = content
    md_content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', md_content)
    md_content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', md_content)
    md_content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n', md_content)
    md_content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', md_content)
    md_content = re.sub(r'<em>(.*?)</em>', r'*\1*', md_content)
    md_content = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', md_content)
    md_content = re.sub(r'</?[uo]l[^>]*>', '', md_content)
    md_content = re.sub(r'<a href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', md_content)
    md_content = re.sub(r'</?[^>]+>', '', md_content)
    md_content = re.sub(r'\n{3,}', '\n\n', md_content)

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
  canonicalUrl: 'https://{domain}/blog/{slug}',
  ogImage: '{og_image}',
  jsonLd: {{
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: '{safe_title}',
    description: '{safe_desc}',
    image: '{og_image}',
    datePublished: '{date_str}',
    dateModified: '{date_str}',
    url: 'https://{domain}/blog/{slug}',
    author: {{ '@type': 'Organization', name: 'Kitchen Cost Estimator' }},
    publisher: {{
      '@type': 'Organization',
      name: 'Kitchen Cost Estimator',
      logo: {{ '@type': 'ImageObject', url: '{og_image}' }},
    }},
  }},
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
    """Publish a blog post to kitchensdirectory's Supabase feature_articles table.

    Includes SEO metadata: canonical URL, OG image, and BlogPosting JSON-LD
    structured data for the front-end to render.

    Args:
        site: Site key.
        slug: URL slug.
        title: Post title.
        content: Article body HTML.
        meta_description: SEO meta description.
        author: Author name.
        date_str: Publication date (YYYY-MM-DD).

    Returns:
        Dict with commit_url, file_path, slug, published_url, and repo.
    """
    from agents.seo_agent.tools.supabase_tools import get_client

    config = BLOG_CONFIG.get("kitchensdirectory", {})
    domain = config.get("domain", "kitchensdirectory.co.uk")
    og_image = config.get("og_image", f"https://{domain}/og-image.png")
    site_name = config.get("site_name", "Kitchens Directory")
    canonical = f"https://{domain}/articles/{slug}"

    json_ld = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "description": meta_description,
        "image": og_image,
        "datePublished": date_str,
        "dateModified": date_str,
        "url": canonical,
        "author": {"@type": "Organization", "name": site_name},
        "publisher": {
            "@type": "Organization",
            "name": site_name,
            "logo": {"@type": "ImageObject", "url": og_image},
        },
    }

    client = get_client()
    record = {
        "slug": slug,
        "content_type": "feature",
        "status": "published",
        "is_published": True,
        "is_featured": False,
        "h1": title,
        "title_tag": f"{title} | {site_name}",
        "meta_description": meta_description,
        "body_html": content,
        "author_name": author,
        "published_at": f"{date_str}T00:00:00Z",
        "canonical_url": canonical,
        "og_image": og_image,
        "json_ld": json.dumps(json_ld),
    }

    resp = client.table("feature_articles").insert(record).execute()
    inserted = resp.data[0] if resp.data else record

    logger.info("Published to Supabase feature_articles: %s", slug)

    return {
        "commit_url": "",
        "file_path": f"supabase:feature_articles/{slug}",
        "slug": slug,
        "published_url": f"https://{domain}/articles/{slug}",
        "repo": "supabase",
    }
