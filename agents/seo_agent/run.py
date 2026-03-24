"""CLI entrypoint for the SEO agent.

Usage::

    python -m agents.seo_agent.run keyword-research --site kitchensdirectory --seed "bespoke kitchens"
    python -m agents.seo_agent.run content-brief --keyword "kitchen makers in Manchester"
    python -m agents.seo_agent.run run-outreach --dry-run

"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

import click
from dotenv import load_dotenv

# Ensure the repo root is on sys.path so agents.seo_agent is importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seo_agent")


def _run_graph(task_type: str, **kwargs: object) -> dict:
    """Build the graph, create initial state, and invoke.

    Args:
        task_type: The task to run.
        **kwargs: Additional keyword arguments for ``create_initial_state``.

    Returns:
        The final state dict after graph execution.
    """
    from agents.seo_agent.agent import build_graph, create_initial_state
    from agents.seo_agent.tools.supabase_tools import ensure_tables, get_weekly_spend

    ensure_tables()
    weekly_spend = get_weekly_spend()

    state = create_initial_state(task_type=task_type, **kwargs)
    state["llm_spend_this_week"] = weekly_spend

    graph = build_graph()
    result = graph.invoke(state)

    if result.get("errors"):
        for err in result["errors"]:
            logger.error("Error: %s", err)

    return result


@click.group()
def cli() -> None:
    """SEO Agent CLI — manage SEO workflows for kitchensdirectory, freeroomplanner, and kitchen estimator."""


# ---------------------------------------------------------------------------
# SEO content commands
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--site", required=True, help="Target site key (e.g. kitchensdirectory)")
@click.option("--seed", default=None, help="Seed keyword for research")
def keyword_research(site: str, seed: str | None) -> None:
    """Run keyword research for a target site."""
    result = _run_graph("keyword_research", target_site=site, seed_keyword=seed)
    opportunities = result.get("keyword_opportunities", [])
    click.echo(f"\nFound {len(opportunities)} keyword opportunities:")
    for kw in opportunities[:20]:
        click.echo(
            f"  {kw.get('keyword', 'N/A'):50s} "
            f"vol={kw.get('volume', 'N/A'):>6} "
            f"KD={kw.get('kd', 'N/A'):>3} "
            f"intent={kw.get('intent', 'N/A')}"
        )


@cli.command()
@click.option("--site", required=True, help="Target site key")
def content_gap(site: str) -> None:
    """Find content gaps vs competitors."""
    result = _run_graph("content_gap", target_site=site)
    gaps = result.get("content_gaps", [])
    click.echo(f"\nFound {len(gaps)} content gaps:")
    for gap in gaps[:20]:
        click.echo(
            f"  {gap.get('keyword', 'N/A'):50s} "
            f"vol={gap.get('volume', 'N/A'):>6} "
            f"stage={gap.get('funnel_stage', 'N/A')}"
        )


@cli.command()
@click.option("--keyword", required=True, help="Target keyword for the brief")
@click.option("--site", default="kitchensdirectory", help="Target site key")
def content_brief(keyword: str, site: str) -> None:
    """Generate a content brief for a keyword."""
    result = _run_graph(
        "content_brief", target_site=site, selected_keyword=keyword
    )
    brief = result.get("content_brief")
    if brief:
        click.echo(f"\nContent brief generated for: {keyword}")
        click.echo(json.dumps(brief, indent=2, default=str))
    else:
        click.echo("Failed to generate content brief.")


@cli.command()
@click.option("--brief-id", required=True, help="Supabase brief ID")
@click.option("--site", default="kitchensdirectory", help="Target site key")
def write_content(brief_id: str, site: str) -> None:
    """Write content from a brief."""
    result = _run_graph("write_content", target_site=site, brief_id=brief_id)
    draft = result.get("content_draft")
    if draft:
        click.echo(f"\nContent draft generated ({len(draft)} chars)")
        click.echo(draft[:500] + "..." if len(draft) > 500 else draft)
    else:
        click.echo("Failed to generate content draft.")


@cli.command()
@click.option("--site", default="all", help="Target site key or 'all'")
def rank_report(site: str) -> None:
    """Generate a rank tracking report."""
    result = _run_graph("rank_report", target_site=site)
    data = result.get("rank_data", [])
    click.echo(f"\nRank data for {len(data)} keywords:")
    for row in data[:20]:
        pos = row.get("position", "N/A")
        prev = row.get("previous_position", "N/A")
        change = ""
        if isinstance(pos, (int, float)) and isinstance(prev, (int, float)):
            diff = prev - pos
            if diff > 0:
                change = f" (+{diff:.0f})"
            elif diff < 0:
                change = f" ({diff:.0f})"
        click.echo(
            f"  {row.get('keyword', 'N/A'):40s} "
            f"pos={pos}{change}"
        )


@cli.command()
@click.option("--email", default=None, help="Email address to send report to")
def weekly_report(email: str | None) -> None:
    """Generate and optionally email the weekly SEO report."""
    if email:
        os.environ["REPORT_EMAIL"] = email
    result = _run_graph("weekly_report", target_site="all")
    report = result.get("report", "")
    if report:
        click.echo("\n" + report)
    else:
        click.echo("No report generated.")


# ---------------------------------------------------------------------------
# Backlink & outreach commands
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--site", required=True, help="Target site key")
@click.option(
    "--method",
    default="all",
    help="Discovery method (all, competitor, content_explorer, unlinked, resource, broken, haro)",
)
def discover_prospects(site: str, method: str) -> None:
    """Discover backlink prospects."""
    result = _run_graph("discover_prospects", target_site=site)
    prospects = result.get("backlink_prospects", [])
    click.echo(f"\nDiscovered {len(prospects)} prospects:")
    for p in prospects[:20]:
        click.echo(
            f"  [{p.get('discovery_method', 'N/A'):20s}] "
            f"DR={p.get('dr', 'N/A'):>3} "
            f"{p.get('domain', 'N/A')}"
        )


@cli.command()
@click.option("--limit", default=50, help="Max prospects to enrich")
def enrich_prospects(limit: int) -> None:
    """Enrich prospect data with author, contact, and page details."""
    result = _run_graph("enrich_prospects", target_site="all")
    enriched = result.get("enriched_prospects", [])
    click.echo(f"\nEnriched {len(enriched)} prospects")


@cli.command()
@click.option("--min-score", default=35, help="Minimum score threshold")
def score_prospects(min_score: int) -> None:
    """Score all enriched prospects."""
    result = _run_graph("score_prospects", target_site="all")
    scored = result.get("scored_prospects", [])
    tier1 = [p for p in scored if p.get("tier") == "tier1"]
    tier2 = [p for p in scored if p.get("tier") == "tier2"]
    rejected = [p for p in scored if p.get("status") == "rejected"]
    click.echo(
        f"\nScored {len(scored)} prospects: "
        f"{len(tier1)} tier 1, {len(tier2)} tier 2, {len(rejected)} rejected"
    )


@cli.command()
@click.option("--tier", default=0, type=int, help="Filter by tier (1 or 2, 0 for all)")
@click.option("--limit", default=10, help="Max emails to generate")
def generate_emails(tier: int, limit: int) -> None:
    """Generate outreach emails for scored prospects."""
    result = _run_graph("generate_emails", target_site="all")
    emails = result.get("emails_generated", [])
    click.echo(f"\nGenerated {len(emails)} outreach emails:")
    for e in emails[:10]:
        click.echo(
            f"  To: {e.get('contact_email', 'N/A'):30s} "
            f"Tier {e.get('tier', 'N/A')} "
            f"| {e.get('subject', 'N/A')[:60]}"
        )


@cli.command()
@click.option("--dry-run", is_flag=True, default=False, help="Show emails without sending")
@click.option("--send", is_flag=True, default=False, help="Actually send emails")
def run_outreach(dry_run: bool, send: bool) -> None:
    """Run the outreach email sequence.

    Use --dry-run to preview what would be sent. Use --send to actually send.
    Default (no flags) does nothing — you must explicitly choose.
    """
    if not dry_run and not send:
        click.echo("Please specify --dry-run or --send")
        return

    if send:
        click.echo("WARNING: This will send real emails. Proceeding...")
    else:
        click.echo("DRY RUN — no emails will be sent.")

    # Set environment flag for the sequencer node
    os.environ["OUTREACH_DRY_RUN"] = "true" if dry_run else "false"
    result = _run_graph("run_outreach", target_site="all")

    if result.get("errors"):
        for err in result["errors"]:
            click.echo(f"  ERROR: {err}")
    else:
        click.echo("Outreach sequence completed.")


@cli.command()
def check_replies() -> None:
    """Check for replies to outreach emails."""
    from agents.seo_agent.tools import supabase_tools

    supabase_tools.ensure_tables()
    # In mock mode, just report
    click.echo("Checking for replies to outreach emails...")
    emails = supabase_tools.query_table(
        "seo_outreach_emails",
        filters={"status": "sent"},
        limit=100,
    )
    click.echo(f"Found {len(emails)} sent emails to check for replies.")


@cli.command()
@click.option("--url", required=True, help="URL of the page that linked to us")
def log_link_acquired(url: str) -> None:
    """Log a backlink that has been acquired."""
    from agents.seo_agent.tools import supabase_tools

    supabase_tools.ensure_tables()

    # Find matching prospect
    prospects = supabase_tools.query_table(
        "seo_backlink_prospects",
        limit=500,
    )
    matched = [p for p in prospects if p.get("page_url") == url]

    if matched:
        for p in matched:
            supabase_tools.upsert_record(
                "seo_backlink_prospects",
                {**p, "status": "link_acquired"},
            )
        click.echo(f"Logged link acquired from: {url}")
    else:
        # Create a new record
        supabase_tools.insert_record(
            "seo_backlink_prospects",
            {
                "page_url": url,
                "domain": url.split("/")[2] if "/" in url else url,
                "status": "link_acquired",
                "discovery_method": "manual",
            },
        )
        click.echo(f"Logged new link acquired from: {url}")


# ---------------------------------------------------------------------------
# Cost & reporting commands
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--period", default="week", help="Period: week or month")
def cost_report(period: str) -> None:
    """Show LLM cost report."""
    from agents.seo_agent.tools import supabase_tools

    supabase_tools.ensure_tables()
    spend = supabase_tools.get_weekly_spend()
    cap = float(os.getenv("MAX_WEEKLY_SPEND_USD", "50.00"))
    remaining = max(0.0, cap - spend)
    pct = (spend / cap * 100) if cap > 0 else 0

    click.echo(f"\nLLM Cost Report ({period})")
    click.echo(f"{'=' * 40}")
    click.echo(f"  Spent:     ${spend:.4f}")
    click.echo(f"  Budget:    ${cap:.2f}")
    click.echo(f"  Remaining: ${remaining:.4f} ({100 - pct:.1f}%)")

    if pct >= 80:
        click.echo("  ⚠ Budget >80% — models will be downgraded")

    # Show breakdown by task
    logs = supabase_tools.query_table("llm_cost_log", limit=500)
    if logs:
        by_task: dict[str, float] = {}
        for row in logs:
            task = row.get("task_type", "unknown")
            by_task[task] = by_task.get(task, 0.0) + row.get("cost_usd", 0.0)
        click.echo(f"\n  Breakdown by task:")
        for task, cost in sorted(by_task.items(), key=lambda x: -x[1]):
            click.echo(f"    {task:30s} ${cost:.4f}")


@cli.command()
def outreach_report() -> None:
    """Generate the weekly outreach report."""
    result = _run_graph("outreach_report", target_site="all")
    report = result.get("report", "")
    if report:
        click.echo("\n" + report)
    else:
        click.echo("No outreach data to report.")


if __name__ == "__main__":
    cli()
