"""Main LangGraph graph definition for the SEO agent.

Builds a StateGraph that routes to the appropriate node based on
``state["task_type"]``.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.seo_agent.state import SEOAgentState


def _router(state: SEOAgentState) -> str:
    """Route to the correct node based on task_type.

    Args:
        state: Current agent state.

    Returns:
        The node name to execute next.
    """
    return state.get("task_type", "END") or "END"


def build_graph() -> Any:
    """Build and compile the SEO agent LangGraph graph.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for invocation.
    """
    from agents.seo_agent.nodes.backlink_prospector import (
        run_backlink_prospector,
    )
    from agents.seo_agent.nodes.content_brief import run_content_brief
    from agents.seo_agent.nodes.content_gap import run_content_gap
    from agents.seo_agent.nodes.content_writer import run_content_writer
    from agents.seo_agent.nodes.email_generator import run_email_generator
    from agents.seo_agent.nodes.internal_linker import run_internal_linker
    from agents.seo_agent.nodes.outreach_reporter import run_outreach_reporter
    from agents.seo_agent.nodes.outreach_sequencer import run_outreach_sequencer
    from agents.seo_agent.nodes.prospect_enrichment import (
        run_prospect_enrichment,
    )
    from agents.seo_agent.nodes.prospect_scorer import run_prospect_scorer
    from agents.seo_agent.nodes.rank_tracker import run_rank_tracker
    from agents.seo_agent.nodes.reporting import run_reporting

    # Lazy import to avoid circular deps
    from agents.seo_agent.nodes.keyword_research import run_keyword_research

    graph = StateGraph(SEOAgentState)

    # ---- Add all nodes ----
    graph.add_node("keyword_research", run_keyword_research)
    graph.add_node("content_gap", run_content_gap)
    graph.add_node("content_brief", run_content_brief)
    graph.add_node("content_writer", run_content_writer)
    graph.add_node("internal_linker", run_internal_linker)
    graph.add_node("rank_tracker", run_rank_tracker)
    graph.add_node("reporting", run_reporting)
    graph.add_node("backlink_prospector", run_backlink_prospector)
    graph.add_node("prospect_enrichment", run_prospect_enrichment)
    graph.add_node("prospect_scorer", run_prospect_scorer)
    graph.add_node("email_generator", run_email_generator)
    graph.add_node("outreach_sequencer", run_outreach_sequencer)
    graph.add_node("outreach_reporter", run_outreach_reporter)

    # ---- Entry: route from START based on task_type ----
    graph.set_conditional_entry_point(
        _router,
        {
            "keyword_research": "keyword_research",
            "content_gap": "content_gap",
            "content_brief": "content_brief",
            "write_content": "content_writer",
            "discover_prospects": "backlink_prospector",
            "enrich_prospects": "prospect_enrichment",
            "score_prospects": "prospect_scorer",
            "generate_emails": "email_generator",
            "run_outreach": "outreach_sequencer",
            "rank_report": "rank_tracker",
            "weekly_report": "reporting",
            "outreach_report": "outreach_reporter",
            "END": END,
        },
    )

    # ---- Chained transitions ----
    # Content pipeline: brief → writer → linker → END
    graph.add_conditional_edges(
        "content_brief",
        lambda s: s.get("next_node", "END"),
        {"content_writer": "content_writer", "END": END},
    )
    graph.add_conditional_edges(
        "content_writer",
        lambda s: s.get("next_node", "END"),
        {"internal_linker": "internal_linker", "END": END},
    )

    # All terminal nodes → END
    for node in [
        "keyword_research",
        "content_gap",
        "internal_linker",
        "rank_tracker",
        "reporting",
        "backlink_prospector",
        "prospect_enrichment",
        "prospect_scorer",
        "email_generator",
        "outreach_sequencer",
        "outreach_reporter",
    ]:
        graph.add_edge(node, END)

    return graph.compile()


def create_initial_state(
    *,
    task_type: str,
    target_site: str = "kitchensdirectory",
    seed_keyword: str | None = None,
    selected_keyword: str | None = None,
    brief_id: str | None = None,
) -> SEOAgentState:
    """Create an initial state dict for invoking the graph.

    Args:
        task_type: The task to run (e.g. ``keyword_research``, ``content_brief``).
        target_site: Site key from SITE_PROFILES.
        seed_keyword: Optional seed keyword for research tasks.
        selected_keyword: Optional keyword for brief/content tasks.
        brief_id: Optional Supabase brief ID for content writing.

    Returns:
        A fully initialised SEOAgentState dict.
    """
    return SEOAgentState(
        target_site=target_site,
        task_type=task_type,
        seed_keyword=seed_keyword,
        keyword_opportunities=[],
        content_gaps=[],
        selected_keyword=selected_keyword,
        content_brief={"id": brief_id} if brief_id else None,
        content_draft=None,
        backlink_prospects=[],
        enriched_prospects=[],
        scored_prospects=[],
        emails_generated=[],
        rank_data=[],
        report=None,
        errors=[],
        llm_spend_this_week=0.0,
        next_node="",
    )
