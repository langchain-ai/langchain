"""SEO Agent state definition for LangGraph."""

from typing import Optional, TypedDict


class SEOAgentState(TypedDict):
    """Shared state passed between all nodes in the SEO agent graph."""

    target_site: str
    task_type: str
    seed_keyword: Optional[str]
    keyword_opportunities: list[dict]
    content_gaps: list[dict]
    selected_keyword: Optional[str]
    content_brief: Optional[dict]
    content_draft: Optional[str]
    backlink_prospects: list[dict]
    enriched_prospects: list[dict]
    scored_prospects: list[dict]
    emails_generated: list[dict]
    rank_data: list[dict]
    report: Optional[str]
    errors: list[str]
    llm_spend_this_week: float
    next_node: str
