"""Supabase tools — database operations and table management.

Set ``SUPABASE_MOCK=true`` in ``.env`` to use an in-memory store.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory mock store
# ---------------------------------------------------------------------------

_mock_store: dict[str, list[dict]] = {}


def _is_mock() -> bool:
    return os.getenv("SUPABASE_MOCK", "false").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Supabase client singleton
# ---------------------------------------------------------------------------

_client: Any = None


def get_client() -> Any:
    """Return a Supabase client (or mock proxy).

    Returns:
        A ``supabase.Client`` instance, or a mock object when mocking.
    """
    if _is_mock():
        return _MockClient()

    global _client  # noqa: PLW0603
    if _client is None:
        from supabase import create_client

        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


# ---------------------------------------------------------------------------
# Table definitions (CREATE TABLE IF NOT EXISTS)
# ---------------------------------------------------------------------------

TABLE_SCHEMAS: dict[str, str] = {
    "seo_keyword_opportunities": """
        CREATE TABLE IF NOT EXISTS seo_keyword_opportunities (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            keyword TEXT NOT NULL,
            volume INTEGER,
            kd INTEGER,
            cpc REAL,
            intent TEXT,
            current_position REAL,
            target_site TEXT,
            suggested_content_type TEXT,
            paa_keywords JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_content_gaps": """
        CREATE TABLE IF NOT EXISTS seo_content_gaps (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            keyword TEXT NOT NULL,
            volume INTEGER,
            kd INTEGER,
            funnel_stage TEXT,
            competitors_ranking JSONB DEFAULT '[]',
            top_url TEXT,
            target_site TEXT,
            competitor_source TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_content_briefs": """
        CREATE TABLE IF NOT EXISTS seo_content_briefs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            keyword TEXT NOT NULL,
            target_site TEXT,
            content_type TEXT,
            title TEXT,
            meta_description TEXT,
            target_word_count INTEGER,
            headings JSONB DEFAULT '[]',
            semantic_keywords JSONB DEFAULT '[]',
            faq_questions JSONB DEFAULT '[]',
            internal_links JSONB DEFAULT '[]',
            cta TEXT,
            brief_json JSONB,
            file_path TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_content_drafts": """
        CREATE TABLE IF NOT EXISTS seo_content_drafts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            brief_id UUID REFERENCES seo_content_briefs(id),
            keyword TEXT NOT NULL,
            target_site TEXT,
            title TEXT,
            word_count INTEGER,
            file_path TEXT,
            self_critique TEXT,
            status TEXT DEFAULT 'draft',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_backlink_prospects": """
        CREATE TABLE IF NOT EXISTS seo_backlink_prospects (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            domain TEXT NOT NULL,
            page_url TEXT,
            page_title TEXT,
            page_summary TEXT,
            author_name TEXT,
            contact_email TEXT,
            dr INTEGER,
            monthly_traffic INTEGER,
            prospect_type TEXT,
            discovery_method TEXT,
            outreach_angle TEXT,
            personalisation_notes TEXT,
            links_to_competitor BOOLEAN DEFAULT false,
            competitor_names JSONB DEFAULT '[]',
            score INTEGER DEFAULT 0,
            tier TEXT,
            status TEXT DEFAULT 'new',
            created_at TIMESTAMPTZ DEFAULT now(),
            last_contacted_at TIMESTAMPTZ,
            follow_up_count INTEGER DEFAULT 0,
            reply_received BOOLEAN DEFAULT false,
            target_site TEXT
        );
    """,
    "seo_rank_history": """
        CREATE TABLE IF NOT EXISTS seo_rank_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            date DATE NOT NULL,
            keyword TEXT NOT NULL,
            url TEXT,
            position REAL,
            previous_position REAL,
            impressions INTEGER,
            clicks INTEGER,
            target_site TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_pr_angles": """
        CREATE TABLE IF NOT EXISTS seo_pr_angles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            angle TEXT NOT NULL,
            target_site TEXT,
            target_publications JSONB DEFAULT '[]',
            status TEXT DEFAULT 'draft',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_outreach_blocklist": """
        CREATE TABLE IF NOT EXISTS seo_outreach_blocklist (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            domain TEXT NOT NULL UNIQUE,
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "haro_responses": """
        CREATE TABLE IF NOT EXISTS haro_responses (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            request_topic TEXT NOT NULL,
            pitch TEXT,
            target_publication TEXT,
            status TEXT DEFAULT 'pending_review',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "llm_cost_log": """
        CREATE TABLE IF NOT EXISTS llm_cost_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            task_type TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cached_tokens INTEGER DEFAULT 0,
            cost_usd REAL NOT NULL,
            site TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "llm_output_cache": """
        CREATE TABLE IF NOT EXISTS llm_output_cache (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            cache_key TEXT NOT NULL,
            task TEXT,
            input_key TEXT,
            result JSONB,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_outreach_emails": """
        CREATE TABLE IF NOT EXISTS seo_outreach_emails (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            prospect_id UUID REFERENCES seo_backlink_prospects(id),
            subject TEXT,
            body TEXT,
            tier INTEGER,
            template_type TEXT,
            sequence_step INTEGER DEFAULT 0,
            status TEXT DEFAULT 'queued',
            sent_at TIMESTAMPTZ,
            opened BOOLEAN DEFAULT false,
            replied BOOLEAN DEFAULT false,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "crm_contacts": """
        CREATE TABLE IF NOT EXISTS crm_contacts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            company_name TEXT NOT NULL,
            contact_name TEXT,
            contact_role TEXT,
            email TEXT,
            phone TEXT,
            website TEXT,
            city TEXT,
            region TEXT,
            postcode TEXT,
            country TEXT DEFAULT 'GB',
            category TEXT NOT NULL,
            subcategory TEXT,
            instagram TEXT,
            facebook TEXT,
            linkedin TEXT,
            outreach_status TEXT DEFAULT 'not_contacted',
            outreach_segment TEXT,
            score INTEGER DEFAULT 0,
            tier TEXT,
            backlink_prospect_id UUID,
            source TEXT,
            tags JSONB DEFAULT '[]',
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now(),
            last_contacted_at TIMESTAMPTZ
        );
    """,
    "crm_interactions": """
        CREATE TABLE IF NOT EXISTS crm_interactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            contact_id UUID NOT NULL,
            interaction_type TEXT NOT NULL,
            direction TEXT DEFAULT 'outbound',
            channel TEXT DEFAULT 'email',
            subject TEXT,
            body_preview TEXT,
            status TEXT DEFAULT 'logged',
            performed_by TEXT DEFAULT 'ralf',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "kitchen_makers": """
        CREATE TABLE IF NOT EXISTS kitchen_makers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            city TEXT,
            region TEXT,
            postcode TEXT,
            country TEXT DEFAULT 'GB',
            founded INTEGER,
            website TEXT,
            email TEXT,
            phone TEXT,
            description TEXT,
            style_categories JSONB DEFAULT '[]',
            budget_tier TEXT,
            verified BOOLEAN DEFAULT false,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    # --- Proactive agent infrastructure tables ---
    "heartbeat_wal": """
        CREATE TABLE IF NOT EXISTS heartbeat_wal (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            cycle_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'running',
            started_at TIMESTAMPTZ DEFAULT now(),
            completed_at TIMESTAMPTZ,
            resumed_from TEXT,
            tasks_json JSONB DEFAULT '[]',
            notes_json JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "working_buffer": """
        CREATE TABLE IF NOT EXISTS working_buffer (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key TEXT NOT NULL UNIQUE,
            value_json JSONB,
            expires_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ DEFAULT now(),
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "ralf_memory": """
        CREATE TABLE IF NOT EXISTS ralf_memory (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            source TEXT DEFAULT 'heartbeat',
            related_site TEXT,
            tags JSONB DEFAULT '[]',
            recall_count INTEGER DEFAULT 0,
            superseded_by UUID,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "ralf_schedule": """
        CREATE TABLE IF NOT EXISTS ralf_schedule (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            cadence TEXT NOT NULL,
            day_of_week INTEGER,
            day_of_month INTEGER,
            skill TEXT NOT NULL,
            boost_amount INTEGER DEFAULT 30,
            label TEXT,
            description TEXT,
            active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "ralf_schedule_log": """
        CREATE TABLE IF NOT EXISTS ralf_schedule_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            schedule_date DATE NOT NULL,
            skill TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            site TEXT,
            summary TEXT,
            heartbeat_id TEXT,
            completed_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    # --- Outreach system tables (Instantly V2 integration) ---
    "outreach_targets": """
        CREATE TABLE IF NOT EXISTS outreach_targets (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            site_id TEXT NOT NULL,
            site_name TEXT NOT NULL,
            url TEXT NOT NULL,
            contact_email TEXT,
            contact_name TEXT,
            article_url TEXT,
            article_title TEXT,
            outreach_type TEXT,
            domain_rating INTEGER,
            notes TEXT,
            status TEXT DEFAULT 'queued',
            created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE(url, site_id)
        );
    """,
    "outreach_emails": """
        CREATE TABLE IF NOT EXISTS outreach_emails (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            site_id TEXT NOT NULL,
            target_id UUID REFERENCES outreach_targets(id),
            instantly_campaign_id TEXT,
            instantly_campaign_name TEXT,
            outreach_type TEXT,
            emails_added INTEGER,
            duplicates_skipped INTEGER,
            launched BOOLEAN DEFAULT false,
            daily_limit INTEGER,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "outreach_replies": """
        CREATE TABLE IF NOT EXISTS outreach_replies (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            site_id TEXT,
            target_id UUID REFERENCES outreach_targets(id),
            instantly_campaign_id TEXT,
            from_address TEXT,
            subject TEXT,
            body TEXT,
            sentiment TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "outreach_links": """
        CREATE TABLE IF NOT EXISTS outreach_links (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            site_id TEXT NOT NULL,
            target_id UUID REFERENCES outreach_targets(id),
            target_url TEXT NOT NULL,
            link_url TEXT NOT NULL,
            anchor_text TEXT,
            domain_rating INTEGER,
            do_follow BOOLEAN DEFAULT true,
            confirmed_at TIMESTAMPTZ DEFAULT now(),
            last_checked_at TIMESTAMPTZ DEFAULT now(),
            is_live BOOLEAN DEFAULT true
        );
    """,
    "agent_turns": """
        CREATE TABLE IF NOT EXISTS agent_turns (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id TEXT NOT NULL DEFAULT 'default',
            agent_name TEXT NOT NULL DEFAULT 'ralf',
            turn_type TEXT NOT NULL,
            input TEXT,
            output TEXT,
            tokens_used INTEGER DEFAULT 0,
            model TEXT,
            duration_ms INTEGER,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "cron_executions": """
        CREATE TABLE IF NOT EXISTS cron_executions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id TEXT NOT NULL,
            fired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            completed_at TIMESTAMPTZ,
            status TEXT NOT NULL DEFAULT 'running',
            tasks_executed INTEGER DEFAULT 0,
            message_sent BOOLEAN DEFAULT false,
            tokens_used INTEGER DEFAULT 0,
            error TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
    "seo_our_rankings": """
        CREATE TABLE IF NOT EXISTS seo_our_rankings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            target_site TEXT NOT NULL,
            keyword TEXT NOT NULL,
            position INTEGER,
            previous_position INTEGER,
            change INTEGER,
            url TEXT,
            volume INTEGER,
            snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
            created_at TIMESTAMPTZ DEFAULT now()
        );
    """,
}


def ensure_tables(client: Any | None = None) -> None:
    """Create all required tables if they do not exist.

    Args:
        client: Optional Supabase client. Uses ``get_client()`` if None.
    """
    if _is_mock():
        for table_name in TABLE_SCHEMAS:
            _mock_store.setdefault(table_name, [])
        logger.info("Mock tables initialised: %s", list(TABLE_SCHEMAS.keys()))
        return

    if client is None:
        client = get_client()

    for table_name, ddl in TABLE_SCHEMAS.items():
        try:
            client.postgrest.rpc("exec_sql", {"query": ddl}).execute()
            logger.info("Ensured table: %s", table_name)
        except Exception:
            logger.warning(
                "Could not ensure table %s via RPC — may already exist",
                table_name,
                exc_info=True,
            )


# ---------------------------------------------------------------------------
# Generic CRUD helpers
# ---------------------------------------------------------------------------


def insert_record(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert a record into a Supabase table.

    Args:
        table: Table name.
        data: Record data.

    Returns:
        The inserted record dict.
    """
    if _is_mock():
        record = {"id": str(uuid.uuid4()), **data}
        _mock_store.setdefault(table, []).append(record)
        return record

    client = get_client()
    resp = client.table(table).insert(data).execute()
    return resp.data[0] if resp.data else data


def upsert_record(
    table: str, data: dict[str, Any], on_conflict: str = "id"
) -> dict[str, Any]:
    """Upsert a record (insert or update on conflict).

    Args:
        table: Table name.
        data: Record data.
        on_conflict: Column(s) to detect conflicts on.

    Returns:
        The upserted record dict.
    """
    if _is_mock():
        store = _mock_store.setdefault(table, [])
        for i, row in enumerate(store):
            if row.get(on_conflict) == data.get(on_conflict):
                store[i] = {**row, **data}
                return store[i]
        record = {"id": str(uuid.uuid4()), **data}
        store.append(record)
        return record

    client = get_client()
    resp = (
        client.table(table)
        .upsert(data, on_conflict=on_conflict)
        .execute()
    )
    return resp.data[0] if resp.data else data


def update_record(
    table: str, record_id: str, data: dict[str, Any], id_column: str = "id"
) -> dict[str, Any]:
    """Update an existing record by ID (partial update).

    Unlike `upsert_record`, this only modifies the specified fields and
    leaves all other columns unchanged.

    Args:
        table: Table name.
        record_id: Value of the ID column to match.
        data: Fields to update (only specified fields are changed).
        id_column: Name of the ID column.

    Returns:
        The updated record dict.

    Raises:
        ValueError: If no record with the given ID exists.
    """
    if _is_mock():
        store = _mock_store.get(table, [])
        for i, row in enumerate(store):
            if row.get(id_column) == record_id:
                store[i] = {**row, **data}
                return store[i]
        msg = f"No record with {id_column}={record_id} in {table}"
        raise ValueError(msg)

    client = get_client()
    resp = (
        client.table(table)
        .update(data)
        .eq(id_column, record_id)
        .execute()
    )
    if not resp.data:
        msg = f"No record with {id_column}={record_id} in {table}"
        raise ValueError(msg)
    return resp.data[0]


def query_table(
    table: str,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
    order_by: str | None = None,
    order_desc: bool = True,
) -> list[dict[str, Any]]:
    """Query records from a Supabase table with optional filters.

    Args:
        table: Table name.
        filters: Column equality filters (e.g. ``{"target_site": "kitchensdirectory"}``).
        limit: Maximum rows to return.
        order_by: Column to sort by.
        order_desc: Sort descending if True.

    Returns:
        List of matching record dicts.
    """
    if _is_mock():
        rows = _mock_store.get(table, [])
        if filters:
            rows = [
                r
                for r in rows
                if all(r.get(k) == v for k, v in filters.items())
            ]
        if order_by:
            rows = sorted(
                rows,
                key=lambda r: r.get(order_by, ""),
                reverse=order_desc,
            )
        return rows[:limit]

    client = get_client()
    query = client.table(table).select("*")
    if filters:
        for col, val in filters.items():
            query = query.eq(col, val)
    if order_by:
        query = query.order(order_by, desc=order_desc)
    resp = query.limit(limit).execute()
    return resp.data or []


# ---------------------------------------------------------------------------
# Specialised helpers
# ---------------------------------------------------------------------------


def log_llm_cost(
    *,
    task_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cost_usd: float,
    site: str = "",
) -> None:
    """Log an LLM API call to the cost log table.

    Args:
        task_type: The task identifier.
        model: Claude model used.
        input_tokens: Total input tokens.
        output_tokens: Total output tokens.
        cached_tokens: Cached input tokens.
        cost_usd: Cost in USD.
        site: Target site name.
    """
    insert_record(
        "llm_cost_log",
        {
            "task_type": task_type,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
            "site": site,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )


def get_weekly_spend() -> float:
    """Get total LLM spend for the current week.

    Returns:
        Total cost in USD for the current ISO week.
    """
    if _is_mock():
        rows = _mock_store.get("llm_cost_log", [])
        now = datetime.now(tz=timezone.utc)
        week_start = now - __import__("datetime").timedelta(
            days=now.weekday()
        )
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        total = 0.0
        for row in rows:
            created = row.get("created_at", "")
            if isinstance(created, str) and created >= week_start.isoformat():
                total += row.get("cost_usd", 0.0)
        return total

    client = get_client()
    now = datetime.now(tz=timezone.utc)
    week_start = now - __import__("datetime").timedelta(days=now.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    resp = (
        client.table("llm_cost_log")
        .select("cost_usd")
        .gte("created_at", week_start.isoformat())
        .execute()
    )
    return sum(row.get("cost_usd", 0.0) for row in (resp.data or []))


def get_makers_by_location(city: str) -> list[dict[str, Any]]:
    """Query kitchen maker listings by city/location.

    Args:
        city: City name to filter by.

    Returns:
        List of maker dicts with name, city, postcode, founded, etc.
    """
    if _is_mock():
        return [
            {
                "name": f"Mock Kitchen Co ({city})",
                "city": city,
                "postcode": "M1 1AA",
                "founded": 2005,
                "description": f"A premium bespoke kitchen maker based in {city}.",
            },
            {
                "name": f"{city} Handmade Kitchens",
                "city": city,
                "postcode": "M2 2BB",
                "founded": 2012,
                "description": f"Handcrafted kitchens designed and built in {city}.",
            },
        ]

    client = get_client()
    resp = (
        client.table("kitchen_makers")
        .select("*")
        .ilike("city", f"%{city}%")
        .execute()
    )
    return resp.data or []


def is_domain_blocked(domain: str) -> bool:
    """Check if a domain is on the permanent outreach blocklist.

    Args:
        domain: The domain to check.

    Returns:
        True if the domain is blocked.
    """
    rows = query_table(
        "seo_outreach_blocklist", filters={"domain": domain}, limit=1
    )
    return len(rows) > 0


def get_last_contact_date(domain: str) -> str | None:
    """Get the last date a domain was contacted for outreach.

    Args:
        domain: The domain to check.

    Returns:
        ISO date string of last contact, or None if never contacted.
    """
    rows = query_table(
        "seo_backlink_prospects",
        filters={"domain": domain},
        order_by="last_contacted_at",
        order_desc=True,
        limit=1,
    )
    if rows and rows[0].get("last_contacted_at"):
        return rows[0]["last_contacted_at"]
    return None


# ---------------------------------------------------------------------------
# Mock client (in-memory Supabase replacement)
# ---------------------------------------------------------------------------


class _MockTable:
    """Minimal mock of Supabase table query builder."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._data = _mock_store.setdefault(name, [])

    def select(self, columns: str = "*") -> "_MockTable":
        return self

    def insert(self, data: dict) -> "_MockTable":
        record = {"id": str(uuid.uuid4()), **data}
        self._data.append(record)
        self._result = [record]
        return self

    def upsert(self, data: dict, on_conflict: str = "id") -> "_MockTable":
        self._result = [data]
        return self

    def eq(self, col: str, val: Any) -> "_MockTable":
        self._data = [r for r in self._data if r.get(col) == val]
        return self

    def ilike(self, col: str, pattern: str) -> "_MockTable":
        clean = pattern.strip("%").lower()
        self._data = [
            r for r in self._data if clean in str(r.get(col, "")).lower()
        ]
        return self

    def gte(self, col: str, val: Any) -> "_MockTable":
        self._data = [r for r in self._data if r.get(col, "") >= val]
        return self

    def order(self, col: str, desc: bool = True) -> "_MockTable":
        self._data.sort(key=lambda r: r.get(col, ""), reverse=desc)
        return self

    def limit(self, n: int) -> "_MockTable":
        self._data = self._data[:n]
        return self

    def execute(self) -> "_MockResult":
        return _MockResult(getattr(self, "_result", self._data))


class _MockResult:
    def __init__(self, data: list[dict]) -> None:
        self.data = data


class _MockClient:
    """In-memory Supabase client replacement for testing."""

    def table(self, name: str) -> _MockTable:
        return _MockTable(name)
