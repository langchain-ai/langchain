"""Thin Supabase client — shared database with SEO agent."""

import logging
from typing import Any
from supabase import create_client, Client

from agents.scraper_agent.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


def insert_record(table: str, record: dict) -> dict:
    client = get_client()
    resp = client.table(table).insert(record).execute()
    return resp.data[0] if resp.data else {}


def query_table(table: str, filters: dict | None = None, limit: int = 100) -> list[dict]:
    client = get_client()
    q = client.table(table).select("*").limit(limit)
    if filters:
        for k, v in filters.items():
            q = q.eq(k, v)
    resp = q.execute()
    return resp.data or []


def upsert_record(table: str, record: dict, on_conflict: str = "") -> dict:
    client = get_client()
    q = client.table(table).upsert(record)
    if on_conflict:
        q = client.table(table).upsert(record, on_conflict=on_conflict)
    resp = q.execute()
    return resp.data[0] if resp.data else {}
