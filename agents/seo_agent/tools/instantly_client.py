"""Instantly V2 API client for email campaign management."""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

INSTANTLY_BASE = "https://api.instantly.ai/api/v2"


def _get_api_key() -> str:
    key = os.environ.get("INSTANTLY_API_KEY", "").strip()
    if not key:
        raise ValueError("INSTANTLY_API_KEY not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, headers=_headers())


def create_campaign(name: str, sender_email: str, schedule: dict | None = None) -> dict:
    """Create a new Instantly campaign."""
    payload = {
        "name": name,
        "schedule": schedule or {
            "days": {
                "monday": True,
                "tuesday": True,
                "wednesday": True,
                "thursday": True,
                "friday": True,
                "saturday": False,
                "sunday": False,
            },
            "timezone": "Europe/London",
            "start_hour": "08:00",
            "end_hour": "17:00",
        },
    }
    with _client() as c:
        resp = c.post(f"{INSTANTLY_BASE}/campaign", json=payload)
        resp.raise_for_status()
        return resp.json()


def launch_campaign(campaign_id: str) -> dict:
    """Launch a draft campaign."""
    with _client() as c:
        resp = c.post(f"{INSTANTLY_BASE}/campaign/{campaign_id}/launch")
        resp.raise_for_status()
        return resp.json()


def pause_campaign(campaign_id: str) -> dict:
    """Pause a running campaign."""
    with _client() as c:
        resp = c.post(f"{INSTANTLY_BASE}/campaign/{campaign_id}/pause")
        resp.raise_for_status()
        return resp.json()


def list_campaigns(limit: int = 20) -> list[dict]:
    """List campaigns from Instantly."""
    with _client() as c:
        resp = c.get(f"{INSTANTLY_BASE}/campaign", params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data if isinstance(data, list) else [])


def add_leads(campaign_id: str, leads: list[dict], skip_if_in_workspace: bool = True) -> dict:
    """Add leads to a campaign. Each lead needs email + custom variables."""
    with _client() as c:
        resp = c.post(f"{INSTANTLY_BASE}/lead", json={
            "campaign_id": campaign_id,
            "leads": leads,
            "skip_if_in_workspace": skip_if_in_workspace,
        })
        resp.raise_for_status()
        return resp.json()


def get_campaign_stats(campaign_id: str) -> dict:
    """Get campaign analytics overview."""
    with _client() as c:
        resp = c.get(
            f"{INSTANTLY_BASE}/analytics/campaign/overview",
            params={"campaign_id": campaign_id},
        )
        resp.raise_for_status()
        return resp.json()


def list_replies(campaign_id: str | None = None, limit: int = 50) -> list[dict]:
    """List replies, optionally filtered by campaign."""
    params: dict = {"limit": limit}
    if campaign_id:
        params["campaign_id"] = campaign_id
    with _client() as c:
        resp = c.get(f"{INSTANTLY_BASE}/reply/list", params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])
