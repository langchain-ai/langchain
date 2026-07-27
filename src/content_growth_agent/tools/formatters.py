"""Helpers to serialize and format outputs for UI or API."""
from __future__ import annotations

from typing import Any
from content_growth_agent.models.output_models import ContentAnalysis


def content_to_dict(content: ContentAnalysis) -> dict[str, Any]:
    return content.model_dump()
