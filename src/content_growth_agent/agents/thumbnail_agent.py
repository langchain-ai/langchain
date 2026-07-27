"""Thumbnail Agent: suggests short text overlays and simple design hints."""
from __future__ import annotations

from typing import Any

from content_growth_agent.agents.base_agent import BaseAgent
from content_growth_agent.prompts.base_prompts import wrap_user_prompt
from content_growth_agent.models.output_models import ThumbnailOutput
from content_growth_agent.tools.logger import get_logger


THUMBNAIL_PROMPT = (
    "Suggest 6 short (3-6 words) thumbnail text overlays and up to 3 suggested color schemes. "
    "Return JSON: {\"texts\": [...], \"color_schemes\": [...]}"
)


class ThumbnailAgent(BaseAgent):
    def __init__(self, llm, memory) -> None:
        super().__init__(llm, memory, name="ThumbnailAgent")
        self.logger = get_logger("ThumbnailAgent")

    async def _build_prompt(self, content: str, **kwargs: Any) -> tuple[str, str | None]:
        prompt = wrap_user_prompt(THUMBNAIL_PROMPT, content)
        return prompt, None

    async def _parse_response(self, raw: str) -> ThumbnailOutput:
        # Simple extraction: try to parse JSON, otherwise return heuristics
        import json

        try:
            start = raw.find("{")
            data = json.loads(raw[start:] if start != -1 else raw)
            texts = data.get("texts", [])
            colors = data.get("color_schemes", [])
            return ThumbnailOutput(texts=texts or [], color_schemes=colors or None)
        except Exception:
            self.logger.exception("Failed to parse thumbnail response")
            return ThumbnailOutput(texts=[], color_schemes=None)
