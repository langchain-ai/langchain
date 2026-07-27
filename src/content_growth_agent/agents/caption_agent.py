"""Caption Agent: generates captions for Instagram, LinkedIn, and X (Twitter threads)."""
from __future__ import annotations

import json
from typing import Any

from content_growth_agent.agents.base_agent import BaseAgent
from content_growth_agent.prompts.base_prompts import wrap_user_prompt
from content_growth_agent.models.output_models import CaptionOutput
from content_growth_agent.tools.logger import get_logger


CAPTION_PROMPT = (
    "Generate an Instagram caption (max 2200 chars), a LinkedIn post (short, professional), "
    "and a Twitter/X thread (list of short tweets) suitable for the video content. "
    "Return a JSON object with keys: instagram, linkedin, x_thread."
)


class CaptionAgent(BaseAgent):
    def __init__(self, llm, memory) -> None:
        super().__init__(llm, memory, name="CaptionAgent")
        self.logger = get_logger("CaptionAgent")

    async def _build_prompt(self, content: str, **kwargs: Any) -> tuple[str, str | None]:
        prompt = wrap_user_prompt(CAPTION_PROMPT, content)
        return prompt, None

    async def _parse_response(self, raw: str) -> CaptionOutput:
        try:
            # naïve JSON extraction
            start = raw.find("{")
            data = json.loads(raw[start:] if start != -1 else raw)
            instagram = data.get("instagram", "")
            linkedin = data.get("linkedin", "")
            x_thread = data.get("x_thread", [])
            return CaptionOutput(instagram=instagram, linkedin=linkedin, x_thread=x_thread)
        except Exception:
            self.logger.exception("Failed to parse CaptionAgent response")
            return CaptionOutput(instagram="", linkedin="", x_thread=[])
