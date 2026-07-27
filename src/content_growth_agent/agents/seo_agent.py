"""SEO Agent: generates titles, descriptions, and keywords."""
from __future__ import annotations

import json
from typing import Any

from content_growth_agent.agents.base_agent import BaseAgent
from content_growth_agent.prompts.seo_prompts import (
    build_title_prompt,
    build_description_prompt,
    build_keywords_prompt,
)
from content_growth_agent.models.output_models import SEOOutput
from content_growth_agent.tools.logger import get_logger


class SEOAgent(BaseAgent):
    def __init__(self, llm, memory) -> None:
        super().__init__(llm, memory, name="SEOAgent")
        self.logger = get_logger("SEOAgent")

    async def _build_prompt(self, content: str, **kwargs: Any) -> tuple[str, str | None]:
        # We'll request titles, descriptions, and keywords separately by returning
        # a concatenated prompt to simplify LLM interaction for MVP.
        prompt = build_title_prompt(content) + "\n\n" + build_description_prompt(content) + "\n\n" + build_keywords_prompt(content)
        system = None
        return prompt, system

    async def _parse_response(self, raw: str) -> SEOOutput:
        # Try to extract JSON arrays from the response. The LLM may include text
        # before/after the JSON; we attempt to find the first JSON array/object.
        try:
            # Find first JSON substring
            start = raw.find("[")
            if start == -1:
                # fallback: try to parse full raw as JSON
                data = json.loads(raw)
            else:
                data = json.loads(raw[start:])

            # Expecting e.g. [titles_array, descriptions_array, keywords_array]
            if isinstance(data, list) and len(data) >= 3:
                titles = data[0]
                descriptions = data[1]
                keywords = data[2]
            else:
                # if single object with keys
                titles = data.get("titles") if isinstance(data, dict) else []
                descriptions = data.get("descriptions") if isinstance(data, dict) else []
                keywords = data.get("keywords") if isinstance(data, dict) else []

            seo = SEOOutput(titles=titles or [], descriptions=descriptions or [], keywords=keywords or [])
            return seo
        except Exception:
            self.logger.exception("Failed to parse SEO agent response")
            # Return empty structure on failure to keep workflow robust
            return SEOOutput(titles=[], descriptions=[], keywords=[])
