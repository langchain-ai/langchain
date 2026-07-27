"""Content workflow orchestration.

Validates input, extracts content, creates agents and runs them (in parallel).
Aggregates outputs into a ContentAnalysis object.
"""
from __future__ import annotations

import asyncio
from typing import Any

from content_growth_agent.config.settings import settings
from content_growth_agent.core.llm_factory import get_llm
from content_growth_agent.core.memory_factory import get_memory
from content_growth_agent.extractors.youtube_extractor import extract_transcript, extract_metadata
from content_growth_agent.agents.seo_agent import SEOAgent
from content_growth_agent.agents.caption_agent import CaptionAgent
from content_growth_agent.agents.thumbnail_agent import ThumbnailAgent
from content_growth_agent.models.output_models import ContentAnalysis, SEOOutput, ThumbnailOutput, CaptionOutput
from content_growth_agent.tools.logger import get_logger
from content_growth_agent.prompts.base_prompts import wrap_user_prompt

logger = get_logger("content_workflow")


class ContentWorkflow:
    def __init__(self) -> None:
        # For MVP we create fresh llm and memory per workflow invocation; in a
        # production system we may reuse instances across requests / pool them.
        self.llm = get_llm()
        self.memory = get_memory()

    async def execute(self, url: str, *, language: str = "en") -> ContentAnalysis:
        logger.info("Starting content workflow for %s", url)

        # 1) Extract content
        try:
            transcript = extract_transcript(url, languages=[language])
        except Exception as exc:
            logger.exception("Transcript extraction failed")
            transcript = ""

        metadata = extract_metadata(url)
        context = ""
        if metadata:
            context += f"Title: {metadata.get('title','')}\n"
            context += f"Author: {metadata.get('author_name','')}\n"
        context += "\nTranscript:\n" + (transcript or "")

        # 2) Create agents
        seo_agent = SEOAgent(self.llm, self.memory)
        caption_agent = CaptionAgent(self.llm, self.memory)
        thumbnail_agent = ThumbnailAgent(self.llm, self.memory)

        # 3) Run agents in parallel
        tasks = [
            asyncio.create_task(seo_agent.run(context)),
            asyncio.create_task(caption_agent.run(context)),
            asyncio.create_task(thumbnail_agent.run(context)),
        ]

        done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        # Collect results with resilience
        seo_res = SEOOutput(titles=[], descriptions=[], keywords=[])
        caption_res = CaptionOutput(instagram="", linkedin="", x_thread=[])
        thumb_res = ThumbnailOutput(texts=[], color_schemes=None)

        for t in done:
            try:
                res = t.result()
                if isinstance(res, SEOOutput):
                    seo_res = res
                elif isinstance(res, CaptionOutput):
                    caption_res = res
                elif isinstance(res, ThumbnailOutput):
                    thumb_res = res
            except Exception:
                logger.exception("Agent task failed")

        # 4) Generate short summary via LLM
        try:
            summary_prompt = wrap_user_prompt("Summarize the following video in 2-3 concise paragraphs.", context)
            summary = await self.llm.agenerate(summary_prompt)
        except Exception:
            logger.exception("Failed to generate summary")
            summary = ""

        # 5) Aggregate
        analysis = ContentAnalysis(
            summary=summary or "",
            seo=seo_res,
            thumbnail=thumb_res,
            captions=caption_res,
            hashtags=seo_res.keywords or [],
            calendar=None,
            competitor_analysis=None,
        )

        logger.info("Content workflow finished for %s", url)
        return analysis
