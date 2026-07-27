"""Public API for the content growth agent.

Provides a simple async function `process_url` that can be called by a web UI
or CLI. Keeps orchestration out of the UI code.
"""
from __future__ import annotations

from content_growth_agent.workflows.content_workflow import ContentWorkflow
from content_growth_agent.models.input_models import VideoInput
from content_growth_agent.tools.logger import get_logger

logger = get_logger("api")


async def process_url(video: VideoInput):
    """Process a video URL and return a ContentAnalysis result.

    Args:
        video: VideoInput Pydantic model
    """
    workflow = ContentWorkflow()
    try:
        result = await workflow.execute(str(video.url), language=video.language)
        return result
    except Exception:
        logger.exception("Failed to process URL")
        raise
