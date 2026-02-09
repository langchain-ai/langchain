"""Async-safe interrupt execution utilities."""

import asyncio
from typing import Any

from langgraph.types import interrupt

from langchain.agents.middleware.human_in_the_loop import HITLRequest


async def execute_interrupt_async(hitl_request: HITLRequest) -> dict[str, Any]:
    """Execute an interrupt request in an async-safe manner.

    This function ensures that interrupt execution preserves the runnable context
    by using asyncio.to_thread() to run the synchronous interrupt function.

    Args:
        hitl_request: The HITL request containing action requests and review configs.

    Returns:
        The interrupt response containing decisions.
    """
    return await asyncio.to_thread(interrupt, hitl_request)
