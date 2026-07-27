"""Base agent class that other specialist agents extend.

Provides a consistent interface: agents are constructed with an LLM and a memory
backend and expose an async `run` method that performs the agent's task.
"""
from __future__ import annotations

from typing import Any, Optional
import asyncio

from content_growth_agent.core.base_memory import BaseMemory, ChatMessage
from content_growth_agent.core.llm_factory import AsyncOpenAIWrapper
from content_growth_agent.tools.logger import get_logger


class BaseAgent:
    """Minimal base class for specialist agents.

    Subclasses should implement the `_build_prompt` and `_parse_response` methods.
    """

    def __init__(self, llm: AsyncOpenAIWrapper, memory: BaseMemory, name: str | None = None) -> None:
        self.llm = llm
        self.memory = memory
        self.logger = get_logger(name or self.__class__.__name__)

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """High-level entrypoint for the agent's job.

        The default implementation builds the prompt, invokes the LLM, stores
        to memory, parses the response and returns a typed result.
        """
        prompt, system = await self._build_prompt(*args, **kwargs)
        self.logger.debug("Prompt built for agent %s", self.__class__.__name__)

        # Store user prompt in memory
        try:
            await self.memory.add_message(ChatMessage(role="user", content=prompt))
        except Exception:
            self.logger.exception("Failed to add user message to memory")

        # Call the LLM
        try:
            raw = await self.llm.agenerate(prompt, system=system)
        except Exception:
            self.logger.exception("LLM invocation failed")
            raise

        # Store assistant reply
        try:
            await self.memory.add_message(ChatMessage(role="assistant", content=raw))
        except Exception:
            self.logger.exception("Failed to add assistant message to memory")

        # Parse and return
        return await self._parse_response(raw)

    async def _build_prompt(self, *args: Any, **kwargs: Any) -> tuple[str, Optional[str]]:
        """Build prompt and optional system message. Must be implemented by subclasses."""
        raise NotImplementedError

    async def _parse_response(self, raw: str) -> Any:
        """Parse LLM output into structured data. Must be implemented by subclasses."""
        raise NotImplementedError
