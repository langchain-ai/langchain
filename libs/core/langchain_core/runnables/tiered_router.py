"""Tiered Semantic Router for LangChain."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.load.serializable import Serializable
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig


class TieredSemanticRouter(Serializable):
    """Proactively routes requests to a primary or fallback model.

    Routes based on a complexity score calculated from the input text.
    """

    primary: BaseChatModel
    fallback: BaseChatModel
    threshold: float = 0.7

    def _get_complexity_score(self, input_data: Any) -> float:
        """Calculate a complexity score for the input."""
        text = str(input_data)
        complexity = min(len(text) / 500, 1.0)
        keywords = ["analyze", "code", "explain", "rewrite", "compare"]
        if any(word in text.lower() for word in keywords):
            complexity += 0.3
        return min(complexity, 1.0)

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage | str:
        """Invoke the appropriate model based on complexity."""
        score = self._get_complexity_score(input)
        if score <= self.threshold:
            return self.primary.invoke(input, config=config, **kwargs)
        return self.fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage | str:
        """Asynchronously invoke the appropriate model based on complexity."""
        score = self._get_complexity_score(input)
        if score <= self.threshold:
            return await self.primary.ainvoke(input, config=config, **kwargs)
        return await self.fallback.ainvoke(input, config=config, **kwargs)
