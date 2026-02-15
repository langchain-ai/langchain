from typing import Any, Optional, Union
from langchain_core.runnables.config import RunnableConfig
from langchain_core.load.serializable import Serializable 
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

class TieredSemanticRouter(Serializable):
    """
    Proactively routes requests to a primary or fallback model 
    based on a complexity score.
    """
    primary: BaseChatModel
    fallback: BaseChatModel
    threshold: float = 0.7

    def _get_complexity_score(self, input_data: Any) -> float:
        # Initial heuristic-based complexity check
        text = str(input_data)
        complexity = min(len(text) / 500, 1.0) 
        keywords = ["analyze", "code", "explain", "rewrite", "compare"]
        if any(word in text.lower() for word in keywords):
            complexity += 0.3
        return min(complexity, 1.0)

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Union[BaseMessage, str]:
        score = self._get_complexity_score(input)
        if score <= self.threshold:
            # Fix: Pass config and kwargs
            return self.primary.invoke(input, config=config, **kwargs)
        return self.fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Union[BaseMessage, str]:
        score = self._get_complexity_score(input)
        if score <= self.threshold:
            # Fix: Pass config and kwargs
            return await self.primary.ainvoke(input, config=config, **kwargs)
        return await self.fallback.ainvoke(input, config=config, **kwargs)