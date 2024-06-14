from abc import abstractmethod
from typing import Any, List, Optional

from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import Callbacks

from langchain_experimental.plan_and_execute.schema import Plan, PlanOutputParser
from langchain_experimental.pydantic_v1 import BaseModel


class BasePlanner(BaseModel):
    """Base planner."""

    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""

    @abstractmethod
    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Given input, asynchronously decide what to do."""


class LLMPlanner(BasePlanner):
    """LLM planner."""

    llm_chain: LLMChain
    """The LLM chain to use."""
    output_parser: PlanOutputParser
    """The output parser to use."""
    stop: Optional[List] = None
    """The stop list to use."""

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return self.output_parser.parse(llm_response)

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Given input, asynchronously decide what to do."""
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response)
