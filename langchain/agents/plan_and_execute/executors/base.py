from abc import abstractmethod
from typing import Any

from pydantic import BaseModel

from langchain.agents.agent import AgentExecutor
from langchain.agents.plan_and_execute.schema import Step
from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain


class BaseExecutor(BaseModel):
    @abstractmethod
    def step(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Step:
        """Take step."""

    @abstractmethod
    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Step:
        """Take step."""


class ChainExecutor(BaseExecutor):
    chain: Chain

    def step(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Step:
        """Take step."""
        response = self.chain.run(**inputs, callbacks=callbacks)
        return Step(response=response)

    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Step:
        """Take step."""
        response = await self.chain.arun(**inputs, callbacks=callbacks)
        return Step(response=response)
