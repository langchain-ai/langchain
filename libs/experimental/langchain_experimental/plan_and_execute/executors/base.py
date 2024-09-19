from abc import abstractmethod
from typing import Any

from langchain.chains.base import Chain
from langchain_core.callbacks.manager import Callbacks
from pydantic import BaseModel

from langchain_experimental.plan_and_execute.schema import StepResponse


class BaseExecutor(BaseModel):
    """Base executor."""

    @abstractmethod
    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""

    @abstractmethod
    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take async step."""


class ChainExecutor(BaseExecutor):
    """Chain executor."""

    chain: Chain
    """The chain to use."""

    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = self.chain.run(**inputs, callbacks=callbacks)
        return StepResponse(response=response)

    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = await self.chain.arun(**inputs, callbacks=callbacks)
        return StepResponse(response=response)
