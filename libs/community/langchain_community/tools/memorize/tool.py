from abc import abstractmethod
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.llms.gradient_ai import TrainResult


@runtime_checkable
class TrainableLLM(Protocol):
    """Protocol for trainable language models."""

    @abstractmethod
    def train_unsupervised(
        self,
        inputs: Sequence[str],
        **kwargs: Any,
    ) -> TrainResult:
        ...

    @abstractmethod
    async def atrain_unsupervised(
        self,
        inputs: Sequence[str],
        **kwargs: Any,
    ) -> TrainResult:
        ...


class Memorize(BaseTool):
    """Tool that trains a language model."""

    name: str = "Memorize"
    description: str = (
        "Useful whenever you observed novel information "
        "from previous conversation history, "
        "i.e., another tool's action outputs or human comments. "
        "The action input should include observed information in detail, "
        "then the tool will fine-tune yourself to remember it."
    )
    llm: TrainableLLM = Field()

    def _run(
        self,
        information_to_learn: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        train_result = self.llm.train_unsupervised((information_to_learn,))
        return f"Train complete. Loss: {train_result['loss']}"

    async def _arun(
        self,
        information_to_learn: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        train_result = await self.llm.atrain_unsupervised((information_to_learn,))
        return f"Train complete. Loss: {train_result['loss']}"
