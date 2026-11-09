"""Base class for decision models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, TypeAdapter
from typing_extensions import override

from langchain_core.decisions.types import (
    Box,
    Choice,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    Question,
    Score,
    State,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config, run_in_executor

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_QUESTION_ADAPTER: TypeAdapter[Noul | Choice | Score | Box] = TypeAdapter(Question)


class BaseDecisionModel(
    RunnableSerializable[ClassifierRequest, ClassifierResponse], ABC
):
    """Answer typed questions about text, messages, or structured state.

    A decision model evaluates shared state against one or more questions and returns
    typed answers: a probability for a `Noul`, a selected label for a `Choice`, an
    expected level for a `Score`, and a location for a `Box`. Answers keep their
    probability distributions and confidence so applications can choose their own
    thresholds.

    Subclasses implement `_classify`, and optionally `_aclassify`, to call a provider.
    This class validates questions and handles callbacks, tracing metadata, and
    token-usage reporting.

    ??? example "Implement a decision model"

        ```python
        from langchain_core.decisions import BaseDecisionModel, ClassifierResponse


        class MyDecisionModel(BaseDecisionModel):
            model: str = "my-model"

            @property
            def _provider(self) -> str:
                return "my_provider"

            def _classify(self, state, questions):
                body = my_client.classify(state=state, questions=questions)
                return ClassifierResponse.model_validate(body)
        ```
    """

    model: str
    """Provider model name used to answer questions."""

    @property
    @abstractmethod
    def _provider(self) -> str:
        """Provider name reported to LangSmith as `ls_provider`."""

    @abstractmethod
    def _classify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        """Answer one request synchronously.

        Args:
            state: State to evaluate.
            questions: Validated questions keyed by answer ID.

        Returns:
            Typed answers keyed by answer ID, with request metadata.
        """

    async def _aclassify(
        self, state: State, questions: dict[str, Question]
    ) -> ClassifierResponse:
        """Answer one request asynchronously.

        The default implementation runs `_classify` in an executor. Override it to use
        a native asynchronous client.

        Args:
            state: State to evaluate.
            questions: Validated questions keyed by answer ID.

        Returns:
            Typed answers keyed by answer ID, with request metadata.
        """
        return await run_in_executor(None, self._classify, state, questions)

    @override
    def invoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        """Classify one request synchronously.

        Args:
            input: State and questions to evaluate.
            config: Optional runnable configuration for callbacks, tags, metadata,
                and tracing.
            **_: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            Typed answers keyed by answer ID, with request metadata.

        Raises:
            ValueError: If no questions are supplied or a question is invalid.
        """
        return self._call_with_config(
            self._classify_and_record,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    @override
    async def ainvoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> ClassifierResponse:
        """Classify one request asynchronously.

        Args:
            input: State and questions to evaluate.
            config: Optional runnable configuration for callbacks, tags, metadata,
                and tracing.
            **_: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            Typed answers keyed by answer ID, with request metadata.

        Raises:
            ValueError: If no questions are supplied or a question is invalid.
        """
        return await self._acall_with_config(
            self._aclassify_and_record,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    def _classify_and_record(self, request: ClassifierRequest) -> ClassifierResponse:
        questions = _validate_questions(request["questions"])
        return _record_usage(self._classify(request["state"], questions))

    async def _aclassify_and_record(
        self, request: ClassifierRequest
    ) -> ClassifierResponse:
        questions = _validate_questions(request["questions"])
        return _record_usage(await self._aclassify(request["state"], questions))

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """Set `ls_provider` and `ls_model_name` when the run is created."""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": self._provider,
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config


def _validate_questions(
    questions: Mapping[str, Question | Mapping[str, Any]],
) -> dict[str, Question]:
    """Validate questions supplied as question objects, other models, or dicts.

    Raises:
        ValueError: If no questions are supplied or a question is invalid.
    """
    if not questions:
        msg = "At least one question is required."
        raise ValueError(msg)
    return {name: _validate_question(question) for name, question in questions.items()}


def _validate_question(question: object) -> Question:
    if isinstance(question, (Noul, Choice, Score, Box)):
        return question
    if isinstance(question, BaseModel):
        question = question.model_dump()
    return _QUESTION_ADAPTER.validate_python(question)


def _record_usage(response: ClassifierResponse) -> ClassifierResponse:
    """Attach token usage to the active run, if there is one.

    Nothing is written when tracing is disabled, and a tracing failure never fails an
    otherwise successful request.
    """
    if response.usage is None:
        return response
    try:
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.extra.setdefault("metadata", {})["usage_metadata"] = dict(
                response.usage
            )
    except Exception:  # tracing must not break classification
        logger.debug("Could not attach decision model usage.", exc_info=True)
    return response


__all__ = ["BaseDecisionModel"]
