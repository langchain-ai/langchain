from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config

from langchain_ai21.ai21_base import AI21Base

ANSWER_NOT_IN_CONTEXT_RESPONSE = "Answer not in context"

ContextType = Union[str, List[Union[Document, str]]]


class ContextualAnswerInput(TypedDict):
    """Input for the ContextualAnswers runnable."""

    context: ContextType
    question: str


class AI21ContextualAnswers(RunnableSerializable[ContextualAnswerInput, str], AI21Base):
    """Runnable for the AI21 Contextual Answers API."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def InputType(self) -> Type[ContextualAnswerInput]:
        """Get the input type for this runnable."""
        return ContextualAnswerInput

    @property
    def OutputType(self) -> Type[str]:
        """Get the input type for this runnable."""
        return str

    def invoke(
        self,
        input: ContextualAnswerInput,
        config: Optional[RunnableConfig] = None,
        response_if_no_answer_found: str = ANSWER_NOT_IN_CONTEXT_RESPONSE,
        **kwargs: Any,
    ) -> str:
        config = ensure_config(config)
        return self._call_with_config(
            func=lambda inner_input: self._call_contextual_answers(
                inner_input, response_if_no_answer_found
            ),
            input=input,
            config=config,
            run_type="llm",
        )

    def _call_contextual_answers(
        self,
        input: ContextualAnswerInput,
        response_if_no_answer_found: str,
    ) -> str:
        context, question = self._convert_input(input)
        response = self.client.answer.create(context=context, question=question)

        if response.answer is None:
            return response_if_no_answer_found

        return response.answer

    def _convert_input(self, input: ContextualAnswerInput) -> Tuple[str, str]:
        context, question = self._extract_context_and_question(input)

        context = self._parse_context(context)

        return context, question

    def _extract_context_and_question(
        self,
        input: ContextualAnswerInput,
    ) -> Tuple[ContextType, str]:
        context = input.get("context")
        question = input.get("question")

        if not context or not question:
            raise ValueError(
                f"Input must contain a 'context' and 'question' fields. Got {input}"
            )

        if not isinstance(context, list) and not isinstance(context, str):
            raise ValueError(
                f"Expected input to be a list of strings or Documents."
                f" Received {type(input)}"
            )

        return context, question

    def _parse_context(self, context: ContextType) -> str:
        if isinstance(context, str):
            return context

        docs = [
            item.page_content if isinstance(item, Document) else item
            for item in context
        ]

        return "\n".join(docs)
