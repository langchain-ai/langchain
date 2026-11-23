"""Structured output parsers raise `ModelRefusalError` for refusals."""

from typing import Any

import pytest
from pydantic import BaseModel

from langchain_core.exceptions import ModelRefusalError
from langchain_core.language_models import GenericFakeChatModel, LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    JsonOutputToolsParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.outputs import ChatGeneration
from langchain_core.runnables import Runnable


class Answer(BaseModel):
    answer: str


REFUSAL = AIMessage(
    content=[],
    response_metadata={
        "model_provider": "anthropic",
        "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": None},
    },
)

PARSERS: list[BaseOutputParser[Any]] = [
    JsonOutputParser(),
    PydanticOutputParser(pydantic_object=Answer),
    JsonOutputToolsParser(),
    JsonOutputKeyToolsParser(key_name="Answer", first_tool_only=True),
    PydanticToolsParser(tools=[Answer], first_tool_only=True),
]


@pytest.mark.parametrize("parser", PARSERS, ids=lambda p: type(p).__name__)
@pytest.mark.parametrize("partial", [False, True])
def test_parser_raises_on_refusal(
    parser: BaseOutputParser[Any], *, partial: bool
) -> None:
    with pytest.raises(ModelRefusalError) as exc_info:
        parser.parse_result([ChatGeneration(message=REFUSAL)], partial=partial)
    assert "cyber" in str(exc_info.value)
    assert exc_info.value.raw_message is REFUSAL


class _ToolFakeChatModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def test_with_structured_output_raises_on_refusal() -> None:
    model = _ToolFakeChatModel(messages=iter([REFUSAL]))
    with pytest.raises(ModelRefusalError):
        model.with_structured_output(Answer).invoke("hi")


def test_with_structured_output_include_raw_reports_refusal() -> None:
    model = _ToolFakeChatModel(messages=iter([REFUSAL]))
    result = model.with_structured_output(Answer, include_raw=True).invoke("hi")
    assert isinstance(result, dict)
    assert isinstance(result["parsing_error"], ModelRefusalError)
    assert result["parsed"] is None


def test_tool_parser_raises_when_refusal_follows_tool_call() -> None:
    message = AIMessage(
        content=[],
        tool_calls=[{"name": "Answer", "args": {"answer": "x"}, "id": "1"}],
        response_metadata={"model_provider": "anthropic", "stop_reason": "refusal"},
    )
    with pytest.raises(ModelRefusalError):
        PydanticToolsParser(tools=[Answer]).parse_result(
            [ChatGeneration(message=message)]
        )


@pytest.mark.parametrize(
    "message",
    [
        AIMessage(
            '{"answer": "no"}',
            response_metadata={
                "model_provider": "anthropic",
                "stop_reason": "end_turn",
            },
        ),
        AIMessage(
            '{"answer": "no"}',
            additional_kwargs={"refusal": None},
            response_metadata={"model_provider": "openai"},
        ),
    ],
    ids=["anthropic", "openai"],
)
def test_parser_parses_non_refusal(message: AIMessage) -> None:
    assert JsonOutputParser().parse_result([ChatGeneration(message=message)]) == {
        "answer": "no"
    }
