"""Tests for parsing structured output wrapped in reasoning tags."""

import json
from typing import Any

import pytest
from pydantic import BaseModel

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    parse_tool_call,
)
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation

INNER_JSON = '{"result": [{"name": "promo", "information": "optimize pricing"}]}'
REASONING = "<think>\nThe user wants a plan. I should not emit raw JSON here.\n</think>"

WRAPPED_ARGUMENTS = [
    pytest.param(f"{REASONING}\n{INNER_JSON}", id="think-prefix"),
    pytest.param(f"<tool_call>\n{INNER_JSON}\n</tool_call>", id="tool-call-wrapper"),
    pytest.param(f"{REASONING}\n<tool_call>\n{INNER_JSON}\n</tool_call>", id="both"),
    pytest.param(
        f"<tool_call>\n{REASONING}\n{INNER_JSON}\n</tool_call>",
        id="think-inside-wrapper",
    ),
    pytest.param(f"{REASONING}\n<tool_call>\n{INNER_JSON}", id="unclosed-wrapper"),
]


class Section(BaseModel):
    name: str
    information: str


class AnalysisResult(BaseModel):
    result: list[Section]


class NameOnly(BaseModel):
    name: str


class _ReasoningToolModel(BaseChatModel):
    """Fake model that emits a raw tool call whose arguments contain reasoning."""

    tool_name: str
    arguments: str

    @property
    def _llm_type(self) -> str:
        return "reasoning-tool-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = (messages, stop, run_manager, kwargs)
        message = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": self.arguments,
                        },
                    }
                ]
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)


def _generation(name: str, arguments: str) -> ChatGeneration:
    return ChatGeneration(
        message=AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
            },
        )
    )


@pytest.mark.parametrize("arguments", WRAPPED_ARGUMENTS)
def test_parse_tool_call_reasoning_full(arguments: str) -> None:
    raw = {"function": {"name": "AnalysisResult", "arguments": arguments}}

    parsed = parse_tool_call(raw, return_id=False)

    assert parsed is not None
    assert parsed["args"] == json.loads(INNER_JSON)


@pytest.mark.parametrize("arguments", WRAPPED_ARGUMENTS)
def test_parse_tool_call_reasoning_partial(arguments: str) -> None:
    truncated = arguments[: arguments.rindex("]")]
    raw = {"function": {"name": "AnalysisResult", "arguments": truncated}}

    parsed = parse_tool_call(raw, partial=True, return_id=False)

    assert parsed is not None
    assert parsed["args"]["result"][0]["name"] == "promo"


def test_parse_tool_call_preserves_tag_literals_in_json_strings() -> None:
    payload = {
        "note": "<think>keep</think>",
        "wrapped": "<tool_call>keep me</tool_call>",
    }
    raw = {"function": {"name": "F", "arguments": json.dumps(payload)}}

    parsed = parse_tool_call(raw, return_id=False)

    assert parsed is not None
    assert parsed["args"] == payload


def test_parse_tool_call_wrapped_payload_with_tag_literals() -> None:
    payload = {"note": "<think>keep</think>", "wrapped": "<tool_call></tool_call>"}
    arguments = f"<tool_call>{json.dumps(payload)}</tool_call>"
    raw = {"function": {"name": "F", "arguments": arguments}}

    parsed = parse_tool_call(raw, return_id=False)

    assert parsed is not None
    assert parsed["args"] == payload


def test_parse_tool_call_invalid_json_preserves_original_arguments() -> None:
    arguments = '<think>reasoning</think>\n{"broken": }'
    raw = {"function": {"name": "F", "arguments": arguments}}

    with pytest.raises(OutputParserException) as exc_info:
        parse_tool_call(raw)

    assert arguments in str(exc_info.value)
    assert exc_info.value.llm_output == arguments


def test_parse_tool_call_unclosed_reasoning_partial_returns_none() -> None:
    raw = {"function": {"name": "F", "arguments": "<think>still thinking"}}

    assert parse_tool_call(raw, partial=True, return_id=False) is None


def test_parse_tool_call_unclosed_reasoning_full_raises() -> None:
    arguments = "<think>still thinking"
    raw = {"function": {"name": "F", "arguments": arguments}}

    with pytest.raises(OutputParserException) as exc_info:
        parse_tool_call(raw)

    assert exc_info.value.llm_output == arguments


@pytest.mark.parametrize("arguments", WRAPPED_ARGUMENTS)
def test_pydantic_tools_parser_reasoning_full(arguments: str) -> None:
    parser = PydanticToolsParser(tools=[AnalysisResult])

    result = parser.parse_result([_generation("AnalysisResult", arguments)])

    assert isinstance(result[0], AnalysisResult)
    assert result[0].result[0].name == "promo"
    assert result[0].result[0].information == "optimize pricing"


def test_pydantic_tools_parser_reasoning_partial() -> None:
    parser = PydanticToolsParser(tools=[NameOnly], first_tool_only=True)
    arguments = f'{REASONING}<tool_call>{{"name": "abc'

    result = parser.parse_result([_generation("NameOnly", arguments)], partial=True)

    assert isinstance(result, NameOnly)
    assert result.name == "abc"


@pytest.mark.parametrize("arguments", WRAPPED_ARGUMENTS)
def test_json_output_key_tools_parser_reasoning(arguments: str) -> None:
    parser = JsonOutputKeyToolsParser(key_name="AnalysisResult", first_tool_only=True)

    result = parser.parse_result([_generation("AnalysisResult", arguments)])

    assert result == json.loads(INNER_JSON)


def test_json_output_key_tools_parser_reasoning_partial() -> None:
    parser = JsonOutputKeyToolsParser(key_name="NameOnly", first_tool_only=True)
    arguments = f'{REASONING}<tool_call>{{"name": "abc'

    result = parser.parse_result([_generation("NameOnly", arguments)], partial=True)

    assert result == {"name": "abc"}


def test_with_structured_output_reasoning_model() -> None:
    model = _ReasoningToolModel(
        tool_name="AnalysisResult",
        arguments=f"{REASONING}\n<tool_call>\n{INNER_JSON}\n</tool_call>",
    )

    result = model.with_structured_output(AnalysisResult).invoke("Analyze this")

    assert isinstance(result, AnalysisResult)
    assert result.result[0].name == "promo"


@pytest.mark.parametrize("arguments", WRAPPED_ARGUMENTS)
def test_json_output_parser_reasoning(arguments: str) -> None:
    expected = json.loads(INNER_JSON)

    assert JsonOutputParser().parse(arguments) == expected


@pytest.mark.parametrize("arguments", WRAPPED_ARGUMENTS)
def test_pydantic_output_parser_reasoning(arguments: str) -> None:
    parser: PydanticOutputParser[AnalysisResult] = PydanticOutputParser(
        pydantic_object=AnalysisResult
    )
    result: AnalysisResult = parser.parse(arguments)

    assert isinstance(result, AnalysisResult)
    assert result.result[0].name == "promo"


def test_json_output_parser_partial_open_think_returns_none() -> None:
    parser = JsonOutputParser()

    assert (
        parser.parse_result([Generation(text="<think>still thinking")], partial=True)
        is None
    )


def test_json_output_parser_partial_closed_think_returns_result() -> None:
    parser = JsonOutputParser()

    result = parser.parse_result(
        [Generation(text=f"{REASONING}\n{INNER_JSON}")], partial=True
    )

    assert result == json.loads(INNER_JSON)


def test_json_output_parser_partial_grows_after_think_closes() -> None:
    parser = JsonOutputParser()
    open_text = '<think>thinking</think>{"a": 1'

    assert parser.parse_result([Generation(text=open_text)], partial=True) == {"a": 1}


def test_pydantic_output_parser_partial_open_think_returns_none() -> None:
    parser: PydanticOutputParser[AnalysisResult] = PydanticOutputParser(
        pydantic_object=AnalysisResult
    )

    result = parser.parse_result(
        [Generation(text="<think>still thinking")], partial=True
    )

    assert result is None


def test_pydantic_output_parser_partial_closed_think_returns_result() -> None:
    parser: PydanticOutputParser[AnalysisResult] = PydanticOutputParser(
        pydantic_object=AnalysisResult
    )

    result = parser.parse_result(
        [Generation(text=f"{REASONING}\n{INNER_JSON}")], partial=True
    )

    assert isinstance(result, AnalysisResult)
    assert result.result[0].name == "promo"


def test_parse_tool_call_truncated_json_preserves_original_arguments() -> None:
    arguments = f'{REASONING}\n{{"result": [{{"name": "promo"'

    with pytest.raises(OutputParserException) as exc_info:
        parse_tool_call({"function": {"name": "F", "arguments": arguments}})

    assert exc_info.value.llm_output == arguments


def test_parse_tool_call_closed_reasoning_without_payload_raises() -> None:
    arguments = REASONING

    with pytest.raises(OutputParserException) as exc_info:
        parse_tool_call({"function": {"name": "F", "arguments": arguments}})

    assert exc_info.value.llm_output == arguments


def test_json_output_parser_closed_think_without_payload_raises() -> None:
    text = REASONING

    with pytest.raises(OutputParserException) as exc_info:
        JsonOutputParser().parse(text)

    assert exc_info.value.llm_output == text


def test_json_output_parser_unrecoverable_json_preserves_text() -> None:
    text = '<think>reasoning</think>\n{"a": 1,,}'

    with pytest.raises(OutputParserException) as exc_info:
        JsonOutputParser().parse(text)

    assert exc_info.value.llm_output == text


def test_json_output_parser_preserves_tag_literals_in_json_strings() -> None:
    text = '{"note": "<think>keep</think>", "wrapped": "<tool_call>x</tool_call>"}'

    assert JsonOutputParser().parse(text) == {
        "note": "<think>keep</think>",
        "wrapped": "<tool_call>x</tool_call>",
    }


def test_json_output_parser_wrapped_payload_preserves_tag_literals() -> None:
    payload = {"note": "<think>keep</think>", "wrapped": "<tool_call></tool_call>"}

    result = JsonOutputParser().parse(f"<tool_call>{json.dumps(payload)}</tool_call>")

    assert result == payload


def test_json_output_parser_invalid_json_preserves_text() -> None:
    text = "<think>reasoning</think>\nnot valid json"

    with pytest.raises(OutputParserException) as exc_info:
        JsonOutputParser().parse(text)

    assert exc_info.value.llm_output == text


def test_pydantic_output_parser_invalid_json_preserves_text() -> None:
    text = "<think>reasoning</think>\nnot valid json"

    with pytest.raises(OutputParserException) as exc_info:
        PydanticOutputParser(pydantic_object=AnalysisResult).parse(text)

    assert exc_info.value.llm_output == text


def test_parse_tool_call_ignores_trailing_text_after_wrapped_payload() -> None:
    arguments = f'<tool_call>\n{INNER_JSON}\n</tool_call>\n{{"extra": true}}'

    parsed = parse_tool_call(
        {"function": {"name": "AnalysisResult", "arguments": arguments}},
        return_id=False,
    )

    assert parsed is not None
    assert parsed["args"] == json.loads(INNER_JSON)


def test_json_output_parser_ignores_trailing_text_after_wrapped_payload() -> None:
    text = f'<tool_call>\n{INNER_JSON}\n</tool_call>\n{{"extra": true}}'

    assert JsonOutputParser().parse(text) == json.loads(INNER_JSON)


def test_pydantic_tools_parser_partial_open_think_returns_none() -> None:
    parser = PydanticToolsParser(tools=[NameOnly], first_tool_only=True)

    result = parser.parse_result(
        [_generation("NameOnly", "<think>still thinking")], partial=True
    )

    assert result is None


def test_pydantic_tools_parser_partial_open_think_returns_empty_list() -> None:
    parser = PydanticToolsParser(tools=[NameOnly])

    result = parser.parse_result(
        [_generation("NameOnly", "<think>still thinking")], partial=True
    )

    assert result == []


def test_json_output_key_tools_parser_partial_open_think_returns_none() -> None:
    parser = JsonOutputKeyToolsParser(key_name="NameOnly", first_tool_only=True)

    result = parser.parse_result(
        [_generation("NameOnly", "<think>still thinking")], partial=True
    )

    assert result is None


def test_pydantic_tools_parser_preserves_llm_output_on_invalid_json() -> None:
    arguments = '<think>reasoning</think>\n{"broken": }'
    parser = PydanticToolsParser(tools=[NameOnly], first_tool_only=True)

    with pytest.raises(OutputParserException) as exc_info:
        parser.parse_result([_generation("NameOnly", arguments)])

    assert exc_info.value.llm_output == arguments


def test_json_output_key_tools_parser_preserves_llm_output_on_invalid_json() -> None:
    arguments = '<think>reasoning</think>\n{"broken": }'
    parser = JsonOutputKeyToolsParser(key_name="NameOnly", first_tool_only=True)

    with pytest.raises(OutputParserException) as exc_info:
        parser.parse_result([_generation("NameOnly", arguments)])

    assert exc_info.value.llm_output == arguments
