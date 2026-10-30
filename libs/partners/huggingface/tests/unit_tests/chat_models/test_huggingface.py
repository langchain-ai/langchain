from typing import Any, Literal
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFacePipeline,
)


class Answer(BaseModel):
    answer: str


@pytest.fixture
def pipeline_chat() -> ChatHuggingFace:
    llm = HuggingFacePipeline(model_id="test/model", pipeline=Mock())
    with patch.object(ChatHuggingFace, "_resolve_model_id"):
        return ChatHuggingFace(llm=llm, tokenizer=Mock())


@pytest.mark.parametrize("method", ["json_schema", "json_mode"])
@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("schema", [Answer, Answer.model_json_schema()])
def test_pipeline_rejects_json_structured_output(
    pipeline_chat: ChatHuggingFace,
    method: Literal["json_schema", "json_mode"],
    include_raw: bool,  # noqa: FBT001
    schema: dict[str, Any] | type[BaseModel],
) -> None:
    with pytest.raises(ValueError) as exc_info:
        pipeline_chat.with_structured_output(
            schema, method=method, include_raw=include_raw
        )

    message = str(exc_info.value)
    assert "HuggingFacePipeline does not support response_format" in message
    assert f"method='{method}'" in message
    assert "HuggingFaceEndpoint" in message
    pipeline_chat.llm.pipeline.assert_not_called()


def test_pipeline_rejects_json_mode_without_schema(
    pipeline_chat: ChatHuggingFace,
) -> None:
    with pytest.raises(ValueError, match=r"HuggingFacePipeline.*response_format"):
        pipeline_chat.with_structured_output(method="json_mode")


@pytest.mark.parametrize("method", ["json_schema", "json_mode"])
def test_endpoint_supports_json_structured_output(
    method: Literal["json_schema", "json_mode"],
) -> None:
    client = Mock()
    client.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": '{"answer": "ok"}'},
                "finish_reason": "stop",
            }
        ]
    }
    llm = HuggingFaceEndpoint.model_construct(
        model="test/model", repo_id="test/model", client=client
    )
    with patch.object(ChatHuggingFace, "_resolve_model_id"):
        chat = ChatHuggingFace(llm=llm, model_id="test/model")
    schema = {
        "title": "Answer",
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = chat.with_structured_output(schema, method=method).invoke("Say ok")

    assert result == {"answer": "ok"}
    client.chat_completion.assert_called_once()
    expected_format: dict[str, Any] = {"type": "json_object"}
    if method == "json_schema":
        expected_format["schema"] = schema
    assert client.chat_completion.call_args.kwargs["response_format"] == expected_format
