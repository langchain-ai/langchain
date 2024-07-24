import json
from typing import Any
from unittest.mock import patch

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from langchain_experimental.llms.ollama_functions import OllamaFunctions


class Schema(BaseModel):
    pass


@patch.object(OllamaFunctions, "_create_stream")
def test_convert_image_prompt(
    _create_stream_mock: Any,
) -> None:
    response = {"message": {"content": '{"tool": "Schema", "tool_input": {}}'}}
    _create_stream_mock.return_value = [json.dumps(response)]

    prompt = ChatPromptTemplate.from_messages(
        [("human", [{"image_url": "data:image/jpeg;base64,{image_url}"}])]
    )

    lmm = prompt | OllamaFunctions().with_structured_output(schema=Schema)

    schema_instance = lmm.invoke(dict(image_url=""))

    assert schema_instance is not None
