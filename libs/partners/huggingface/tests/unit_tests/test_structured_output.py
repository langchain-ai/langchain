from typing import Any
from unittest.mock import Mock, patch
import pytest

from pydantic import BaseModel, Field

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)

class PersonInfo(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")

@pytest.fixture
def mock_llm() -> Mock:
    llm = Mock(spec=HuggingFaceEndpoint)
    llm.repo_id = "test/model"
    llm.model = "test/model"
    return llm

@pytest.fixture
@patch("langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id")
def chat_hugging_face(mock_resolve_id: Any, mock_llm: Any) -> ChatHuggingFace:
    return ChatHuggingFace(llm=mock_llm)

def test_structured_output_function_calling_pydantic(chat_hugging_face: ChatHuggingFace) -> None:
    """Test with_structured_output returns PydanticToolsParser when schema is a Pydantic model and method is function_calling."""
    chain = chat_hugging_face.with_structured_output(schema=PersonInfo, method="function_calling")
    
    output_parser = chain.last  # Getting the output parser from the pipeline
    assert isinstance(output_parser, PydanticToolsParser)

def test_structured_output_json_schema_pydantic(chat_hugging_face: ChatHuggingFace) -> None:
    """Test with_structured_output returns PydanticOutputParser when schema is a Pydantic model and method is json_schema."""
    chain = chat_hugging_face.with_structured_output(schema=PersonInfo, method="json_schema")
    
    output_parser = chain.last
    assert isinstance(output_parser, PydanticOutputParser)
    assert output_parser.pydantic_object == PersonInfo

def test_structured_output_json_mode_pydantic(chat_hugging_face: ChatHuggingFace) -> None:
    """Test with_structured_output returns PydanticOutputParser when schema is a Pydantic model and method is json_mode."""
    chain = chat_hugging_face.with_structured_output(schema=PersonInfo, method="json_mode")
    
    output_parser = chain.last
    assert isinstance(output_parser, PydanticOutputParser)
    assert output_parser.pydantic_object == PersonInfo

def test_structured_output_function_calling_dict(chat_hugging_face: ChatHuggingFace) -> None:
    """Test with_structured_output returns JsonOutputKeyToolsParser when schema is a dict."""
    schema_dict = convert_to_openai_tool(PersonInfo)
    
    chain = chat_hugging_face.with_structured_output(schema=schema_dict, method="function_calling")
    
    output_parser = chain.last
    assert isinstance(output_parser, JsonOutputKeyToolsParser)
