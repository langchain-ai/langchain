import base64
import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.pydantic_v1 import Field
from langchain.schema.output_parser import StrOutputParser
from langserve import CustomUserType

from .prompts import (
    AI_REPONSE_DICT,
    FULL_PROMPT,
    USER_EXAMPLE_DICT,
    create_prompt,
)
from .utils import parse_llm_output

llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(FULL_PROMPT),
        ("human", "{user_example}"),
        ("ai", "{ai_response}"),
        ("human", "{input}"),
    ],
)


# ATTENTION: Inherit from CustomUserType instead of BaseModel otherwise
# the server will decode it into a dict instead of a pydantic model.
class FileProcessingRequest(CustomUserType):
    """Request including a base64 encoded file."""

    # The extra field is used to specify a widget for the playground UI.
    file: str = Field(..., extra={"widget": {"type": "base64file"}})
    num_plates: int = None
    num_rows: int = 8
    num_cols: int = 12


def _load_file(request: FileProcessingRequest):
    return base64.b64decode(request.file.encode("utf-8")).decode("utf-8")


def _load_prompt(request: FileProcessingRequest):
    return create_prompt(
        num_plates=request.num_plates,
        num_rows=request.num_rows,
        num_cols=request.num_cols,
    )


def _get_col_range_str(request: FileProcessingRequest):
    if request.num_cols:
        return f"from 1 to {request.num_cols}"
    else:
        return ""


def _get_json_format(request: FileProcessingRequest):
    return json.dumps(
        [
            {
                "row_start": 12,
                "row_end": 12 + request.num_rows - 1,
                "col_start": 1,
                "col_end": 1 + request.num_cols - 1,
                "contents": "Entity ID",
            }
        ]
    )


chain = (
    {
        # Should add validation to ensure numeric indices
        "input": _load_file,
        "hint": _load_prompt,
        "col_range_str": _get_col_range_str,
        "json_format": _get_json_format,
        "user_example": lambda x: USER_EXAMPLE_DICT[x.num_rows * x.num_cols],
        "ai_response": lambda x: AI_REPONSE_DICT[x.num_rows * x.num_cols],
    }
    | prompt
    | llm
    | StrOutputParser()
    | parse_llm_output
).with_types(input_type=FileProcessingRequest)
