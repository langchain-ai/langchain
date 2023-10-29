import json
import base64

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.pydantic_v1 import BaseModel, Field
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


chain = (
    {
        # Should add validation to ensure numeric indices
        "input": lambda x: base64.b64decode(x.file.encode("utf-8")),
        "hint": lambda x: create_prompt(
            num_plates=x.num_plates,
            num_rows=x.num_rows,
            num_cols=x.num_cols,
        ),
        "col_range_str": lambda x: f"from 1 to {x.num_cols}"
        if x.num_cols
        else "",
        "json_format": lambda x: json.dumps(
            [
                {
                    "row_start": 12,
                    "row_end": 12 + x.num_rows - 1,
                    "col_start": 1,
                    "col_end": 1 + x.num_cols - 1,
                    "contents": "Entity ID",
                }
            ]
        ),
        "user_example": lambda x: USER_EXAMPLE_DICT[
            x.num_rows * x.num_cols
        ],
        "ai_response": lambda x: AI_REPONSE_DICT[
            x.num_rows * x.num_cols
        ],
    }
    | prompt
    | llm
    | StrOutputParser()
    | parse_llm_output
).with_types(input_type=FileProcessingRequest)
