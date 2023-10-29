import json

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.pydantic_v1 import BaseModel

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


class PlateChainInput(BaseModel):
    num_plates: int = None
    num_rows: int = None
    num_cols: int = None
    input: str


chain = (
    {
        # Should add validation to ensure numeric indices
        "input": lambda x: x["input"].to_csv(header=False),
        "hint": lambda x: create_prompt(
            num_plates=x.get("num_plates"),
            num_rows=x.get("num_rows"),
            num_cols=x.get("num_cols"),
        ),
        "col_range_str": lambda x: f"from 1 to {x.get('num_cols')}"
        if x.get("num_cols")
        else "",
        "json_format": lambda x: json.dumps(
            [
                {
                    "row_start": 12,
                    "row_end": 12 + x.get("num_rows", 8) - 1,
                    "col_start": 1,
                    "col_end": 1 + x.get("num_cols", 12) - 1,
                    "contents": "Entity ID",
                }
            ]
        ),
        "user_example": lambda x: USER_EXAMPLE_DICT[
            x.get("num_rows", 8) * x.get("num_cols", 12)
        ],
        "ai_response": lambda x: AI_REPONSE_DICT[
            x.get("num_rows", 8) * x.get("num_cols", 12)
        ],
    }
    | prompt
    | llm
    | StrOutputParser()
    | parse_llm_output
).with_types(input_type=PlateChainInput)
