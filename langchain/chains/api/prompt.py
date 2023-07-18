# flake8: noqa
from typing import Optional

from pydantic import BaseModel, Field, validator

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate


API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
{api_docs}

Using this documentation, generate the full API url to call for answering the user question.
You should build the API url in order to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

{format_instructions}

Question:{question}
Request:"""


class RequestParams(BaseModel):
    """Represents the format of an API request."""

    method: str = Field(
        description="One of: GET, POST, PUT, DELETE, OPTIONS, HEAD, or PATCH"
    )
    url: str = Field(
        description="URL that you want to access. DO NOT include any params here"
    )
    params: Optional[dict] = Field(
        description="Parameters that you want to include in your request, should be a dict."
    )

    @validator("method")
    def verify_request_method(cls, field: str) -> str:
        """Ensure method is a valid HTTP method."""
        if field not in ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"]:
            raise ValueError("Invalid request method.")
        return field

    @validator("params")
    def verify_params(cls, field: Optional[str]) -> Optional[str]:
        """Ensure params is a dict or empty."""
        if not field:
            return field
        if not isinstance(field, dict):
            raise ValueError("Params must be a dict.")
        return field


request_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=RequestParams
)

API_URL_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
    ],
    template=API_URL_PROMPT_TEMPLATE,
    partial_variables={"format_instructions": request_parser.get_format_instructions()},
)

API_RESPONSE_PROMPT_TEMPLATE = (
    API_URL_PROMPT_TEMPLATE
    + """
{request_params}
Here is the response from the API:
{api_response}

Summarize this response to answer the original question.

Summary:"""
)

API_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["api_docs", "question", "request_params", "api_response"],
    template=API_RESPONSE_PROMPT_TEMPLATE,
    partial_variables={"format_instructions": request_parser.get_format_instructions()},
)
