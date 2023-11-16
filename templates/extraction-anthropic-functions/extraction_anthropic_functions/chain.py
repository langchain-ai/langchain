from typing import List, Optional

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain_experimental.llms.anthropic_functions import AnthropicFunctions

template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])


# Function output schema
class Paper(BaseModel):
    """Information about papers mentioned."""

    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""

    papers: List[Paper]


# Function definition
model = AnthropicFunctions()
function = [convert_pydantic_to_openai_function(Info)]

chain = (
    prompt
    | model.bind(functions=function, function_call={"name": "Info"})
    | JsonKeyOutputFunctionsParser(key_name="papers")
)
