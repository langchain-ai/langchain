from typing import Any, List

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import _convert_schema
from langchain.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate

EXTRACTION_NAME = "information_extraction"
EXTRACTION_KWARGS = {"function_call": {"name": "information_extraction"}}


def _get_tagging_functions(schema: dict) -> List[dict]:
    return [
        {
            "name": EXTRACTION_NAME,
            "description": "Extracts the relevant information from the passage.",
            "parameters": _convert_schema(schema),
        }
    ]


_TAGGING_TEMPLATE = """Extract the desired information from the following passage.

Passage:
{input}
"""


def create_tagging_chain(schema: dict, llm: BaseLanguageModel) -> Chain:
    functions = _get_tagging_functions(schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = JsonOutputFunctionsParser()
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={**{"functions": functions}, **EXTRACTION_KWARGS},
        output_parser=output_parser,
    )
    return chain


def create_tagging_chain_pydantic(
    pydantic_schema: Any, llm: BaseLanguageModel
) -> Chain:
    openai_schema = pydantic_schema.schema()

    functions = _get_tagging_functions(openai_schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={**{"functions": functions}, **EXTRACTION_KWARGS},
        output_parser=output_parser,
    )
    return chain
