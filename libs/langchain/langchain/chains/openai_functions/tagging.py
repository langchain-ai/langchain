from typing import Any, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import _convert_schema, get_llm_kwargs


def _get_tagging_function(schema: dict) -> dict:
    return {
        "name": "information_extraction",
        "description": "Extracts the relevant information from the passage.",
        "parameters": _convert_schema(schema),
    }


_TAGGING_TEMPLATE = """Extract the desired information from the following passage.

Only extract the properties mentioned in the 'information_extraction' function.

Passage:
{input}
"""


def create_tagging_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[ChatPromptTemplate] = None,
    **kwargs: Any,
) -> Chain:
    """Creates a chain that extracts information from a passage
     based on a schema.

    Args:
        schema: The schema of the entities to extract.
        llm: The language model to use.

    Returns:
        Chain (LLMChain) that can be used to extract information from a passage.
    """
    function = _get_tagging_function(schema)
    prompt = prompt or ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = JsonOutputFunctionsParser()
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        **kwargs,
    )
    return chain


def create_tagging_chain_pydantic(
    pydantic_schema: Any,
    llm: BaseLanguageModel,
    prompt: Optional[ChatPromptTemplate] = None,
    **kwargs: Any,
) -> Chain:
    """Creates a chain that extracts information from a passage
     based on a pydantic schema.

    Args:
        pydantic_schema: The pydantic schema of the entities to extract.
        llm: The language model to use.

    Returns:
        Chain (LLMChain) that can be used to extract information from a passage.
    """
    openai_schema = pydantic_schema.schema()
    function = _get_tagging_function(openai_schema)
    prompt = prompt or ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        **kwargs,
    )
    return chain
