from typing import Any, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)


def _get_extraction_function(entity_schema: dict) -> dict:
    return {
        "name": "information_extraction",
        "description": "Extracts the relevant information from the passage.",
        "parameters": {
            "type": "object",
            "properties": {
                "info": {"type": "array", "items": _convert_schema(entity_schema)}
            },
            "required": ["info"],
        },
    }


_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned \
in the following passage together with their properties.

Only extract the properties mentioned in the 'information_extraction' function.

If a property is not present and is not required in the function parameters, do not include it in the output.

Passage:
{input}
"""  # noqa: E501


def create_extraction_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    tags: Optional[List[str]] = None,
    verbose: bool = False,
) -> Chain:
    """Creates a chain that extracts information from a passage.

    Args:
        schema: The schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console. Defaults to the global `verbose` value,
            accessible via `langchain.globals.get_verbose()`.

    Returns:
        Chain that can be used to extract information from a passage.
    """
    function = _get_extraction_function(schema)
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = JsonKeyOutputFunctionsParser(key_name="info")
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        tags=tags,
        verbose=verbose,
    )
    return chain


def create_extraction_chain_pydantic(
    pydantic_schema: Any,
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    verbose: bool = False,
) -> Chain:
    """Creates a chain that extracts information from a passage using pydantic schema.

    Args:
        pydantic_schema: The pydantic schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console. Defaults to the global `verbose` value,
            accessible via `langchain.globals.get_verbose()`

    Returns:
        Chain that can be used to extract information from a passage.
    """

    class PydanticSchema(BaseModel):
        info: List[pydantic_schema]  # type: ignore

    openai_schema = pydantic_schema.schema()
    openai_schema = _resolve_schema_references(
        openai_schema, openai_schema.get("definitions", {})
    )

    function = _get_extraction_function(openai_schema)
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = PydanticAttrOutputFunctionsParser(
        pydantic_schema=PydanticSchema, attr_name="info"
    )
    llm_kwargs = get_llm_kwargs(function)
    chain = LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        verbose=verbose,
    )
    return chain
