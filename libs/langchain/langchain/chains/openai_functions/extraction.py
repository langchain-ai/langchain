from typing import Any, List, Optional

from langchain_core._api import deprecated
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


@deprecated(
    since="0.1.14",
    message=(
        "LangChain has introduced a method called `with_structured_output` that"
        "is available on ChatModels capable of tool calling."
        "You can read more about the method here: "
        "https://python.langchain.com/docs/modules/model_io/chat/structured_output/"
        "Please follow our extraction use case documentation for more guidelines"
        "on how to do information extraction with LLMs."
        "https://python.langchain.com/docs/use_cases/extraction/."
        "If you notice other issues, please provide "
        "feedback here:"
        "https://github.com/langchain-ai/langchain/discussions/18154"
    ),
    removal="0.3.0",
    alternative=(
        """
            from langchain_core.pydantic_v1 import BaseModel, Field
            from langchain_anthropic import ChatAnthropic
    
            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke") 
    
            # Or any other chat model that supports tools.
            # Please reference to to the documentation of structured_output
            # to see an up to date list of which models support 
            # with_structured_output.
            model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
            structured_llm = model.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats. 
                Make sure to call the Joke function.")
            """
    ),
)
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


@deprecated(
    since="0.1.14",
    message=(
        "LangChain has introduced a method called `with_structured_output` that"
        "is available on ChatModels capable of tool calling."
        "You can read more about the method here: "
        "https://python.langchain.com/docs/modules/model_io/chat/structured_output/"
        "Please follow our extraction use case documentation for more guidelines"
        "on how to do information extraction with LLMs."
        "https://python.langchain.com/docs/use_cases/extraction/."
        "If you notice other issues, please provide "
        "feedback here:"
        "https://github.com/langchain-ai/langchain/discussions/18154"
    ),
    removal="0.3.0",
    alternative=(
        """
            from langchain_core.pydantic_v1 import BaseModel, Field
            from langchain_anthropic import ChatAnthropic
    
            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke") 
    
            # Or any other chat model that supports tools.
            # Please reference to to the documentation of structured_output
            # to see an up to date list of which models support 
            # with_structured_output.
            model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
            structured_llm = model.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats. 
                Make sure to call the Joke function.")
            """
    ),
)
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
