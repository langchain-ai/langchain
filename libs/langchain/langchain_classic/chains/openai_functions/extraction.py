from typing import Any

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel

from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.openai_functions.utils import (
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
                "info": {"type": "array", "items": _convert_schema(entity_schema)},
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
        "<https://docs.langchain.com/oss/python/langchain/models#structured-outputs>."
    ),
    removal="1.0",
    alternative=(
        """
            from pydantic import BaseModel, Field
            from langchain_anthropic import ChatAnthropic

            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            # Or any other chat model that supports tools.
            # Please reference to the documentation of structured_output
            # to see an up to date list of which models support
            # with_structured_output.
            model = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0)
            structured_model = model.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats.
                Make sure to call the Joke function.")
            """
    ),
)
def create_extraction_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate | None = None,
    tags: list[str] | None = None,
    verbose: bool = False,  # noqa: FBT001,FBT002
) -> Chain:
    """Creates a chain that extracts information from a passage.

    Args:
        schema: The schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        tags: Optional list of tags to associate with the chain.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console.

    Returns:
        Chain that can be used to extract information from a passage.
    """
    function = _get_extraction_function(schema)
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = JsonKeyOutputFunctionsParser(key_name="info")
    llm_kwargs = get_llm_kwargs(function)
    return LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        tags=tags,
        verbose=verbose,
    )


@deprecated(
    since="0.1.14",
    message=(
        "LangChain has introduced a method called `with_structured_output` that"
        "is available on ChatModels capable of tool calling."
        "You can read more about the method here: "
        "<https://docs.langchain.com/oss/python/langchain/models#structured-outputs>. "
        "Please follow our extraction use case documentation for more guidelines"
        "on how to do information extraction with LLMs."
        "<https://python.langchain.com/docs/use_cases/extraction/>. "
        "If you notice other issues, please provide "
        "feedback here:"
        "<https://github.com/langchain-ai/langchain/discussions/18154>"
    ),
    removal="1.0",
    alternative=(
        """
            from pydantic import BaseModel, Field
            from langchain_anthropic import ChatAnthropic

            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            # Or any other chat model that supports tools.
            # Please reference to the documentation of structured_output
            # to see an up to date list of which models support
            # with_structured_output.
            model = ChatAnthropic(model="claude-opus-4-1-20250805", temperature=0)
            structured_model = model.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats.
                Make sure to call the Joke function.")
            """
    ),
)
def create_extraction_chain_pydantic(
    pydantic_schema: Any,
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate | None = None,
    verbose: bool = False,  # noqa: FBT001,FBT002
) -> Chain:
    """Creates a chain that extracts information from a passage using Pydantic schema.

    Args:
        pydantic_schema: The Pydantic schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console.

    Returns:
        Chain that can be used to extract information from a passage.
    """

    class PydanticSchema(BaseModel):
        info: list[pydantic_schema]

    if hasattr(pydantic_schema, "model_json_schema"):
        openai_schema = pydantic_schema.model_json_schema()
    else:
        openai_schema = pydantic_schema.schema()

    openai_schema = _resolve_schema_references(
        openai_schema,
        openai_schema.get("definitions", {}),
    )

    function = _get_extraction_function(openai_schema)
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = PydanticAttrOutputFunctionsParser(
        pydantic_schema=PydanticSchema,
        attr_name="info",
    )
    llm_kwargs = get_llm_kwargs(function)
    return LLMChain(
        llm=llm,
        prompt=extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        verbose=verbose,
    )
