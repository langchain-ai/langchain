from typing import Any, Optional

from langchain_core._api import deprecated
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


@deprecated(
    since="0.2.13",
    message=(
        "LangChain has introduced a method called `with_structured_output` that "
        "is available on ChatModels capable of tool calling. "
        "See API reference for this function for replacement: "
        "<https://api.python.langchain.com/en/latest/chains/langchain.chains.openai_functions.tagging.create_tagging_chain.html> "  # noqa: E501
        "You can read more about `with_structured_output` here: "
        "<https://python.langchain.com/docs/how_to/structured_output/>. "
        "If you notice other issues, please provide "
        "feedback here: "
        "<https://github.com/langchain-ai/langchain/discussions/18154>"
    ),
    removal="1.0",
)
def create_tagging_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[ChatPromptTemplate] = None,
    **kwargs: Any,
) -> Chain:
    """Create a chain that extracts information from a passage
     based on a schema.

     This function is deprecated. Please use `with_structured_output` instead.
     See example usage below:

        .. code-block:: python

            from typing_extensions import Annotated, TypedDict
            from langchain_anthropic import ChatAnthropic

            class Joke(TypedDict):
                \"\"\"Tagged joke.\"\"\"

                setup: Annotated[str, ..., "The setup of the joke"]
                punchline: Annotated[str, ..., "The punchline of the joke"]

            # Or any other chat model that supports tools.
            # Please reference to to the documentation of structured_output
            # to see an up to date list of which models support
            # with_structured_output.
            model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
            structured_llm = model.with_structured_output(Joke)
            structured_llm.invoke(
                "Why did the cat cross the road? To get to the other "
                "side... and then lay down in the middle of it!"
            )
    Read more here: https://python.langchain.com/docs/how_to/structured_output/

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


@deprecated(
    since="0.2.13",
    message=(
        "LangChain has introduced a method called `with_structured_output` that "
        "is available on ChatModels capable of tool calling. "
        "See API reference for this function for replacement: "
        "<https://api.python.langchain.com/en/latest/chains/langchain.chains.openai_functions.tagging.create_tagging_chain_pydantic.html> "  # noqa: E501
        "You can read more about `with_structured_output` here: "
        "<https://python.langchain.com/docs/how_to/structured_output/>. "
        "If you notice other issues, please provide "
        "feedback here: "
        "<https://github.com/langchain-ai/langchain/discussions/18154>"
    ),
    removal="1.0",
)
def create_tagging_chain_pydantic(
    pydantic_schema: Any,
    llm: BaseLanguageModel,
    prompt: Optional[ChatPromptTemplate] = None,
    **kwargs: Any,
) -> Chain:
    """Create a chain that extracts information from a passage
     based on a pydantic schema.

     This function is deprecated. Please use `with_structured_output` instead.
     See example usage below:

     .. code-block:: python

            from pydantic import BaseModel, Field
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
            structured_llm.invoke(
                "Why did the cat cross the road? To get to the other "
                "side... and then lay down in the middle of it!"
            )
    Read more here: https://python.langchain.com/docs/how_to/structured_output/

    Args:
        pydantic_schema: The pydantic schema of the entities to extract.
        llm: The language model to use.

    Returns:
        Chain (LLMChain) that can be used to extract information from a passage.
    """
    if hasattr(pydantic_schema, "model_json_schema"):
        openai_schema = pydantic_schema.model_json_schema()
    else:
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
