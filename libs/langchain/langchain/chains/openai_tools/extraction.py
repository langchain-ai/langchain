from typing import List, Type, Union

from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from pydantic import BaseModel

_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned \
in the following passage together with their properties.

If a property is not present and is not required in the function parameters, do not include it in the output."""  # noqa: E501


@deprecated(
    since="0.1.14",
    message=(
        "LangChain has introduced a method called `with_structured_output` that"
        "is available on ChatModels capable of tool calling."
        "You can read more about the method here: "
        "<https://python.langchain.com/docs/modules/model_io/chat/structured_output/>. "
        "Please follow our extraction use case documentation for more guidelines"
        "on how to do information extraction with LLMs."
        "<https://python.langchain.com/docs/use_cases/extraction/>. "
        "with_structured_output does not currently support a list of pydantic schemas. "
        "If this is a blocker or if you notice other issues, please provide "
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
    pydantic_schemas: Union[List[Type[BaseModel]], Type[BaseModel]],
    llm: BaseLanguageModel,
    system_message: str = _EXTRACTION_TEMPLATE,
) -> Runnable:
    """Creates a chain that extracts information from a passage.

    Args:
        pydantic_schemas: The schema of the entities to extract.
        llm: The language model to use.
        system_message: The system message to use for extraction.

    Returns:
        A runnable that extracts information from a passage.
    """
    if not isinstance(pydantic_schemas, list):
        pydantic_schemas = [pydantic_schemas]
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "{input}")]
    )
    functions = [convert_pydantic_to_openai_function(p) for p in pydantic_schemas]
    tools = [{"type": "function", "function": d} for d in functions]
    model = llm.bind(tools=tools)
    chain = prompt | model | PydanticToolsParser(tools=pydantic_schemas)
    return chain
