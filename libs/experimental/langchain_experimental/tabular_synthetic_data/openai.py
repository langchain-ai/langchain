from typing import Any, Dict, Optional, Type, Union

from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BaseLLMOutputParser, BasePromptTemplate

from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")


def create_openai_data_generator(
    output_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: ChatOpenAI,
    prompt: BasePromptTemplate,
    output_parser: Optional[BaseLLMOutputParser] = None,
    **kwargs: Any
) -> SyntheticDataGenerator:
    """
    Create an instance of SyntheticDataGenerator tailored for OpenAI models.

    This function creates an LLM chain designed for structured output based on the
    provided schema, language model, and prompt template. The resulting chain is then
    used to instantiate and return a SyntheticDataGenerator.

    Args:
        output_schema (Union[Dict[str, Any], Type[BaseModel]]): Schema for expected
        output. This can be either a dictionary representing a valid JsonSchema or a
        Pydantic BaseModel class.


        llm (ChatOpenAI): OpenAI language model to use.

        prompt (BasePromptTemplate): Template to be used for generating prompts.


        output_parser (Optional[BaseLLMOutputParser], optional): Parser for
        processing model outputs. If none is provided, a default will be inferred
        from the function types.


        **kwargs: Additional keyword arguments to be passed to
        `create_structured_output_chain`.


    Returns: SyntheticDataGenerator: An instance of the data generator set up with
    the constructed chain.

    Usage:
        To generate synthetic data with a structured output, first define your desired
        output schema. Then, use this function to create a SyntheticDataGenerator
        instance. After obtaining the generator, you can utilize its methods to produce
        the desired synthetic data.
    """
    # Create function calling chain to ensure structured output
    chain = create_structured_output_chain(
        output_schema, llm, prompt, output_parser=output_parser, **kwargs
    )

    # Create the SyntheticDataGenerator instance with the created chain
    generator = SyntheticDataGenerator(template=prompt, llm_chain=chain)
    return generator
