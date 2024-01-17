from typing import Any, List, Optional, Sequence, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.sql_constructor.base import (
    StructuredQueryOutputParser,
    construct_examples,
    fix_filter_directive,
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)
from langchain_core.sql_constructor.ir import (
    Comparator,
    Operator,
)
from langchain_core.sql_constructor.schema import AttributeInfo

from langchain.chains import LLMChain

__all__ = [
    "StructuredQueryOutputParser",
    "construct_examples",
    "fix_filter_directive",
    "get_query_constructor_prompt",
    "load_query_constructor_runnable",
]


def load_query_constructor_chain(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: Sequence[Union[AttributeInfo, dict]],
    examples: Optional[List] = None,
    allowed_comparators: Sequence[Comparator] = tuple(Comparator),
    allowed_operators: Sequence[Operator] = tuple(Operator),
    enable_limit: bool = False,
    schema_prompt: Optional[BasePromptTemplate] = None,
    **kwargs: Any,
) -> LLMChain:
    """Load a query constructor chain.

    Args:
        llm: BaseLanguageModel to use for the chain.
        document_contents: The contents of the document to be queried.
        attribute_info: Sequence of attributes in the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators. Defaults to all
            Comparators.
        allowed_operators: Sequence of allowed operators. Defaults to all Operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        **kwargs: Arbitrary named params to pass to LLMChain.

    Returns:
        A LLMChain that can be used to construct queries.
    """
    prompt = get_query_constructor_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        enable_limit=enable_limit,
        schema_prompt=schema_prompt,
    )
    allowed_attributes = []
    for ainfo in attribute_info:
        allowed_attributes.append(
            ainfo.name if isinstance(ainfo, AttributeInfo) else ainfo["name"]
        )
    output_parser = StructuredQueryOutputParser.from_components(
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        allowed_attributes=allowed_attributes,
    )
    # For backwards compatibility.
    prompt.output_parser = output_parser
    return LLMChain(llm=llm, prompt=prompt, output_parser=output_parser, **kwargs)
