"""LLM Chain for turning a user text query into a structured query."""
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.runnables import Runnable

from langchain.chains.llm import LLMChain
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    FilterDirective,
    Operation,
    Operator,
    StructuredQuery,
)
from langchain.chains.query_constructor.parser import get_parser
from langchain.chains.query_constructor.prompt import (
    DEFAULT_EXAMPLES,
    DEFAULT_PREFIX,
    DEFAULT_SCHEMA_PROMPT,
    DEFAULT_SUFFIX,
    EXAMPLE_PROMPT,
    EXAMPLES_WITH_LIMIT,
    PREFIX_WITH_DATA_SOURCE,
    SCHEMA_WITH_LIMIT_PROMPT,
    SUFFIX_WITHOUT_DATA_SOURCE,
    USER_SPECIFIED_EXAMPLE_PROMPT,
)
from langchain.chains.query_constructor.schema import AttributeInfo


class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery]):
    """Output parser that parses a structured query."""

    ast_parse: Callable
    """Callable that parses dict into internal representation of query language."""

    def parse(self, text: str) -> StructuredQuery:
        try:
            expected_keys = ["query", "filter"]
            allowed_keys = ["query", "filter", "limit"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if parsed["query"] is None or len(parsed["query"]) == 0:
                parsed["query"] = " "
            if parsed["filter"] == "NO_FILTER" or not parsed["filter"]:
                parsed["filter"] = None
            else:
                parsed["filter"] = self.ast_parse(parsed["filter"])
            if not parsed.get("limit"):
                parsed.pop("limit", None)
            return StructuredQuery(
                **{k: v for k, v in parsed.items() if k in allowed_keys}
            )
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )

    @classmethod
    def from_components(
        cls,
        allowed_comparators: Optional[Sequence[Comparator]] = None,
        allowed_operators: Optional[Sequence[Operator]] = None,
        allowed_attributes: Optional[Sequence[str]] = None,
        fix_invalid: bool = False,
    ) -> StructuredQueryOutputParser:
        """
        Create a structured query output parser from components.

        Args:
            allowed_comparators: allowed comparators
            allowed_operators: allowed operators

        Returns:
            a structured query output parser
        """
        ast_parse: Callable
        if fix_invalid:

            def ast_parse(raw_filter: str) -> Optional[FilterDirective]:
                filter = cast(Optional[FilterDirective], get_parser().parse(raw_filter))
                fixed = fix_filter_directive(
                    filter,
                    allowed_comparators=allowed_comparators,
                    allowed_operators=allowed_operators,
                    allowed_attributes=allowed_attributes,
                )
                return fixed

        else:
            ast_parse = get_parser(
                allowed_comparators=allowed_comparators,
                allowed_operators=allowed_operators,
                allowed_attributes=allowed_attributes,
            ).parse
        return cls(ast_parse=ast_parse)


def fix_filter_directive(
    filter: Optional[FilterDirective],
    *,
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
    allowed_attributes: Optional[Sequence[str]] = None,
) -> Optional[FilterDirective]:
    """Fix invalid filter directive.

    Args:
        filter: Filter directive to fix.
        allowed_comparators: allowed comparators. Defaults to all comparators.
        allowed_operators: allowed operators. Defaults to all operators.
        allowed_attributes: allowed attributes. Defaults to all attributes.

    Returns:
        Fixed filter directive.
    """
    if (
        not (allowed_comparators or allowed_operators or allowed_attributes)
    ) or not filter:
        return filter

    elif isinstance(filter, Comparison):
        if allowed_comparators and filter.comparator not in allowed_comparators:
            return None
        if allowed_attributes and filter.attribute not in allowed_attributes:
            return None
        return filter
    elif isinstance(filter, Operation):
        if allowed_operators and filter.operator not in allowed_operators:
            return None
        args = [
            fix_filter_directive(
                arg,
                allowed_comparators=allowed_comparators,
                allowed_operators=allowed_operators,
                allowed_attributes=allowed_attributes,
            )
            for arg in filter.arguments
        ]
        args = [arg for arg in args if arg is not None]
        if not args:
            return None
        elif len(args) == 1 and filter.operator in (Operator.AND, Operator.OR):
            return args[0]
        else:
            return Operation(
                operator=filter.operator,
                arguments=args,
            )
    else:
        return filter


def _format_attribute_info(info: Sequence[Union[AttributeInfo, dict]]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=4).replace("{", "{{").replace("}", "}}")


def construct_examples(input_output_pairs: Sequence[Tuple[str, dict]]) -> List[dict]:
    """Construct examples from input-output pairs.

    Args:
        input_output_pairs: Sequence of input-output pairs.

    Returns:
        List of examples.
    """
    examples = []
    for i, (_input, output) in enumerate(input_output_pairs):
        structured_request = (
            json.dumps(output, indent=4).replace("{", "{{").replace("}", "}}")
        )
        example = {
            "i": i + 1,
            "user_query": _input,
            "structured_request": structured_request,
        }
        examples.append(example)
    return examples


def get_query_constructor_prompt(
    document_contents: str,
    attribute_info: Sequence[Union[AttributeInfo, dict]],
    *,
    examples: Optional[Sequence] = None,
    allowed_comparators: Sequence[Comparator] = tuple(Comparator),
    allowed_operators: Sequence[Operator] = tuple(Operator),
    enable_limit: bool = False,
    schema_prompt: Optional[BasePromptTemplate] = None,
    **kwargs: Any,
) -> BasePromptTemplate:
    """Create query construction prompt.

    Args:
        document_contents: The contents of the document to be queried.
        attribute_info: A list of AttributeInfo objects describing
            the attributes of the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators.
        allowed_operators: Sequence of allowed operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        **kwargs: Additional named params to pass to FewShotPromptTemplate init.

    Returns:
        A prompt template that can be used to construct queries.
    """
    default_schema_prompt = (
        SCHEMA_WITH_LIMIT_PROMPT if enable_limit else DEFAULT_SCHEMA_PROMPT
    )
    schema_prompt = schema_prompt or default_schema_prompt
    attribute_str = _format_attribute_info(attribute_info)
    schema = schema_prompt.format(
        allowed_comparators=" | ".join(allowed_comparators),
        allowed_operators=" | ".join(allowed_operators),
    )
    if examples and isinstance(examples[0], tuple):
        examples = construct_examples(examples)
        example_prompt = USER_SPECIFIED_EXAMPLE_PROMPT
        prefix = PREFIX_WITH_DATA_SOURCE.format(
            schema=schema, content=document_contents, attributes=attribute_str
        )
        suffix = SUFFIX_WITHOUT_DATA_SOURCE.format(i=len(examples) + 1)
    else:
        examples = examples or (
            EXAMPLES_WITH_LIMIT if enable_limit else DEFAULT_EXAMPLES
        )
        example_prompt = EXAMPLE_PROMPT
        prefix = DEFAULT_PREFIX.format(schema=schema)
        suffix = DEFAULT_SUFFIX.format(
            i=len(examples) + 1, content=document_contents, attributes=attribute_str
        )
    return FewShotPromptTemplate(
        examples=list(examples),
        example_prompt=example_prompt,
        input_variables=["query"],
        suffix=suffix,
        prefix=prefix,
        **kwargs,
    )


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


def load_query_constructor_runnable(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: Sequence[Union[AttributeInfo, dict]],
    *,
    examples: Optional[Sequence] = None,
    allowed_comparators: Sequence[Comparator] = tuple(Comparator),
    allowed_operators: Sequence[Operator] = tuple(Operator),
    enable_limit: bool = False,
    schema_prompt: Optional[BasePromptTemplate] = None,
    fix_invalid: bool = False,
    **kwargs: Any,
) -> Runnable:
    """Load a query constructor runnable chain.

    Args:
        llm: BaseLanguageModel to use for the chain.
        document_contents: Description of the page contents of the document to be
            queried.
        attribute_info: Sequence of attributes in the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: Sequence of allowed comparators. Defaults to all
            Comparators.
        allowed_operators: Sequence of allowed operators. Defaults to all Operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        schema_prompt: Prompt for describing query schema. Should have string input
            variables allowed_comparators and allowed_operators.
        fix_invalid: Whether to fix invalid filter directives by ignoring invalid
            operators, comparators and attributes.
        **kwargs: Additional named params to pass to FewShotPromptTemplate init.

    Returns:
        A Runnable that can be used to construct queries.
    """
    prompt = get_query_constructor_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        enable_limit=enable_limit,
        schema_prompt=schema_prompt,
        **kwargs,
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
        fix_invalid=fix_invalid,
    )
    return prompt | llm | output_parser
