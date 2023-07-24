"""LLM Chain for turning a user text query into a structured query."""
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Sequence

from langchain import FewShotPromptTemplate, LLMChain
from langchain.chains.query_constructor.ir import (
    Comparator,
    Operator,
    StructuredQuery,
)
from langchain.chains.query_constructor.parser import get_parser
from langchain.chains.query_constructor.prompt import (
    DEFAULT_EXAMPLES,
    DEFAULT_PREFIX,
    DEFAULT_SCHEMA,
    DEFAULT_SUFFIX,
    EXAMPLE_PROMPT,
    EXAMPLES_WITH_LIMIT,
    SCHEMA_WITH_LIMIT,
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.output_parsers.json import parse_and_check_json_markdown
from langchain.schema import BaseOutputParser, BasePromptTemplate, OutputParserException
from langchain.schema.language_model import BaseLanguageModel


class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery]):
    """Output parser that parses a structured query."""

    ast_parse: Callable
    """Callable that parses dict into internal representation of query language."""

    def parse(self, text: str) -> StructuredQuery:
        try:
            expected_keys = ["query", "filter"]
            allowed_keys = ["query", "filter", "limit"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if len(parsed["query"]) == 0:
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
    ) -> StructuredQueryOutputParser:
        """
        Create a structured query output parser from components.

        Args:
            allowed_comparators: allowed comparators
            allowed_operators: allowed operators

        Returns:
            a structured query output parser
        """
        ast_parser = get_parser(
            allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
        )
        return cls(ast_parse=ast_parser.parse)


def _format_attribute_info(info: Sequence[AttributeInfo]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=4).replace("{", "{{").replace("}", "}}")


def _get_prompt(
    document_contents: str,
    attribute_info: Sequence[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
    enable_limit: bool = False,
) -> BasePromptTemplate:
    attribute_str = _format_attribute_info(attribute_info)
    allowed_comparators = allowed_comparators or list(Comparator)
    allowed_operators = allowed_operators or list(Operator)
    if enable_limit:
        schema = SCHEMA_WITH_LIMIT.format(
            allowed_comparators=" | ".join(allowed_comparators),
            allowed_operators=" | ".join(allowed_operators),
        )

        examples = examples or EXAMPLES_WITH_LIMIT
    else:
        schema = DEFAULT_SCHEMA.format(
            allowed_comparators=" | ".join(allowed_comparators),
            allowed_operators=" | ".join(allowed_operators),
        )

        examples = examples or DEFAULT_EXAMPLES
    prefix = DEFAULT_PREFIX.format(schema=schema)
    suffix = DEFAULT_SUFFIX.format(
        i=len(examples) + 1, content=document_contents, attributes=attribute_str
    )
    output_parser = StructuredQueryOutputParser.from_components(
        allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
    )
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=EXAMPLE_PROMPT,
        input_variables=["query"],
        suffix=suffix,
        prefix=prefix,
        output_parser=output_parser,
    )


def load_query_constructor_chain(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: List[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
    enable_limit: bool = False,
    **kwargs: Any,
) -> LLMChain:
    """Load a query constructor chain.

    Args:
        llm: BaseLanguageModel to use for the chain.
        document_contents: The contents of the document to be queried.
        attribute_info: A list of AttributeInfo objects describing
            the attributes of the document.
        examples: Optional list of examples to use for the chain.
        allowed_comparators: An optional list of allowed comparators.
        allowed_operators: An optional list of allowed operators.
        enable_limit: Whether to enable the limit operator. Defaults to False.
        **kwargs:

    Returns:
        A LLMChain that can be used to construct queries.
    """
    prompt = _get_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
        enable_limit=enable_limit,
    )
    return LLMChain(llm=llm, prompt=prompt, **kwargs)
