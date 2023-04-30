"""LLM Chain for turning a user text query into a structured query."""
from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Sequence

from langchain import BasePromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
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
)
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException


class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery]):
    ast_parse: Callable
    """Callable that parses dict into internal representation of query language."""

    def parse(self, text: str) -> StructuredQuery:
        try:
            expected_keys = ["query", "filter"]
            parsed = parse_json_markdown(text, expected_keys)
            if len(parsed["query"]) == 0:
                parsed["query"] = " "
            if parsed["filter"] == "NO_FILTER" or not parsed["filter"]:
                parsed["filter"] = None
            else:
                parsed["filter"] = self.ast_parse(parsed["filter"])
            return StructuredQuery(query=parsed["query"], filter=parsed["filter"])
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
        ast_parser = get_parser(
            allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
        )
        return cls(ast_parse=ast_parser.parse)


def _format_attribute_info(info: Sequence[AttributeInfo]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=2).replace("{", "{{").replace("}", "}}")


def _get_prompt(
    document_contents: str,
    attribute_info: Sequence[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[Sequence[Comparator]] = None,
    allowed_operators: Optional[Sequence[Operator]] = None,
) -> BasePromptTemplate:
    attribute_str = _format_attribute_info(attribute_info)
    examples = examples or DEFAULT_EXAMPLES
    allowed_comparators = allowed_comparators or list(Comparator)
    allowed_operators = allowed_operators or list(Operator)
    schema = DEFAULT_SCHEMA.format(
        allowed_comparators=" | ".join(allowed_comparators),
        allowed_operators=" | ".join(allowed_operators),
    )
    prefix = DEFAULT_PREFIX.format(schema=schema)
    suffix = DEFAULT_SUFFIX.format(
        i=len(examples) + 1, content=document_contents, attributes=attribute_str
    )
    output_parser = StructuredQueryOutputParser.from_components(
        allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
    )
    return FewShotPromptTemplate(
        examples=DEFAULT_EXAMPLES,
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
    **kwargs: Any,
) -> LLMChain:
    prompt = _get_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
    )
    return LLMChain(llm=llm, prompt=prompt, **kwargs)
