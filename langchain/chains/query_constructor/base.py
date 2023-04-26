""""""
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from langchain import BasePromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.chains.query_constructor.prompt import (
    default_examples,
    default_prefix,
    default_schema,
    default_suffix,
    example_prompt,
)
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseLanguageModel, BaseOutputParser, OutputParserException


class Operator(str, Enum):
    AND = "and"
    OR = "or"
    NOT = "not"


class Comparator(str, Enum):
    EQ = "eq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class Comparison(BaseModel):
    comparator: Comparator
    attribute: str
    value: Any


class Operation(BaseModel):
    operator: Operator
    arguments: List[Union["Operation", Comparison]]


def parse_comparison(comparison: str) -> Comparison:
    comp, attr, val = re.match("(.*)\((.*), ?(.*)\)", comparison).groups()
    try:
        val = float(val)
    except ValueError:
        val = val.strip("\"'")
    return Comparison(comparator=comp, attribute=attr.strip("\"'"), value=val)


def parse_filter(_filter: str) -> Union[Operation, Comparison]:
    num_left = len(re.findall("\(", _filter))
    num_right = len(re.findall("\)", _filter))
    if num_left != num_right:
        raise ValueError(
            f"Invalid filter string. Expected equal number of left and "
            "right parentheses. Received filter {_filter}"
        )
    if num_left == 1:
        return parse_comparison(_filter)

    to_parse = _filter
    op_stack = []
    curr = None
    while to_parse:
        next_paren = re.search("[()]", to_parse)
        if next_paren is None:
            ValueError("filter string expected to contain parentheses.")
        next_paren_idx = next_paren.span()[0]
        if next_paren.group() == "(":
            fn = to_parse[:next_paren_idx]
            if fn in set(Operator):
                op_stack.append((fn, []))
                to_parse = to_parse[next_paren_idx + 1 :]
            elif fn in set(Comparator):
                closed_paren_idx = to_parse.find(")")
                comparison = parse_comparison(to_parse[: closed_paren_idx + 1])
                op_stack[-1][1].append(comparison)
                to_parse = to_parse[closed_paren_idx + 1 :].strip(", ")
            else:
                raise ValueError("invalid fn: " + fn)
        else:  # paren.group() == ")"
            curr_op, curr_args = op_stack.pop()
            curr = Operation(operator=curr_op, arguments=curr_args)
            if len(op_stack):
                op_stack[-1][1].append(curr)
            to_parse = to_parse[next_paren_idx + 1 :].strip(", ")
    if curr is None:
        raise ValueError
    return curr


class QueryConstructorOutputParser(BaseOutputParser[Dict]):
    def parse(self, text: str) -> Dict:
        try:
            expected_keys = ["query", "filter"]
            parsed = parse_json_markdown(text, expected_keys)
            if len(parsed["query"]) == 0:
                parsed["filter"] = " "
            if parsed["filter"] == "NO_FILTER":
                parsed["filter"] = None
            else:
                parsed["filter"] = parse_filter(parsed["filter"])
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )


class AttributeInfo(BaseModel):
    """Information about a data source attribute."""

    name: str
    description: str
    type: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


def format_attribute_info(info: List[AttributeInfo]) -> str:
    info_dicts = {}
    for i in info:
        i_dict = dict(i)
        info_dicts[i_dict.pop("name")] = i_dict
    return json.dumps(info_dicts, indent=2).replace("{", "{{").replace("}", "}}")


def _get_prompt(
    document_contents: str,
    attribute_info: List[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[List[Comparator]] = None,
    allowed_operators: Optional[List[Operator]] = None,
) -> BasePromptTemplate:
    attribute_str = format_attribute_info(attribute_info)
    examples = examples or default_examples
    allowed_comparators = allowed_comparators or list(Comparator)
    allowed_operators = allowed_operators or list(Operator)
    schema = default_schema.format(
        allowed_comparators=" | ".join(allowed_comparators),
        allowed_operators=" | ".join(allowed_operators),
    )
    prefix = default_prefix.format(schema=schema)
    suffix = default_suffix.format(
        i=len(examples) + 1, content=document_contents, attributes=attribute_str
    )
    return FewShotPromptTemplate(
        examples=default_examples,
        example_prompt=example_prompt,
        input_variables=["query"],
        suffix=suffix,
        prefix=prefix,
        output_parser=QueryConstructorOutputParser(),
    )


def load_query_constructor_chain(
    llm: BaseLanguageModel,
    document_contents: str,
    attribute_info: List[AttributeInfo],
    examples: Optional[List] = None,
    allowed_comparators: Optional[List[Comparator]] = None,
    allowed_operators: Optional[List[Operator]] = None,
) -> LLMChain:
    prompt = _get_prompt(
        document_contents,
        attribute_info,
        examples=examples,
        allowed_comparators=allowed_comparators,
        allowed_operators=allowed_operators,
    )
    return LLMChain(llm=llm, prompt=prompt)
