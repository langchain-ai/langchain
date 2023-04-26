""""""
import json
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel

from langchain import BasePromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.chains.query_constructor.prompt import (
    default_examples,
    default_prefix,
    default_schema,
    default_suffix,
    example_prompt,
)
from langchain.chains.query_constructor.query_language import (
    Comparator,
    Operator,
    get_parser,
)
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseLanguageModel, BaseOutputParser, OutputParserException


class QueryConstructorOutputParser(BaseOutputParser[Dict]):
    ast_parse: Callable
    """"""

    def parse(self, text: str) -> Dict:
        try:
            expected_keys = ["query", "filter"]
            parsed = parse_json_markdown(text, expected_keys)
            if len(parsed["query"]) == 0:
                parsed["filter"] = " "
            if parsed["filter"] == "NO_FILTER":
                parsed["filter"] = None
            else:
                parsed["filter"] = self.ast_parse(parsed["filter"])
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )

    @classmethod
    def from_components(
        cls,
        allowed_comparators: Optional[List[Comparator]] = None,
        allowed_operators: Optional[List[Operator]] = None,
    ) -> "QueryConstructorOutputParser":
        ast_parser = get_parser(
            allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
        )
        return cls(ast_parse=ast_parser.parse)


class AttributeInfo(BaseModel):
    """Information about a data source attribute."""

    name: str
    description: str
    type: str

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


def _format_attribute_info(info: List[AttributeInfo]) -> str:
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
    attribute_str = _format_attribute_info(attribute_info)
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
    output_parser = QueryConstructorOutputParser.from_components(
        allowed_comparators=allowed_comparators, allowed_operators=allowed_operators
    )
    return FewShotPromptTemplate(
        examples=default_examples,
        example_prompt=example_prompt,
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
