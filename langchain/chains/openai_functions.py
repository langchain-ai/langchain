import json
from typing import Any, Dict, List

from pydantic import BaseModel

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseLLMOutputParser, ChatGeneration, Generation

EXTRACTION_NAME = "information_extraction"
EXTRACTION_KWARGS = {"function_call": {"name": "information_extraction"}}


def _resolve_schema_references(schema: Any, definitions: Dict[str, Any]) -> Any:
    """
    Resolves the $ref keys in a JSON schema object using the provided definitions.
    """
    if isinstance(schema, list):
        for i, item in enumerate(schema):
            schema[i] = _resolve_schema_references(item, definitions)
    elif isinstance(schema, dict):
        if "$ref" in schema:
            ref_key = schema.pop("$ref").split("/")[-1]
            ref = definitions.get(ref_key, {})
            schema.update(ref)
        else:
            for key, value in schema.items():
                schema[key] = _resolve_schema_references(value, definitions)
    return schema


class OutputFunctionsParser(BaseLLMOutputParser[Any]):
    def parse_result(self, result: List[Generation]) -> Any:
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            raise ValueError(
                "This output parser can only be used with a chat generation."
            )
        message = generation.message
        try:
            func_call = message.additional_kwargs["function_call"]
        except ValueError as exc:
            raise ValueError(f"Could not parse function call: {exc}")

        return func_call["arguments"]


class JsonOutputFunctionsParser(OutputFunctionsParser):
    def parse_result(self, result: List[Generation]) -> Any:
        _args = super().parse_result(result)
        return json.loads(_args)


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    key_name: str

    def parse_result(self, result: List[Generation]) -> Any:
        res = super().parse_result(result)
        return res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    pydantic_schema: Any

    def parse_result(self, result: List[Generation]) -> Any:
        _args = super().parse_result(result)
        pydantic_args = self.pydantic_schema.parse_raw(_args)
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    attr_name: str

    def parse_result(self, result: List[Generation]) -> Any:
        result = super().parse_result(result)
        return getattr(result, self.attr_name)


def _convert_schema(schema: dict) -> dict:
    props = {k: {"title": k, **v} for k, v in schema["properties"].items()}
    return {
        "type": "object",
        "properties": props,
        "required": schema.get("required", []),
    }


def _get_extraction_functions(entity_schema: dict) -> List[dict]:
    return [
        {
            "name": EXTRACTION_NAME,
            "description": "Extracts the relevant information from the passage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info": {"type": "array", "items": _convert_schema(entity_schema)}
                },
                "required": ["info"],
            },
        }
    ]


def _get_tagging_functions(schema: dict) -> List[dict]:
    return [
        {
            "name": EXTRACTION_NAME,
            "description": "Extracts the relevant information from the passage.",
            "parameters": _convert_schema(schema),
        }
    ]


_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned\
 in the following passage together with their properties.

Passage:
{input}
"""


def create_extraction_chain(schema: dict, llm: BaseLanguageModel) -> Chain:
    functions = _get_extraction_functions(schema)
    prompt = ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = JsonKeyOutputFunctionsParser(key_name="info")
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={**{"functions": functions}, **EXTRACTION_KWARGS},
        output_parser=output_parser,
    )
    return chain


def create_extraction_chain_pydantic(
    pydantic_schema: Any, llm: BaseLanguageModel
) -> Chain:
    class PydanticSchema(BaseModel):
        info: List[pydantic_schema]  # type: ignore

    openai_schema = PydanticSchema.schema()
    openai_schema = _resolve_schema_references(
        openai_schema, openai_schema["definitions"]
    )

    functions = _get_extraction_functions(openai_schema)
    prompt = ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    output_parser = PydanticAttrOutputFunctionsParser(
        pydantic_schema=PydanticSchema, attr_name="info"
    )
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={**{"functions": functions}, **EXTRACTION_KWARGS},
        output_parser=output_parser,
    )
    return chain


_TAGGING_TEMPLATE = """Extract the desired information from the following passage.

Passage:
{input}
"""


def create_tagging_chain(schema: dict, llm: BaseLanguageModel) -> Chain:
    functions = _get_tagging_functions(schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = JsonOutputFunctionsParser()
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={**{"functions": functions}, **EXTRACTION_KWARGS},
        output_parser=output_parser,
    )
    return chain


def create_tagging_chain_pydantic(
    pydantic_schema: Any, llm: BaseLanguageModel
) -> Chain:
    openai_schema = pydantic_schema.schema()

    functions = _get_tagging_functions(openai_schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    output_parser = PydanticOutputFunctionsParser(pydantic_schema=pydantic_schema)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        llm_kwargs={**{"functions": functions}, **EXTRACTION_KWARGS},
        output_parser=output_parser,
    )
    return chain
