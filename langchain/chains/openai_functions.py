import json
from functools import partial
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.sequential import SimpleSequentialChain
from langchain.chains.transform import TransformChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

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


def _get_function_arguments(inputs: dict) -> str:
    message = inputs["input"]
    try:
        func_call = message.additional_kwargs["function_call"]
    except ValueError as exc:
        raise ValueError(f"Could not parse function call: {exc}")

    return func_call["arguments"]


def _parse_tag(inputs: dict) -> dict:
    args = _get_function_arguments(inputs)
    return {"output": json.loads(args)}


def _parse_tag_pydantic(inputs: dict, pydantic_schema: Any) -> dict:
    args = _get_function_arguments(inputs)
    args = pydantic_schema.parse_raw(args)
    return {"output": args}


def _parse_entities(inputs: dict) -> dict:
    args = _get_function_arguments(inputs)
    return {"output": json.loads(args)["info"]}


def _parse_entities_pydantic(inputs: dict, pydantic_schema: Any) -> dict:
    args = _get_function_arguments(inputs)
    pydantic_args = pydantic_schema.parse_raw(args)
    return {"output": pydantic_args.info}


class OpenAIFunctionsChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    functions: List[Dict]
    kwargs: Dict = Field(default_factory=dict)

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _inputs = {k: v for k, v in inputs.items() if k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**_inputs)
        messages = prompt.to_messages()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        predicted_message = self.llm.predict_messages(
            messages, functions=self.functions, callbacks=callbacks, **self.kwargs
        )
        return {"output": predicted_message}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _inputs = {k: v for k, v in inputs.items() if k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**_inputs)
        messages = prompt.to_messages()
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        predicted_message = await self.llm.apredict_messages(
            messages, functions=self.functions, callbacks=callbacks, **self.kwargs
        )
        return {"output": predicted_message}


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
    chain = OpenAIFunctionsChain(
        llm=llm, prompt=prompt, functions=functions, kwargs=EXTRACTION_KWARGS
    )
    parsing_chain = TransformChain(
        transform=_parse_entities,
        input_variables=["input"],
        output_variables=["output"],
    )
    return SimpleSequentialChain(chains=[chain, parsing_chain])


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
    chain = OpenAIFunctionsChain(
        llm=llm, prompt=prompt, functions=functions, kwargs=EXTRACTION_KWARGS
    )
    pydantic_parsing_chain = TransformChain(
        transform=partial(_parse_entities_pydantic, pydantic_schema=PydanticSchema),
        input_variables=["input"],
        output_variables=["output"],
    )
    return SimpleSequentialChain(chains=[chain, pydantic_parsing_chain])


_TAGGING_TEMPLATE = """Extract the desired information from the following passage.

Passage:
{input}
"""


def create_tagging_chain(schema: dict, llm: BaseLanguageModel) -> Chain:
    functions = _get_tagging_functions(schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    chain = OpenAIFunctionsChain(
        llm=llm, prompt=prompt, functions=functions, kwargs=EXTRACTION_KWARGS
    )
    parsing_chain = TransformChain(
        transform=_parse_tag, input_variables=["input"], output_variables=["output"]
    )
    return SimpleSequentialChain(chains=[chain, parsing_chain])


def create_tagging_chain_pydantic(
    pydantic_schema: Any, llm: BaseLanguageModel
) -> Chain:
    openai_schema = pydantic_schema.schema()

    functions = _get_tagging_functions(openai_schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    chain = OpenAIFunctionsChain(
        llm=llm, prompt=prompt, functions=functions, kwargs=EXTRACTION_KWARGS
    )
    pydantic_parsing_chain = TransformChain(
        transform=partial(_parse_tag_pydantic, pydantic_schema=pydantic_schema),
        input_variables=["input"],
        output_variables=["output"],
    )

    return SimpleSequentialChain(chains=[chain, pydantic_parsing_chain])
