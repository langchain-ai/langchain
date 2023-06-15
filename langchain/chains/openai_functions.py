import json
from typing import Any, Dict, List, Optional

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


def _parse_tag(inputs: dict) -> dict:
    message = inputs["input"]
    args = message.additional_kwargs["function_call"]["arguments"]
    return {"output": json.loads(args)}


def _parse_entities(inputs: dict) -> dict:
    message = inputs["input"]
    args = message.additional_kwargs["function_call"]["arguments"]
    return {"output": json.loads(args)["info"]}


class OpenAIFunctionsChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    functions: List[Dict]

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
            messages, functions=self.functions, callbacks=callbacks
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
            messages, functions=self.functions, callbacks=callbacks
        )
        return {"output": predicted_message}


def _convert_schema(schema: dict) -> dict:
    props = {k: {"title": k, "type": v} for k, v in schema["properties"].items()}
    return {
        "type": "object",
        "properties": props,
        "required": schema.get("required", []),
    }


def _get_extraction_functions(entity_schema: dict) -> List[dict]:
    return [
        {
            "name": "information_extraction",
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
            "name": "information_extraction",
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
    chain = OpenAIFunctionsChain(llm=llm, prompt=prompt, functions=functions)
    parsing_chain = TransformChain(
        transform=_parse_entities,
        input_variables=["input"],
        output_variables=["output"],
    )
    return SimpleSequentialChain(chains=[chain, parsing_chain])


_TAGGING_TEMPLATE = """Extract the desired information from the following passage.

Passage:
{input}
"""


def create_tagging_chain(schema: dict, llm: BaseLanguageModel) -> Chain:
    functions = _get_tagging_functions(schema)
    prompt = ChatPromptTemplate.from_template(_TAGGING_TEMPLATE)
    chain = OpenAIFunctionsChain(llm=llm, prompt=prompt, functions=functions)
    parsing_chain = TransformChain(
        transform=_parse_tag, input_variables=["input"], output_variables=["output"]
    )
    return SimpleSequentialChain(chains=[chain, parsing_chain])
