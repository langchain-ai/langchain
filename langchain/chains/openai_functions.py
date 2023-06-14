from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate


class OpenAIFunctionsChain(Chain):

    prompt: BasePromptTemplate
    llm: BaseLanguageModel

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables + ["entity_schema"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]
    
    def _get_functions(self, entity_schema): 
        entity_name = entity_schema['entity_name']
        properties = entity_schema['properties']
        required = entity_schema['required']

        func = {}
        func['name'] = f"get_{entity_name}_info"
        func['description'] = f"Use this function to save the information of each {entity_name} that were mentioned in the passage"

        parameters = {"type": "object"}
        parameters["required"] = required
        parameters["properties"] = {}

        for k,v in properties.items():
            parameters["properties"][k] = {'title': k, 'type': v}

        func['parameters'] = parameters

        return func

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _inputs = {k: v for k,v in inputs.items() if k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**_inputs)
        messages = prompt.to_messages()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        function = self._get_functions(inputs['entity_schema'])
        predicted_message = self.llm.predict_messages(
            messages, functions=[function], callbacks=callbacks
        )
        return {"output": predicted_message}

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None) -> \
    Dict[str, Any]:
        _inputs = {k: v for k, v in inputs.items() if k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**_inputs)
        messages = prompt.to_messages()
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        function = self._get_functions(inputs['entity_schema'])
        predicted_message = await self.llm.apredict_messages(
            messages, functions=[function], callbacks=callbacks
        )
        return {"output": predicted_message}