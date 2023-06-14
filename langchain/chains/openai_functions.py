from typing import Dict, Any, Optional, List

from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.base import Chain

from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate


class OpenAIFunctionsChain(Chain):

    prompt: BasePromptTemplate
    llm: BaseLanguageModel

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables + ["__tools__"]

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        _inputs = {k: v for k,v in inputs.items() if k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**_inputs)
        messages = prompt.to_messages()
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        predicted_message = self.llm.predict_messages(
            messages, functions=inputs["__tools__"], callbacks=callbacks
        )
        return {"output": predicted_message}

    async def _acall(self, inputs: Dict[str, Any], run_manager: Optional[AsyncCallbackManagerForChainRun] = None) -> \
    Dict[str, Any]:
        _inputs = {k: v for k, v in inputs.items() if k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**_inputs)
        messages = prompt.to_messages()
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        predicted_message = await self.llm.apredict_messages(
            messages, functions=inputs["__tools__"], callbacks=callbacks
        )
        return {"output": predicted_message}