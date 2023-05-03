from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import root_validator

from langchain import BasePromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.router.base import RouterChain


class LLMRouterChain(RouterChain):
    llm_chain: LLMChain

    @root_validator()
    def validate_prompt(cls, values: dict) -> dict:
        prompt = values["llm_chain"].prompt
        if prompt.output_parser is None:
            raise ValueError(
                "LLMRouterChain requires base llm_chain prompt to have an output"
                " parser that converts LLM text output to a dictionary with keys"
                " 'destination' and 'next_inputs'. Received a prompt with no output"
                " parser."
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        return self.llm_chain.input_keys

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        super()._validate_outputs(outputs)
        if not isinstance(outputs["next_inputs"], dict):
            raise ValueError

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        output = self.llm_chain.predict_and_parse(callbacks=callbacks, **inputs)
        return output

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: BasePromptTemplate, **kwargs: Any
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
