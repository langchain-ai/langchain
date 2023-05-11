from __future__ import annotations

from typing import Any, Coroutine, List, Mapping, Optional
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import LLM, BaseLanguageModel
from langchain.schema import LLMResult


class LLMFacade(LLM):
    chat_model: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return self.chat_model._llm_type

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        return self.chat_model.call_as_llm(prompt, stop=stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return self._chat._identifying_params

    @staticmethod
    def of(llm) -> LLMFacade:
        if isinstance(llm, BaseChatModel):
            return LLMFacade(chat_model=llm)
        elif isinstance(llm, BaseLanguageModel):
            return llm
        else:
            raise ValueError(
                f"Invalid llm type: {type(llm)}. Must be a chat model or language model."
            )

    def _agenerate(
        self,
        prompts: List[str],
        stop: List[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
    ) -> Coroutine[Any, Any, LLMResult]:
        return super()._agenerate(prompts, stop, run_manager)

    # `_type` req'd to pass `test_all_subclasses_implement_unique_type` test: FAILED tests/unit_tests/output_parsers/test_base_output_parser.py::test_all_subclasses_implement_unique_type - AssertionError: Duplicate types: {<property object at 0x7f300b31a7f0>}

    @property
    def _type(self) -> str:
        return "llm_facade"
