"""This is a template for a custom chain.

Edit this file to implement your chain logic."""

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.schema.language_model import BaseLanguageModel

MY_PROMPT = """You are an assistant that helps to form nice and human understandable answers.

Information:
{my_template_var}

Question: {question}
Helpful Answer:"""


class MyChain(Chain):
    llm_chain: LLMChain

    @property
    def input_keys(self) -> List[str]:
        return self.llm_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return self.llm_chain.output_keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        prompt: str = MY_PROMPT,
    ) -> "MyChain":
        """Initialize from LLM."""
        llm_chain = LLMChain.from_string(llm, prompt)
        return cls(llm_chain=llm_chain)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Call the chain.

        Edit this method to implement your chain logic."""
        return self.llm_chain(
            inputs, callbacks=run_manager.get_child() if run_manager else None
        )
