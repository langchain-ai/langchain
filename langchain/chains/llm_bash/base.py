"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from pydantic import Extra, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_bash.prompt import PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.utilities.bash import BashProcess


class LLMBashChain(Chain):
    """Chain that interprets a prompt and executes bash code to perform bash operations.

    Example:
        .. code-block:: python

            from langchain import LLMBashChain, OpenAI
            llm_bash = LLMBashChain.from_llm(OpenAI())
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated]"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMBashChain with an llm is deprecated. "
                "Please instantiate with llm_chain or using the from_llm class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        bash_executor = BashProcess()
        _run_manager.on_text(inputs[self.input_key], verbose=self.verbose)

        t = self.llm_chain.predict(
            question=inputs[self.input_key], callbacks=_run_manager.get_child()
        )
        _run_manager.on_text(t, color="green", verbose=self.verbose)

        t = t.strip()
        if t.startswith("```bash"):
            # Split the string into a list of substrings
            command_list = t.split("\n")
            print(command_list)

            # Remove the first and last substrings
            command_list = [s for s in command_list[1:-1]]
            output = bash_executor.run(command_list)

            _run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            _run_manager.on_text(output, color="yellow", verbose=self.verbose)

        else:
            raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: output}

    @property
    def _chain_type(self) -> str:
        return "llm_bash_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> LLMBashChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
