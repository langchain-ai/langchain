"""Chain that interprets a prompt and executes bash operations."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.schema import BasePromptTemplate, OutputParserException
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from pydantic import ConfigDict, Field, model_validator

from langchain_experimental.llm_bash.bash import BashProcess
from langchain_experimental.llm_bash.prompt import PROMPT

logger = logging.getLogger(__name__)


class LLMBashChain(Chain):
    """Chain that interprets a prompt and executes bash operations.

    Example:
        .. code-block:: python

            from langchain.chains import LLMBashChain
            from langchain_community.llms import OpenAI
            llm_bash = LLMBashChain.from_llm(OpenAI())
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated]"""
    bash_process: BashProcess = Field(default_factory=BashProcess)  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def raise_deprecation(cls, values: Dict) -> Any:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMBashChain with an llm is deprecated. "
                "Please instantiate with llm_chain or using the from_llm class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, values: Dict) -> Any:
        if values["llm_chain"].prompt.output_parser is None:
            raise ValueError(
                "The prompt used by llm_chain is expected to have an output_parser."
            )
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
        _run_manager.on_text(inputs[self.input_key], verbose=self.verbose)

        t = self.llm_chain.predict(
            question=inputs[self.input_key], callbacks=_run_manager.get_child()
        )
        _run_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()
        try:
            parser = self.llm_chain.prompt.output_parser
            command_list = parser.parse(t)  # type: ignore[union-attr]
        except OutputParserException as e:
            _run_manager.on_chain_error(e, verbose=self.verbose)
            raise e

        if self.verbose:
            _run_manager.on_text("\nCode: ", verbose=self.verbose)
            _run_manager.on_text(
                str(command_list), color="yellow", verbose=self.verbose
            )
        output = self.bash_process.run(command_list)
        _run_manager.on_text("\nAnswer: ", verbose=self.verbose)
        _run_manager.on_text(output, color="yellow", verbose=self.verbose)
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
