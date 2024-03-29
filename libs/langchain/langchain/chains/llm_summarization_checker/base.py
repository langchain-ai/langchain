"""Chain for summarization with self-verification."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain

PROMPTS_DIR = Path(__file__).parent / "prompts"

CREATE_ASSERTIONS_PROMPT = PromptTemplate.from_file(PROMPTS_DIR / "create_facts.txt")
CHECK_ASSERTIONS_PROMPT = PromptTemplate.from_file(PROMPTS_DIR / "check_facts.txt")
REVISED_SUMMARY_PROMPT = PromptTemplate.from_file(PROMPTS_DIR / "revise_summary.txt")
ARE_ALL_TRUE_PROMPT = PromptTemplate.from_file(PROMPTS_DIR / "are_all_true_prompt.txt")


def _load_sequential_chain(
    llm: BaseLanguageModel,
    create_assertions_prompt: PromptTemplate,
    check_assertions_prompt: PromptTemplate,
    revised_summary_prompt: PromptTemplate,
    are_all_true_prompt: PromptTemplate,
    verbose: bool = False,
) -> SequentialChain:
    chain = SequentialChain(
        chains=[
            LLMChain(
                llm=llm,
                prompt=create_assertions_prompt,
                output_key="assertions",
                verbose=verbose,
            ),
            LLMChain(
                llm=llm,
                prompt=check_assertions_prompt,
                output_key="checked_assertions",
                verbose=verbose,
            ),
            LLMChain(
                llm=llm,
                prompt=revised_summary_prompt,
                output_key="revised_summary",
                verbose=verbose,
            ),
            LLMChain(
                llm=llm,
                output_key="all_true",
                prompt=are_all_true_prompt,
                verbose=verbose,
            ),
        ],
        input_variables=["summary"],
        output_variables=["all_true", "revised_summary"],
        verbose=verbose,
    )
    return chain


class LLMSummarizationCheckerChain(Chain):
    """Chain for question-answering with self-verification.

    Example:
        .. code-block:: python

            from langchain_community.llms import OpenAI
            from langchain.chains import LLMSummarizationCheckerChain
            llm = OpenAI(temperature=0.0)
            checker_chain = LLMSummarizationCheckerChain.from_llm(llm)
    """

    sequential_chain: SequentialChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""

    create_assertions_prompt: PromptTemplate = CREATE_ASSERTIONS_PROMPT
    """[Deprecated]"""
    check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT
    """[Deprecated]"""
    revised_summary_prompt: PromptTemplate = REVISED_SUMMARY_PROMPT
    """[Deprecated]"""
    are_all_true_prompt: PromptTemplate = ARE_ALL_TRUE_PROMPT
    """[Deprecated]"""

    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    max_checks: int = 2
    """Maximum number of times to check the assertions. Default to double-checking."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMSummarizationCheckerChain with an llm is "
                "deprecated. Please instantiate with"
                " sequential_chain argument or using the from_llm class method."
            )
            if "sequential_chain" not in values and values["llm"] is not None:
                values["sequential_chain"] = _load_sequential_chain(
                    values["llm"],
                    values.get("create_assertions_prompt", CREATE_ASSERTIONS_PROMPT),
                    values.get("check_assertions_prompt", CHECK_ASSERTIONS_PROMPT),
                    values.get("revised_summary_prompt", REVISED_SUMMARY_PROMPT),
                    values.get("are_all_true_prompt", ARE_ALL_TRUE_PROMPT),
                    verbose=values.get("verbose", False),
                )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        all_true = False
        count = 0
        output = None
        original_input = inputs[self.input_key]
        chain_input = original_input
        while not all_true and count < self.max_checks:
            output = self.sequential_chain(
                {"summary": chain_input}, callbacks=_run_manager.get_child()
            )
            count += 1

            if output["all_true"].strip() == "True":
                break

            if self.verbose:
                print(output["revised_summary"])  # noqa: T201

            chain_input = output["revised_summary"]

        if not output:
            raise ValueError("No output from chain")

        return {self.output_key: output["revised_summary"].strip()}

    @property
    def _chain_type(self) -> str:
        return "llm_summarization_checker_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        create_assertions_prompt: PromptTemplate = CREATE_ASSERTIONS_PROMPT,
        check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT,
        revised_summary_prompt: PromptTemplate = REVISED_SUMMARY_PROMPT,
        are_all_true_prompt: PromptTemplate = ARE_ALL_TRUE_PROMPT,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMSummarizationCheckerChain:
        chain = _load_sequential_chain(
            llm,
            create_assertions_prompt,
            check_assertions_prompt,
            revised_summary_prompt,
            are_all_true_prompt,
            verbose=verbose,
        )
        return cls(sequential_chain=chain, verbose=verbose, **kwargs)
