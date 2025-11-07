"""Chain for question-answering with self-verification."""

from __future__ import annotations

import warnings
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict, model_validator

from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.llm_checker.prompt import (
    CHECK_ASSERTIONS_PROMPT,
    CREATE_DRAFT_ANSWER_PROMPT,
    LIST_ASSERTIONS_PROMPT,
    REVISED_ANSWER_PROMPT,
)
from langchain_classic.chains.sequential import SequentialChain


def _load_question_to_checked_assertions_chain(
    llm: BaseLanguageModel,
    create_draft_answer_prompt: PromptTemplate,
    list_assertions_prompt: PromptTemplate,
    check_assertions_prompt: PromptTemplate,
    revised_answer_prompt: PromptTemplate,
) -> SequentialChain:
    create_draft_answer_chain = LLMChain(
        llm=llm,
        prompt=create_draft_answer_prompt,
        output_key="statement",
    )
    list_assertions_chain = LLMChain(
        llm=llm,
        prompt=list_assertions_prompt,
        output_key="assertions",
    )
    check_assertions_chain = LLMChain(
        llm=llm,
        prompt=check_assertions_prompt,
        output_key="checked_assertions",
    )
    revised_answer_chain = LLMChain(
        llm=llm,
        prompt=revised_answer_prompt,
        output_key="revised_statement",
    )
    chains = [
        create_draft_answer_chain,
        list_assertions_chain,
        check_assertions_chain,
        revised_answer_chain,
    ]
    return SequentialChain(
        chains=chains,
        input_variables=["question"],
        output_variables=["revised_statement"],
        verbose=True,
    )


@deprecated(
    since="0.2.13",
    message=(
        "See LangGraph guides for a variety of self-reflection and corrective "
        "strategies for question-answering and other tasks: "
        "https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/"
    ),
    removal="1.0",
)
class LLMCheckerChain(Chain):
    """Chain for question-answering with self-verification.

    Example:
        ```python
        from langchain_openai import OpenAI
        from langchain_classic.chains import LLMCheckerChain

        model = OpenAI(temperature=0.7)
        checker_chain = LLMCheckerChain.from_llm(model)
        ```
    """

    question_to_checked_assertions_chain: SequentialChain

    llm: BaseLanguageModel | None = None
    """[Deprecated] LLM wrapper to use."""
    create_draft_answer_prompt: PromptTemplate = CREATE_DRAFT_ANSWER_PROMPT
    """[Deprecated]"""
    list_assertions_prompt: PromptTemplate = LIST_ASSERTIONS_PROMPT
    """[Deprecated]"""
    check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT
    """[Deprecated]"""
    revised_answer_prompt: PromptTemplate = REVISED_ANSWER_PROMPT
    """[Deprecated] Prompt to use when questioning the documents."""
    input_key: str = "query"
    output_key: str = "result"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMCheckerChain with an llm is deprecated. "
                "Please instantiate with question_to_checked_assertions_chain "
                "or using the from_llm class method.",
                stacklevel=5,
            )
            if (
                "question_to_checked_assertions_chain" not in values
                and values["llm"] is not None
            ):
                question_to_checked_assertions_chain = (
                    _load_question_to_checked_assertions_chain(
                        values["llm"],
                        values.get(
                            "create_draft_answer_prompt",
                            CREATE_DRAFT_ANSWER_PROMPT,
                        ),
                        values.get("list_assertions_prompt", LIST_ASSERTIONS_PROMPT),
                        values.get("check_assertions_prompt", CHECK_ASSERTIONS_PROMPT),
                        values.get("revised_answer_prompt", REVISED_ANSWER_PROMPT),
                    )
                )
                values["question_to_checked_assertions_chain"] = (
                    question_to_checked_assertions_chain
                )
        return values

    @property
    def input_keys(self) -> list[str]:
        """Return the singular input key."""
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Return the singular output key."""
        return [self.output_key]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        output = self.question_to_checked_assertions_chain(
            {"question": question},
            callbacks=_run_manager.get_child(),
        )
        return {self.output_key: output["revised_statement"]}

    @property
    def _chain_type(self) -> str:
        return "llm_checker_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        create_draft_answer_prompt: PromptTemplate = CREATE_DRAFT_ANSWER_PROMPT,
        list_assertions_prompt: PromptTemplate = LIST_ASSERTIONS_PROMPT,
        check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT,
        revised_answer_prompt: PromptTemplate = REVISED_ANSWER_PROMPT,
        **kwargs: Any,
    ) -> LLMCheckerChain:
        """Create an LLMCheckerChain from a language model.

        Args:
            llm: a language model
            create_draft_answer_prompt: prompt to create a draft answer
            list_assertions_prompt: prompt to list assertions
            check_assertions_prompt: prompt to check assertions
            revised_answer_prompt: prompt to revise the answer
            **kwargs: additional arguments
        """
        question_to_checked_assertions_chain = (
            _load_question_to_checked_assertions_chain(
                llm,
                create_draft_answer_prompt,
                list_assertions_prompt,
                check_assertions_prompt,
                revised_answer_prompt,
            )
        )
        return cls(
            question_to_checked_assertions_chain=question_to_checked_assertions_chain,
            **kwargs,
        )
