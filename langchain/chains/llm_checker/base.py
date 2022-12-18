"""Chain for question-answering with self-verification."""


from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_checker.prompt import (
    CHECK_ASSERTIONS_PROMPT,
    CREATE_DRAFT_ANSWER_PROMPT,
    LIST_ASSERTIONS_PROMPT,
    REVISED_ANSWER_PROMPT,
)
from langchain.chains.sequential import SequentialChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


class LLMCheckerChain(Chain, BaseModel):
    """Chain for question-answering with self-verification.

    Example:
        .. code-block:: python
            from langchain import OpenAI, LLMCheckerChain
            llm = OpenAI(temperature=0.7)
            checker_chain = LLMCheckerChain(llm=llm)
    """

    llm: BaseLLM
    """LLM wrapper to use."""
    create_draft_answer_prompt: PromptTemplate = CREATE_DRAFT_ANSWER_PROMPT
    list_assertions_prompt: PromptTemplate = LIST_ASSERTIONS_PROMPT
    check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT
    revised_answer_prompt: PromptTemplate = REVISED_ANSWER_PROMPT
    """Prompt to use when questioning the documents."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        question = inputs[self.input_key]

        create_draft_answer_chain = LLMChain(
            llm=self.llm, prompt=self.create_draft_answer_prompt, output_key="statement"
        )
        list_assertions_chain = LLMChain(
            llm=self.llm, prompt=self.list_assertions_prompt, output_key="assertions"
        )
        check_assertions_chain = LLMChain(
            llm=self.llm,
            prompt=self.check_assertions_prompt,
            output_key="checked_assertions",
        )

        revised_answer_chain = LLMChain(
            llm=self.llm,
            prompt=self.revised_answer_prompt,
            output_key="revised_statement",
        )

        chains = [
            create_draft_answer_chain,
            list_assertions_chain,
            check_assertions_chain,
            revised_answer_chain,
        ]

        question_to_checked_assertions_chain = SequentialChain(
            chains=chains,
            input_variables=["question"],
            output_variables=["revised_statement"],
            verbose=True,
        )
        output = question_to_checked_assertions_chain({"question": question})
        return {self.output_key: output["revised_statement"]}
