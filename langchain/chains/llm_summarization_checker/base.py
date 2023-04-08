"""Chain for summarization with self-verification."""

from pathlib import Path
from typing import Dict, List

from pydantic import Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.llms.base import BaseLLM
from langchain.prompts.prompt import PromptTemplate

PROMPTS_DIR = Path(__file__).parent / "prompts"

CREATE_ASSERTIONS_PROMPT = PromptTemplate.from_file(
    PROMPTS_DIR / "create_facts.txt", ["summary"]
)
CHECK_ASSERTIONS_PROMPT = PromptTemplate.from_file(
    PROMPTS_DIR / "check_facts.txt", ["assertions"]
)
REVISED_SUMMARY_PROMPT = PromptTemplate.from_file(
    PROMPTS_DIR / "revise_summary.txt", ["checked_assertions", "summary"]
)
ARE_ALL_TRUE_PROMPT = PromptTemplate.from_file(
    PROMPTS_DIR / "are_all_true_prompt.txt", ["checked_assertions"]
)


class LLMSummarizationCheckerChain(Chain):
    """Chain for question-answering with self-verification.

    Example:
        .. code-block:: python

            from langchain import OpenAI, LLMSummarizationCheckerChain
            llm = OpenAI(temperature=0.0)
            checker_chain = LLMSummarizationCheckerChain(llm=llm)
    """

    llm: BaseLLM
    """LLM wrapper to use."""

    create_assertions_prompt: PromptTemplate = CREATE_ASSERTIONS_PROMPT
    check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT
    revised_summary_prompt: PromptTemplate = REVISED_SUMMARY_PROMPT
    are_all_true_prompt: PromptTemplate = ARE_ALL_TRUE_PROMPT

    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    max_checks: int = 2
    """Maximum number of times to check the assertions. Default to double-checking."""

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
        all_true = False
        count = 0
        output = None
        original_input = inputs[self.input_key]
        chain_input = original_input

        while not all_true and count < self.max_checks:
            chain = SequentialChain(
                chains=[
                    LLMChain(
                        llm=self.llm,
                        prompt=self.create_assertions_prompt,
                        output_key="assertions",
                        verbose=self.verbose,
                    ),
                    LLMChain(
                        llm=self.llm,
                        prompt=self.check_assertions_prompt,
                        output_key="checked_assertions",
                        verbose=self.verbose,
                    ),
                    LLMChain(
                        llm=self.llm,
                        prompt=self.revised_summary_prompt,
                        output_key="revised_summary",
                        verbose=self.verbose,
                    ),
                    LLMChain(
                        llm=self.llm,
                        output_key="all_true",
                        prompt=self.are_all_true_prompt,
                        verbose=self.verbose,
                    ),
                ],
                input_variables=["summary"],
                output_variables=["all_true", "revised_summary"],
                verbose=self.verbose,
            )
            output = chain({"summary": chain_input})
            count += 1

            if output["all_true"].strip() == "True":
                break

            if self.verbose:
                print(output["revised_summary"])

            chain_input = output["revised_summary"]

        if not output:
            raise ValueError("No output from chain")

        return {self.output_key: output["revised_summary"].strip()}

    @property
    def _chain_type(self) -> str:
        return "llm_summarization_checker_chain"
