"""Chain for checking the return of an LLM call that asks for an action in a format for validity and correctness."""

from typing import Dict, List

from pydantic import BaseModel, Extra

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.llm_return_format_checker.prompt import (
    CHECK_ACTION_FORMAT_PROMPT,
    CHECK_ACTION_VALIDITY_PROMPT,
    CHECK_FORMAT_VALIDITY_PROMPT,
    CREATE_DRAFT_ACTION_PROMPT,
)
from langchain.chains.sequential import SequentialChain
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate


class LLMReturnFormatCheckerChain(Chain, BaseModel):
    """
    Chain for checking the return of an LLM call that asks for an action in a format for validity and correctness.

    Example:
        .. code-block:: python
            from langchain import OpenAI, LLMReturnFormatCheckerChain
            llm = OpenAI(temperature=0.5)
            return_format_checker_chain = LLMReturnFormatCheckerChain(llm=llm)
    """

    llm: BaseLLM
    """LLM wrapper to use."""

    create_draft_action_prompt: PromptTemplate = CREATE_DRAFT_ACTION_PROMPT
    check_action_validity_prompt: PromptTemplate = CHECK_ACTION_VALIDITY_PROMPT
    check_action_format_prompt: PromptTemplate = CHECK_ACTION_FORMAT_PROMPT
    check_format_validity_prompt: PromptTemplate = CHECK_FORMAT_VALIDITY_PROMPT
    """Prompts to use when checking the return of an LLM call that asks for a correct action in a given format."""

    input_key: List[str] = [
        "situation",
        "valid_actions",
        "call_to_action",
        "action_format",
    ]  #: :meta private:
    output_key: str = "formatted_validated_action"  #: :meta private:

    class Config:
        """Pydantic config."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return self.input_key

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Call the chain.

        :meta private:
        """
        situation = inputs.get("situation")
        valid_actions = inputs.get("valid_actions")
        call_to_action = inputs.get("call_to_action")
        action_format = inputs.get("action_format")

        create_draft_action_chain = LLMChain(
            llm=self.llm,
            prompt=self.create_draft_action_prompt,
            output_key="initial_action",
        )
        check_action_validity_chain = LLMChain(
            llm=self.llm,
            prompt=self.check_action_validity_prompt,
            output_key="validated_action",
        )
        check_action_format_chain = LLMChain(
            llm=self.llm,
            prompt=self.check_action_format_prompt,
            output_key="initial_format_validated_action",
        )
        check_format_validity_chain = LLMChain(
            llm=self.llm,
            prompt=self.check_format_validity_prompt,
            output_key="formatted_validated_action",
        )
        chains = [
            create_draft_action_chain,
            check_action_validity_chain,
            check_action_format_chain,
            check_format_validity_chain,
        ]

        validate_and_format_chain = SequentialChain(
            chains=chains,
            input_variables=self.input_keys,
            output_variables=[self.output_key],
            verbose=True,
        )

        output = validate_and_format_chain(inputs)
        return {self.output_key: output.get("formatted_validated_action")}

    @property
    def _chain_type(self) -> str:
        return "llm_return_format_checker"
