from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.reflection.prompts import (
    _ASSISTANT_AGENT_ACTUAL_RESPONSE_TEMPLATE,
    _ASSISTANT_AGENT_REFLECTION_TEMPLATE,
    _ASSISTANT_HYPOTHETICAL_RESPONSE_TEMPLATE,
)
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate


class SelfReflectionChain(Chain):
    """
    A self reflection chain using Chain-of-thought (COT) reasoning, inspired by:
    "Reflexion: an autonomous agent with dynamic memory and self-reflection"
    Authors: Noah Shinn, Beck Labash and Ashwin Gopinath
    Source: https://arxiv.org/abs/2303.11366
    """

    llm: BaseLanguageModel
    input_variables: List[str] = ["question"]
    output_variables: List[str] = ["text"]  #: :meta private:
    """Output key to use."""
    template: str = "{question}"
    """Base question template."""
    _ASSISTANT_HYPOTHETICAL_RESPONSE_TEMPLATE: str = (
        _ASSISTANT_HYPOTHETICAL_RESPONSE_TEMPLATE
    )
    """Template to use for agent hypothetical response."""
    _ASSISTANT_AGENT_REFLECTION_TEMPLATE: str = _ASSISTANT_AGENT_REFLECTION_TEMPLATE
    """Template to use for agent reflection."""
    _ASSISTANT_AGENT_ACTUAL_RESPONSE_TEMPLATE: str = (
        _ASSISTANT_AGENT_ACTUAL_RESPONSE_TEMPLATE
    )
    return_intermediate_steps: bool = False
    """Return the results of the refine steps in the output."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return expected input keys to the chain.

        :meta private:
        """
        return self.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.output_variables + ["intermediate_steps"]
        return self.output_variables

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Call the chain.
        """

        # Get run manager
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        # Set input variables
        self.input_variables = list(inputs.keys())

        # Replace <_USER_PROMPT_> with the user template
        _ASSISTANT_HYPOTHETICAL_RESPONSE_TEMPLATE = (
            self._ASSISTANT_HYPOTHETICAL_RESPONSE_TEMPLATE.replace(
                "<_USER_PROMPT_>", self.template
            )
        )
        _ASSISTANT_AGENT_REFLECTION_TEMPLATE = (
            self._ASSISTANT_AGENT_REFLECTION_TEMPLATE.replace(
                "<_USER_PROMPT_>", self.template
            )
        )
        _ASSISTANT_AGENT_ACTUAL_RESPONSE_TEMPLATE = (
            self._ASSISTANT_AGENT_ACTUAL_RESPONSE_TEMPLATE.replace(
                "<_USER_PROMPT_>", self.template
            )
        )

        # Create prompt templates
        _agent_hypothetical_response_prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=_ASSISTANT_HYPOTHETICAL_RESPONSE_TEMPLATE,
        )
        _agent_reflection_prompt = PromptTemplate(
            input_variables=self.input_variables + ["agent_hypothetical_response"],
            template=_ASSISTANT_AGENT_REFLECTION_TEMPLATE,
        )
        _agent_actual_response_prompt = PromptTemplate(
            input_variables=self.input_variables + ["agent_reflection_response"],
            template=_ASSISTANT_AGENT_ACTUAL_RESPONSE_TEMPLATE,
        )

        # Create LLM chains
        _agent_hypothetical = LLMChain(
            llm=self.llm,
            prompt=_agent_hypothetical_response_prompt,
            output_key="agent_hypothetical_response",
        )
        _agent_reflection = LLMChain(
            llm=self.llm,
            prompt=_agent_reflection_prompt,
            output_key="agent_reflection_response",
        )
        _agent_actual = LLMChain(
            llm=self.llm,
            prompt=_agent_actual_response_prompt,
            output_key="agent_actual_response",
        )

        # Create final chain
        _final_chain = SequentialChain(
            chains=[_agent_hypothetical, _agent_reflection, _agent_actual],
            input_variables=self.input_variables,
            output_variables=[
                "agent_hypothetical_response",
                "agent_reflection_response",
                "agent_actual_response",
            ],
        )

        # Run final chain
        response = _final_chain(
            inputs=inputs, callbacks=_run_manager.get_child(), **kwargs
        )
        output = {"text": response["agent_actual_response"]}

        if self.return_intermediate_steps:
            output["intermediate_steps"] = {
                "agent_hypothetical_response": response["agent_hypothetical_response"],
                "agent_reflection_response": response["agent_reflection_response"],
            }
        return output

    @property
    def _chain_type(self) -> str:
        return "self_reflection_chain"
