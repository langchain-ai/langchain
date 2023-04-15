from typing import Any, List, Optional, Sequence

from langchain.agents.agent import AgentOutputParser, LLMSingleActionAgent
from langchain.agents.chat_v2.prompt import (
    FORMAT_INSTRUCTIONS,
    PREFIX,
    SUFFIX,
    create_prompt,
)
from langchain.agents.output_parsers.chat_v2_agent_output_parser import ChatOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import BaseLanguageModel, BaseOutputParser
from langchain.tools import BaseTool


class ChatAgentV2(LLMSingleActionAgent):
    output_parser: BaseOutputParser

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[AgentOutputParser] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMSingleActionAgent:
        """Construct an agent from an LLM and tools."""
        _stop = stop or ["Observation:"]
        _output_parser = output_parser or ChatOutputParser(llm, tools, FORMAT_INSTRUCTIONS=FORMAT_INSTRUCTIONS)
        prompt = create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        return cls(
            llm_chain=llm_chain, output_parser=_output_parser, stop=_stop, **kwargs
        )

    @property
    def _agent_type(self) -> str:
        raise ValueError
