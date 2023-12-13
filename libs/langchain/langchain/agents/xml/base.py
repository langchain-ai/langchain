from typing import Any, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate
from langchain_core.tools import BaseTool

from langchain.agents.agent import BaseSingleActionAgent
from langchain.agents.output_parsers.xml import XMLAgentOutputParser
from langchain.agents.xml.prompt import agent_instructions
from langchain.callbacks.base import Callbacks
from langchain.chains.llm import LLMChain


class XMLAgent(BaseSingleActionAgent):
    """Agent that uses XML tags.

    This agent only works with LLMs not chat models!

    Ability of agent to invoke tools varies a lot depending on how good the underlying
    LLM is!

    Args:
        tools: list of tools the agent can choose from
        llm_chain: The LLMChain to call to predict the next action

    Examples:

        .. code-block:: python

            from langchain.agents import AgentExecutor, XMLAgent
            from langchain.chains import LLMChain

            chain = LLMChain(
                llm=model,
                prompt=XMLAgent.get_default_prompt(),
                output_parser=XMLAgent.get_default_output_parser(),
            )

            agent = XMLAgent(tools=tools, llm_chain=chain)

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )

            agent_executor.invoke({"input": "what's the weather in New york?"})
    """

    tools: List[BaseTool]
    """List of tools this agent has access to."""
    llm_chain: LLMChain
    """Chain to use to predict action."""

    @property
    def input_keys(self) -> List[str]:
        """Get the input keys."""
        return ["input"]

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            agent_instructions
        ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")

    @staticmethod
    def get_default_output_parser() -> XMLAgentOutputParser:
        """Get the default output parser."""
        return XMLAgentOutputParser()

    def _format_intermediate_steps(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> str:
        """Format the steps."""
        log = ""
        for action, observation in intermediate_steps:
            if action.tool == "_Exception":
                # This only works correctly when handle_parsing_errors=True
                log += action.log  # Will contain the llm output from the exception
                log += "\n{observation}\n"
                pass
            else:
                log += (
                    f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                    f"</tool_input>\n<observation>{observation}</observation>\n"
                )
        return log

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": self._format_intermediate_steps(intermediate_steps),
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = self.llm_chain(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": self._format_intermediate_steps(intermediate_steps),
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = await self.llm_chain.acall(inputs, callbacks=callbacks)
        return response[self.llm_chain.output_key]
