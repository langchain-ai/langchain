import xml.etree.ElementTree as ET
from typing import Any, List, Tuple, Union

from langchain.agents.agent import (
    AgentExecutor,
    AgentOutputParser,
    BaseSingleActionAgent,
)
from langchain.agents.xml.prompt import agent_instructions
from langchain.callbacks.base import Callbacks
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains.llm import LLMChain
from langchain.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, BaseRetriever, Document
from langchain.tools.base import BaseTool


class XMLAgentOutputParser(AgentOutputParser):
    """Output parser for XMLAgent."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "</tool>" in text:
            tool, tool_input = text.split("</tool>")
            _tool = tool.split("<tool>")[1]
            _tool_input = tool_input.split("<tool_input>")[1]
            return AgentAction(tool=_tool, tool_input=_tool_input, log=text)
        elif "<final_answer>" in text:
            _, answer = text.split("<final_answer>")
            return AgentFinish(return_values={"output": answer}, log=text)
        else:
            raise ValueError

    def get_format_instructions(self) -> str:
        raise NotImplementedError

    @property
    def _type(self) -> str:
        return "xml-retrieval-agent"


class XMLRetrievalAgent(BaseSingleActionAgent):
    """Agent that uses XML tags to do retrieval.

    Args:
        tools: list of tools the agent can choose from
        llm_chain: The LLMChain to call to predict the next action

    Examples:

        .. code-block:: python

            from langchain.agents import XMLAgent
            from langchain

            tools = ...
            model =


    """

    tools: List[BaseTool]
    """List of tools this agent has access to."""
    llm_chain: LLMChain
    """Chain to use to predict action."""

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @staticmethod
    def get_default_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            agent_instructions
        ) + AIMessagePromptTemplate.from_template("{intermediate_steps}")

    @staticmethod
    def get_default_output_parser() -> XMLAgentOutputParser:
        return XMLAgentOutputParser()

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        log = ""
        _id = 0
        doc_mapping = {}
        for action, observation in intermediate_steps:
            doc_string = ""
            for doc in observation:
                doc_mapping[_id] = doc
                doc_string += f"<id>{_id}</id><content>{doc.page_content}</content>"
                _id += 1
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{doc_string}</observation>"
            )
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": log,
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = self.llm_chain(inputs, callbacks=callbacks)
        result = response[self.llm_chain.output_key]
        if isinstance(result, AgentAction):
            return result
        else:
            root = ET.fromstring("<root>" + result.return_values["output"] + "</root>")
            ids = [elem.text for elem in root.findall("id")]
            docs = [doc_mapping[int(i)] for i in ids]
            return AgentFinish(return_values={"output": docs}, log=result.log)

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        log = ""
        _id = 0
        doc_mapping = {}
        for action, observation in intermediate_steps:
            doc_string = ""
            for doc in observation:
                doc_mapping[_id] = doc
                doc_string += f"<id>{_id}</id><content>{doc.page_content}</content>"
                _id += 1
            log += (
                f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
                f"</tool_input><observation>{doc_string}</observation>"
            )
        tools = ""
        for tool in self.tools:
            tools += f"{tool.name}: {tool.description}\n"
        inputs = {
            "intermediate_steps": log,
            "tools": tools,
            "question": kwargs["input"],
            "stop": ["</tool_input>", "</final_answer>"],
        }
        response = await self.llm_chain.acall(inputs, callbacks=callbacks)
        result = response[self.llm_chain.output_key]
        if isinstance(result, AgentAction):
            return result
        else:
            root = ET.fromstring("<root>" + result.return_values["output"] + "</root>")
            ids = [elem.text for elem in root.findall("id")]
            docs = [doc_mapping[int(i)] for i in ids]
            return AgentFinish(return_values={"output": docs}, log=result.log)


class AgentRetriever(BaseRetriever, AgentExecutor):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        result = self({"input": query}, callbacks=run_manager.get_child())
        return result["output"]
