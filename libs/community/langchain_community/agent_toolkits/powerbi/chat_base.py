"""Power BI agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models.chat_models import BaseChatModel

from langchain_community.agent_toolkits.powerbi.prompt import (
    POWERBI_CHAT_PREFIX,
    POWERBI_CHAT_SUFFIX,
)
from langchain_community.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain_community.utilities.powerbi import PowerBIDataset

if TYPE_CHECKING:
    from langchain.agents import AgentExecutor
    from langchain.agents.agent import AgentOutputParser
    from langchain.memory.chat_memory import BaseChatMemory


def create_pbi_chat_agent(
    llm: BaseChatModel,
    toolkit: Optional[PowerBIToolkit] = None,
    powerbi: Optional[PowerBIDataset] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    output_parser: Optional[AgentOutputParser] = None,
    prefix: str = POWERBI_CHAT_PREFIX,
    suffix: str = POWERBI_CHAT_SUFFIX,
    examples: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    memory: Optional[BaseChatMemory] = None,
    top_k: int = 10,
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a Power BI agent from a Chat LLM and tools.

    If you supply only a toolkit and no Power BI dataset, the same LLM is used for both.

    Args:
        llm: The language model to use.
        toolkit: Optional. The Power BI toolkit. Default is None.
        powerbi: Optional. The Power BI dataset. Default is None.
        callback_manager: Optional. The callback manager. Default is None.
        output_parser: Optional. The output parser. Default is None.
        prefix: Optional. The prefix for the prompt. Default is POWERBI_CHAT_PREFIX.
        suffix: Optional. The suffix for the prompt. Default is POWERBI_CHAT_SUFFIX.
        examples: Optional. The examples for the prompt. Default is None.
        input_variables: Optional. The input variables for the prompt. Default is None.
        memory: Optional. The memory. Default is None.
        top_k: Optional. The top k for the prompt. Default is 10.
        verbose: Optional. Whether to print verbose output. Default is False.
        agent_executor_kwargs: Optional. The agent executor kwargs. Default is None.
        kwargs: Any. Additional keyword arguments.

    Returns:
        The agent executor.
    """
    from langchain.agents import AgentExecutor
    from langchain.agents.conversational_chat.base import ConversationalChatAgent
    from langchain.memory import ConversationBufferMemory

    if toolkit is None:
        if powerbi is None:
            raise ValueError("Must provide either a toolkit or powerbi dataset")
        toolkit = PowerBIToolkit(powerbi=powerbi, llm=llm, examples=examples)
    tools = toolkit.get_tools()
    tables = powerbi.table_names if powerbi else toolkit.powerbi.table_names
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=prefix.format(top_k=top_k).format(tables=tables),
        human_message=suffix,
        input_variables=input_variables,
        callback_manager=callback_manager,
        output_parser=output_parser,
        verbose=verbose,
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        memory=memory
        or ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
