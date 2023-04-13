"""Power BI agent."""
from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor
from langchain.agents.agent_toolkits.powerbi.prompt import (
    POWERBI_CHAT_PREFIX,
    POWERBI_CHAT_SUFFIX,
)
from langchain.agents.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.powerbi import PowerBIDataset


def create_pbi_chat_agent(
    llm: BaseChatModel,
    toolkit: PowerBIToolkit | None,
    powerbi: PowerBIDataset | None = None,
    callback_manager: BaseCallbackManager | None = None,
    prefix: str = POWERBI_CHAT_PREFIX,
    suffix: str = POWERBI_CHAT_SUFFIX,
    input_variables: list[str] | None = None,
    top_k: int = 10,
    verbose: bool = False,
    agent_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a pbi agent from an Chat LLM and tools."""
    if toolkit is None:
        if powerbi is None:
            raise ValueError("Must provide either a toolkit or powerbi dataset")
        toolkit = PowerBIToolkit(powerbi=powerbi, llm=llm)
    tools = toolkit.get_tools()

    prefix = prefix.format(top_k=top_k)
    agent_kwargs = agent_kwargs or {}
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=prefix,
        user_message=suffix,
        input_variables=input_variables,
        callback_manager=callback_manager,
        verbose=verbose,
        **agent_kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ),
        verbose=verbose,
        **kwargs,
    )
