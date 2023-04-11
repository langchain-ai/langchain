"""Power BI agent."""
from __future__ import annotations

from typing import Any

from langchain.agents import Agent, ConversationalChatAgent
from langchain.agents.agent_toolkits.powerbi.prompt import (
    POWERBI_CHAT_PREFIX,
    POWERBI_CHAT_SUFFIX,
)
from langchain.agents.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory


def create_pbi_chat_agent(
    llm: BaseChatModel,
    toolkit: PowerBIToolkit,
    callback_manager: BaseCallbackManager | None = None,
    prefix: str = POWERBI_CHAT_PREFIX,
    suffix: str = POWERBI_CHAT_SUFFIX,
    input_variables: list[str] | None = None,
    top_k: int = 10,
    verbose: bool = False,
    **kwargs: Any,
) -> Agent:
    """Construct a pbi agent from an Chat LLM and tools."""
    prefix = prefix.format(top_k=top_k)
    return ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=toolkit.get_tools(),
        system_message=prefix,
        user_message=suffix,
        input_variables=input_variables,
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ),
        callback_manager=callback_manager,
        verbose=verbose,
        **kwargs,
    )
