"""
**Agent** is a class that uses an LLM to choose a sequence of actions to take.

In Chains, a sequence of actions is hardcoded. In Agents,
a language model is used as a reasoning engine to determine which actions
to take and in which order.

Agents select and use **Tools** and **Toolkits** for actions.

**Class hierarchy:**

.. code-block::

    BaseSingleActionAgent --> LLMSingleActionAgent
                              OpenAIFunctionsAgent
                              XMLAgent
                              Agent --> <name>Agent  # Examples: ZeroShotAgent, ChatAgent
                                        

    BaseMultiActionAgent  --> OpenAIMultiFunctionsAgent
    
    
**Main helpers:**

.. code-block::

    AgentType, AgentExecutor, AgentOutputParser, AgentExecutorIterator,
    AgentAction, AgentFinish
    
"""  # noqa: E501
from langchain_xfyun.agents.agent import (
    Agent,
    AgentExecutor,
    AgentOutputParser,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    LLMSingleActionAgent,
)
from langchain_xfyun.agents.agent_iterator import AgentExecutorIterator
from langchain_xfyun.agents.agent_toolkits import (
    create_csv_agent,
    create_json_agent,
    create_openapi_agent,
    create_pandas_dataframe_agent,
    create_pbi_agent,
    create_pbi_chat_agent,
    create_spark_dataframe_agent,
    create_spark_sql_agent,
    create_sql_agent,
    create_vectorstore_agent,
    create_vectorstore_router_agent,
    create_xorbits_agent,
)
from langchain_xfyun.agents.agent_types import AgentType
from langchain_xfyun.agents.conversational.base import ConversationalAgent
from langchain_xfyun.agents.conversational_chat.base import ConversationalChatAgent
from langchain_xfyun.agents.initialize import initialize_agent
from langchain_xfyun.agents.load_tools import (
    get_all_tool_names,
    load_huggingface_tool,
    load_tools,
)
from langchain_xfyun.agents.loading import load_agent
from langchain_xfyun.agents.mrkl.base import MRKLChain, ZeroShotAgent
from langchain_xfyun.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_xfyun.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent
from langchain_xfyun.agents.react.base import ReActChain, ReActTextWorldAgent
from langchain_xfyun.agents.self_ask_with_search.base import SelfAskWithSearchChain
from langchain_xfyun.agents.structured_chat.base import StructuredChatAgent
from langchain_xfyun.agents.tools import Tool, tool
from langchain_xfyun.agents.xml.base import XMLAgent

__all__ = [
    "Agent",
    "AgentExecutor",
    "AgentExecutorIterator",
    "AgentOutputParser",
    "AgentType",
    "BaseMultiActionAgent",
    "BaseSingleActionAgent",
    "ConversationalAgent",
    "ConversationalChatAgent",
    "LLMSingleActionAgent",
    "MRKLChain",
    "OpenAIFunctionsAgent",
    "OpenAIMultiFunctionsAgent",
    "ReActChain",
    "ReActTextWorldAgent",
    "SelfAskWithSearchChain",
    "StructuredChatAgent",
    "Tool",
    "ZeroShotAgent",
    "create_csv_agent",
    "create_json_agent",
    "create_openapi_agent",
    "create_pandas_dataframe_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_spark_dataframe_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
    "create_vectorstore_agent",
    "create_vectorstore_router_agent",
    "get_all_tool_names",
    "initialize_agent",
    "load_agent",
    "load_huggingface_tool",
    "load_tools",
    "tool",
    "create_xorbits_agent",
    "XMLAgent",
]
