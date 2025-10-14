"""**Agent** is a class that uses an LLM to choose a sequence of actions to take.

In Chains, a sequence of actions is hardcoded. In Agents,
a language model is used as a reasoning engine to determine which actions
to take and in which order.

Agents select and use **Tools** and **Toolkits** for actions.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core._api.path import as_import_path
from langchain_core.tools import Tool
from langchain_core.tools.convert import tool

from langchain_classic._api import create_importer
from langchain_classic.agents.agent import (
    Agent,
    AgentExecutor,
    AgentOutputParser,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    LLMSingleActionAgent,
)
from langchain_classic.agents.agent_iterator import AgentExecutorIterator
from langchain_classic.agents.agent_toolkits.vectorstore.base import (
    create_vectorstore_agent,
    create_vectorstore_router_agent,
)
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.conversational.base import ConversationalAgent
from langchain_classic.agents.conversational_chat.base import ConversationalChatAgent
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.agents.json_chat.base import create_json_chat_agent
from langchain_classic.agents.loading import load_agent
from langchain_classic.agents.mrkl.base import MRKLChain, ZeroShotAgent
from langchain_classic.agents.openai_functions_agent.base import (
    OpenAIFunctionsAgent,
    create_openai_functions_agent,
)
from langchain_classic.agents.openai_functions_multi_agent.base import (
    OpenAIMultiFunctionsAgent,
)
from langchain_classic.agents.openai_tools.base import create_openai_tools_agent
from langchain_classic.agents.react.agent import create_react_agent
from langchain_classic.agents.react.base import ReActChain, ReActTextWorldAgent
from langchain_classic.agents.self_ask_with_search.base import (
    SelfAskWithSearchChain,
    create_self_ask_with_search_agent,
)
from langchain_classic.agents.structured_chat.base import (
    StructuredChatAgent,
    create_structured_chat_agent,
)
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_classic.agents.xml.base import XMLAgent, create_xml_agent

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.json.base import create_json_agent
    from langchain_community.agent_toolkits.load_tools import (
        get_all_tool_names,
        load_huggingface_tool,
        load_tools,
    )
    from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
    from langchain_community.agent_toolkits.powerbi.base import create_pbi_agent
    from langchain_community.agent_toolkits.powerbi.chat_base import (
        create_pbi_chat_agent,
    )
    from langchain_community.agent_toolkits.spark_sql.base import create_spark_sql_agent
    from langchain_community.agent_toolkits.sql.base import create_sql_agent

DEPRECATED_CODE = [
    "create_csv_agent",
    "create_pandas_dataframe_agent",
    "create_spark_dataframe_agent",
    "create_xorbits_agent",
]

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "create_json_agent": "langchain_community.agent_toolkits.json.base",
    "create_openapi_agent": "langchain_community.agent_toolkits.openapi.base",
    "create_pbi_agent": "langchain_community.agent_toolkits.powerbi.base",
    "create_pbi_chat_agent": "langchain_community.agent_toolkits.powerbi.chat_base",
    "create_spark_sql_agent": "langchain_community.agent_toolkits.spark_sql.base",
    "create_sql_agent": "langchain_community.agent_toolkits.sql.base",
    "load_tools": "langchain_community.agent_toolkits.load_tools",
    "load_huggingface_tool": "langchain_community.agent_toolkits.load_tools",
    "get_all_tool_names": "langchain_community.agent_toolkits.load_tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Get attr name."""
    if name in DEPRECATED_CODE:
        # Get directory of langchain package
        here = Path(__file__).parents[1]
        relative_path = as_import_path(
            Path(__file__).parent,
            suffix=name,
            relative_to=here,
        )
        old_path = "langchain_classic." + relative_path
        new_path = "langchain_experimental." + relative_path
        msg = (
            f"{name} has been moved to langchain experimental. "
            "See https://github.com/langchain-ai/langchain/discussions/11680"
            "for more information.\n"
            f"Please update your import statement from: `{old_path}` to `{new_path}`."
        )
        raise ImportError(msg)
    return _import_attribute(name)


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
    "XMLAgent",
    "ZeroShotAgent",
    "create_json_agent",
    "create_json_chat_agent",
    "create_openai_functions_agent",
    "create_openai_tools_agent",
    "create_openapi_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_react_agent",
    "create_self_ask_with_search_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
    "create_structured_chat_agent",
    "create_tool_calling_agent",
    "create_vectorstore_agent",
    "create_vectorstore_router_agent",
    "create_xml_agent",
    "get_all_tool_names",
    "initialize_agent",
    "load_agent",
    "load_huggingface_tool",
    "load_tools",
    "tool",
]
