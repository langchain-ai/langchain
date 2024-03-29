"""Agent toolkits contain integrations with various resources and services.

LangChain has a large ecosystem of integrations with various external resources
like local and remote file systems, APIs and databases.

These integrations allow developers to create versatile applications that combine the
power of LLMs with the ability to access, interact with and manipulate external
resources.

When developing an application, developers should inspect the capabilities and
permissions of the tools that underlie the given agent toolkit, and determine
whether permissions of the given toolkit are appropriate for the application.

See [Security](https://python.langchain.com/docs/security) for more information.
"""
import warnings
from pathlib import Path
from typing import Any

from langchain_core._api import LangChainDeprecationWarning
from langchain_core._api.path import as_import_path

from langchain.agents.agent_toolkits.conversational_retrieval.openai_functions import (
    create_conversational_retrieval_agent,
)
from langchain.agents.agent_toolkits.vectorstore.base import (
    create_vectorstore_agent,
    create_vectorstore_router_agent,
)
from langchain.agents.agent_toolkits.vectorstore.toolkit import (
    VectorStoreInfo,
    VectorStoreRouterToolkit,
    VectorStoreToolkit,
)
from langchain.tools.retriever import create_retriever_tool
from langchain.utils.interactive_env import is_interactive_env

DEPRECATED_AGENTS = [
    "create_csv_agent",
    "create_pandas_dataframe_agent",
    "create_xorbits_agent",
    "create_python_agent",
    "create_spark_dataframe_agent",
]


def __getattr__(name: str) -> Any:
    """Get attr name."""
    if name in DEPRECATED_AGENTS:
        relative_path = as_import_path(Path(__file__).parent, suffix=name)
        old_path = "langchain." + relative_path
        new_path = "langchain_experimental." + relative_path
        raise ImportError(
            f"{name} has been moved to langchain experimental. "
            "See https://github.com/langchain-ai/langchain/discussions/11680"
            "for more information.\n"
            f"Please update your import statement from: `{old_path}` to `{new_path}`."
        )

    from langchain_community import agent_toolkits

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing this agent toolkit from langchain is deprecated. Importing it "
            "from langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.agent_toolkits import {name}`.\n\n"
            "To install langchain-community run `pip install -U langchain-community`.",
            category=LangChainDeprecationWarning,
        )

    return getattr(agent_toolkits, name)


__all__ = [
    "AINetworkToolkit",
    "AmadeusToolkit",
    "AzureCognitiveServicesToolkit",
    "FileManagementToolkit",
    "GmailToolkit",
    "JiraToolkit",
    "JsonToolkit",
    "MultionToolkit",
    "NasaToolkit",
    "NLAToolkit",
    "O365Toolkit",
    "OpenAPIToolkit",
    "PlayWrightBrowserToolkit",
    "PowerBIToolkit",
    "SlackToolkit",
    "SteamToolkit",
    "SQLDatabaseToolkit",
    "SparkSQLToolkit",
    "VectorStoreInfo",
    "VectorStoreRouterToolkit",
    "VectorStoreToolkit",
    "ZapierToolkit",
    "create_json_agent",
    "create_openapi_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
    "create_vectorstore_agent",
    "create_vectorstore_router_agent",
    "create_conversational_retrieval_agent",
    "create_retriever_tool",
]
