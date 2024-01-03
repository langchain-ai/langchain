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
from pathlib import Path
from typing import Any

from langchain_community.agent_toolkits.ainetwork.toolkit import AINetworkToolkit
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain_community.agent_toolkits.azure_cognitive_services import (
    AzureCognitiveServicesToolkit,
)
from langchain_community.agent_toolkits.file_management.toolkit import (
    FileManagementToolkit,
)
from langchain_community.agent_toolkits.gmail.toolkit import GmailToolkit
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.agent_toolkits.multion.toolkit import MultionToolkit
from langchain_community.agent_toolkits.nasa.toolkit import NasaToolkit
from langchain_community.agent_toolkits.nla.toolkit import NLAToolkit
from langchain_community.agent_toolkits.office365.toolkit import O365Toolkit
from langchain_community.agent_toolkits.openapi.base import create_openapi_agent
from langchain_community.agent_toolkits.openapi.toolkit import OpenAPIToolkit
from langchain_community.agent_toolkits.playwright.toolkit import (
    PlayWrightBrowserToolkit,
)
from langchain_community.agent_toolkits.powerbi.base import create_pbi_agent
from langchain_community.agent_toolkits.powerbi.chat_base import create_pbi_chat_agent
from langchain_community.agent_toolkits.powerbi.toolkit import PowerBIToolkit
from langchain_community.agent_toolkits.slack.toolkit import SlackToolkit
from langchain_community.agent_toolkits.spark_sql.base import create_spark_sql_agent
from langchain_community.agent_toolkits.spark_sql.toolkit import SparkSQLToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.steam.toolkit import SteamToolkit
from langchain_community.agent_toolkits.zapier.toolkit import ZapierToolkit
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
    raise AttributeError(f"{name} does not exist")


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
