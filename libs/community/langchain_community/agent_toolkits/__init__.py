"""**Toolkits** are sets of tools that can be used to interact with
various services and APIs.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.agent_toolkits.ainetwork.toolkit import (
        AINetworkToolkit,
    )
    from langchain_community.agent_toolkits.amadeus.toolkit import (
        AmadeusToolkit,
    )
    from langchain_community.agent_toolkits.azure_ai_services import (
        AzureAiServicesToolkit,
    )
    from langchain_community.agent_toolkits.azure_cognitive_services import (
        AzureCognitiveServicesToolkit,
    )
    from langchain_community.agent_toolkits.cassandra_database.toolkit import (
        CassandraDatabaseToolkit,  # noqa: F401
    )
    from langchain_community.agent_toolkits.cogniswitch.toolkit import (
        CogniswitchToolkit,
    )
    from langchain_community.agent_toolkits.connery import (
        ConneryToolkit,
    )
    from langchain_community.agent_toolkits.file_management.toolkit import (
        FileManagementToolkit,
    )
    from langchain_community.agent_toolkits.gmail.toolkit import (
        GmailToolkit,
    )
    from langchain_community.agent_toolkits.jira.toolkit import (
        JiraToolkit,
    )
    from langchain_community.agent_toolkits.json.base import (
        create_json_agent,
    )
    from langchain_community.agent_toolkits.json.toolkit import (
        JsonToolkit,
    )
    from langchain_community.agent_toolkits.multion.toolkit import (
        MultionToolkit,
    )
    from langchain_community.agent_toolkits.nasa.toolkit import (
        NasaToolkit,
    )
    from langchain_community.agent_toolkits.nla.toolkit import (
        NLAToolkit,
    )
    from langchain_community.agent_toolkits.office365.toolkit import (
        O365Toolkit,
    )
    from langchain_community.agent_toolkits.openapi.base import (
        create_openapi_agent,
    )
    from langchain_community.agent_toolkits.openapi.toolkit import (
        OpenAPIToolkit,
    )
    from langchain_community.agent_toolkits.playwright.toolkit import (
        PlayWrightBrowserToolkit,
    )
    from langchain_community.agent_toolkits.polygon.toolkit import (
        PolygonToolkit,
    )
    from langchain_community.agent_toolkits.powerbi.base import (
        create_pbi_agent,
    )
    from langchain_community.agent_toolkits.powerbi.chat_base import (
        create_pbi_chat_agent,
    )
    from langchain_community.agent_toolkits.powerbi.toolkit import (
        PowerBIToolkit,
    )
    from langchain_community.agent_toolkits.slack.toolkit import (
        SlackToolkit,
    )
    from langchain_community.agent_toolkits.spark_sql.base import (
        create_spark_sql_agent,
    )
    from langchain_community.agent_toolkits.spark_sql.toolkit import (
        SparkSQLToolkit,
    )
    from langchain_community.agent_toolkits.sql.base import (
        create_sql_agent,
    )
    from langchain_community.agent_toolkits.sql.toolkit import (
        SQLDatabaseToolkit,
    )
    from langchain_community.agent_toolkits.steam.toolkit import (
        SteamToolkit,
    )
    from langchain_community.agent_toolkits.zapier.toolkit import (
        ZapierToolkit,
    )

__all__ = [
    "AINetworkToolkit",
    "AmadeusToolkit",
    "AzureAiServicesToolkit",
    "AzureCognitiveServicesToolkit",
    "CogniswitchToolkit",
    "ConneryToolkit",
    "FileManagementToolkit",
    "GmailToolkit",
    "JiraToolkit",
    "JsonToolkit",
    "MultionToolkit",
    "NLAToolkit",
    "NasaToolkit",
    "O365Toolkit",
    "OpenAPIToolkit",
    "PlayWrightBrowserToolkit",
    "PolygonToolkit",
    "PowerBIToolkit",
    "SQLDatabaseToolkit",
    "SlackToolkit",
    "SparkSQLToolkit",
    "SteamToolkit",
    "ZapierToolkit",
    "create_json_agent",
    "create_openapi_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
]


_module_lookup = {
    "AINetworkToolkit": "langchain_community.agent_toolkits.ainetwork.toolkit",
    "AmadeusToolkit": "langchain_community.agent_toolkits.amadeus.toolkit",
    "AzureAiServicesToolkit": "langchain_community.agent_toolkits.azure_ai_services",
    "AzureCognitiveServicesToolkit": "langchain_community.agent_toolkits.azure_cognitive_services",  # noqa: E501
    "CogniswitchToolkit": "langchain_community.agent_toolkits.cogniswitch.toolkit",
    "ConneryToolkit": "langchain_community.agent_toolkits.connery",
    "FileManagementToolkit": "langchain_community.agent_toolkits.file_management.toolkit",  # noqa: E501
    "GmailToolkit": "langchain_community.agent_toolkits.gmail.toolkit",
    "JiraToolkit": "langchain_community.agent_toolkits.jira.toolkit",
    "JsonToolkit": "langchain_community.agent_toolkits.json.toolkit",
    "MultionToolkit": "langchain_community.agent_toolkits.multion.toolkit",
    "NLAToolkit": "langchain_community.agent_toolkits.nla.toolkit",
    "NasaToolkit": "langchain_community.agent_toolkits.nasa.toolkit",
    "O365Toolkit": "langchain_community.agent_toolkits.office365.toolkit",
    "OpenAPIToolkit": "langchain_community.agent_toolkits.openapi.toolkit",
    "PlayWrightBrowserToolkit": "langchain_community.agent_toolkits.playwright.toolkit",
    "PolygonToolkit": "langchain_community.agent_toolkits.polygon.toolkit",
    "PowerBIToolkit": "langchain_community.agent_toolkits.powerbi.toolkit",
    "SQLDatabaseToolkit": "langchain_community.agent_toolkits.sql.toolkit",
    "SlackToolkit": "langchain_community.agent_toolkits.slack.toolkit",
    "SparkSQLToolkit": "langchain_community.agent_toolkits.spark_sql.toolkit",
    "SteamToolkit": "langchain_community.agent_toolkits.steam.toolkit",
    "ZapierToolkit": "langchain_community.agent_toolkits.zapier.toolkit",
    "create_json_agent": "langchain_community.agent_toolkits.json.base",
    "create_openapi_agent": "langchain_community.agent_toolkits.openapi.base",
    "create_pbi_agent": "langchain_community.agent_toolkits.powerbi.base",
    "create_pbi_chat_agent": "langchain_community.agent_toolkits.powerbi.chat_base",
    "create_spark_sql_agent": "langchain_community.agent_toolkits.spark_sql.base",
    "create_sql_agent": "langchain_community.agent_toolkits.sql.base",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
