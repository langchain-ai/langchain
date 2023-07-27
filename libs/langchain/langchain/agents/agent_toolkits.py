"""DEPRECATED: Kept for backwards compatibility."""
from langchain.toolkits.amadeus.toolkit import AmadeusToolkit
from langchain.toolkits.azure_cognitive_services import (
    AzureCognitiveServicesToolkit,
)
from langchain.toolkits.csv.base import create_csv_agent
from langchain.toolkits.file_management.toolkit import (
    FileManagementToolkit,
)
from langchain.toolkits.gmail.toolkit import GmailToolkit
from langchain.toolkits.jira.toolkit import JiraToolkit
from langchain.toolkits.json.base import create_json_agent
from langchain.toolkits.json.toolkit import JsonToolkit
from langchain.toolkits.nla.toolkit import NLAToolkit
from langchain.toolkits.office365.toolkit import O365Toolkit
from langchain.toolkits.openapi.base import create_openapi_agent
from langchain.toolkits.openapi.toolkit import OpenAPIToolkit
from langchain.toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain.toolkits.powerbi.base import create_pbi_agent
from langchain.toolkits.powerbi.chat_base import create_pbi_chat_agent
from langchain.toolkits.powerbi.toolkit import PowerBIToolkit
from langchain.toolkits.python.base import create_python_agent
from langchain.toolkits.spark.base import create_spark_dataframe_agent
from langchain.toolkits.spark_sql.base import create_spark_sql_agent
from langchain.toolkits.spark_sql.toolkit import SparkSQLToolkit
from langchain.toolkits.sql.base import create_sql_agent
from langchain.toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.toolkits.vectorstore.base import (
    create_vectorstore_agent,
    create_vectorstore_router_agent,
)
from langchain.toolkits.vectorstore.toolkit import (
    VectorStoreInfo,
    VectorStoreRouterToolkit,
    VectorStoreToolkit,
)
from langchain.toolkits.xorbits.base import create_xorbits_agent
from langchain.toolkits.zapier.toolkit import ZapierToolkit

__all__ = [
    "AmadeusToolkit",
    "AzureCognitiveServicesToolkit",
    "FileManagementToolkit",
    "GmailToolkit",
    "JiraToolkit",
    "JsonToolkit",
    "NLAToolkit",
    "O365Toolkit",
    "OpenAPIToolkit",
    "PlayWrightBrowserToolkit",
    "PowerBIToolkit",
    "SQLDatabaseToolkit",
    "SparkSQLToolkit",
    "VectorStoreInfo",
    "VectorStoreRouterToolkit",
    "VectorStoreToolkit",
    "ZapierToolkit",
    "create_csv_agent",
    "create_json_agent",
    "create_openapi_agent",
    "create_pandas_dataframe_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_python_agent",
    "create_spark_dataframe_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
    "create_vectorstore_agent",
    "create_vectorstore_router_agent",
    "create_xorbits_agent",
]
