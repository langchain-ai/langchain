from langchain_community.agent_toolkits import __all__, _module_lookup

EXPECTED_ALL = [
    "AINetworkToolkit",
    "AmadeusToolkit",
    "AzureAiServicesToolkit",
    "AzureCognitiveServicesToolkit",
    "ConneryToolkit",
    "FileManagementToolkit",
    "GmailToolkit",
    "GoogleCalendarToolkit",
    "JiraToolkit",
    "JsonToolkit",
    "MultionToolkit",
    "NasaToolkit",
    "NLAToolkit",
    "O365Toolkit",
    "OpenAPIToolkit",
    "PlayWrightBrowserToolkit",
    "PolygonToolkit",
    "PowerBIToolkit",
    "SlackToolkit",
    "SteamToolkit",
    "SQLDatabaseToolkit",
    "SparkSQLToolkit",
    "ZapierToolkit",
    "create_json_agent",
    "create_openapi_agent",
    "create_pbi_agent",
    "create_pbi_chat_agent",
    "create_spark_sql_agent",
    "create_sql_agent",
    "CogniswitchToolkit",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
    assert set(__all__) == set(_module_lookup.keys())
