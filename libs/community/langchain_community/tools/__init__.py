"""**Tools** are classes that an Agent uses to interact with the world.

Each tool has a **description**. Agent uses the description to choose the right
tool for the job.

**Class hierarchy:**

.. code-block::

    ToolMetaclass --> BaseTool --> <name>Tool  # Examples: AIPluginTool, BaseGraphQLTool
                                   <name>      # Examples: BraveSearch, HumanInputRun

**Main helpers:**

.. code-block::

    CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
"""
import importlib
from typing import Any

# Used for internal purposes
_DEPRECATED_TOOLS = {"PythonAstREPLTool", "PythonREPLTool"}

_module_lookup = {
    "AINAppOps": "langchain_community.tools.ainetwork.app",
    "AINOwnerOps": "langchain_community.tools.ainetwork.owner",
    "AINRuleOps": "langchain_community.tools.ainetwork.rule",
    "AINTransfer": "langchain_community.tools.ainetwork.transfer",
    "AINValueOps": "langchain_community.tools.ainetwork.value",
    "AIPluginTool": "langchain_community.tools.plugin",
    "APIOperation": "langchain_community.tools.openapi.utils.api_models",
    "ArxivQueryRun": "langchain_community.tools.arxiv.tool",
    "AzureCogsFormRecognizerTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsImageAnalysisTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsSpeech2TextTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsText2SpeechTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsTextAnalyticsHealthTool": "langchain_community.tools.azure_cognitive_services",  # noqa: E501
    "BaseGraphQLTool": "langchain_community.tools.graphql.tool",
    "BaseRequestsTool": "langchain_community.tools.requests.tool",
    "BaseSQLDatabaseTool": "langchain_community.tools.sql_database.tool",
    "BaseSparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "BaseTool": "langchain_core.tools",
    "BearlyInterpreterTool": "langchain_community.tools.bearly.tool",
    "BingSearchResults": "langchain_community.tools.bing_search.tool",
    "BingSearchRun": "langchain_community.tools.bing_search.tool",
    "BraveSearch": "langchain_community.tools.brave_search.tool",
    "ClickTool": "langchain_community.tools.playwright",
    "CogniswitchKnowledgeRequest": "langchain_community.tools.cogniswitch.tool",
    "CogniswitchKnowledgeSourceFile": "langchain_community.tools.cogniswitch.tool",
    "CogniswitchKnowledgeSourceURL": "langchain_community.tools.cogniswitch.tool",
    "CogniswitchKnowledgeStatus": "langchain_community.tools.cogniswitch.tool",
    "ConneryAction": "langchain_community.tools.connery",
    "CopyFileTool": "langchain_community.tools.file_management",
    "CurrentWebPageTool": "langchain_community.tools.playwright",
    "DeleteFileTool": "langchain_community.tools.file_management",
    "DuckDuckGoSearchResults": "langchain_community.tools.ddg_search.tool",
    "DuckDuckGoSearchRun": "langchain_community.tools.ddg_search.tool",
    "E2BDataAnalysisTool": "langchain_community.tools.e2b_data_analysis.tool",
    "EdenAiExplicitImageTool": "langchain_community.tools.edenai",
    "EdenAiObjectDetectionTool": "langchain_community.tools.edenai",
    "EdenAiParsingIDTool": "langchain_community.tools.edenai",
    "EdenAiParsingInvoiceTool": "langchain_community.tools.edenai",
    "EdenAiSpeechToTextTool": "langchain_community.tools.edenai",
    "EdenAiTextModerationTool": "langchain_community.tools.edenai",
    "EdenAiTextToSpeechTool": "langchain_community.tools.edenai",
    "EdenaiTool": "langchain_community.tools.edenai",
    "ElevenLabsText2SpeechTool": "langchain_community.tools.eleven_labs.text2speech",
    "ExtractHyperlinksTool": "langchain_community.tools.playwright",
    "ExtractTextTool": "langchain_community.tools.playwright",
    "FileSearchTool": "langchain_community.tools.file_management",
    "GetElementsTool": "langchain_community.tools.playwright",
    "GmailCreateDraft": "langchain_community.tools.gmail",
    "GmailGetMessage": "langchain_community.tools.gmail",
    "GmailGetThread": "langchain_community.tools.gmail",
    "GmailSearch": "langchain_community.tools.gmail",
    "GmailSendMessage": "langchain_community.tools.gmail",
    "GoogleCloudTextToSpeechTool": "langchain_community.tools.google_cloud.texttospeech",  # noqa: E501
    "GooglePlacesTool": "langchain_community.tools.google_places.tool",
    "GoogleSearchResults": "langchain_community.tools.google_search.tool",
    "GoogleSearchRun": "langchain_community.tools.google_search.tool",
    "GoogleSerperResults": "langchain_community.tools.google_serper.tool",
    "GoogleSerperRun": "langchain_community.tools.google_serper.tool",
    "HumanInputRun": "langchain_community.tools.human.tool",
    "IFTTTWebhook": "langchain_community.tools.ifttt",
    "InfoPowerBITool": "langchain_community.tools.powerbi.tool",
    "InfoSQLDatabaseTool": "langchain_community.tools.sql_database.tool",
    "InfoSparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "JiraAction": "langchain_community.tools.jira.tool",
    "JsonGetValueTool": "langchain_community.tools.json.tool",
    "JsonListKeysTool": "langchain_community.tools.json.tool",
    "ListDirectoryTool": "langchain_community.tools.file_management",
    "ListPowerBITool": "langchain_community.tools.powerbi.tool",
    "ListSQLDatabaseTool": "langchain_community.tools.sql_database.tool",
    "ListSparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "MerriamWebsterQueryRun": "langchain_community.tools.merriam_webster.tool",
    "MetaphorSearchResults": "langchain_community.tools.metaphor_search",
    "MoveFileTool": "langchain_community.tools.file_management",
    "NasaAction": "langchain_community.tools.nasa.tool",
    "NavigateBackTool": "langchain_community.tools.playwright",
    "NavigateTool": "langchain_community.tools.playwright",
    "O365CreateDraftMessage": "langchain_community.tools.office365.create_draft_message",  # noqa: E501
    "O365SearchEmails": "langchain_community.tools.office365.messages_search",
    "O365SearchEvents": "langchain_community.tools.office365.events_search",
    "O365SendEvent": "langchain_community.tools.office365.send_event",
    "O365SendMessage": "langchain_community.tools.office365.send_message",
    "OpenAPISpec": "langchain_community.tools.openapi.utils.openapi_utils",
    "OpenWeatherMapQueryRun": "langchain_community.tools.openweathermap.tool",
    "PolygonAggregates": "langchain_community.tools.polygon.aggregates",
    "PolygonFinancials": "langchain_community.tools.polygon.financials",
    "PolygonLastQuote": "langchain_community.tools.polygon.last_quote",
    "PolygonTickerNews": "langchain_community.tools.polygon.ticker_news",
    "PubmedQueryRun": "langchain_community.tools.pubmed.tool",
    "QueryCheckerTool": "langchain_community.tools.spark_sql.tool",
    "QueryPowerBITool": "langchain_community.tools.powerbi.tool",
    "QuerySQLCheckerTool": "langchain_community.tools.sql_database.tool",
    "QuerySQLDataBaseTool": "langchain_community.tools.sql_database.tool",
    "QuerySparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "ReadFileTool": "langchain_community.tools.file_management",
    "RedditSearchRun": "langchain_community.tools.reddit_search.tool",
    "RedditSearchSchema": "langchain_community.tools.reddit_search.tool",
    "RequestsDeleteTool": "langchain_community.tools.requests.tool",
    "RequestsGetTool": "langchain_community.tools.requests.tool",
    "RequestsPatchTool": "langchain_community.tools.requests.tool",
    "RequestsPostTool": "langchain_community.tools.requests.tool",
    "RequestsPutTool": "langchain_community.tools.requests.tool",
    "SceneXplainTool": "langchain_community.tools.scenexplain.tool",
    "SearchAPIResults": "langchain_community.tools.searchapi.tool",
    "SearchAPIRun": "langchain_community.tools.searchapi.tool",
    "SearxSearchResults": "langchain_community.tools.searx_search.tool",
    "SearxSearchRun": "langchain_community.tools.searx_search.tool",
    "ShellTool": "langchain_community.tools.shell.tool",
    "SlackGetChannel": "langchain_community.tools.slack.get_channel",
    "SlackGetMessage": "langchain_community.tools.slack.get_message",
    "SlackScheduleMessage": "langchain_community.tools.slack.schedule_message",
    "SlackSendMessage": "langchain_community.tools.slack.send_message",
    "SleepTool": "langchain_community.tools.sleep.tool",
    "StackExchangeTool": "langchain_community.tools.stackexchange.tool",
    "StdInInquireTool": "langchain_community.tools.interaction.tool",
    "SteamWebAPIQueryRun": "langchain_community.tools.steam.tool",
    "SteamshipImageGenerationTool": "langchain_community.tools.steamship_image_generation",  # noqa: E501
    "StructuredTool": "langchain_core.tools",
    "Tool": "langchain_core.tools",
    "VectorStoreQATool": "langchain_community.tools.vectorstore.tool",
    "VectorStoreQAWithSourcesTool": "langchain_community.tools.vectorstore.tool",
    "WikipediaQueryRun": "langchain_community.tools.wikipedia.tool",
    "WolframAlphaQueryRun": "langchain_community.tools.wolfram_alpha.tool",
    "WriteFileTool": "langchain_community.tools.file_management",
    "YahooFinanceNewsTool": "langchain_community.tools.yahoo_finance_news",
    "YouSearchTool": "langchain_community.tools.you.tool",
    "YouTubeSearchTool": "langchain_community.tools.youtube.search",
    "ZapierNLAListActions": "langchain_community.tools.zapier.tool",
    "ZapierNLARunAction": "langchain_community.tools.zapier.tool",
    "authenticate": "langchain_community.tools.office365.utils",
    "format_tool_to_openai_function": "langchain_community.tools.convert_to_openai",
    "tool": "langchain_core.tools",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
