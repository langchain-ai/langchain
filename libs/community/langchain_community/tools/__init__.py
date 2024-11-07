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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import (
        BaseTool as BaseTool,
    )
    from langchain_core.tools import (
        StructuredTool as StructuredTool,
    )
    from langchain_core.tools import (
        Tool as Tool,
    )
    from langchain_core.tools.convert import tool as tool

    from langchain_community.tools.ainetwork.app import (
        AINAppOps,
    )
    from langchain_community.tools.ainetwork.owner import (
        AINOwnerOps,
    )
    from langchain_community.tools.ainetwork.rule import (
        AINRuleOps,
    )
    from langchain_community.tools.ainetwork.transfer import (
        AINTransfer,
    )
    from langchain_community.tools.ainetwork.value import (
        AINValueOps,
    )
    from langchain_community.tools.arxiv.tool import (
        ArxivQueryRun,
    )
    from langchain_community.tools.asknews.tool import (
        AskNewsSearch,
    )
    from langchain_community.tools.azure_ai_services import (
        AzureAiServicesDocumentIntelligenceTool,
        AzureAiServicesImageAnalysisTool,
        AzureAiServicesSpeechToTextTool,
        AzureAiServicesTextAnalyticsForHealthTool,
        AzureAiServicesTextToSpeechTool,
    )
    from langchain_community.tools.azure_cognitive_services import (
        AzureCogsFormRecognizerTool,
        AzureCogsImageAnalysisTool,
        AzureCogsSpeech2TextTool,
        AzureCogsText2SpeechTool,
        AzureCogsTextAnalyticsHealthTool,
    )
    from langchain_community.tools.bearly.tool import (
        BearlyInterpreterTool,
    )
    from langchain_community.tools.bing_search.tool import (
        BingSearchResults,
        BingSearchRun,
    )
    from langchain_community.tools.brave_search.tool import (
        BraveSearch,
    )
    from langchain_community.tools.cassandra_database.tool import (
        GetSchemaCassandraDatabaseTool,  # noqa: F401
        GetTableDataCassandraDatabaseTool,  # noqa: F401
        QueryCassandraDatabaseTool,  # noqa: F401
    )
    from langchain_community.tools.cogniswitch.tool import (
        CogniswitchKnowledgeRequest,
        CogniswitchKnowledgeSourceFile,
        CogniswitchKnowledgeSourceURL,
        CogniswitchKnowledgeStatus,
    )
    from langchain_community.tools.connery import (
        ConneryAction,
    )
    from langchain_community.tools.convert_to_openai import (
        format_tool_to_openai_function,
    )
    from langchain_community.tools.dataherald import DataheraldTextToSQL
    from langchain_community.tools.ddg_search.tool import (
        DuckDuckGoSearchResults,
        DuckDuckGoSearchRun,
    )
    from langchain_community.tools.e2b_data_analysis.tool import (
        E2BDataAnalysisTool,
    )
    from langchain_community.tools.edenai import (
        EdenAiExplicitImageTool,
        EdenAiObjectDetectionTool,
        EdenAiParsingIDTool,
        EdenAiParsingInvoiceTool,
        EdenAiSpeechToTextTool,
        EdenAiTextModerationTool,
        EdenAiTextToSpeechTool,
        EdenaiTool,
    )
    from langchain_community.tools.eleven_labs.text2speech import (
        ElevenLabsText2SpeechTool,
    )
    from langchain_community.tools.file_management import (
        CopyFileTool,
        DeleteFileTool,
        FileSearchTool,
        ListDirectoryTool,
        MoveFileTool,
        ReadFileTool,
        WriteFileTool,
    )
    from langchain_community.tools.financial_datasets.balance_sheets import (
        BalanceSheets,
    )
    from langchain_community.tools.financial_datasets.cash_flow_statements import (
        CashFlowStatements,
    )
    from langchain_community.tools.financial_datasets.income_statements import (
        IncomeStatements,
    )
    from langchain_community.tools.gmail import (
        GmailCreateDraft,
        GmailGetMessage,
        GmailGetThread,
        GmailSearch,
        GmailSendMessage,
    )
    from langchain_community.tools.google_books import (
        GoogleBooksQueryRun,
    )
    from langchain_community.tools.google_cloud.texttospeech import (
        GoogleCloudTextToSpeechTool,
    )
    from langchain_community.tools.google_places.tool import (
        GooglePlacesTool,
    )
    from langchain_community.tools.google_search.tool import (
        GoogleSearchResults,
        GoogleSearchRun,
    )
    from langchain_community.tools.google_serper.tool import (
        GoogleSerperResults,
        GoogleSerperRun,
    )
    from langchain_community.tools.graphql.tool import (
        BaseGraphQLTool,
    )
    from langchain_community.tools.human.tool import (
        HumanInputRun,
    )
    from langchain_community.tools.ifttt import (
        IFTTTWebhook,
    )
    from langchain_community.tools.interaction.tool import (
        StdInInquireTool,
    )
    from langchain_community.tools.jina_search.tool import JinaSearch
    from langchain_community.tools.jira.tool import (
        JiraAction,
    )
    from langchain_community.tools.json.tool import (
        JsonGetValueTool,
        JsonListKeysTool,
    )
    from langchain_community.tools.merriam_webster.tool import (
        MerriamWebsterQueryRun,
    )
    from langchain_community.tools.metaphor_search import (
        MetaphorSearchResults,
    )
    from langchain_community.tools.mojeek_search.tool import (
        MojeekSearch,
    )
    from langchain_community.tools.nasa.tool import (
        NasaAction,
    )
    from langchain_community.tools.office365.create_draft_message import (
        O365CreateDraftMessage,
    )
    from langchain_community.tools.office365.events_search import (
        O365SearchEvents,
    )
    from langchain_community.tools.office365.messages_search import (
        O365SearchEmails,
    )
    from langchain_community.tools.office365.send_event import (
        O365SendEvent,
    )
    from langchain_community.tools.office365.send_message import (
        O365SendMessage,
    )
    from langchain_community.tools.office365.utils import (
        authenticate,
    )
    from langchain_community.tools.openapi.utils.api_models import (
        APIOperation,
    )
    from langchain_community.tools.openapi.utils.openapi_utils import (
        OpenAPISpec,
    )
    from langchain_community.tools.openweathermap.tool import (
        OpenWeatherMapQueryRun,
    )
    from langchain_community.tools.playwright import (
        ClickTool,
        CurrentWebPageTool,
        ExtractHyperlinksTool,
        ExtractTextTool,
        GetElementsTool,
        NavigateBackTool,
        NavigateTool,
    )
    from langchain_community.tools.plugin import (
        AIPluginTool,
    )
    from langchain_community.tools.polygon.aggregates import (
        PolygonAggregates,
    )
    from langchain_community.tools.polygon.financials import (
        PolygonFinancials,
    )
    from langchain_community.tools.polygon.last_quote import (
        PolygonLastQuote,
    )
    from langchain_community.tools.polygon.ticker_news import (
        PolygonTickerNews,
    )
    from langchain_community.tools.powerbi.tool import (
        InfoPowerBITool,
        ListPowerBITool,
        QueryPowerBITool,
    )
    from langchain_community.tools.pubmed.tool import (
        PubmedQueryRun,
    )
    from langchain_community.tools.reddit_search.tool import (
        RedditSearchRun,
        RedditSearchSchema,
    )
    from langchain_community.tools.requests.tool import (
        BaseRequestsTool,
        RequestsDeleteTool,
        RequestsGetTool,
        RequestsPatchTool,
        RequestsPostTool,
        RequestsPutTool,
    )
    from langchain_community.tools.scenexplain.tool import (
        SceneXplainTool,
    )
    from langchain_community.tools.searchapi.tool import (
        SearchAPIResults,
        SearchAPIRun,
    )
    from langchain_community.tools.searx_search.tool import (
        SearxSearchResults,
        SearxSearchRun,
    )
    from langchain_community.tools.shell.tool import (
        ShellTool,
    )
    from langchain_community.tools.slack.get_channel import (
        SlackGetChannel,
    )
    from langchain_community.tools.slack.get_message import (
        SlackGetMessage,
    )
    from langchain_community.tools.slack.schedule_message import (
        SlackScheduleMessage,
    )
    from langchain_community.tools.slack.send_message import (
        SlackSendMessage,
    )
    from langchain_community.tools.sleep.tool import (
        SleepTool,
    )
    from langchain_community.tools.spark_sql.tool import (
        BaseSparkSQLTool,
        InfoSparkSQLTool,
        ListSparkSQLTool,
        QueryCheckerTool,
        QuerySparkSQLTool,
    )
    from langchain_community.tools.sql_database.tool import (
        BaseSQLDatabaseTool,
        InfoSQLDatabaseTool,
        ListSQLDatabaseTool,
        QuerySQLCheckerTool,
        QuerySQLDataBaseTool,
    )
    from langchain_community.tools.stackexchange.tool import (
        StackExchangeTool,
    )
    from langchain_community.tools.steam.tool import (
        SteamWebAPIQueryRun,
    )
    from langchain_community.tools.steamship_image_generation import (
        SteamshipImageGenerationTool,
    )
    from langchain_community.tools.tavily_search import (
        TavilyAnswer,
        TavilySearchResults,
    )
    from langchain_community.tools.vectorstore.tool import (
        VectorStoreQATool,
        VectorStoreQAWithSourcesTool,
    )
    from langchain_community.tools.wikipedia.tool import (
        WikipediaQueryRun,
    )
    from langchain_community.tools.wolfram_alpha.tool import (
        WolframAlphaQueryRun,
    )
    from langchain_community.tools.yahoo_finance_news import (
        YahooFinanceNewsTool,
    )
    from langchain_community.tools.you.tool import (
        YouSearchTool,
    )
    from langchain_community.tools.youtube.search import (
        YouTubeSearchTool,
    )
    from langchain_community.tools.zapier.tool import (
        ZapierNLAListActions,
        ZapierNLARunAction,
    )
    from langchain_community.tools.zenguard.tool import (
        Detector,
        ZenGuardInput,
        ZenGuardTool,
    )

__all__ = [
    "BaseTool",
    "Tool",
    "tool",
    "StructuredTool",
    "AINAppOps",
    "AINOwnerOps",
    "AINRuleOps",
    "AINTransfer",
    "AINValueOps",
    "AIPluginTool",
    "APIOperation",
    "ArxivQueryRun",
    "AskNewsSearch",
    "AzureAiServicesDocumentIntelligenceTool",
    "AzureAiServicesImageAnalysisTool",
    "AzureAiServicesSpeechToTextTool",
    "AzureAiServicesTextAnalyticsForHealthTool",
    "AzureAiServicesTextToSpeechTool",
    "AzureCogsFormRecognizerTool",
    "AzureCogsImageAnalysisTool",
    "AzureCogsSpeech2TextTool",
    "AzureCogsText2SpeechTool",
    "AzureCogsTextAnalyticsHealthTool",
    "BalanceSheets",
    "BaseGraphQLTool",
    "BaseRequestsTool",
    "BaseSQLDatabaseTool",
    "BaseSparkSQLTool",
    "BearlyInterpreterTool",
    "BingSearchResults",
    "BingSearchRun",
    "BraveSearch",
    "CashFlowStatements",
    "ClickTool",
    "CogniswitchKnowledgeRequest",
    "CogniswitchKnowledgeSourceFile",
    "CogniswitchKnowledgeSourceURL",
    "CogniswitchKnowledgeStatus",
    "ConneryAction",
    "CopyFileTool",
    "CurrentWebPageTool",
    "DeleteFileTool",
    "DataheraldTextToSQL",
    "DuckDuckGoSearchResults",
    "DuckDuckGoSearchRun",
    "E2BDataAnalysisTool",
    "EdenAiExplicitImageTool",
    "EdenAiObjectDetectionTool",
    "EdenAiParsingIDTool",
    "EdenAiParsingInvoiceTool",
    "EdenAiSpeechToTextTool",
    "EdenAiTextModerationTool",
    "EdenAiTextToSpeechTool",
    "EdenaiTool",
    "ElevenLabsText2SpeechTool",
    "ExtractHyperlinksTool",
    "ExtractTextTool",
    "FileSearchTool",
    "GetElementsTool",
    "GmailCreateDraft",
    "GmailGetMessage",
    "GmailGetThread",
    "GmailSearch",
    "GmailSendMessage",
    "GoogleBooksQueryRun",
    "GoogleCloudTextToSpeechTool",
    "GooglePlacesTool",
    "GoogleSearchResults",
    "GoogleSearchRun",
    "GoogleSerperResults",
    "GoogleSerperRun",
    "HumanInputRun",
    "IFTTTWebhook",
    "IncomeStatements",
    "InfoPowerBITool",
    "InfoSQLDatabaseTool",
    "InfoSparkSQLTool",
    "JiraAction",
    "JinaSearch",
    "JsonGetValueTool",
    "JsonListKeysTool",
    "ListDirectoryTool",
    "ListPowerBITool",
    "ListSQLDatabaseTool",
    "ListSparkSQLTool",
    "MerriamWebsterQueryRun",
    "MetaphorSearchResults",
    "MojeekSearch",
    "MoveFileTool",
    "NasaAction",
    "NavigateBackTool",
    "NavigateTool",
    "O365CreateDraftMessage",
    "O365SearchEmails",
    "O365SearchEvents",
    "O365SendEvent",
    "O365SendMessage",
    "OpenAPISpec",
    "OpenWeatherMapQueryRun",
    "PolygonAggregates",
    "PolygonFinancials",
    "PolygonLastQuote",
    "PolygonTickerNews",
    "PubmedQueryRun",
    "QueryCheckerTool",
    "QueryPowerBITool",
    "QuerySQLCheckerTool",
    "QuerySQLDataBaseTool",
    "QuerySparkSQLTool",
    "ReadFileTool",
    "RedditSearchRun",
    "RedditSearchSchema",
    "RequestsDeleteTool",
    "RequestsGetTool",
    "RequestsPatchTool",
    "RequestsPostTool",
    "RequestsPutTool",
    "SceneXplainTool",
    "SearchAPIResults",
    "SearchAPIRun",
    "SearxSearchResults",
    "SearxSearchRun",
    "ShellTool",
    "SlackGetChannel",
    "SlackGetMessage",
    "SlackScheduleMessage",
    "SlackSendMessage",
    "SleepTool",
    "StackExchangeTool",
    "StdInInquireTool",
    "SteamWebAPIQueryRun",
    "SteamshipImageGenerationTool",
    "TavilyAnswer",
    "TavilySearchResults",
    "VectorStoreQATool",
    "VectorStoreQAWithSourcesTool",
    "WikipediaQueryRun",
    "WolframAlphaQueryRun",
    "WriteFileTool",
    "YahooFinanceNewsTool",
    "YouSearchTool",
    "YouTubeSearchTool",
    "ZapierNLAListActions",
    "ZapierNLARunAction",
    "Detector",
    "ZenGuardInput",
    "ZenGuardTool",
    "authenticate",
    "format_tool_to_openai_function",
]

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
    "AskNewsSearch": "langchain_community.tools.asknews.tool",
    "AzureAiServicesDocumentIntelligenceTool": "langchain_community.tools.azure_ai_services",  # noqa: E501
    "AzureAiServicesImageAnalysisTool": "langchain_community.tools.azure_ai_services",
    "AzureAiServicesSpeechToTextTool": "langchain_community.tools.azure_ai_services",
    "AzureAiServicesTextToSpeechTool": "langchain_community.tools.azure_ai_services",
    "AzureAiServicesTextAnalyticsForHealthTool": "langchain_community.tools.azure_ai_services",  # noqa: E501
    "AzureCogsFormRecognizerTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsImageAnalysisTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsSpeech2TextTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsText2SpeechTool": "langchain_community.tools.azure_cognitive_services",
    "AzureCogsTextAnalyticsHealthTool": "langchain_community.tools.azure_cognitive_services",  # noqa: E501
    "BalanceSheets": "langchain_community.tools.financial_datasets.balance_sheets",
    "BaseGraphQLTool": "langchain_community.tools.graphql.tool",
    "BaseRequestsTool": "langchain_community.tools.requests.tool",
    "BaseSQLDatabaseTool": "langchain_community.tools.sql_database.tool",
    "BaseSparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "BaseTool": "langchain_core.tools",
    "BearlyInterpreterTool": "langchain_community.tools.bearly.tool",
    "BingSearchResults": "langchain_community.tools.bing_search.tool",
    "BingSearchRun": "langchain_community.tools.bing_search.tool",
    "BraveSearch": "langchain_community.tools.brave_search.tool",
    "CashFlowStatements": "langchain_community.tools.financial_datasets.cash_flow_statements",  # noqa: E501
    "ClickTool": "langchain_community.tools.playwright",
    "CogniswitchKnowledgeRequest": "langchain_community.tools.cogniswitch.tool",
    "CogniswitchKnowledgeSourceFile": "langchain_community.tools.cogniswitch.tool",
    "CogniswitchKnowledgeSourceURL": "langchain_community.tools.cogniswitch.tool",
    "CogniswitchKnowledgeStatus": "langchain_community.tools.cogniswitch.tool",
    "ConneryAction": "langchain_community.tools.connery",
    "CopyFileTool": "langchain_community.tools.file_management",
    "CurrentWebPageTool": "langchain_community.tools.playwright",
    "DataheraldTextToSQL": "langchain_community.tools.dataherald.tool",
    "DeleteFileTool": "langchain_community.tools.file_management",
    "Detector": "langchain_community.tools.zenguard.tool",
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
    "GoogleBooksQueryRun": "langchain_community.tools.google_books",
    "GoogleCloudTextToSpeechTool": "langchain_community.tools.google_cloud.texttospeech",  # noqa: E501
    "GooglePlacesTool": "langchain_community.tools.google_places.tool",
    "GoogleSearchResults": "langchain_community.tools.google_search.tool",
    "GoogleSearchRun": "langchain_community.tools.google_search.tool",
    "GoogleSerperResults": "langchain_community.tools.google_serper.tool",
    "GoogleSerperRun": "langchain_community.tools.google_serper.tool",
    "HumanInputRun": "langchain_community.tools.human.tool",
    "IFTTTWebhook": "langchain_community.tools.ifttt",
    "IncomeStatements": "langchain_community.tools.financial_datasets.income_statements",  # noqa: E501
    "InfoPowerBITool": "langchain_community.tools.powerbi.tool",
    "InfoSQLDatabaseTool": "langchain_community.tools.sql_database.tool",
    "InfoSparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "JiraAction": "langchain_community.tools.jira.tool",
    "JinaSearch": "langchain_community.tools.jina_search.tool",
    "JsonGetValueTool": "langchain_community.tools.json.tool",
    "JsonListKeysTool": "langchain_community.tools.json.tool",
    "ListDirectoryTool": "langchain_community.tools.file_management",
    "ListPowerBITool": "langchain_community.tools.powerbi.tool",
    "ListSQLDatabaseTool": "langchain_community.tools.sql_database.tool",
    "ListSparkSQLTool": "langchain_community.tools.spark_sql.tool",
    "MerriamWebsterQueryRun": "langchain_community.tools.merriam_webster.tool",
    "MetaphorSearchResults": "langchain_community.tools.metaphor_search",
    "MojeekSearch": "langchain_community.tools.mojeek_search.tool",
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
    "TavilyAnswer": "langchain_community.tools.tavily_search",
    "TavilySearchResults": "langchain_community.tools.tavily_search",
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
    "ZenGuardInput": "langchain_community.tools.zenguard.tool",
    "ZenGuardTool": "langchain_community.tools.zenguard.tool",
    "authenticate": "langchain_community.tools.office365.utils",
    "format_tool_to_openai_function": "langchain_community.tools.convert_to_openai",
    "tool": "langchain_core.tools",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
