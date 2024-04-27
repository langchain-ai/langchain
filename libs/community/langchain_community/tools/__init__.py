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
        BaseTool,  # noqa: F401
        StructuredTool,  # noqa: F401
        Tool,  # noqa: F401
        tool,  # noqa: F401
    )

    from langchain_community.tools.ainetwork.app import (
        AINAppOps,  # noqa: F401
    )
    from langchain_community.tools.ainetwork.owner import (
        AINOwnerOps,  # noqa: F401
    )
    from langchain_community.tools.ainetwork.rule import (
        AINRuleOps,  # noqa: F401
    )
    from langchain_community.tools.ainetwork.transfer import (
        AINTransfer,  # noqa: F401
    )
    from langchain_community.tools.ainetwork.value import (
        AINValueOps,  # noqa: F401
    )
    from langchain_community.tools.arxiv.tool import (
        ArxivQueryRun,  # noqa: F401
    )
    from langchain_community.tools.azure_ai_services import (
        AzureAiServicesDocumentIntelligenceTool,  # noqa: F401
        AzureAiServicesImageAnalysisTool,  # noqa: F401
        AzureAiServicesSpeechToTextTool,  # noqa: F401
        AzureAiServicesTextAnalyticsForHealthTool,  # noqa: F401
        AzureAiServicesTextToSpeechTool,  # noqa: F401
    )
    from langchain_community.tools.azure_cognitive_services import (
        AzureCogsFormRecognizerTool,  # noqa: F401
        AzureCogsImageAnalysisTool,  # noqa: F401
        AzureCogsSpeech2TextTool,  # noqa: F401
        AzureCogsText2SpeechTool,  # noqa: F401
        AzureCogsTextAnalyticsHealthTool,  # noqa: F401
    )
    from langchain_community.tools.bearly.tool import (
        BearlyInterpreterTool,  # noqa: F401
    )
    from langchain_community.tools.bing_search.tool import (
        BingSearchResults,  # noqa: F401
        BingSearchRun,  # noqa: F401
    )
    from langchain_community.tools.brave_search.tool import (
        BraveSearch,  # noqa: F401
    )
    from langchain_community.tools.cogniswitch.tool import (
        CogniswitchKnowledgeRequest,  # noqa: F401
        CogniswitchKnowledgeSourceFile,  # noqa: F401
        CogniswitchKnowledgeSourceURL,  # noqa: F401
        CogniswitchKnowledgeStatus,  # noqa: F401
    )
    from langchain_community.tools.connery import (
        ConneryAction,  # noqa: F401
    )
    from langchain_community.tools.convert_to_openai import (
        format_tool_to_openai_function,  # noqa: F401
    )
    from langchain_community.tools.ddg_search.tool import (
        DuckDuckGoSearchResults,  # noqa: F401
        DuckDuckGoSearchRun,  # noqa: F401
    )
    from langchain_community.tools.e2b_data_analysis.tool import (
        E2BDataAnalysisTool,  # noqa: F401
    )
    from langchain_community.tools.edenai import (
        EdenAiExplicitImageTool,  # noqa: F401
        EdenAiObjectDetectionTool,  # noqa: F401
        EdenAiParsingIDTool,  # noqa: F401
        EdenAiParsingInvoiceTool,  # noqa: F401
        EdenAiSpeechToTextTool,  # noqa: F401
        EdenAiTextModerationTool,  # noqa: F401
        EdenAiTextToSpeechTool,  # noqa: F401
        EdenaiTool,  # noqa: F401
    )
    from langchain_community.tools.eleven_labs.text2speech import (
        ElevenLabsText2SpeechTool,  # noqa: F401
    )
    from langchain_community.tools.file_management import (
        CopyFileTool,  # noqa: F401
        DeleteFileTool,  # noqa: F401
        FileSearchTool,  # noqa: F401
        ListDirectoryTool,  # noqa: F401
        MoveFileTool,  # noqa: F401
        ReadFileTool,  # noqa: F401
        WriteFileTool,  # noqa: F401
    )
    from langchain_community.tools.gmail import (
        GmailCreateDraft,  # noqa: F401
        GmailGetMessage,  # noqa: F401
        GmailGetThread,  # noqa: F401
        GmailSearch,  # noqa: F401
        GmailSendMessage,  # noqa: F401
    )
    from langchain_community.tools.google_cloud.texttospeech import (
        GoogleCloudTextToSpeechTool,  # noqa: F401
    )
    from langchain_community.tools.google_places.tool import (
        GooglePlacesTool,  # noqa: F401
    )
    from langchain_community.tools.google_search.tool import (
        GoogleSearchResults,  # noqa: F401
        GoogleSearchRun,  # noqa: F401
    )
    from langchain_community.tools.google_serper.tool import (
        GoogleSerperResults,  # noqa: F401
        GoogleSerperRun,  # noqa: F401
    )
    from langchain_community.tools.graphql.tool import (
        BaseGraphQLTool,  # noqa: F401
    )
    from langchain_community.tools.human.tool import (
        HumanInputRun,  # noqa: F401
    )
    from langchain_community.tools.ifttt import (
        IFTTTWebhook,  # noqa: F401
    )
    from langchain_community.tools.interaction.tool import (
        StdInInquireTool,  # noqa: F401
    )
    from langchain_community.tools.jira.tool import (
        JiraAction,  # noqa: F401
    )
    from langchain_community.tools.json.tool import (
        JsonGetValueTool,  # noqa: F401
        JsonListKeysTool,  # noqa: F401
    )
    from langchain_community.tools.merriam_webster.tool import (
        MerriamWebsterQueryRun,  # noqa: F401
    )
    from langchain_community.tools.metaphor_search import (
        MetaphorSearchResults,  # noqa: F401
    )
    from langchain_community.tools.nasa.tool import (
        NasaAction,  # noqa: F401
    )
    from langchain_community.tools.office365.create_draft_message import (
        O365CreateDraftMessage,  # noqa: F401
    )
    from langchain_community.tools.office365.events_search import (
        O365SearchEvents,  # noqa: F401
    )
    from langchain_community.tools.office365.messages_search import (
        O365SearchEmails,  # noqa: F401
    )
    from langchain_community.tools.office365.send_event import (
        O365SendEvent,  # noqa: F401
    )
    from langchain_community.tools.office365.send_message import (
        O365SendMessage,  # noqa: F401
    )
    from langchain_community.tools.office365.utils import (
        authenticate,  # noqa: F401
    )
    from langchain_community.tools.openapi.utils.api_models import (
        APIOperation,  # noqa: F401
    )
    from langchain_community.tools.openapi.utils.openapi_utils import (
        OpenAPISpec,  # noqa: F401
    )
    from langchain_community.tools.openweathermap.tool import (
        OpenWeatherMapQueryRun,  # noqa: F401
    )
    from langchain_community.tools.playwright import (
        ClickTool,  # noqa: F401
        CurrentWebPageTool,  # noqa: F401
        ExtractHyperlinksTool,  # noqa: F401
        ExtractTextTool,  # noqa: F401
        GetElementsTool,  # noqa: F401
        NavigateBackTool,  # noqa: F401
        NavigateTool,  # noqa: F401
    )
    from langchain_community.tools.plugin import (
        AIPluginTool,  # noqa: F401
    )
    from langchain_community.tools.polygon.aggregates import (
        PolygonAggregates,  # noqa: F401
    )
    from langchain_community.tools.polygon.financials import (
        PolygonFinancials,  # noqa: F401
    )
    from langchain_community.tools.polygon.last_quote import (
        PolygonLastQuote,  # noqa: F401
    )
    from langchain_community.tools.polygon.ticker_news import (
        PolygonTickerNews,  # noqa: F401
    )
    from langchain_community.tools.powerbi.tool import (
        InfoPowerBITool,  # noqa: F401
        ListPowerBITool,  # noqa: F401
        QueryPowerBITool,  # noqa: F401
    )
    from langchain_community.tools.pubmed.tool import (
        PubmedQueryRun,  # noqa: F401
    )
    from langchain_community.tools.reddit_search.tool import (
        RedditSearchRun,  # noqa: F401
        RedditSearchSchema,  # noqa: F401
    )
    from langchain_community.tools.requests.tool import (
        BaseRequestsTool,  # noqa: F401
        RequestsDeleteTool,  # noqa: F401
        RequestsGetTool,  # noqa: F401
        RequestsPatchTool,  # noqa: F401
        RequestsPostTool,  # noqa: F401
        RequestsPutTool,  # noqa: F401
    )
    from langchain_community.tools.scenexplain.tool import (
        SceneXplainTool,  # noqa: F401
    )
    from langchain_community.tools.searchapi.tool import (
        SearchAPIResults,  # noqa: F401
        SearchAPIRun,  # noqa: F401
    )
    from langchain_community.tools.searx_search.tool import (
        SearxSearchResults,  # noqa: F401
        SearxSearchRun,  # noqa: F401
    )
    from langchain_community.tools.shell.tool import (
        ShellTool,  # noqa: F401
    )
    from langchain_community.tools.slack.get_channel import (
        SlackGetChannel,  # noqa: F401
    )
    from langchain_community.tools.slack.get_message import (
        SlackGetMessage,  # noqa: F401
    )
    from langchain_community.tools.slack.schedule_message import (
        SlackScheduleMessage,  # noqa: F401
    )
    from langchain_community.tools.slack.send_message import (
        SlackSendMessage,  # noqa: F401
    )
    from langchain_community.tools.sleep.tool import (
        SleepTool,  # noqa: F401
    )
    from langchain_community.tools.spark_sql.tool import (
        BaseSparkSQLTool,  # noqa: F401
        InfoSparkSQLTool,  # noqa: F401
        ListSparkSQLTool,  # noqa: F401
        QueryCheckerTool,  # noqa: F401
        QuerySparkSQLTool,  # noqa: F401
    )
    from langchain_community.tools.sql_database.tool import (
        BaseSQLDatabaseTool,  # noqa: F401
        InfoSQLDatabaseTool,  # noqa: F401
        ListSQLDatabaseTool,  # noqa: F401
        QuerySQLCheckerTool,  # noqa: F401
        QuerySQLDataBaseTool,  # noqa: F401
    )
    from langchain_community.tools.stackexchange.tool import (
        StackExchangeTool,  # noqa: F401
    )
    from langchain_community.tools.steam.tool import (
        SteamWebAPIQueryRun,  # noqa: F401
    )
    from langchain_community.tools.steamship_image_generation import (
        SteamshipImageGenerationTool,  # noqa: F401
    )
    from langchain_community.tools.vectorstore.tool import (
        VectorStoreQATool,  # noqa: F401
        VectorStoreQAWithSourcesTool,  # noqa: F401
    )
    from langchain_community.tools.wikipedia.tool import (
        WikipediaQueryRun,  # noqa: F401
    )
    from langchain_community.tools.wolfram_alpha.tool import (
        WolframAlphaQueryRun,  # noqa: F401
    )
    from langchain_community.tools.yahoo_finance_news import (
        YahooFinanceNewsTool,  # noqa: F401
    )
    from langchain_community.tools.you.tool import (
        YouSearchTool,  # noqa: F401
    )
    from langchain_community.tools.youtube.search import (
        YouTubeSearchTool,  # noqa: F401
    )
    from langchain_community.tools.zapier.tool import (
        ZapierNLAListActions,  # noqa: F401
        ZapierNLARunAction,  # noqa: F401
    )

__all__ = [
    "AINAppOps",
    "AINOwnerOps",
    "AINRuleOps",
    "AINTransfer",
    "AINValueOps",
    "AIPluginTool",
    "APIOperation",
    "ArxivQueryRun",
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
    "BaseGraphQLTool",
    "BaseRequestsTool",
    "BaseSQLDatabaseTool",
    "BaseSparkSQLTool",
    "BaseTool",
    "BearlyInterpreterTool",
    "BingSearchResults",
    "BingSearchRun",
    "BraveSearch",
    "ClickTool",
    "CogniswitchKnowledgeRequest",
    "CogniswitchKnowledgeSourceFile",
    "CogniswitchKnowledgeSourceURL",
    "CogniswitchKnowledgeStatus",
    "ConneryAction",
    "CopyFileTool",
    "CurrentWebPageTool",
    "DeleteFileTool",
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
    "GoogleCloudTextToSpeechTool",
    "GooglePlacesTool",
    "GoogleSearchResults",
    "GoogleSearchRun",
    "GoogleSerperResults",
    "GoogleSerperRun",
    "HumanInputRun",
    "IFTTTWebhook",
    "InfoPowerBITool",
    "InfoSQLDatabaseTool",
    "InfoSparkSQLTool",
    "JiraAction",
    "JsonGetValueTool",
    "JsonListKeysTool",
    "ListDirectoryTool",
    "ListPowerBITool",
    "ListSQLDatabaseTool",
    "ListSparkSQLTool",
    "MerriamWebsterQueryRun",
    "MetaphorSearchResults",
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
    "StructuredTool",
    "Tool",
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
    "authenticate",
    "format_tool_to_openai_function",
    "tool",
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
    "DataheraldTextToSQL": "langchain_community.tools.dataherald.tool",
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
