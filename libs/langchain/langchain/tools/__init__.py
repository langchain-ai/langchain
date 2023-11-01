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
from typing import Any

from langchain.tools.base import BaseTool, StructuredTool, Tool, tool

# Used for internal purposes
_DEPRECATED_TOOLS = {"PythonAstREPLTool", "PythonREPLTool"}


def _import_ainetwork_app() -> Any:
    from langchain.tools.ainetwork.app import AINAppOps

    return AINAppOps


def _import_ainetwork_owner() -> Any:
    from langchain.tools.ainetwork.owner import AINOwnerOps

    return AINOwnerOps


def _import_ainetwork_rule() -> Any:
    from langchain.tools.ainetwork.rule import AINRuleOps

    return AINRuleOps


def _import_ainetwork_transfer() -> Any:
    from langchain.tools.ainetwork.transfer import AINTransfer

    return AINTransfer


def _import_ainetwork_value() -> Any:
    from langchain.tools.ainetwork.value import AINValueOps

    return AINValueOps


def _import_arxiv_tool() -> Any:
    from langchain.tools.arxiv.tool import ArxivQueryRun

    return ArxivQueryRun


def _import_azure_cognitive_services_AzureCogsFormRecognizerTool() -> Any:
    from langchain.tools.azure_cognitive_services import AzureCogsFormRecognizerTool

    return AzureCogsFormRecognizerTool


def _import_azure_cognitive_services_AzureCogsImageAnalysisTool() -> Any:
    from langchain.tools.azure_cognitive_services import AzureCogsImageAnalysisTool

    return AzureCogsImageAnalysisTool


def _import_azure_cognitive_services_AzureCogsSpeech2TextTool() -> Any:
    from langchain.tools.azure_cognitive_services import AzureCogsSpeech2TextTool

    return AzureCogsSpeech2TextTool


def _import_azure_cognitive_services_AzureCogsText2SpeechTool() -> Any:
    from langchain.tools.azure_cognitive_services import AzureCogsText2SpeechTool

    return AzureCogsText2SpeechTool


def _import_bing_search_tool_BingSearchResults() -> Any:
    from langchain.tools.bing_search.tool import BingSearchResults

    return BingSearchResults


def _import_bing_search_tool_BingSearchRun() -> Any:
    from langchain.tools.bing_search.tool import BingSearchRun

    return BingSearchRun


def _import_brave_search_tool() -> Any:
    from langchain.tools.brave_search.tool import BraveSearch

    return BraveSearch


def _import_ddg_search_tool_DuckDuckGoSearchResults() -> Any:
    from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults

    return DuckDuckGoSearchResults


def _import_ddg_search_tool_DuckDuckGoSearchRun() -> Any:
    from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun

    return DuckDuckGoSearchRun


def _import_edenai_EdenAiExplicitImageTool() -> Any:
    from langchain.tools.edenai import EdenAiExplicitImageTool

    return EdenAiExplicitImageTool


def _import_edenai_EdenAiObjectDetectionTool() -> Any:
    from langchain.tools.edenai import EdenAiObjectDetectionTool

    return EdenAiObjectDetectionTool


def _import_edenai_EdenAiParsingIDTool() -> Any:
    from langchain.tools.edenai import EdenAiParsingIDTool

    return EdenAiParsingIDTool


def _import_edenai_EdenAiParsingInvoiceTool() -> Any:
    from langchain.tools.edenai import EdenAiParsingInvoiceTool

    return EdenAiParsingInvoiceTool


def _import_edenai_EdenAiSpeechToTextTool() -> Any:
    from langchain.tools.edenai import EdenAiSpeechToTextTool

    return EdenAiSpeechToTextTool


def _import_edenai_EdenAiTextModerationTool() -> Any:
    from langchain.tools.edenai import EdenAiTextModerationTool

    return EdenAiTextModerationTool


def _import_edenai_EdenAiTextToSpeechTool() -> Any:
    from langchain.tools.edenai import EdenAiTextToSpeechTool

    return EdenAiTextToSpeechTool


def _import_edenai_EdenaiTool() -> Any:
    from langchain.tools.edenai import EdenaiTool

    return EdenaiTool


def _import_eleven_labs_text2speech() -> Any:
    from langchain.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool

    return ElevenLabsText2SpeechTool


def _import_file_management_CopyFileTool() -> Any:
    from langchain.tools.file_management import CopyFileTool

    return CopyFileTool


def _import_file_management_DeleteFileTool() -> Any:
    from langchain.tools.file_management import DeleteFileTool

    return DeleteFileTool


def _import_file_management_FileSearchTool() -> Any:
    from langchain.tools.file_management import FileSearchTool

    return FileSearchTool


def _import_file_management_ListDirectoryTool() -> Any:
    from langchain.tools.file_management import ListDirectoryTool

    return ListDirectoryTool


def _import_file_management_MoveFileTool() -> Any:
    from langchain.tools.file_management import MoveFileTool

    return MoveFileTool


def _import_file_management_ReadFileTool() -> Any:
    from langchain.tools.file_management import ReadFileTool

    return ReadFileTool


def _import_file_management_WriteFileTool() -> Any:
    from langchain.tools.file_management import WriteFileTool

    return WriteFileTool


def _import_gmail_GmailCreateDraft() -> Any:
    from langchain.tools.gmail import GmailCreateDraft

    return GmailCreateDraft


def _import_gmail_GmailGetMessage() -> Any:
    from langchain.tools.gmail import GmailGetMessage

    return GmailGetMessage


def _import_gmail_GmailGetThread() -> Any:
    from langchain.tools.gmail import GmailGetThread

    return GmailGetThread


def _import_gmail_GmailSearch() -> Any:
    from langchain.tools.gmail import GmailSearch

    return GmailSearch


def _import_gmail_GmailSendMessage() -> Any:
    from langchain.tools.gmail import GmailSendMessage

    return GmailSendMessage


def _import_google_cloud_texttospeech() -> Any:
    from langchain.tools.google_cloud.texttospeech import GoogleCloudTextToSpeechTool

    return GoogleCloudTextToSpeechTool


def _import_google_places_tool() -> Any:
    from langchain.tools.google_places.tool import GooglePlacesTool

    return GooglePlacesTool


def _import_google_search_tool_GoogleSearchResults() -> Any:
    from langchain.tools.google_search.tool import GoogleSearchResults

    return GoogleSearchResults


def _import_google_search_tool_GoogleSearchRun() -> Any:
    from langchain.tools.google_search.tool import GoogleSearchRun

    return GoogleSearchRun


def _import_google_serper_tool_GoogleSerperResults() -> Any:
    from langchain.tools.google_serper.tool import GoogleSerperResults

    return GoogleSerperResults


def _import_google_serper_tool_GoogleSerperRun() -> Any:
    from langchain.tools.google_serper.tool import GoogleSerperRun

    return GoogleSerperRun


def _import_graphql_tool() -> Any:
    from langchain.tools.graphql.tool import BaseGraphQLTool

    return BaseGraphQLTool


def _import_human_tool() -> Any:
    from langchain.tools.human.tool import HumanInputRun

    return HumanInputRun


def _import_ifttt() -> Any:
    from langchain.tools.ifttt import IFTTTWebhook

    return IFTTTWebhook


def _import_interaction_tool() -> Any:
    from langchain.tools.interaction.tool import StdInInquireTool

    return StdInInquireTool


def _import_jira_tool() -> Any:
    from langchain.tools.jira.tool import JiraAction

    return JiraAction


def _import_json_tool_JsonGetValueTool() -> Any:
    from langchain.tools.json.tool import JsonGetValueTool

    return JsonGetValueTool


def _import_json_tool_JsonListKeysTool() -> Any:
    from langchain.tools.json.tool import JsonListKeysTool

    return JsonListKeysTool


def _import_metaphor_search() -> Any:
    from langchain.tools.metaphor_search import MetaphorSearchResults

    return MetaphorSearchResults


def _import_office365_create_draft_message() -> Any:
    from langchain.tools.office365.create_draft_message import O365CreateDraftMessage

    return O365CreateDraftMessage


def _import_office365_events_search() -> Any:
    from langchain.tools.office365.events_search import O365SearchEvents

    return O365SearchEvents


def _import_office365_messages_search() -> Any:
    from langchain.tools.office365.messages_search import O365SearchEmails

    return O365SearchEmails


def _import_office365_send_event() -> Any:
    from langchain.tools.office365.send_event import O365SendEvent

    return O365SendEvent


def _import_office365_send_message() -> Any:
    from langchain.tools.office365.send_message import O365SendMessage

    return O365SendMessage


def _import_office365_utils() -> Any:
    from langchain.tools.office365.utils import authenticate

    return authenticate


def _import_openapi_utils_api_models() -> Any:
    from langchain.tools.openapi.utils.api_models import APIOperation

    return APIOperation


def _import_openapi_utils_openapi_utils() -> Any:
    from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec

    return OpenAPISpec


def _import_openweathermap_tool() -> Any:
    from langchain.tools.openweathermap.tool import OpenWeatherMapQueryRun

    return OpenWeatherMapQueryRun


def _import_playwright_ClickTool() -> Any:
    from langchain.tools.playwright import ClickTool

    return ClickTool


def _import_playwright_CurrentWebPageTool() -> Any:
    from langchain.tools.playwright import CurrentWebPageTool

    return CurrentWebPageTool


def _import_playwright_ExtractHyperlinksTool() -> Any:
    from langchain.tools.playwright import ExtractHyperlinksTool

    return ExtractHyperlinksTool


def _import_playwright_ExtractTextTool() -> Any:
    from langchain.tools.playwright import ExtractTextTool

    return ExtractTextTool


def _import_playwright_GetElementsTool() -> Any:
    from langchain.tools.playwright import GetElementsTool

    return GetElementsTool


def _import_playwright_NavigateBackTool() -> Any:
    from langchain.tools.playwright import NavigateBackTool

    return NavigateBackTool


def _import_playwright_NavigateTool() -> Any:
    from langchain.tools.playwright import NavigateTool

    return NavigateTool


def _import_plugin() -> Any:
    from langchain.tools.plugin import AIPluginTool

    return AIPluginTool


def _import_powerbi_tool_InfoPowerBITool() -> Any:
    from langchain.tools.powerbi.tool import InfoPowerBITool

    return InfoPowerBITool


def _import_powerbi_tool_ListPowerBITool() -> Any:
    from langchain.tools.powerbi.tool import ListPowerBITool

    return ListPowerBITool


def _import_powerbi_tool_QueryPowerBITool() -> Any:
    from langchain.tools.powerbi.tool import QueryPowerBITool

    return QueryPowerBITool


def _import_pubmed_tool() -> Any:
    from langchain.tools.pubmed.tool import PubmedQueryRun

    return PubmedQueryRun


def _import_python_tool_PythonAstREPLTool() -> Any:
    raise ImportError(
        "This tool has been moved to langchain experiment. "
        "This tool has access to a python REPL. "
        "For best practices make sure to sandbox this tool. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "To keep using this code as is, install langchain experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )


def _import_python_tool_PythonREPLTool() -> Any:
    raise ImportError(
        "This tool has been moved to langchain experiment. "
        "This tool has access to a python REPL. "
        "For best practices make sure to sandbox this tool. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "To keep using this code as is, install langchain experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )


def _import_render() -> Any:
    from langchain.tools.render import format_tool_to_openai_function

    return format_tool_to_openai_function


def _import_requests_tool_BaseRequestsTool() -> Any:
    from langchain.tools.requests.tool import BaseRequestsTool

    return BaseRequestsTool


def _import_requests_tool_RequestsDeleteTool() -> Any:
    from langchain.tools.requests.tool import RequestsDeleteTool

    return RequestsDeleteTool


def _import_requests_tool_RequestsGetTool() -> Any:
    from langchain.tools.requests.tool import RequestsGetTool

    return RequestsGetTool


def _import_requests_tool_RequestsPatchTool() -> Any:
    from langchain.tools.requests.tool import RequestsPatchTool

    return RequestsPatchTool


def _import_requests_tool_RequestsPostTool() -> Any:
    from langchain.tools.requests.tool import RequestsPostTool

    return RequestsPostTool


def _import_requests_tool_RequestsPutTool() -> Any:
    from langchain.tools.requests.tool import RequestsPutTool

    return RequestsPutTool


def _import_scenexplain_tool() -> Any:
    from langchain.tools.scenexplain.tool import SceneXplainTool

    return SceneXplainTool


def _import_searx_search_tool_SearxSearchResults() -> Any:
    from langchain.tools.searx_search.tool import SearxSearchResults

    return SearxSearchResults


def _import_searx_search_tool_SearxSearchRun() -> Any:
    from langchain.tools.searx_search.tool import SearxSearchRun

    return SearxSearchRun


def _import_shell_tool() -> Any:
    from langchain.tools.shell.tool import ShellTool

    return ShellTool


def _import_sleep_tool() -> Any:
    from langchain.tools.sleep.tool import SleepTool

    return SleepTool


def _import_spark_sql_tool_BaseSparkSQLTool() -> Any:
    from langchain.tools.spark_sql.tool import BaseSparkSQLTool

    return BaseSparkSQLTool


def _import_spark_sql_tool_InfoSparkSQLTool() -> Any:
    from langchain.tools.spark_sql.tool import InfoSparkSQLTool

    return InfoSparkSQLTool


def _import_spark_sql_tool_ListSparkSQLTool() -> Any:
    from langchain.tools.spark_sql.tool import ListSparkSQLTool

    return ListSparkSQLTool


def _import_spark_sql_tool_QueryCheckerTool() -> Any:
    from langchain.tools.spark_sql.tool import QueryCheckerTool

    return QueryCheckerTool


def _import_spark_sql_tool_QuerySparkSQLTool() -> Any:
    from langchain.tools.spark_sql.tool import QuerySparkSQLTool

    return QuerySparkSQLTool


def _import_sql_database_tool_BaseSQLDatabaseTool() -> Any:
    from langchain.tools.sql_database.tool import BaseSQLDatabaseTool

    return BaseSQLDatabaseTool


def _import_sql_database_tool_InfoSQLDatabaseTool() -> Any:
    from langchain.tools.sql_database.tool import InfoSQLDatabaseTool

    return InfoSQLDatabaseTool


def _import_sql_database_tool_ListSQLDatabaseTool() -> Any:
    from langchain.tools.sql_database.tool import ListSQLDatabaseTool

    return ListSQLDatabaseTool


def _import_sql_database_tool_QuerySQLCheckerTool() -> Any:
    from langchain.tools.sql_database.tool import QuerySQLCheckerTool

    return QuerySQLCheckerTool


def _import_sql_database_tool_QuerySQLDataBaseTool() -> Any:
    from langchain.tools.sql_database.tool import QuerySQLDataBaseTool

    return QuerySQLDataBaseTool


def _import_steamship_image_generation() -> Any:
    from langchain.tools.steamship_image_generation import SteamshipImageGenerationTool

    return SteamshipImageGenerationTool


def _import_vectorstore_tool_VectorStoreQATool() -> Any:
    from langchain.tools.vectorstore.tool import VectorStoreQATool

    return VectorStoreQATool


def _import_vectorstore_tool_VectorStoreQAWithSourcesTool() -> Any:
    from langchain.tools.vectorstore.tool import VectorStoreQAWithSourcesTool

    return VectorStoreQAWithSourcesTool


def _import_wikipedia_tool() -> Any:
    from langchain.tools.wikipedia.tool import WikipediaQueryRun

    return WikipediaQueryRun


def _import_wolfram_alpha_tool() -> Any:
    from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun

    return WolframAlphaQueryRun


def _import_yahoo_finance_news() -> Any:
    from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

    return YahooFinanceNewsTool


def _import_youtube_search() -> Any:
    from langchain.tools.youtube.search import YouTubeSearchTool

    return YouTubeSearchTool


def _import_zapier_tool_ZapierNLAListActions() -> Any:
    from langchain.tools.zapier.tool import ZapierNLAListActions

    return ZapierNLAListActions


def _import_zapier_tool_ZapierNLARunAction() -> Any:
    from langchain.tools.zapier.tool import ZapierNLARunAction

    return ZapierNLARunAction


def _import_bearly_tool() -> Any:
    from langchain.tools.bearly.tool import BearlyInterpreterTool

    return BearlyInterpreterTool


def _import_e2b_data_analysis() -> Any:
    from langchain.tools.e2b_data_analysis.tool import E2BDataAnalysisTool

    return E2BDataAnalysisTool


def __getattr__(name: str) -> Any:
    if name == "AINAppOps":
        return _import_ainetwork_app()
    elif name == "AINOwnerOps":
        return _import_ainetwork_owner()
    elif name == "AINRuleOps":
        return _import_ainetwork_rule()
    elif name == "AINTransfer":
        return _import_ainetwork_transfer()
    elif name == "AINValueOps":
        return _import_ainetwork_value()
    elif name == "ArxivQueryRun":
        return _import_arxiv_tool()
    elif name == "AzureCogsFormRecognizerTool":
        return _import_azure_cognitive_services_AzureCogsFormRecognizerTool()
    elif name == "AzureCogsImageAnalysisTool":
        return _import_azure_cognitive_services_AzureCogsImageAnalysisTool()
    elif name == "AzureCogsSpeech2TextTool":
        return _import_azure_cognitive_services_AzureCogsSpeech2TextTool()
    elif name == "AzureCogsText2SpeechTool":
        return _import_azure_cognitive_services_AzureCogsText2SpeechTool()
    elif name == "BingSearchResults":
        return _import_bing_search_tool_BingSearchResults()
    elif name == "BingSearchRun":
        return _import_bing_search_tool_BingSearchRun()
    elif name == "BraveSearch":
        return _import_brave_search_tool()
    elif name == "DuckDuckGoSearchResults":
        return _import_ddg_search_tool_DuckDuckGoSearchResults()
    elif name == "DuckDuckGoSearchRun":
        return _import_ddg_search_tool_DuckDuckGoSearchRun()
    elif name == "EdenAiExplicitImageTool":
        return _import_edenai_EdenAiExplicitImageTool()
    elif name == "EdenAiObjectDetectionTool":
        return _import_edenai_EdenAiObjectDetectionTool()
    elif name == "EdenAiParsingIDTool":
        return _import_edenai_EdenAiParsingIDTool()
    elif name == "EdenAiParsingInvoiceTool":
        return _import_edenai_EdenAiParsingInvoiceTool()
    elif name == "EdenAiSpeechToTextTool":
        return _import_edenai_EdenAiSpeechToTextTool()
    elif name == "EdenAiTextModerationTool":
        return _import_edenai_EdenAiTextModerationTool()
    elif name == "EdenAiTextToSpeechTool":
        return _import_edenai_EdenAiTextToSpeechTool()
    elif name == "EdenaiTool":
        return _import_edenai_EdenaiTool()
    elif name == "ElevenLabsText2SpeechTool":
        return _import_eleven_labs_text2speech()
    elif name == "CopyFileTool":
        return _import_file_management_CopyFileTool()
    elif name == "DeleteFileTool":
        return _import_file_management_DeleteFileTool()
    elif name == "FileSearchTool":
        return _import_file_management_FileSearchTool()
    elif name == "ListDirectoryTool":
        return _import_file_management_ListDirectoryTool()
    elif name == "MoveFileTool":
        return _import_file_management_MoveFileTool()
    elif name == "ReadFileTool":
        return _import_file_management_ReadFileTool()
    elif name == "WriteFileTool":
        return _import_file_management_WriteFileTool()
    elif name == "GmailCreateDraft":
        return _import_gmail_GmailCreateDraft()
    elif name == "GmailGetMessage":
        return _import_gmail_GmailGetMessage()
    elif name == "GmailGetThread":
        return _import_gmail_GmailGetThread()
    elif name == "GmailSearch":
        return _import_gmail_GmailSearch()
    elif name == "GmailSendMessage":
        return _import_gmail_GmailSendMessage()
    elif name == "GoogleCloudTextToSpeechTool":
        return _import_google_cloud_texttospeech()
    elif name == "GooglePlacesTool":
        return _import_google_places_tool()
    elif name == "GoogleSearchResults":
        return _import_google_search_tool_GoogleSearchResults()
    elif name == "GoogleSearchRun":
        return _import_google_search_tool_GoogleSearchRun()
    elif name == "GoogleSerperResults":
        return _import_google_serper_tool_GoogleSerperResults()
    elif name == "GoogleSerperRun":
        return _import_google_serper_tool_GoogleSerperRun()
    elif name == "BaseGraphQLTool":
        return _import_graphql_tool()
    elif name == "HumanInputRun":
        return _import_human_tool()
    elif name == "IFTTTWebhook":
        return _import_ifttt()
    elif name == "StdInInquireTool":
        return _import_interaction_tool()
    elif name == "JiraAction":
        return _import_jira_tool()
    elif name == "JsonGetValueTool":
        return _import_json_tool_JsonGetValueTool()
    elif name == "JsonListKeysTool":
        return _import_json_tool_JsonListKeysTool()
    elif name == "MetaphorSearchResults":
        return _import_metaphor_search()
    elif name == "O365CreateDraftMessage":
        return _import_office365_create_draft_message()
    elif name == "O365SearchEvents":
        return _import_office365_events_search()
    elif name == "O365SearchEmails":
        return _import_office365_messages_search()
    elif name == "O365SendEvent":
        return _import_office365_send_event()
    elif name == "O365SendMessage":
        return _import_office365_send_message()
    elif name == "authenticate":
        return _import_office365_utils()
    elif name == "APIOperation":
        return _import_openapi_utils_api_models()
    elif name == "OpenAPISpec":
        return _import_openapi_utils_openapi_utils()
    elif name == "OpenWeatherMapQueryRun":
        return _import_openweathermap_tool()
    elif name == "ClickTool":
        return _import_playwright_ClickTool()
    elif name == "CurrentWebPageTool":
        return _import_playwright_CurrentWebPageTool()
    elif name == "ExtractHyperlinksTool":
        return _import_playwright_ExtractHyperlinksTool()
    elif name == "ExtractTextTool":
        return _import_playwright_ExtractTextTool()
    elif name == "GetElementsTool":
        return _import_playwright_GetElementsTool()
    elif name == "NavigateBackTool":
        return _import_playwright_NavigateBackTool()
    elif name == "NavigateTool":
        return _import_playwright_NavigateTool()
    elif name == "AIPluginTool":
        return _import_plugin()
    elif name == "InfoPowerBITool":
        return _import_powerbi_tool_InfoPowerBITool()
    elif name == "ListPowerBITool":
        return _import_powerbi_tool_ListPowerBITool()
    elif name == "QueryPowerBITool":
        return _import_powerbi_tool_QueryPowerBITool()
    elif name == "PubmedQueryRun":
        return _import_pubmed_tool()
    elif name == "PythonAstREPLTool":
        return _import_python_tool_PythonAstREPLTool()
    elif name == "PythonREPLTool":
        return _import_python_tool_PythonREPLTool()
    elif name == "format_tool_to_openai_function":
        return _import_render()
    elif name == "BaseRequestsTool":
        return _import_requests_tool_BaseRequestsTool()
    elif name == "RequestsDeleteTool":
        return _import_requests_tool_RequestsDeleteTool()
    elif name == "RequestsGetTool":
        return _import_requests_tool_RequestsGetTool()
    elif name == "RequestsPatchTool":
        return _import_requests_tool_RequestsPatchTool()
    elif name == "RequestsPostTool":
        return _import_requests_tool_RequestsPostTool()
    elif name == "RequestsPutTool":
        return _import_requests_tool_RequestsPutTool()
    elif name == "SceneXplainTool":
        return _import_scenexplain_tool()
    elif name == "SearxSearchResults":
        return _import_searx_search_tool_SearxSearchResults()
    elif name == "SearxSearchRun":
        return _import_searx_search_tool_SearxSearchRun()
    elif name == "ShellTool":
        return _import_shell_tool()
    elif name == "SleepTool":
        return _import_sleep_tool()
    elif name == "BaseSparkSQLTool":
        return _import_spark_sql_tool_BaseSparkSQLTool()
    elif name == "InfoSparkSQLTool":
        return _import_spark_sql_tool_InfoSparkSQLTool()
    elif name == "ListSparkSQLTool":
        return _import_spark_sql_tool_ListSparkSQLTool()
    elif name == "QueryCheckerTool":
        return _import_spark_sql_tool_QueryCheckerTool()
    elif name == "QuerySparkSQLTool":
        return _import_spark_sql_tool_QuerySparkSQLTool()
    elif name == "BaseSQLDatabaseTool":
        return _import_sql_database_tool_BaseSQLDatabaseTool()
    elif name == "InfoSQLDatabaseTool":
        return _import_sql_database_tool_InfoSQLDatabaseTool()
    elif name == "ListSQLDatabaseTool":
        return _import_sql_database_tool_ListSQLDatabaseTool()
    elif name == "QuerySQLCheckerTool":
        return _import_sql_database_tool_QuerySQLCheckerTool()
    elif name == "QuerySQLDataBaseTool":
        return _import_sql_database_tool_QuerySQLDataBaseTool()
    elif name == "SteamshipImageGenerationTool":
        return _import_steamship_image_generation()
    elif name == "VectorStoreQATool":
        return _import_vectorstore_tool_VectorStoreQATool()
    elif name == "VectorStoreQAWithSourcesTool":
        return _import_vectorstore_tool_VectorStoreQAWithSourcesTool()
    elif name == "WikipediaQueryRun":
        return _import_wikipedia_tool()
    elif name == "WolframAlphaQueryRun":
        return _import_wolfram_alpha_tool()
    elif name == "YahooFinanceNewsTool":
        return _import_yahoo_finance_news()
    elif name == "YouTubeSearchTool":
        return _import_youtube_search()
    elif name == "ZapierNLAListActions":
        return _import_zapier_tool_ZapierNLAListActions()
    elif name == "ZapierNLARunAction":
        return _import_zapier_tool_ZapierNLARunAction()
    elif name == "BearlyInterpreterTool":
        return _import_bearly_tool()
    elif name == "E2BDataAnalysisTool":
        return _import_e2b_data_analysis()
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "AINAppOps",
    "AINOwnerOps",
    "AINRuleOps",
    "AINTransfer",
    "AINValueOps",
    "AIPluginTool",
    "APIOperation",
    "ArxivQueryRun",
    "AzureCogsFormRecognizerTool",
    "AzureCogsImageAnalysisTool",
    "AzureCogsSpeech2TextTool",
    "AzureCogsText2SpeechTool",
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
    "MetaphorSearchResults",
    "MoveFileTool",
    "NavigateBackTool",
    "NavigateTool",
    "O365CreateDraftMessage",
    "O365SearchEmails",
    "O365SearchEvents",
    "O365SendEvent",
    "O365SendMessage",
    "OpenAPISpec",
    "OpenWeatherMapQueryRun",
    "PubmedQueryRun",
    "QueryCheckerTool",
    "QueryPowerBITool",
    "QuerySQLCheckerTool",
    "QuerySQLDataBaseTool",
    "QuerySparkSQLTool",
    "ReadFileTool",
    "RequestsDeleteTool",
    "RequestsGetTool",
    "RequestsPatchTool",
    "RequestsPostTool",
    "RequestsPutTool",
    "SceneXplainTool",
    "SearxSearchResults",
    "SearxSearchRun",
    "ShellTool",
    "SleepTool",
    "StdInInquireTool",
    "SteamshipImageGenerationTool",
    "StructuredTool",
    "Tool",
    "VectorStoreQATool",
    "VectorStoreQAWithSourcesTool",
    "WikipediaQueryRun",
    "WolframAlphaQueryRun",
    "WriteFileTool",
    "YahooFinanceNewsTool",
    "YouTubeSearchTool",
    "ZapierNLAListActions",
    "ZapierNLARunAction",
    "authenticate",
    "format_tool_to_openai_function",
    "tool",
]
