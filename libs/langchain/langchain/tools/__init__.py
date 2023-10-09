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

from langchain.tools.ainetwork.app import AINAppOps
from langchain.tools.ainetwork.owner import AINOwnerOps
from langchain.tools.ainetwork.rule import AINRuleOps
from langchain.tools.ainetwork.transfer import AINTransfer
from langchain.tools.ainetwork.value import AINValueOps
from langchain.tools.arxiv.tool import ArxivQueryRun
from langchain.tools.azure_cognitive_services import (
    AzureCogsFormRecognizerTool,
    AzureCogsImageAnalysisTool,
    AzureCogsSpeech2TextTool,
    AzureCogsText2SpeechTool,
)
from langchain.tools.base import BaseTool, StructuredTool, Tool, tool
from langchain.tools.bing_search.tool import BingSearchResults, BingSearchRun
from langchain.tools.brave_search.tool import BraveSearch
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain.tools.edenai import (
    EdenAiExplicitImageTool,
    EdenAiObjectDetectionTool,
    EdenAiParsingIDTool,
    EdenAiParsingInvoiceTool,
    EdenAiSpeechToTextTool,
    EdenAiTextModerationTool,
    EdenAiTextToSpeechTool,
    EdenaiTool,
)
from langchain.tools.eleven_labs.text2speech import ElevenLabsText2SpeechTool
from langchain.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)
from langchain.tools.gmail import (
    GmailCreateDraft,
    GmailGetMessage,
    GmailGetThread,
    GmailSearch,
    GmailSendMessage,
)
from langchain.tools.google_places.tool import GooglePlacesTool
from langchain.tools.google_search.tool import GoogleSearchResults, GoogleSearchRun
from langchain.tools.google_serper.tool import GoogleSerperResults, GoogleSerperRun
from langchain.tools.graphql.tool import BaseGraphQLTool
from langchain.tools.human.tool import HumanInputRun
from langchain.tools.ifttt import IFTTTWebhook
from langchain.tools.interaction.tool import StdInInquireTool
from langchain.tools.jira.tool import JiraAction
from langchain.tools.json.tool import JsonGetValueTool, JsonListKeysTool
from langchain.tools.metaphor_search import MetaphorSearchResults
from langchain.tools.office365.create_draft_message import O365CreateDraftMessage
from langchain.tools.office365.events_search import O365SearchEvents
from langchain.tools.office365.messages_search import O365SearchEmails
from langchain.tools.office365.send_event import O365SendEvent
from langchain.tools.office365.send_message import O365SendMessage
from langchain.tools.office365.utils import authenticate
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain.tools.openweathermap.tool import OpenWeatherMapQueryRun
from langchain.tools.playwright import (
    ClickTool,
    CurrentWebPageTool,
    ExtractHyperlinksTool,
    ExtractTextTool,
    GetElementsTool,
    NavigateBackTool,
    NavigateTool,
)
from langchain.tools.plugin import AIPluginTool
from langchain.tools.powerbi.tool import (
    InfoPowerBITool,
    ListPowerBITool,
    QueryPowerBITool,
)
from langchain.tools.pubmed.tool import PubmedQueryRun
from langchain.tools.python.tool import PythonAstREPLTool, PythonREPLTool
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools.requests.tool import (
    BaseRequestsTool,
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
)
from langchain.tools.scenexplain.tool import SceneXplainTool
from langchain.tools.searx_search.tool import SearxSearchResults, SearxSearchRun
from langchain.tools.shell.tool import ShellTool
from langchain.tools.sleep.tool import SleepTool
from langchain.tools.spark_sql.tool import (
    BaseSparkSQLTool,
    InfoSparkSQLTool,
    ListSparkSQLTool,
    QueryCheckerTool,
    QuerySparkSQLTool,
)
from langchain.tools.sql_database.tool import (
    BaseSQLDatabaseTool,
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain.tools.steamship_image_generation import SteamshipImageGenerationTool
from langchain.tools.vectorstore.tool import (
    VectorStoreQATool,
    VectorStoreQAWithSourcesTool,
)
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.tools.youtube.search import YouTubeSearchTool
from langchain.tools.zapier.tool import ZapierNLAListActions, ZapierNLARunAction

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
    "BingSearchResults",
    "BingSearchRun",
    "BraveSearch",
    "ClickTool",
    "CopyFileTool",
    "CurrentWebPageTool",
    "DeleteFileTool",
    "DuckDuckGoSearchResults",
    "DuckDuckGoSearchRun",
    "EdenAiExplicitImageTool",
    "EdenAiObjectDetectionTool",
    "EdenAiParsingIDTool",
    "EdenAiParsingInvoiceTool",
    "EdenAiTextToSpeechTool",
    "EdenAiSpeechToTextTool",
    "EdenAiTextModerationTool",
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
    "O365SearchEmails",
    "O365SearchEvents",
    "O365CreateDraftMessage",
    "O365SendMessage",
    "O365SendEvent",
    "authenticate",
    "OpenAPISpec",
    "OpenWeatherMapQueryRun",
    "PubmedQueryRun",
    "PythonAstREPLTool",
    "PythonREPLTool",
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
    "YouTubeSearchTool",
    "ZapierNLAListActions",
    "ZapierNLARunAction",
    "tool",
    "format_tool_to_openai_function",
]
