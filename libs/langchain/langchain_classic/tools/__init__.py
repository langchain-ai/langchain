"""**Tools** are classes that an Agent uses to interact with the world.

Each tool has a **description**. Agent uses the description to choose the right
tool for the job.
"""

import warnings
from typing import Any

from langchain_core._api import LangChainDeprecationWarning
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

from langchain_classic._api.interactive_env import is_interactive_env

# Used for internal purposes
_DEPRECATED_TOOLS = {"PythonAstREPLTool", "PythonREPLTool"}


def _import_python_tool_python_ast_repl_tool() -> Any:
    msg = (
        "This tool has been moved to langchain experiment. "
        "This tool has access to a python REPL. "
        "For best practices make sure to sandbox this tool. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "To keep using this code as is, install langchain experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )
    raise ImportError(msg)


def _import_python_tool_python_repl_tool() -> Any:
    msg = (
        "This tool has been moved to langchain experiment. "
        "This tool has access to a python REPL. "
        "For best practices make sure to sandbox this tool. "
        "Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md "
        "To keep using this code as is, install langchain experimental and "
        "update relevant imports replacing 'langchain' with 'langchain_experimental'"
    )
    raise ImportError(msg)


def __getattr__(name: str) -> Any:
    if name == "PythonAstREPLTool":
        return _import_python_tool_python_ast_repl_tool()
    if name == "PythonREPLTool":
        return _import_python_tool_python_repl_tool()
    from langchain_community import tools

    # If not in interactive env, raise warning.
    if not is_interactive_env():
        warnings.warn(
            "Importing tools from langchain is deprecated. Importing from "
            "langchain will no longer be supported as of langchain==0.2.0. "
            "Please import from langchain-community instead:\n\n"
            f"`from langchain_community.tools import {name}`.\n\n"
            "To install langchain-community run "
            "`pip install -U langchain-community`.",
            stacklevel=2,
            category=LangChainDeprecationWarning,
        )

    return getattr(tools, name)


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
    "PubmedQueryRun",
    "QueryCheckerTool",
    "QueryPowerBITool",
    "QuerySQLCheckerTool",
    "QuerySQLDataBaseTool",
    "QuerySparkSQLTool",
    "ReadFileTool",
    "RedditSearchRun",
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
    "YouTubeSearchTool",
    "ZapierNLAListActions",
    "ZapierNLARunAction",
    "format_tool_to_openai_function",
    "tool",
]
