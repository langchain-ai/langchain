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

from langchain_core.tools import BaseTool, StructuredTool, Tool, tool

# Used for internal purposes
_DEPRECATED_TOOLS = {"PythonAstREPLTool", "PythonREPLTool"}


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


DEPRECATED_IMPORTS = [
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
    "SearchAPIResults",
    "SearchAPIRun",
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
    "RedditSearchRun",
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
    "SteamWebAPIQueryRun",
    "SceneXplainTool",
    "SearxSearchResults",
    "SearxSearchRun",
    "ShellTool",
    "SlackGetChannel",
    "SlackGetMessage",
    "SlackScheduleMessage",
    "SlackSendMessage",
    "SleepTool",
    "StdInInquireTool",
    "StackExchangeTool",
    "SteamshipImageGenerationTool",
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
]


def __getattr__(name: str) -> Any:
    if name == "PythonAstREPLTool":
        return _import_python_tool_PythonAstREPLTool()
    elif name == "PythonREPLTool":
        return _import_python_tool_PythonREPLTool()
    else:
        if name in DEPRECATED_IMPORTS:
            raise ImportError(
                f"{name} has been moved to the langchain-community package. "
                f"See https://github.com/langchain-ai/langchain/discussions/19083 for more "
                f"information.\n\nTo use it install langchain-community:\n\n"
                f"`pip install -U langchain-community`\n\n"
                f"then import with:\n\n"
                f"`from langchain_community.tools import {name}`"
            )

    raise AttributeError()


__all__ = [
    "StructuredTool",
    "Tool",
    "BaseTool",
    "tool",
]
