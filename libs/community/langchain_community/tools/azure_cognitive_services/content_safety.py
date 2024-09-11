from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class AzureContentSafetyTextTool(BaseTool):
    """
    A tool that interacts with the Azure AI Content Safety API.

    This tool queries the Azure AI Content Safety API to analyze text for harmful 
    content and identify sentiment. It requires an API key and endpoint, 
    which can be set up as described in the following guide:
    
    https://learn.microsoft.com/python/api/overview/azure/ai-contentsafety-readme?view=azure-python

    Attributes:
        content_safety_key (str): 
            The API key used to authenticate requests with Azure Content Safety API.
        content_safety_endpoint (str): 
            The endpoint URL for the Azure Content Safety API.
        content_safety_client (Any): 
            An instance of the Azure Content Safety Client used for making API requests.

    Methods:
        _sentiment_analysis(text: str) -> Dict:
            Analyzes the provided text to assess its sentiment and safety, 
            returning the analysis results.

        _run(query: str, 
            run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
            Uses the tool to analyze the given query and returns the result. 
            Raises a RuntimeError if an exception occurs.
    """

    content_safety_key: str = ""  #: :meta private:
    content_safety_endpoint: str = ""  #: :meta private:
    content_safety_client: Any  #: :meta private:

    name: str = "azure_content_safety_tool"
    description: str = (
        "A wrapper around Azure AI Content Safety. "
        '''Useful for when you need to identify the sentiment of text
        and whether or not a text is harmful. '''
        "Input should be text."
    )

    def __init__(
            self, 
                *,
            content_safety_key: Optional[str] = None,
            content_safety_endpoint: Optional[str] = None,
            ) -> None:
        """
        Initialize the AzureContentSafetyTextTool with the given API key and endpoint.

        This constructor sets up the API key and endpoint, and initializes
        the Azure Content Safety Client. If API key or endpoint is not provided,
        they are fetched from environment variables.

        Args:
            content_safety_key (Optional[str]): 
                The API key for Azure Content Safety API. If not provided, 
                it will be fetched from the environment 
                variable 'CONTENT_SAFETY_API_KEY'.
            content_safety_endpoint (Optional[str]): 
                The endpoint URL for Azure Content Safety API. If not provided, 
                it will be fetched from the environment 
                variable 'CONTENT_SAFETY_ENDPOINT'.

        Raises:
            ImportError: If the 'azure-ai-contentsafety' package is not installed.
            ValueError: 
                If API key or endpoint is not provided 
                and environment variables are missing.
        """
        content_safety_key = (content_safety_key or 
                                os.environ['CONTENT_SAFETY_API_KEY'])
        content_safety_endpoint = (content_safety_endpoint or
                                    os.environ['CONTENT_SAFETY_ENDPOINT'])
        try:
            import azure.ai.contentsafety as sdk
            from azure.core.credentials import AzureKeyCredential

            content_safety_client = sdk.ContentSafetyClient(
                endpoint=content_safety_endpoint,
                credential=AzureKeyCredential(content_safety_key),
            )

        except ImportError:
            raise ImportError(
                "azure-ai-contentsafety is not installed. "
                "Run `pip install azure-ai-contentsafety` to install."
            )
        super().__init__(content_safety_key=content_safety_key,
                        content_safety_endpoint=content_safety_endpoint,
                        content_safety_client=content_safety_client)

    def _sentiment_analysis(self, text: str) -> Dict:
        """
        Perform sentiment analysis on the provided text.

        This method uses the Azure Content Safety Client to analyze
        the text and determine its sentiment and safety categories.

        Args:
            text (str): The text to be analyzed.

        Returns:
            Dict: The analysis results containing sentiment and safety categories.
        """
        
        from azure.ai.contentsafety.models import AnalyzeTextOptions

        request = AnalyzeTextOptions(text=text)
        response = self.content_safety_client.analyze_text(request)
        result = response.categories_analysis
        return result

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Analyze the given query using the tool.

        This method calls `_sentiment_analysis` to process the 
        query and returns the result. It raises a RuntimeError if an 
        exception occurs during analysis.

        Args:
            query (str): 
                The query text to be analyzed.
            run_manager (Optional[CallbackManagerForToolRun], optional): 
                A callback manager for tracking the tool run. Defaults to None.

        Returns:
            str: The result of the sentiment analysis.

        Raises:
            RuntimeError: If an error occurs while running the tool.
        """
        try:
            return self._sentiment_analysis(query)
        except Exception as e:
            raise RuntimeError(
                f"Error while running AzureContentSafetyTextTool: {e}"
            )