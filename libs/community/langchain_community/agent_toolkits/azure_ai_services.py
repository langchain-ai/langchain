from __future__ import annotations

import sys
from typing import List

from langchain_core.tools import BaseTool

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.azure_ai_services import (
    AzureAiServicesDocumentIntelligenceTool,
    AzureAiServicesImageAnalysisTool,
    AzureAiServicesSpeechToTextTool,
    AzureAiServicesTextToSpeechTool,
    AzureAiServicesTextAnalyticsForHealthTool,
)


class AzureAiServicesToolkit(BaseToolkit):
    """Toolkit for Azure AI Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        tools: List[BaseTool] = [
            AzureAiServicesDocumentIntelligenceTool(),
            AzureAiServicesImageAnalysisTool(),
            AzureAiServicesSpeechToTextTool(),
            AzureAiServicesTextToSpeechTool(),
            AzureAiServicesTextAnalyticsForHealthTool(),
        ]
