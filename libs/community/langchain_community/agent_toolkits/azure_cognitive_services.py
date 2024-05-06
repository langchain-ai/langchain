from __future__ import annotations

import sys
from typing import List

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_community.tools.azure_cognitive_services import (
    AzureCogsFormRecognizerTool,
    AzureCogsImageAnalysisTool,
    AzureCogsSpeech2TextTool,
    AzureCogsText2SpeechTool,
    AzureCogsTextAnalyticsHealthTool,
)


class AzureCognitiveServicesToolkit(BaseToolkit):
    """Toolkit for Azure Cognitive Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        tools: List[BaseTool] = [
            AzureCogsFormRecognizerTool(),
            AzureCogsSpeech2TextTool(),
            AzureCogsText2SpeechTool(),
            AzureCogsTextAnalyticsHealthTool(),
        ]

        # TODO: Remove check once azure-ai-vision supports MacOS.
        if sys.platform.startswith("linux") or sys.platform.startswith("win"):
            tools.append(AzureCogsImageAnalysisTool())
        return tools
