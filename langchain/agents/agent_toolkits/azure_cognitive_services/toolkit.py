from __future__ import annotations

from typing import List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools.azure_cognitive_services import (
    AzureCogsImageAnalysisTool,
    AzureCogsFormRecognizerTool,
    AzureCogsSpeech2TextTool,
    AzureCogsText2SpeechTool,
)
from langchain.tools.base import BaseTool


class AzureCognitiveServicesToolkit(BaseToolkit):
    """Toolkit for Azure Cognitive Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            AzureCogsImageAnalysisTool(),
            AzureCogsFormRecognizerTool(),
            AzureCogsSpeech2TextTool(),
            AzureCogsText2SpeechTool(),
        ]