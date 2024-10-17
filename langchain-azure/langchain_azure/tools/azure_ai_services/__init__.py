"""Azure AI Services Tools."""

from langchain_community.tools.azure_ai_services.document_intelligence import (
    AzureAiServicesDocumentIntelligenceTool,
)
from langchain_community.tools.azure_ai_services.image_analysis import (
    AzureAiServicesImageAnalysisTool,
)
from langchain_community.tools.azure_ai_services.speech_to_text import (
    AzureAiServicesSpeechToTextTool,
)
from langchain_community.tools.azure_ai_services.text_analytics_for_health import (
    AzureAiServicesTextAnalyticsForHealthTool,
)
from langchain_community.tools.azure_ai_services.text_to_speech import (
    AzureAiServicesTextToSpeechTool,
)

__all__ = [
    "AzureAiServicesDocumentIntelligenceTool",
    "AzureAiServicesImageAnalysisTool",
    "AzureAiServicesSpeechToTextTool",
    "AzureAiServicesTextToSpeechTool",
    "AzureAiServicesTextAnalyticsForHealthTool",
]