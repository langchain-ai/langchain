"""Azure AI Services Tools."""

from langchain_community.tools.azure_ai_services.form_recognizer import (
    AzureAiServicesDocumentIntelligenceTool,
)
from langchain_community.tools.azure_ai_services.image_analysis import (
    AzureAiServicesImageAnalysisTool,
)
from libs.community.langchain_community.tools.azure_ai_services.speech_to_text import (
    AzureAiServicesSpeechToTextTool,
)
from libs.community.langchain_community.tools.azure_ai_services.text_to_speech import (
    AzureAiServicesTextToSpeechTool,
)
from libs.community.langchain_community.tools.azure_ai_services.text_analytics_for_health import (
    AzureAiServicesTextAnalyticsForHealthTool,
)

__all__ = [
    "AzureAiServicesImageAnalysisTool",
    "AzureAiServicesDocumentIntelligenceTool",
    "AzureAiServicesSpeechToTextTool",
    "AzureAiServicesTextToSpeechTool",
    "AzureAiServicesTextAnalyticsForHealthTool",
]
