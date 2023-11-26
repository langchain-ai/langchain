"""Azure Cognitive Services Tools."""

from langchain.tools.azure_cognitive_services.form_recognizer import (
    AzureCogsFormRecognizerTool,
)
from langchain.tools.azure_cognitive_services.image_analysis import (
    AzureCogsImageAnalysisTool,
)
from langchain.tools.azure_cognitive_services.speech2text import (
    AzureCogsSpeech2TextTool,
)
from langchain.tools.azure_cognitive_services.text2speech import (
    AzureCogsText2SpeechTool,
)
from langchain.tools.azure_cognitive_services.text_analytics_health import (
    AzureCogsTextAnalyticsHealthTool,
)

__all__ = [
    "AzureCogsImageAnalysisTool",
    "AzureCogsFormRecognizerTool",
    "AzureCogsSpeech2TextTool",
    "AzureCogsText2SpeechTool",
    "AzureCogsTextAnalyticsHealthTool",
]
