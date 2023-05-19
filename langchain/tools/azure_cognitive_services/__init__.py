"""Azure Cognitive Services Tools."""

from langchain.tools.azure_cognitive_services.image_analysis import AzureCogsImageAnalysisTool
from langchain.tools.azure_cognitive_services.form_recognizer import AzureCogsFormRecognizerTool

__all__ = [
    "AzureCogsImageAnalysisTool",
    "AzureCogsFormRecognizerTool",
]