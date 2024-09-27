from __future__ import annotations

import os
import logging
import docx

from typing import Optional
from azure.ai.translation.text.models import InputTextItem
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient

logger = logging.getLogger(__name__)

class AzureDocumentTranslateTool:
    """
    A tool that uses Azure Text Translation API to translate a text document from
    any language into a target language.
    """

    text_translation_key: str = ""
    text_translation_region: str = ""
    text_translation_endpoint: str = ""
    client: Optional[TextTranslationClient] = None

    name: str = "azure_document_translation"
    description: str = (
        """
        This tool can be used if you want to translate a document into a specific language.
        It reads the text froma file, processes it and then outputs with the desired language.
        """
    )

    def __init__(self, *,
                 text_translation_key: Optional[str] = None,
                 text_translation_region: Optional[str] = None,
                 text_translation_endpoint: Optional[str] = None
                 ) -> None:
        super().__init__()
        self.text_translation_key = text_translation_key or os.getenv("TEXT_TRANSLATION_KEY")
        self.text_translation_region = text_translation_region or os.getenv("TEXT_TRANSLATION_REGION")
        self.text_translation_endpoint = text_translation_endpoint or os.getenv("TEXT_TRANSLATION_ENDPOINT")

        if not all([self.text_translation_key, self.text_translation_region, self.text_translation_endpoint]):
            raise ValueError("Azure Cognitive Services key, region, and endpoint must be provided")

        try:
            self.client = TextTranslationClient(
                endpoint=self.text_translation_endpoint,
                credential=AzureKeyCredential(self.text_translation_key)
            )
            logger.info("TextTranslationClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TextTranslationClient: {e}")
            raise

    def read_text_from_file(self, file_path: str) -> str:
        """
        Read and return text from the specified file, supporting PDF, TXT, and DOCX formats.

        Args:
            file_path (str): Path to the input file.

        Returns:
            str: Extracted text from the file.

        Raises:
            ValueError: If the file type is unsupported.
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            return self.read_pdf(file_path)
        elif file_extension == ".txt":
            return self.read_text(file_path)
        elif file_extension == ".docx":
            return self.read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def read_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        import PyPDF2
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text.strip()

    def read_text(self, file_path: str) -> str:
        """Read text from a plain text file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def read_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs]).strip()

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate the input text to the target language using the Azure Text Translation API.

        Args:
            text (str): The text to be translated.
            target_language (str): The target language for translation.

        Returns:
            str: Translated text.

        Raises:
            RuntimeError: If the translation request fails.
        """
        try:
            request_body = [InputTextItem(text=text)]
            response = self.client.translate(content=request_body, to=[target_language])

            translations = response[0].translations
            if translations:
                return translations[0].text
            return ""  # No translations found
        except Exception as e:
            logger.error(f"Error occurred during translation: {e}")
            raise

    def _run(self, query: str, target_language: str) -> str:
        """
        Perform translation of the input query into the specified target language.

        Args:
            query (str): The text to be translated.
            target_language (str): The target language for translation.

        Returns:
            str: Translated text.

        Raises:
            RuntimeError: If an error occurs during the translation process.
        """
        try:
            return self.translate_text(query, target_language)
        except Exception as e:
            logger.error(f"Error while running AzureDocumentTranslateTool: {e}")
            raise RuntimeError(f"Error while running AzureDocumentTranslateTool: {e}")
