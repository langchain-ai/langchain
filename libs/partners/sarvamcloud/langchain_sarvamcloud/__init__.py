"""Sarvam AI integration for LangChain.

This package provides LangChain integrations for all Sarvam AI services:

- `ChatSarvam`: Chat completions with tool calling and streaming.
- `SarvamSTT`: Speech-to-Text (REST, up to 30 seconds).
- `SarvamBatchSTT`: Batch Speech-to-Text for long-form audio.
- `SarvamTTS`: Text-to-Speech synthesis.
- `SarvamTranslator`: Translation across 22 Indian languages.
- `SarvamTransliterator`: Script conversion (e.g. Devanagari ↔ Roman).
- `SarvamLanguageDetector`: Language and script identification.
- `SarvamDocumentIntelligence`: OCR and document digitization.
"""

from langchain_sarvamcloud.chat_models import ChatSarvam
from langchain_sarvamcloud.document_loaders import SarvamDocumentIntelligence
from langchain_sarvamcloud.speech import SarvamBatchSTT, SarvamSTT, SarvamTTS
from langchain_sarvamcloud.text import (
    SarvamLanguageDetector,
    SarvamTransliterator,
    SarvamTranslator,
)
from langchain_sarvamcloud.version import __version__

__all__ = [
    "ChatSarvam",
    "SarvamSTT",
    "SarvamBatchSTT",
    "SarvamTTS",
    "SarvamTranslator",
    "SarvamTransliterator",
    "SarvamLanguageDetector",
    "SarvamDocumentIntelligence",
    "__version__",
]
