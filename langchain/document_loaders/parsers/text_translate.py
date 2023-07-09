"""Module contains common parsers for PDFs."""
from typing import Any, Iterator, Mapping, Optional, Union

from langchain.document_loaders.base import BaseBlobParser
from langchain.utils import get_from_env
from doctran import Doctran
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class DoctranTranslateParser(BaseBlobParser):
    """Translates text documents into other languages using doctran."""

    def __init__(self, language: str, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.language = language

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        if self.openai_api_key:
            openai_api_key = self.openai_api_key
        else:
            openai_api_key = get_from_env("openai_api_key", "OPENAI_API_KEY")
        doctran = Doctran(openai_api_key=openai_api_key)
        doctran_doc = doctran.parse(content=blob.as_string()).translate(language=self.language).execute()
        yield Document(page_content=doctran_doc.transformed_content)
