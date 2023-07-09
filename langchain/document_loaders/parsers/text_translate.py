"""Module contains common parsers for PDFs."""
from typing import Any, Iterator, Mapping, Optional, Union

from langchain.document_loaders.base import BaseBlobParser
from langchain.utils import get_from_dict_or_env
from doctran import Doctran
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class DoctranTranslateParser(BaseBlobParser):
    """Translates text documents into other languages using doctran."""

    def lazy_parse(self, blob: Blob, language: str, **kwargs) -> Iterator[Document]:
        """Lazily parse the blob."""
        openai_api_key = get_from_dict_or_env(kwargs, "openai_api_key", "OPENAI_API_KEY")
        doctran = Doctran(openai_api_key=openai_api_key)
        doctran_doc = doctran.parse(content=blob.as_string()).translate(language=language).execute()
        yield Document(page_content=doctran_doc.transformed_content)
