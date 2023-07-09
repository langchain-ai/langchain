"""Module contains common parsers for PDFs."""
import json
from typing import Any, Iterator, Mapping, Optional, Union

from langchain.document_loaders.base import BaseBlobParser
from langchain.utils import get_from_dict_or_env
from doctran import Doctran
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class DoctranQAParser(BaseBlobParser):
    """Extracts metadata from text documents using doctran."""

    def lazy_parse(self, blob: Blob, **kwargs) -> Iterator[Document]:
        """Lazily parse the blob."""
        openai_api_key = get_from_dict_or_env(kwargs, "openai_api_key", "OPENAI_API_KEY")
        doctran = Doctran(openai_api_key=openai_api_key)
        doctran_doc = doctran.parse(content=blob.as_string()).interrogate().execute()
        questions_and_answers = doctran_doc.extracted_properties.get("questions_and_answers")
        yield Document(page_content=blob.as_string(), metadata={"questions_and_answers": questions_and_answers, "source": blob.source})
