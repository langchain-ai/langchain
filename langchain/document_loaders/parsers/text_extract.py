"""Module contains common parsers for PDFs."""
from typing import Iterator, List, Optional
from pydantic import ValidationError

from langchain.document_loaders.base import BaseBlobParser
from langchain.utils import get_from_env
from doctran import Doctran, ExtractProperty
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class DoctranExtractParser(BaseBlobParser):
    """Extracts metadata from text documents using doctran."""

    def __init__(self, properties: List[dict], openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        try:
            self.properties = [ExtractProperty(**property) for property in properties]
        except ValidationError as e:
            raise e

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        if self.openai_api_key:
            openai_api_key = self.openai_api_key
        else:
            openai_api_key = get_from_env("openai_api_key", "OPENAI_API_KEY")
        doctran = Doctran(openai_api_key=openai_api_key)
        doctran_doc = doctran.parse(content=blob.as_string()).extract(properties=self.properties).execute()
        yield Document(page_content=blob.as_string(), metadata={"extracted_properties": doctran_doc.extracted_properties, "source": blob.source})