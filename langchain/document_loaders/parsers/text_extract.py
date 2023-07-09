"""Module contains common parsers for PDFs."""
from typing import Iterator, List
from pydantic import ValidationError

from langchain.document_loaders.base import BaseBlobParser
from langchain.utils import get_from_dict_or_env
from doctran import Doctran, ExtractProperty
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class DoctranExtractParser(BaseBlobParser):
    """Extracts metadata from text documents using doctran."""

    def lazy_parse(self, blob: Blob, properties: List[dict], **kwargs) -> Iterator[Document]:
        """Lazily parse the blob."""
        openai_api_key = get_from_dict_or_env(kwargs, "openai_api_key", "OPENAI_API_KEY")
        try:
            properties = [ExtractProperty(**property) for property in properties]
        except ValidationError as e:
            raise e
        doctran = Doctran(openai_api_key=openai_api_key)
        doctran_doc = doctran.parse(content=blob.as_string()).extract(properties=properties).execute()
        yield Document(page_content=blob.as_string(), metadata={"extracted_properties": doctran_doc.extracted_properties, "source": blob.source})