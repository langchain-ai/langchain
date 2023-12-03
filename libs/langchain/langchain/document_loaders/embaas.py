import base64
import warnings
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from typing_extensions import NotRequired, TypedDict

from langchain.document_loaders.base import BaseBlobParser, BaseLoader
from langchain.document_loaders.blob_loaders import Blob
from langchain.text_splitter import TextSplitter
from langchain.utils import get_from_dict_or_env

EMBAAS_DOC_API_URL = "https://api.embaas.io/v1/document/extract-text/bytes/"


class EmbaasDocumentExtractionParameters(TypedDict):
    """Parameters for the embaas document extraction API."""

    mime_type: NotRequired[str]
    """The mime type of the document."""
    file_extension: NotRequired[str]
    """The file extension of the document."""
    file_name: NotRequired[str]
    """The file name of the document."""

    should_chunk: NotRequired[bool]
    """Whether to chunk the document into pages."""
    chunk_size: NotRequired[int]
    """The maximum size of the text chunks."""
    chunk_overlap: NotRequired[int]
    """The maximum overlap allowed between chunks."""
    chunk_splitter: NotRequired[str]
    """The text splitter class name for creating chunks."""
    separators: NotRequired[List[str]]
    """The separators for chunks."""

    should_embed: NotRequired[bool]
    """Whether to create embeddings for the document in the response."""
    model: NotRequired[str]
    """The model to pass to the Embaas document extraction API."""
    instruction: NotRequired[str]
    """The instruction to pass to the Embaas document extraction API."""


class EmbaasDocumentExtractionPayload(EmbaasDocumentExtractionParameters):
    """Payload for the Embaas document extraction API."""

    bytes: str
    """The base64 encoded bytes of the document to extract text from."""


class BaseEmbaasLoader(BaseModel):
    """Base loader for `Embaas` document extraction API."""

    embaas_api_key: Optional[str] = None
    """The API key for the Embaas document extraction API."""
    api_url: str = EMBAAS_DOC_API_URL
    """The URL of the Embaas document extraction API."""
    params: EmbaasDocumentExtractionParameters = EmbaasDocumentExtractionParameters()
    """Additional parameters to pass to the Embaas document extraction API."""

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        embaas_api_key = get_from_dict_or_env(
            values, "embaas_api_key", "EMBAAS_API_KEY"
        )
        values["embaas_api_key"] = embaas_api_key
        return values


class EmbaasBlobLoader(BaseEmbaasLoader, BaseBlobParser):
    """Load `Embaas` blob.

    To use, you should have the
    environment variable ``EMBAAS_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            # Default parsing
            from langchain.document_loaders.embaas import EmbaasBlobLoader
            loader = EmbaasBlobLoader()
            blob = Blob.from_path(path="example.mp3")
            documents = loader.parse(blob=blob)

            # Custom api parameters (create embeddings automatically)
            from langchain.document_loaders.embaas import EmbaasBlobLoader
            loader = EmbaasBlobLoader(
                params={
                    "should_embed": True,
                    "model": "e5-large-v2",
                    "chunk_size": 256,
                    "chunk_splitter": "CharacterTextSplitter"
                }
            )
            blob = Blob.from_path(path="example.pdf")
            documents = loader.parse(blob=blob)
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parses the blob lazily.

        Args:
            blob: The blob to parse.
        """
        yield from self._get_documents(blob=blob)

    @staticmethod
    def _api_response_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert the API response to a list of documents."""
        docs = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            if chunk.get("embedding", None) is not None:
                metadata["embedding"] = chunk["embedding"]
            doc = Document(page_content=chunk["text"], metadata=metadata)
            docs.append(doc)

        return docs

    def _generate_payload(self, blob: Blob) -> EmbaasDocumentExtractionPayload:
        """Generates payload for the API request."""
        base64_byte_str = base64.b64encode(blob.as_bytes()).decode()
        payload: EmbaasDocumentExtractionPayload = EmbaasDocumentExtractionPayload(
            bytes=base64_byte_str,
            # Workaround for mypy issue: https://github.com/python/mypy/issues/9408
            # type: ignore
            **self.params,
        )

        if blob.mimetype is not None and payload.get("mime_type", None) is None:
            payload["mime_type"] = blob.mimetype

        return payload

    def _handle_request(
        self, payload: EmbaasDocumentExtractionPayload
    ) -> List[Document]:
        """Sends a request to the embaas API and handles the response."""
        headers = {
            "Authorization": f"Bearer {self.embaas_api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        parsed_response = response.json()
        return EmbaasBlobLoader._api_response_to_documents(
            chunks=parsed_response["data"]["chunks"]
        )

    def _get_documents(self, blob: Blob) -> Iterator[Document]:
        """Get the documents from the blob."""
        payload = self._generate_payload(blob=blob)

        try:
            documents = self._handle_request(payload=payload)
        except requests.exceptions.RequestException as e:
            if e.response is None or not e.response.text:
                raise ValueError(
                    f"Error raised by Embaas document text extraction API: {e}"
                )

            parsed_response = e.response.json()
            if "message" in parsed_response:
                raise ValueError(
                    f"Validation Error raised by Embaas document text extraction API:"
                    f" {parsed_response['message']}"
                )
            raise

        yield from documents


class EmbaasLoader(BaseEmbaasLoader, BaseLoader):
    """Load from `Embaas`.

    To use, you should have the
    environment variable ``EMBAAS_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            # Default parsing
            from langchain.document_loaders.embaas import EmbaasLoader
            loader = EmbaasLoader(file_path="example.mp3")
            documents = loader.load()

            # Custom api parameters (create embeddings automatically)
            from langchain.document_loaders.embaas import EmbaasBlobLoader
            loader = EmbaasBlobLoader(
                file_path="example.pdf",
                params={
                    "should_embed": True,
                    "model": "e5-large-v2",
                    "chunk_size": 256,
                    "chunk_splitter": "CharacterTextSplitter"
                }
            )
            documents = loader.load()
    """

    file_path: str
    """The path to the file to load."""
    blob_loader: Optional[EmbaasBlobLoader]
    """The blob loader to use. If not provided, a default one will be created."""

    @validator("blob_loader", always=True)
    def validate_blob_loader(
        cls, v: EmbaasBlobLoader, values: Dict
    ) -> EmbaasBlobLoader:
        return v or EmbaasBlobLoader(
            embaas_api_key=values["embaas_api_key"],
            api_url=values["api_url"],
            params=values["params"],
        )

    def lazy_load(self) -> Iterator[Document]:
        """Load the documents from the file path lazily."""
        blob = Blob.from_path(path=self.file_path)

        assert self.blob_loader is not None
        # Should never be None, but mypy doesn't know that.
        yield from self.blob_loader.lazy_parse(blob=blob)

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        if self.params.get("should_embed", False):
            warnings.warn(
                "Embeddings are not supported with load_and_split."
                " Use the API splitter to properly generate embeddings."
                " For more information see embaas.io docs."
            )
        return super().load_and_split(text_splitter=text_splitter)
