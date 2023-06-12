import base64
import os
import requests
from typing import Any, Dict, List, Optional, Iterator
from pydantic import BaseModel, root_validator, validator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader, BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.utils import get_from_dict_or_env

EMBAAS_API_URL = "https://api.embaas.io/v1/document/extract-text/bytes"


class BaseEmbaasLoader(BaseModel):
    embaas_api_key: Optional[str] = None
    api_url: str = EMBAAS_API_URL
    """The URL of the embaas document extraction API."""
    params: Dict[str, Any] = {}
    """Additional parameters to pass to the embaas document extraction API."""

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        embaas_api_key = get_from_dict_or_env(
            values, "embaas_api_key", "EMBAAS_API_KEY"
        )
        values["embaas_api_key"] = embaas_api_key
        return values


class EmbaasBlobLoader(BaseEmbaasLoader, BaseBlobParser):
    """Wrapper around embaas's document loader service."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
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

    def _get_documents(self, blob: Blob) -> Iterator[Document]:
        """Get the documents from the blob."""
        byte_param = blob.as_string()

        payload = {
            **self.params,
            "bytes": byte_param,
        }

        if blob.mimetype is not None and payload.get("mimetype", None) is None:
            payload["mimetype"] = blob.mimetype

        try:
            documents = self._handle_request(payload=payload)
        except requests.exceptions.RequestException as e:
            if e.response is None or not e.response.text:
                raise ValueError(
                    f"Error raised by embaas document text extraction API: {e}"
                )

            parsed_response = e.response.json()
            if "message" in parsed_response:
                raise ValueError(
                    f"Validation Error raised by embaas document text extraction API: {parsed_response['message']}"
                )
            raise

        yield from documents

    def _handle_request(self, payload: Dict[str, Any]) -> List[Document]:
        """Sends a request to the Embaas API and handles the response."""

        headers = {
            "Authorization": f"Bearer {self.embaas_api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        print(response)
        response.raise_for_status()

        parsed_response = response.json()
        return EmbaasBlobLoader._api_response_to_documents(
            chunks=parsed_response["data"]
        )


class EmbaasLoader(BaseEmbaasLoader, BaseLoader):
    file_path: str
    """The path to the file to load."""
    blob_loader: Optional[EmbaasBlobLoader]

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
        yield from self.blob_loader.lazy_parse(blob=blob)

    def load(self) -> List[Document]:
        return list(self.lazy_load())
