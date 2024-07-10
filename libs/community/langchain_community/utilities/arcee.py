# This module contains utility classes and functions for interacting with Arcee API.
# For more information and updates, refer to the Arcee utils page:
# [https://github.com/arcee-ai/arcee-python/blob/main/arcee/dalm.py]

from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.retrievers import Document


class ArceeRoute(str, Enum):
    """Routes available for the Arcee API as enumerator."""

    generate = "models/generate"
    retrieve = "models/retrieve"
    model_training_status = "models/status/{id_or_name}"


class DALMFilterType(str, Enum):
    """Filter types available for a DALM retrieval as enumerator."""

    fuzzy_search = "fuzzy_search"
    strict_search = "strict_search"


class DALMFilter(BaseModel):
    """Filters available for a DALM retrieval and generation.

    Arguments:
        field_name: The field to filter on. Can be 'document' or 'name' to filter
            on your document's raw text or title. Any other field will be presumed
            to be a metadata field you included when uploading your context data
        filter_type: Currently 'fuzzy_search' and 'strict_search' are supported.
            'fuzzy_search' means a fuzzy search on the provided field is performed.
            The exact strict doesn't need to exist in the document
            for this to find a match.
            Very useful for scanning a document for some keyword terms.
            'strict_search' means that the exact string must appear
            in the provided field.
            This is NOT an exact eq filter. ie a document with content
            "the happy dog crossed the street" will match on a strict_search of
            "dog" but won't match on "the dog".
            Python equivalent of `return search_string in full_string`.
        value: The actual value to search for in the context data/metadata
    """

    field_name: str
    filter_type: DALMFilterType
    value: str
    _is_metadata: bool = False

    @root_validator(pre=True)
    def set_meta(cls, values: Dict) -> Dict:
        """document and name are reserved arcee keys. Anything else is metadata"""
        values["_is_meta"] = values.get("field_name") not in ["document", "name"]
        return values


class ArceeDocumentSource(BaseModel):
    """Source of an Arcee document."""

    document: str
    name: str
    id: str


class ArceeDocument(BaseModel):
    """Arcee document."""

    index: str
    id: str
    score: float
    source: ArceeDocumentSource


class ArceeDocumentAdapter:
    """Adapter for Arcee documents"""

    @classmethod
    def adapt(cls, arcee_document: ArceeDocument) -> Document:
        """Adapts an `ArceeDocument` to a langchain's `Document` object."""
        return Document(
            page_content=arcee_document.source.document,
            metadata={
                # arcee document; source metadata
                "name": arcee_document.source.name,
                "source_id": arcee_document.source.id,
                # arcee document metadata
                "index": arcee_document.index,
                "id": arcee_document.id,
                "score": arcee_document.score,
            },
        )


class ArceeWrapper:
    """Wrapper for Arcee API.

    For more details, see: https://www.arcee.ai/
    """

    def __init__(
        self,
        arcee_api_key: Union[str, SecretStr],
        arcee_api_url: str,
        arcee_api_version: str,
        model_kwargs: Optional[Dict[str, Any]],
        model_name: str,
    ):
        """Initialize ArceeWrapper.

        Arguments:
            arcee_api_key: API key for Arcee API.
            arcee_api_url: URL for Arcee API.
            arcee_api_version: Version of Arcee API.
            model_kwargs: Keyword arguments for Arcee API.
            model_name: Name of an Arcee model.
        """
        if isinstance(arcee_api_key, str):
            arcee_api_key_ = SecretStr(arcee_api_key)
        else:
            arcee_api_key_ = arcee_api_key
        self.arcee_api_key: SecretStr = arcee_api_key_
        self.model_kwargs = model_kwargs
        self.arcee_api_url = arcee_api_url
        self.arcee_api_version = arcee_api_version

        try:
            route = ArceeRoute.model_training_status.value.format(id_or_name=model_name)
            response = self._make_request("get", route)
            self.model_id = response.get("model_id")
            self.model_training_status = response.get("status")
        except Exception as e:
            raise ValueError(
                f"Error while validating model training status for '{model_name}': {e}"
            ) from e

    def validate_model_training_status(self) -> None:
        if self.model_training_status != "training_complete":
            raise Exception(
                f"Model {self.model_id} is not ready. "
                "Please wait for training to complete."
            )

    def _make_request(
        self,
        method: Literal["post", "get"],
        route: Union[ArceeRoute, str],
        body: Optional[Mapping[str, Any]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        """Make a request to the Arcee API
        Args:
            method: The HTTP method to use
            route: The route to call
            body: The body of the request
            params: The query params of the request
            headers: The headers of the request
        """
        headers = self._make_request_headers(headers=headers)
        url = self._make_request_url(route=route)

        req_type = getattr(requests, method)

        response = req_type(url, json=body, params=params, headers=headers)
        if response.status_code not in (200, 201):
            raise Exception(f"Failed to make request. Response: {response.text}")
        return response.json()

    def _make_request_headers(self, headers: Optional[Dict] = None) -> Dict:
        headers = headers or {}
        if not isinstance(self.arcee_api_key, SecretStr):
            raise TypeError(
                f"arcee_api_key must be a SecretStr. Got {type(self.arcee_api_key)}"
            )
        api_key = self.arcee_api_key.get_secret_value()
        internal_headers = {
            "X-Token": api_key,
            "Content-Type": "application/json",
        }
        headers.update(internal_headers)
        return headers

    def _make_request_url(self, route: Union[ArceeRoute, str]) -> str:
        return f"{self.arcee_api_url}/{self.arcee_api_version}/{route}"

    def _make_request_body_for_models(
        self, prompt: str, **kwargs: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Make the request body for generate/retrieve models endpoint"""
        _model_kwargs = self.model_kwargs or {}
        _params = {**_model_kwargs, **kwargs}

        filters = [DALMFilter(**f) for f in _params.get("filters", [])]
        return dict(
            model_id=self.model_id,
            query=prompt,
            size=_params.get("size", 3),
            filters=filters,
            id=self.model_id,
        )

    def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text from Arcee DALM.

        Args:
            prompt: Prompt to generate text from.
            size: The max number of context results to retrieve. Defaults to 3.
              (Can be less if filters are provided).
            filters: Filters to apply to the context dataset.
        """

        response = self._make_request(
            method="post",
            route=ArceeRoute.generate.value,
            body=self._make_request_body_for_models(
                prompt=prompt,
                **kwargs,
            ),
        )
        return response["text"]

    def retrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve {size} contexts with your retriever for a given query

        Args:
            query: Query to submit to the model
            size: The max number of context results to retrieve. Defaults to 3.
              (Can be less if filters are provided).
            filters: Filters to apply to the context dataset.
        """

        response = self._make_request(
            method="post",
            route=ArceeRoute.retrieve.value,
            body=self._make_request_body_for_models(
                prompt=query,
                **kwargs,
            ),
        )
        return [
            ArceeDocumentAdapter.adapt(ArceeDocument(**doc))
            for doc in response["results"]
        ]
