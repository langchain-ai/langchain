# This module contains utility classes and functions for interacting with Arcee API.
# For more information and updates, refer to the Arcee utils page:
# [https://github.com/arcee-ai/arcee-python/blob/main/arcee/dalm.py]

from enum import Enum
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import requests

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.retriever import Document


class ArceeRoute(str, Enum):
    generate = "models/generate"
    retrieve = "models/retrieve"
    model_training_status = "models/status/{id_or_name}"


class DALMFilterType(str, Enum):
    fuzzy_search = "fuzzy_search"
    strict_search = "strict_search"


class DALMFilter(BaseModel):
    """Filters available for a dalm retrieval and generation

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

    @root_validator()
    def set_meta(cls, values: Dict) -> Dict:
        """document and name are reserved arcee keys. Anything else is metadata"""
        values["_is_meta"] = values.get("field_name") not in ["document", "name"]
        return values


class ArceeWrapper:
    def __init__(
        self,
        arcee_api_key: str,
        arcee_api_url: str,
        arcee_api_version: str,
        model_kwargs: Optional[Dict[str, Any]],
        model_name: str,
    ):
        self.arcee_api_key = arcee_api_key
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
        internal_headers = {
            "X-Token": self.arcee_api_key,
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
            route=ArceeRoute.generate,
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
            route=ArceeRoute.retrieve,
            body=self._make_request_body_for_models(
                prompt=query,
                **kwargs,
            ),
        )
        return [Document(**doc) for doc in response["documents"]]
