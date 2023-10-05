from enum import Enum

from langchain.pydantic_v1 import BaseModel, root_validator
from typing import Any, Dict, TYPE_CHECKING, List, Mapping, Union, Optional, Literal
import requests


class ArceeRoute(str, Enum):
    generate = "models/generate"
    retrieve = "models/retrieve"
    model_training_status = "models/status/{model_id}"


class DALMFilterType(str, Enum):
    fuzzy_search = "fuzzy_search"
    strict_search = "strict_search"


class DALMFilter(BaseModel):
    """Filters available for a dalm retrieval and generation

    Arguments:
        field_name: The field to filter on. Can be 'document' or 'name' to filter on your document's raw text or title
            Any other field will be presumed to be a metadata field you included when uploading your context data
        filter_type: Currently 'fuzzy_search' and 'strict_search' are supported. More to come soon!
            'fuzzy_search' means a fuzzy search on the provided field will be performed. The exact strict doesn't
            need to exist in the document for this to find a match. Very useful for scanning a document for some
            keyword terms
            'strict_search' means that the exact string must appear in the provided field. This is NOT an exact eq
            filter. ie a document with content "the happy dog crossed the street" will match on a strict_search of "dog"
            but won't match on "the dog". Python equivalent of `return search_string in full_string`
        value: The actual value to search for in the context data/metadata
    """

    field_name: str
    filter_type: DALMFilterType
    value: str
    _is_metadata: bool = False

    @root_validator
    def set_meta(self) -> "DALMFilter":
        """document and name are reserved arcee keys. Anything else is metadata"""
        self._is_metadata = self.field_name not in ["document", "name"]
        return self


class ArceeWrapper(BaseModel):
    """Wrapper for Arcee APIs"""

    arcee_api_url: str = "https://api.arcee.ai"
    """Arcee API URL"""

    arcee_api_version: str = "v2"
    """Arcee API Version"""

    arcee_app_url: str = "https://app.arcee.ai"
    """Arcee App URL"""

    model_id: str
    """Arcee Model ID"""

    model_kwargs: Mapping[str, Any] = None
    """Keyword arguments to pass to the model."""

    @root_validator()
    def validate_model_kwargs(cls, values: Dict) -> Dict:
        """Validate that model kwargs are valid."""

        print("validating kwargs: ", values.get("model_kwargs"))

        if values.get("model_kwargs") is not None:
            kw = values.get("model_kwargs")

            # validate size
            if kw.get("size") is not None:
                if not kw.get("size") >= 0:
                    raise ValueError("`size` must be positive")

            # validate filters
            if kw.get("filters") is not None:
                if not isinstance(kw.get("filters"), List):
                    raise ValueError("`filters` must be a list")
                for f in kw.get("filters"):
                    DALMFilter.validate(f)

    @classmethod
    def _make_request_headers(cls, headers: Optional[Dict] = None) -> Dict:
        """Make the request headers"""
        headers = headers or {}
        internal_headers = {
            "X-Token": f"{cls.arcee_api_key}",
            "Content-Type": "application/json",
        }
        headers.update(**internal_headers)
        return headers

    @classmethod
    def _make_request_url(cls, route: ArceeRoute) -> str:
        """Make the request url"""
        return f"{cls.arcee_api_url}/{cls.arcee_api_version}/{route}"

    def _make_request_body_for_models(
        self, prompt: str, **kwargs: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Build the kwargs for the Post request, used by sync

        Args:
            prompt (str): prompt used in query
            kwargs (dict): model kwargs in payload

        Returns:
            Dict[str, Union[str,dict]]: _description_
        """
        _model_kwargs = self.model_kwargs or {}
        _params = {**_model_kwargs, **kwargs}

        # validate filters
        filters = [DALMFilter.validate(f) for f in _params.get("filters", [])]

        return dict(
            model_id=self.model_id,
            query=prompt,
            size=_params.get("size", 3),
            filters=filters,
            id=self.model_id,
        )

    @classmethod
    def make_request(
        cls,
        request: Literal["post", "get"],
        route: ArceeRoute,
        body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, Any]] = None,
    ) -> dict[str, str]:
        """Makes the request"""
        headers = cls._make_request_headers(headers)
        url = cls.make_request_url(route)

        req_type = getattr(requests, request)

        response = req_type(url, json=body, params=params, headers=headers)
        if response.status_code not in (200, 201):
            raise Exception(f"Failed to make request. Response: {response.text}")
        return response.json()
