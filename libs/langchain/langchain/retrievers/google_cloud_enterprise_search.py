"""Retriever wrapper for Google Cloud Enterprise Search on Gen App Builder."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.schema import BaseRetriever, Document
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from google.cloud.discoveryengine_v1beta import (
        SearchRequest,
        SearchResult,
        SearchServiceClient,
    )


class GoogleCloudEnterpriseSearchRetriever(BaseRetriever):
    """Retriever for the Google Cloud Enterprise Search Service API.

    For the detailed explanation of the Enterprise Search concepts
    and configuration parameters refer to the product documentation.

    https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction
    """

    project_id: str
    """Google Cloud Project ID."""
    search_engine_id: str
    """Enterprise Search engine ID."""
    serving_config_id: str = "default_config"
    """Enterprise Search serving config ID."""
    location_id: str = "global"
    """Enterprise Search engine location."""
    filter: Optional[str] = None
    """Filter expression."""
    get_extractive_answers: bool = False
    """If True return Extractive Answers, otherwise return Extractive Segments."""
    max_documents: int = Field(default=5, ge=1, le=100)
    """The maximum number of documents to return."""
    max_extractive_answer_count: int = Field(default=1, ge=1, le=5)
    """The maximum number of extractive answers returned in each search result.
    At most 5 answers will be returned for each SearchResult.
    """
    max_extractive_segment_count: int = Field(default=1, ge=1, le=1)
    """The maximum number of extractive segments returned in each search result.
    Currently one segment will be returned for each SearchResult.
    """
    query_expansion_condition: int = Field(default=1, ge=0, le=2)
    """Specification to determine under which conditions query expansion should occur.
    0 - Unspecified query expansion condition. In this case, server behavior defaults 
        to disabled
    1 - Disabled query expansion. Only the exact search query is used, even if 
        SearchResponse.total_size is zero.
    2 - Automatic query expansion built by the Search API.
    """
    spell_correction_mode: int = Field(default=2, ge=0, le=2)
    """Specification to determine under which conditions query expansion should occur.
    0 - Unspecified spell correction mode. In this case, server behavior defaults 
        to auto.
    1 - Suggestion only. Search API will try to find a spell suggestion if there is any
        and put in the `SearchResponse.corrected_query`.
        The spell suggestion will not be used as the search query.
    2 - Automatic spell correction built by the Search API.
        Search will be based on the corrected query if found.
    """
    credentials: Any = None
    """The default custom credentials (google.auth.credentials.Credentials) to use
    when making API calls. If not provided, credentials will be ascertained from
    the environment."""

    _client: SearchServiceClient
    _serving_config: str

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates the environment."""
        try:
            from google.cloud import discoveryengine_v1beta  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "google.cloud.discoveryengine is not installed. "
                "Please install it with pip install google-cloud-discoveryengine"
            ) from exc

        values["project_id"] = get_from_dict_or_env(values, "project_id", "PROJECT_ID")
        values["search_engine_id"] = get_from_dict_or_env(
            values, "search_engine_id", "SEARCH_ENGINE_ID"
        )

        return values

    def __init__(self, **data: Any) -> None:
        """Initializes private fields."""
        from google.cloud.discoveryengine_v1beta import SearchServiceClient

        super().__init__(**data)
        self._client = SearchServiceClient(credentials=self.credentials)
        self._serving_config = self._client.serving_config_path(
            project=self.project_id,
            location=self.location_id,
            data_store=self.search_engine_id,
            serving_config=self.serving_config_id,
        )

    def _convert_search_response(
        self, results: Sequence[SearchResult]
    ) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        from google.protobuf.json_format import MessageToDict

        documents: List[Document] = []

        for result in results:
            document_dict = MessageToDict(
                result.document._pb, preserving_proto_field_name=True
            )
            derived_struct_data = document_dict.get("derived_struct_data", None)
            if not derived_struct_data:
                continue

            doc_metadata = document_dict.get("struct_data", {})
            doc_metadata["id"] = document_dict["id"]

            chunk_type = (
                "extractive_answers"
                if self.get_extractive_answers
                else "extractive_segments"
            )

            for chunk in getattr(derived_struct_data, chunk_type, []):
                doc_metadata["source"] = derived_struct_data.get("link", "")

                if chunk_type == "extractive_answers":
                    doc_metadata["source"] += f":{chunk.get('pageNumber', '')}"

                documents.append(
                    Document(
                        page_content=chunk.get("content", ""), metadata=doc_metadata
                    )
                )

        return documents

    def _create_search_request(self, query: str) -> SearchRequest:
        """Prepares a SearchRequest object."""
        from google.cloud.discoveryengine_v1beta import SearchRequest

        query_expansion_spec = SearchRequest.QueryExpansionSpec(
            condition=self.query_expansion_condition,
        )

        spell_correction_spec = SearchRequest.SpellCorrectionSpec(
            mode=self.spell_correction_mode
        )

        if self.get_extractive_answers:
            extractive_content_spec = (
                SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count=self.max_extractive_answer_count,
                )
            )
        else:
            extractive_content_spec = (
                SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_segment_count=self.max_extractive_segment_count,
                )
            )

        content_search_spec = SearchRequest.ContentSearchSpec(
            extractive_content_spec=extractive_content_spec,
        )

        return SearchRequest(
            query=query,
            filter=self.filter,
            serving_config=self._serving_config,
            page_size=self.max_documents,
            content_search_spec=content_search_spec,
            query_expansion_spec=query_expansion_spec,
            spell_correction_spec=spell_correction_spec,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant for a query."""
        search_request = self._create_search_request(query)
        response = self._client.search(search_request)
        documents = self._convert_search_response(response.results)

        return documents
