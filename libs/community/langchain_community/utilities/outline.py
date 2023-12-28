"""Util that calls Outline."""
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)

OUTLINE_MAX_QUERY_LENGTH = 300


class OutlineAPIWrapper(BaseModel):
    """Wrapper around OutlineAPI.

    This wrapper will use the Outline API to query the documents of your instance.
    By default it will return the document content of the top-k results.
    It limits the document content by doc_content_chars_max.
    """

    top_k_results: int = 3
    load_all_available_meta: bool = False
    doc_content_chars_max: int = 4000
    outline_instance_url: Optional[str] = None
    outline_api_key: Optional[str] = None
    outline_search_endpoint: str = "/api/documents.search"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that instance url and api key exists in environment."""
        outline_instance_url = get_from_dict_or_env(
            values, "outline_instance_url", "OUTLINE_INSTANCE_URL"
        )
        values["outline_instance_url"] = outline_instance_url

        outline_api_key = get_from_dict_or_env(
            values, "outline_api_key", "OUTLINE_API_KEY"
        )
        values["outline_api_key"] = outline_api_key

        return values

    def _result_to_document(self, outline_res: Any) -> Document:
        main_meta = {
            "title": outline_res["document"]["title"],
            "source": self.outline_instance_url + outline_res["document"]["url"],
        }
        add_meta = (
            {
                "id": outline_res["document"]["id"],
                "ranking": outline_res["ranking"],
                "collection_id": outline_res["document"]["collectionId"],
                "parent_document_id": outline_res["document"]["parentDocumentId"],
                "revision": outline_res["document"]["revision"],
                "created_by": outline_res["document"]["createdBy"]["name"],
            }
            if self.load_all_available_meta
            else {}
        )
        doc = Document(
            page_content=outline_res["document"]["text"][: self.doc_content_chars_max],
            metadata={
                **main_meta,
                **add_meta,
            },
        )
        return doc

    def _outline_api_query(self, query: str) -> List:
        raw_result = requests.post(
            f"{self.outline_instance_url}{self.outline_search_endpoint}",
            data={"query": query, "limit": self.top_k_results},
            headers={"Authorization": f"Bearer {self.outline_api_key}"},
        )

        if not raw_result.ok:
            raise ValueError("Outline API returned an error: ", raw_result.text)

        return raw_result.json()["data"]

    def run(self, query: str) -> List[Document]:
        """
        Run Outline search and get the document content plus the meta information.

        Returns: a list of documents.

        """
        results = self._outline_api_query(query[:OUTLINE_MAX_QUERY_LENGTH])
        docs = []
        for result in results[: self.top_k_results]:
            if doc := self._result_to_document(result):
                docs.append(doc)
        return docs
