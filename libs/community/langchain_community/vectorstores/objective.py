import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import UpsertResponse
from langchain_core.vectorstores import VST, VectorStore

CPU_COUNT = os.cpu_count() or 6
CONNECTION_POOL_SIZE = int(
    os.getenv(
        "OBJECTIVE_CONNECTION_POOL_SIZE",
        CPU_COUNT * 12,
    )
)
API_BASE_URL = "https://api.objective.inc/v1/"
logger = logging.getLogger(__name__)


class ObjectiveError(ValueError):
    pass


class Objective(VectorStore):
    def __init__(self, api_key: str):
        self.api_key = api_key.strip('"')
        self.http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=CONNECTION_POOL_SIZE)
        self.http_session.mount("https://", adapter)
        self.http_session.mount("http://", adapter)

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> VST:
        api_key = kwargs.pop("api_key")
        objective = Objective(api_key)
        objective.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return objective  # type: ignore

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError

    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert document metadata into the vector store."""

        def _upsert(document: Document) -> Tuple[str, bool]:
            try:
                obj_data = {
                    "page_content": document.page_content,
                    "metadata": document.metadata,
                }

                if document.id:
                    self._request(
                        method="PUT",
                        endpoint=f"objects/{document.id}",
                        data=obj_data,
                    )
                else:
                    obj_response = self._request(
                        method="POST", endpoint="objects", data=obj_data
                    )
                    document.id = obj_response["id"]
                # unknown id condition should never happen in practice
                # but UpsertResponse is non-optional string typed
                # see UpsertResponse docstring for failure case
                return document.id or "unknown id", True
            except Exception as e:
                logger.error(f"Failed to upsert document: {document.id}", e)
                return document.id or "unknown id", False

        success = []
        failures = []
        with ThreadPoolExecutor(max_workers=CONNECTION_POOL_SIZE) as executor:
            for result in executor.map(_upsert, items):
                if result[1]:
                    success.append(result[0])
                else:
                    failures.append(result[0])

        return UpsertResponse(
            succeeded=success,
            failed=failures,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ObjectiveError("ids parameter is required for delete")

        def _delete(id: str) -> bool:
            try:
                self._request(
                    method="DELETE",
                    endpoint=f"objects/{id}",
                )
                return True
            except Exception as e:
                logger.error(f"Failed to delete document: {id}", e)
                return False

        success = True
        with ThreadPoolExecutor(max_workers=CONNECTION_POOL_SIZE) as executor:
            for result in executor.map(_delete, ids):
                success = success and result

        return success

    @staticmethod
    def _doc_from_response(obj_response: Dict[str, Any]) -> Document:
        obj_id = obj_response["id"]
        obj_data = obj_response["object"]
        content = obj_data.pop("page_content")
        metadata = obj_data.get("metadata", {})
        return Document(id=obj_id, page_content=content, metadata=metadata)

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        def _get_by_id(obj_id: str) -> Optional[Document]:
            try:
                obj_response = self._request(
                    method="GET",
                    endpoint=f"objects/{obj_id}",
                )
                return self._doc_from_response(obj_response)
            except Exception as e:
                logger.error(f"Failed to get document: {obj_id}", e)
                return None

        successes = []
        with ThreadPoolExecutor(max_workers=CONNECTION_POOL_SIZE) as executor:
            for result in executor.map(_get_by_id, ids):
                if result:
                    successes.append(result)

        return successes

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        if "index_id" not in kwargs:
            raise ObjectiveError("index_id is required for search")
        if "filter_query" in kwargs:
            logger.warning("Filter queries are not yet supported and will be ignored.")

        index_id = kwargs.pop("index_id")
        params = {"query": query}
        if "k" in kwargs:
            params["limit"] = kwargs["k"]

        response = self._request(
            method="GET",
            endpoint=f"indexes/{index_id}/search?object_fields=page_content,metadata",
            params=params,
        )
        return [self._doc_from_response(obj) for obj in response["results"]]

    def create_index(self) -> str:
        response = self._request(
            method="POST",
            endpoint="indexes",
            data={
                "configuration": {
                    "index_type": {"name": "text"},
                    "fields": {
                        "searchable": {"allow": ["page_content"]},
                    },
                }
            },
        )
        if "id" not in response:
            raise ObjectiveError("Failed to create index")
        return response["id"]

    def index_status(self, index_id: str) -> Dict[str, int]:
        response = self._request(
            "GET",
            f"indexes/{index_id}/status",
        )
        if "status" not in response:
            raise ObjectiveError("Failed to retrieve index")
        return response["status"]

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Issue a request to the Objective API

        Returns the JSON from the request."""
        url = API_BASE_URL + endpoint

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.get_user_agent(),
        }

        MAX_RETRIES = 3
        BACKOFF_FACTOR = 1.5
        for attempt in range(MAX_RETRIES):
            try:
                response = self.http_session.request(
                    method,
                    url,
                    headers=headers,
                    json=data,
                    params=params,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:  # i.e. if it's not the last attempt
                    sleep_time = BACKOFF_FACTOR * (2**attempt)
                    time.sleep(sleep_time)
                    continue
                else:
                    if e.response is not None:
                        raise ObjectiveError(e.response.text) from e
                    raise e
        raise ObjectiveError(f"Unknown failure on request [{method}] {url}")

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__

        return f"langchain-py/{__version__}"
