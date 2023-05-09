"""Wrapper for retrieving documents from Vespa."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

from langchain.schema import BaseRetriever, Document

if TYPE_CHECKING:
    from vespa.application import Vespa


class VespaRetriever(BaseRetriever):
    def __init__(self, app: Vespa, body: dict, content_field: str):
        self._application = app
        self._query_body = body
        self._content_field = content_field

    def get_relevant_documents(self, query: str) -> List[Document]:
        body = self._query_body.copy()
        body["query"] = query
        response = self._application.query(body)

        if not str(response.status_code).startswith("2"):
            raise RuntimeError(
                "Could not retrieve data from Vespa. Error code: {}".format(
                    response.status_code
                )
            )

        root = response.json["root"]
        if "errors" in root:
            raise RuntimeError(json.dumps(root["errors"]))

        hits = []
        for child in response.hits:
            page_content = child["fields"][self._content_field]
            metadata = {"id": child["id"]}
            hits.append(Document(page_content=page_content, metadata=metadata))
        return hits

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
