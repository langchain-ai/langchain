"""Wrapper for retrieving documents from Vespa."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from langchain.schema import BaseRetriever, Document

if TYPE_CHECKING:
    from vespa.application import Vespa


class VespaRetriever(BaseRetriever):
    def __init__(
        self,
        app: Vespa,
        body: Dict,
        content_field: str,
        metadata_fields: Optional[Sequence[str]] = None,
    ):
        self._application = app
        self._query_body = body
        self._content_field = content_field
        self._metadata_fields = metadata_fields or ()

    def _query(self, body: Dict) -> List[Document]:
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

        docs = []
        for child in response.hits:
            page_content = child["fields"].pop(self._content_field, "")
            if self._metadata_fields == "*":
                metadata = child["fields"]
            else:
                metadata = {mf: child["fields"].get(mf) for mf in self._metadata_fields}
            metadata["id"] = child["id"]
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        body = self._query_body.copy()
        body["query"] = query
        return self._query(body)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    def get_relevant_documents_with_filter(
        self, query: str, _filter: Optional[str] = None
    ) -> List[Document]:
        body = self._query_body.copy()
        _filter = f" and {_filter}" if _filter else ""
        body["yql"] = body["yql"] + _filter
        body["query"] = query
        return self._query(body)

    @classmethod
    def from_params(
        cls,
        url: str,
        content_field: str,
        k: Optional[int] = None,
        metadata_fields: Union[Sequence[str], str] = (),
        sources: Union[Sequence[str], str, None] = None,
        _filter: Optional[str] = None,
        yql: Optional[str] = None,
        **kwargs: Any,
    ) -> VespaRetriever:
        try:
            from vespa.application import Vespa
        except ImportError:
            raise ImportError(
                "pyvespa is not installed, please install with `pip install pyvespa`"
            )
        app = Vespa(url)
        if yql:
            if sources or _filter:
                raise ValueError("")
        else:
            if metadata_fields == "*":
                _fields = "*"
            else:
                _fields = ", ".join([content_field] + list(metadata_fields or []))
            _sources = ", ".join(sources) if isinstance(sources, Sequence) else "*"
            _filter = f" and {_filter}" if _filter else ""
            yql = f"select {_fields} from sources {_sources} where userQuery(){_filter}"
        body: Dict[str, Any] = {"yql": yql, **kwargs}
        if k:
            body["hits"] = k
        return cls(app, body, content_field, metadata_fields=metadata_fields)
