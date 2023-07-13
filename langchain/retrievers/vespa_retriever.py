"""Wrapper for retrieving documents from Vespa."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document

if TYPE_CHECKING:
    from vespa.application import Vespa


class VespaRetriever(BaseRetriever):
    """Retriever that uses the Vespa."""

    app: Vespa
    body: Dict
    content_field: str
    metadata_fields: Sequence[str]

    def _query(self, body: Dict) -> List[Document]:
        response = self.app.query(body)

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
            page_content = child["fields"].pop(self.content_field, "")
            if self.metadata_fields == "*":
                metadata = child["fields"]
            else:
                metadata = {mf: child["fields"].get(mf) for mf in self.metadata_fields}
            metadata["id"] = child["id"]
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        body = self.body.copy()
        body["query"] = query
        return self._query(body)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError

    def get_relevant_documents_with_filter(
        self, query: str, *, _filter: Optional[str] = None
    ) -> List[Document]:
        body = self.body.copy()
        _filter = f" and {_filter}" if _filter else ""
        body["yql"] = body["yql"] + _filter
        body["query"] = query
        return self._query(body)

    @classmethod
    def from_params(
        cls,
        url: str,
        content_field: str,
        *,
        k: Optional[int] = None,
        metadata_fields: Union[Sequence[str], Literal["*"]] = (),
        sources: Union[Sequence[str], Literal["*"], None] = None,
        _filter: Optional[str] = None,
        yql: Optional[str] = None,
        **kwargs: Any,
    ) -> VespaRetriever:
        """Instantiate retriever from params.

        Args:
            url (str): Vespa app URL.
            content_field (str): Field in results to return as Document page_content.
            k (Optional[int]): Number of Documents to return. Defaults to None.
            metadata_fields(Sequence[str] or "*"): Fields in results to include in
                document metadata. Defaults to empty tuple ().
            sources (Sequence[str] or "*" or None): Sources to retrieve
                from. Defaults to None.
            _filter (Optional[str]): Document filter condition expressed in YQL.
                Defaults to None.
            yql (Optional[str]): Full YQL query to be used. Should not be specified
                if _filter or sources are specified. Defaults to None.
            kwargs (Any): Keyword arguments added to query body.
        """
        try:
            from vespa.application import Vespa
        except ImportError:
            raise ImportError(
                "pyvespa is not installed, please install with `pip install pyvespa`"
            )
        app = Vespa(url)
        body = kwargs.copy()
        if yql and (sources or _filter):
            raise ValueError(
                "yql should only be specified if both sources and _filter are not "
                "specified."
            )
        else:
            if metadata_fields == "*":
                _fields = "*"
                body["summary"] = "short"
            else:
                _fields = ", ".join([content_field] + list(metadata_fields or []))
            _sources = ", ".join(sources) if isinstance(sources, Sequence) else "*"
            _filter = f" and {_filter}" if _filter else ""
            yql = f"select {_fields} from sources {_sources} where userQuery(){_filter}"
        body["yql"] = yql
        if k:
            body["hits"] = k
        return cls(
            app=app,
            body=body,
            content_field=content_field,
            metadata_fields=metadata_fields,
        )
