from __future__ import annotations

import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.astradb import _AstraDBEnvironment

if TYPE_CHECKING:
    from astrapy.db import AstraDB, AsyncAstraDB

logger = logging.getLogger(__name__)


class AstraDBLoader(BaseLoader):
    def __init__(
        self,
        collection_name: str,
        *,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        find_options: Optional[Dict[str, Any]] = None,
        nb_prefetched: int = 1000,
        extraction_function: Callable[[Dict], str] = json.dumps,
    ) -> None:
        """Load DataStax Astra DB documents.

        Args:
            collection_name: name of the Astra DB collection to use.
            token: API token for Astra DB usage.
            api_endpoint: full URL to the API endpoint,
                such as `https://<DB-ID>-us-east1.apps.astra.datastax.com`.
            astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            async_astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance.
            namespace: namespace (aka keyspace) where the
                collection is. Defaults to the database's "default namespace".
            filter_criteria: Criteria to filter documents.
            projection: Specifies the fields to return.
            find_options: Additional options for the query.
            nb_prefetched: Max number of documents to pre-fetch. Defaults to 1000.
            extraction_function: Function applied to collection documents to create
                the `page_content` of the LangChain Document. Defaults to `json.dumps`.
        """
        astra_env = _AstraDBEnvironment(
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
        )
        self.astra_env = astra_env
        self.collection = astra_env.astra_db.collection(collection_name)
        self.collection_name = collection_name
        self.filter = filter_criteria
        self.projection = projection
        self.find_options = find_options or {}
        self.nb_prefetched = nb_prefetched
        self.extraction_function = extraction_function

    def lazy_load(self) -> Iterator[Document]:
        for doc in self.collection.paginated_find(
            filter=self.filter,
            options=self.find_options,
            projection=self.projection,
            sort=None,
            prefetched=self.nb_prefetched,
        ):
            yield Document(
                page_content=self.extraction_function(doc),
                metadata={
                    "namespace": self.collection.astra_db.namespace,
                    "api_endpoint": self.collection.astra_db.base_url,
                    "collection": self.collection_name,
                },
            )

    async def aload(self) -> List[Document]:
        """Load data into Document objects."""
        return [doc async for doc in self.alazy_load()]

    async def alazy_load(self) -> AsyncIterator[Document]:
        async_collection = await self.astra_env.async_astra_db.collection(
            self.collection_name
        )
        async for doc in async_collection.paginated_find(
            filter=self.filter,
            options=self.find_options,
            projection=self.projection,
            sort=None,
            prefetched=self.nb_prefetched,
        ):
            yield Document(
                page_content=self.extraction_function(doc),
                metadata={
                    "namespace": async_collection.astra_db.namespace,
                    "api_endpoint": async_collection.astra_db.base_url,
                    "collection": self.collection_name,
                },
            )
