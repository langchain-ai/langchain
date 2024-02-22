"""Astra DB - based chat message history, based on astrapy."""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, List, Optional, Sequence

from langchain_community.utilities.astradb import (
    SetupMode,
    _AstraDBCollectionEnvironment,
)

if TYPE_CHECKING:
    from astrapy.db import AstraDB, AsyncAstraDB

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

DEFAULT_COLLECTION_NAME = "langchain_message_store"


class AstraDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        *,
        session_id: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[AstraDB] = None,
        async_astra_db_client: Optional[AsyncAstraDB] = None,
        namespace: Optional[str] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        pre_delete_collection: bool = False,
    ) -> None:
        """Chat message history that stores history in Astra DB.

        Args:
            session_id: arbitrary key that is used to store the messages
                of a single chat session.
            collection_name: name of the Astra DB collection to create/use.
            token: API token for Astra DB usage.
            api_endpoint: full URL to the API endpoint,
                such as "https://<DB-ID>-us-east1.apps.astra.datastax.com".
            astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AstraDB' instance.
            async_astra_db_client: *alternative to token+api_endpoint*,
                you can pass an already-created 'astrapy.db.AsyncAstraDB' instance.
            namespace: namespace (aka keyspace) where the
                collection is created. Defaults to the database's "default namespace".
            setup_mode: mode used to create the Astra DB collection (SYNC, ASYNC or
                OFF).
            pre_delete_collection: whether to delete the collection
                before creating it. If False and the collection already exists,
                the collection will be used as is.
        """
        self.astra_env = _AstraDBCollectionEnvironment(
            collection_name=collection_name,
            token=token,
            api_endpoint=api_endpoint,
            astra_db_client=astra_db_client,
            async_astra_db_client=async_astra_db_client,
            namespace=namespace,
            setup_mode=setup_mode,
            pre_delete_collection=pre_delete_collection,
        )

        self.collection = self.astra_env.collection
        self.async_collection = self.astra_env.async_collection

        self.session_id = session_id
        self.collection_name = collection_name

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all session messages from DB"""
        self.astra_env.ensure_db_setup()
        message_blobs = [
            doc["body_blob"]
            for doc in sorted(
                self.collection.paginated_find(
                    filter={
                        "session_id": self.session_id,
                    },
                    projection={
                        "timestamp": 1,
                        "body_blob": 1,
                    },
                ),
                key=lambda _doc: _doc["timestamp"],
            )
        ]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        messages = messages_from_dict(items)
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError("Use add_messages instead")

    async def aget_messages(self) -> List[BaseMessage]:
        await self.astra_env.aensure_db_setup()
        docs = self.async_collection.paginated_find(
            filter={
                "session_id": self.session_id,
            },
            projection={
                "timestamp": 1,
                "body_blob": 1,
            },
        )
        sorted_docs = sorted(
            [doc async for doc in docs],
            key=lambda _doc: _doc["timestamp"],
        )
        message_blobs = [doc["body_blob"] for doc in sorted_docs]
        items = [json.loads(message_blob) for message_blob in message_blobs]
        messages = messages_from_dict(items)
        return messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.astra_env.ensure_db_setup()
        docs = [
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        self.collection.chunked_insert_many(docs)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        await self.astra_env.aensure_db_setup()
        docs = [
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
            for message in messages
        ]
        await self.async_collection.chunked_insert_many(docs)

    def clear(self) -> None:
        self.astra_env.ensure_db_setup()
        self.collection.delete_many(filter={"session_id": self.session_id})

    async def aclear(self) -> None:
        await self.astra_env.aensure_db_setup()
        await self.async_collection.delete_many(filter={"session_id": self.session_id})
