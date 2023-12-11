"""Astra DB - based chat message history, based on astrapy."""
from __future__ import annotations

import json
import time
import typing
from typing import List, Optional

if typing.TYPE_CHECKING:
    from astrapy.db import AstraDB as LibAstraDB

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

DEFAULT_COLLECTION_NAME = "langchain_message_store"


class AstraDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in Astra DB.

    Args (only keyword-arguments accepted):
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        collection_name (str): name of the Astra DB collection to create/use.
        token (Optional[str]): API token for Astra DB usage.
        api_endpoint (Optional[str]): full URL to the API endpoint,
            such as "https://<DB-ID>-us-east1.apps.astra.datastax.com".
        astra_db_client (Optional[Any]): *alternative to token+api_endpoint*,
            you can pass an already-created 'astrapy.db.AstraDB' instance.
        namespace (Optional[str]): namespace (aka keyspace) where the
            collection is created. Defaults to the database's "default namespace".
    """

    def __init__(
        self,
        *,
        session_id: str,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        token: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        astra_db_client: Optional[LibAstraDB] = None,  # type 'astrapy.db.AstraDB'
        namespace: Optional[str] = None,
    ) -> None:
        """Create an Astra DB chat message history."""
        try:
            from astrapy.db import AstraDB as LibAstraDB
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "Could not import a recent astrapy python package. "
                "Please install it with `pip install --upgrade astrapy`."
            )

        # Conflicting-arg checks:
        if astra_db_client is not None:
            if token is not None or api_endpoint is not None:
                raise ValueError(
                    "You cannot pass 'astra_db_client' to AstraDB if passing "
                    "'token' and 'api_endpoint'."
                )

        self.session_id = session_id
        self.collection_name = collection_name
        self.token = token
        self.api_endpoint = api_endpoint
        self.namespace = namespace
        if astra_db_client is not None:
            self.astra_db = astra_db_client
        else:
            self.astra_db = LibAstraDB(
                token=self.token,
                api_endpoint=self.api_endpoint,
                namespace=self.namespace,
            )
        self.collection = self.astra_db.create_collection(self.collection_name)

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve all session messages from DB"""
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

    def add_message(self, message: BaseMessage) -> None:
        """Write a message to the table"""
        self.collection.insert_one(
            {
                "timestamp": time.time(),
                "session_id": self.session_id,
                "body_blob": json.dumps(message_to_dict(message)),
            }
        )

    def clear(self) -> None:
        """Clear session memory from DB"""
        self.collection.delete_many(filter={"session_id": self.session_id})
