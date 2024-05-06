import json
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)


class XataChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Xata database."""

    def __init__(
        self,
        session_id: str,
        db_url: str,
        api_key: str,
        branch_name: str = "main",
        table_name: str = "messages",
        create_table: bool = True,
    ) -> None:
        """Initialize with Xata client."""
        try:
            from xata.client import XataClient
        except ImportError:
            raise ImportError(
                "Could not import xata python package. "
                "Please install it with `pip install xata`."
            )
        self._client = XataClient(
            api_key=api_key, db_url=db_url, branch_name=branch_name
        )
        self._table_name = table_name
        self._session_id = session_id

        if create_table:
            self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        r = self._client.table().get_schema(self._table_name)
        if r.status_code <= 299:
            return
        if r.status_code != 404:
            raise Exception(
                f"Error checking if table exists in Xata: {r.status_code} {r}"
            )
        r = self._client.table().create(self._table_name)
        if r.status_code > 299:
            raise Exception(f"Error creating table in Xata: {r.status_code} {r}")
        r = self._client.table().set_schema(
            self._table_name,
            payload={
                "columns": [
                    {"name": "sessionId", "type": "string"},
                    {"name": "type", "type": "string"},
                    {"name": "role", "type": "string"},
                    {"name": "content", "type": "text"},
                    {"name": "name", "type": "string"},
                    {"name": "additionalKwargs", "type": "json"},
                ]
            },
        )
        if r.status_code > 299:
            raise Exception(f"Error setting table schema in Xata: {r.status_code} {r}")

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the Xata table"""
        msg = message_to_dict(message)
        r = self._client.records().insert(
            self._table_name,
            {
                "sessionId": self._session_id,
                "type": msg["type"],
                "content": message.content,
                "additionalKwargs": json.dumps(message.additional_kwargs),
                "role": msg["data"].get("role"),
                "name": msg["data"].get("name"),
            },
        )
        if r.status_code > 299:
            raise Exception(f"Error adding message to Xata: {r.status_code} {r}")

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        r = self._client.data().query(
            self._table_name,
            payload={
                "filter": {
                    "sessionId": self._session_id,
                },
                "sort": {"xata.createdAt": "asc"},
            },
        )
        if r.status_code != 200:
            raise Exception(f"Error running query: {r.status_code} {r}")
        msgs = messages_from_dict(
            [
                {
                    "type": m["type"],
                    "data": {
                        "content": m["content"],
                        "role": m.get("role"),
                        "name": m.get("name"),
                        "additional_kwargs": json.loads(m["additionalKwargs"]),
                    },
                }
                for m in r["records"]
            ]
        )
        return msgs

    def clear(self) -> None:
        """Delete session from Xata table."""
        while True:
            r = self._client.data().query(
                self._table_name,
                payload={
                    "columns": ["id"],
                    "filter": {
                        "sessionId": self._session_id,
                    },
                },
            )
            if r.status_code != 200:
                raise Exception(f"Error running query: {r.status_code} {r}")
            ids = [rec["id"] for rec in r["records"]]
            if len(ids) == 0:
                break
            operations = [
                {"delete": {"table": self._table_name, "id": id}} for id in ids
            ]
            self._client.records().transaction(payload={"operations": operations})
