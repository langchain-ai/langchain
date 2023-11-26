from datetime import datetime
from time import sleep
from typing import Any, Callable, List, Union
from uuid import uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)


class RocksetChatMessageHistory(BaseChatMessageHistory):
    """Uses Rockset to store chat messages.

    To use, ensure that the `rockset` python package installed.

    Example:
        .. code-block:: python

            from langchain.memory.chat_message_histories import (
                RocksetChatMessageHistory
            )
            from rockset import RocksetClient

            history = RocksetChatMessageHistory(
                session_id="MySession",
                client=RocksetClient(),
                collection="langchain_demo",
                sync=True
            )

            history.add_user_message("hi!")
            history.add_ai_message("whats up?")

            print(history.messages)
    """

    # You should set these values based on your VI.
    # These values are configured for the typical
    # free VI. Read more about VIs here:
    # https://rockset.com/docs/instances
    SLEEP_INTERVAL_MS: int = 5
    ADD_TIMEOUT_MS: int = 5000
    CREATE_TIMEOUT_MS: int = 20000

    def _wait_until(self, method: Callable, timeout: int, **method_params: Any) -> None:
        """Sleeps until meth() evaluates to true. Passes kwargs into
        meth.
        """
        start = datetime.now()
        while not method(**method_params):
            curr = datetime.now()
            if (curr - start).total_seconds() * 1000 > timeout:
                raise TimeoutError(f"{method} timed out at {timeout} ms")
            sleep(RocksetChatMessageHistory.SLEEP_INTERVAL_MS / 1000)

    def _query(self, query: str, **query_params: Any) -> List[Any]:
        """Executes an SQL statement and returns the result
        Args:
            - query: The SQL string
            - **query_params: Parameters to pass into the query
        """
        return self.client.sql(query, params=query_params).results

    def _create_collection(self) -> None:
        """Creates a collection for this message history"""
        self.client.Collections.create_s3_collection(
            name=self.collection, workspace=self.workspace
        )

    def _collection_exists(self) -> bool:
        """Checks whether a collection exists for this message history"""
        try:
            self.client.Collections.get(collection=self.collection)
        except self.rockset.exceptions.NotFoundException:
            return False
        return True

    def _collection_is_ready(self) -> bool:
        """Checks whether the collection for this message history is ready
        to be queried
        """
        return (
            self.client.Collections.get(collection=self.collection).data.status
            == "READY"
        )

    def _document_exists(self) -> bool:
        return (
            len(
                self._query(
                    f"""
                        SELECT 1
                        FROM {self.location} 
                        WHERE _id=:session_id
                        LIMIT 1
                    """,
                    session_id=self.session_id,
                )
            )
            != 0
        )

    def _wait_until_collection_created(self) -> None:
        """Sleeps until the collection for this message history is ready
        to be queried
        """
        self._wait_until(
            lambda: self._collection_is_ready(),
            RocksetChatMessageHistory.CREATE_TIMEOUT_MS,
        )

    def _wait_until_message_added(self, message_id: str) -> None:
        """Sleeps until a message is added to the messages list"""
        self._wait_until(
            lambda message_id: len(
                self._query(
                    f"""
                        SELECT * 
                        FROM UNNEST((
                            SELECT {self.messages_key}
                            FROM {self.location}
                            WHERE _id = :session_id
                        )) AS message
                        WHERE message.data.additional_kwargs.id = :message_id
                        LIMIT 1
                    """,
                    session_id=self.session_id,
                    message_id=message_id,
                ),
            )
            != 0,
            RocksetChatMessageHistory.ADD_TIMEOUT_MS,
            message_id=message_id,
        )

    def _create_empty_doc(self) -> None:
        """Creates or replaces a document for this message history with no
        messages"""
        self.client.Documents.add_documents(
            collection=self.collection,
            workspace=self.workspace,
            data=[{"_id": self.session_id, self.messages_key: []}],
        )

    def __init__(
        self,
        session_id: str,
        client: Any,
        collection: str,
        workspace: str = "commons",
        messages_key: str = "messages",
        sync: bool = False,
        message_uuid_method: Callable[[], Union[str, int]] = lambda: str(uuid4()),
    ) -> None:
        """Constructs a new RocksetChatMessageHistory.

        Args:
            - session_id: The ID of the chat session
            - client: The RocksetClient object to use to query
            - collection: The name of the collection to use to store chat
                          messages. If a collection with the given name
                          does not exist in the workspace, it is created.
            - workspace: The workspace containing `collection`. Defaults
                         to `"commons"`
            - messages_key: The DB column containing message history.
                            Defaults to `"messages"`
            - sync: Whether to wait for messages to be added. Defaults
                    to `False`. NOTE: setting this to `True` will slow
                    down performance.
            - message_uuid_method: The method that generates message IDs.
                    If set, all messages will have an `id` field within the
                    `additional_kwargs` property. If this param is not set
                    and `sync` is `False`, message IDs will not be created.
                    If this param is not set and `sync` is `True`, the
                    `uuid.uuid4` method will be used to create message IDs.
        """
        try:
            import rockset
        except ImportError:
            raise ImportError(
                "Could not import rockset client python package. "
                "Please install it with `pip install rockset`."
            )

        if not isinstance(client, rockset.RocksetClient):
            raise ValueError(
                f"client should be an instance of rockset.RocksetClient, "
                f"got {type(client)}"
            )

        self.session_id = session_id
        self.client = client
        self.collection = collection
        self.workspace = workspace
        self.location = f'"{self.workspace}"."{self.collection}"'
        self.rockset = rockset
        self.messages_key = messages_key
        self.message_uuid_method = message_uuid_method
        self.sync = sync

        try:
            self.client.set_application("langchain")
        except AttributeError:
            # ignore
            pass

        if not self._collection_exists():
            self._create_collection()
            self._wait_until_collection_created()
            self._create_empty_doc()
        elif not self._document_exists():
            self._create_empty_doc()

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Messages in this chat history."""
        return messages_from_dict(
            self._query(
                f"""
                    SELECT *
                    FROM UNNEST ((
                        SELECT "{self.messages_key}"
                        FROM {self.location}
                        WHERE _id = :session_id
                    ))
                """,
                session_id=self.session_id,
            )
        )

    def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the history.

        Args:
            message: A BaseMessage object to store.
        """
        if self.sync and "id" not in message.additional_kwargs:
            message.additional_kwargs["id"] = self.message_uuid_method()
        self.client.Documents.patch_documents(
            collection=self.collection,
            workspace=self.workspace,
            data=[
                self.rockset.model.patch_document.PatchDocument(
                    id=self.session_id,
                    patch=[
                        self.rockset.model.patch_operation.PatchOperation(
                            op="ADD",
                            path=f"/{self.messages_key}/-",
                            value=message_to_dict(message),
                        )
                    ],
                )
            ],
        )
        if self.sync:
            self._wait_until_message_added(message.additional_kwargs["id"])

    def clear(self) -> None:
        """Removes all messages from the chat history"""
        self._create_empty_doc()
        if self.sync:
            self._wait_until(
                lambda: not self.messages,
                RocksetChatMessageHistory.ADD_TIMEOUT_MS,
            )
