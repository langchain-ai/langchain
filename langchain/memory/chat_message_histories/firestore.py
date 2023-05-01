"""Firestore Chat Message History."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from google.cloud.firestore import DocumentReference


class FirestoreChatMessageHistory(BaseChatMessageHistory):
    """Chat history backed by Google Firestore."""

    def __init__(
        self,
        collection_name: str,
        session_id: str,
        user_id: str,
    ):
        """
        Initialize a new instance of the FirestoreChatMessageHistory class.

        :param collection_name: The name of the collection to use.
        :param session_id: The session ID for the chat..
        :param user_id: The user ID for the chat.
        """
        self.collection_name = collection_name
        self.session_id = session_id
        self.user_id = user_id

        self._document: Optional[DocumentReference] = None
        self.messages: List[BaseMessage] = []

        self.prepare_firestore()

    def prepare_firestore(self) -> None:
        """Prepare the Firestore client.

        Use this function to make sure your database is ready.
        """
        try:
            import firebase_admin
            from firebase_admin import firestore
        except ImportError as e:
            logger.error(
                "Failed to import Firebase and Firestore: %s. "
                "Make sure to install the 'firebase-admin' module.",
                e,
            )
            raise e

        # For multiple instances, only initialize the app once.
        try:
            firebase_admin.get_app()
        except ValueError as e:
            logger.debug("Initializing Firebase app: %s", e)
            firebase_admin.initialize_app()

        self.firestore_client = firestore.client()
        self._document = self.firestore_client.collection(
            self.collection_name
        ).document(self.session_id)
        self.load_messages()

    def load_messages(self) -> None:
        """Retrieve the messages from Firestore"""
        if not self._document:
            raise ValueError("Document not initialized")
        doc = self._document.get()
        if doc.exists:
            data = doc.to_dict()
            if "messages" in data and len(data["messages"]) > 0:
                self.messages = messages_from_dict(data["messages"])

    def add_user_message(self, message: str) -> None:
        """Add a user message to the memory."""
        self.upsert_messages(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Add a AI message to the memory."""
        self.upsert_messages(AIMessage(content=message))

    def upsert_messages(self, new_message: Optional[BaseMessage] = None) -> None:
        """Update the Firestore document."""
        if new_message:
            self.messages.append(new_message)
        if not self._document:
            raise ValueError("Document not initialized")
        self._document.set(
            {
                "id": self.session_id,
                "user_id": self.user_id,
                "messages": messages_to_dict(self.messages),
            }
        )

    def clear(self) -> None:
        """Clear session memory from this memory and Firestore."""
        self.messages = []
        if self._document:
            self._document.delete()
