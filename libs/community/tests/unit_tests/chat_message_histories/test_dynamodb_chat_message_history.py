from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
from pytest_mock import MockerFixture

from langchain_community.chat_message_histories.dynamodb import (
    DynamoDBChatMessageHistory,
)

HISTORY_KEY = "ChatHistory"
TTL_KEY = "TimeToLive"


def dict_to_key(Key: dict) -> tuple:
    return tuple(sorted(Key.items()))


class MockDynamoDBChatHistoryTable:
    """Contains the table for the mock DynamoDB resource."""

    class Table:
        """Contains methods to mock Boto's DynamoDB calls"""

        def __init__(self, *args: tuple, **kwargs: dict[str, Any]) -> None:
            self.items: dict = dict()

        def get_item(self, Key: dict) -> dict:
            return self.items.get(dict_to_key(Key), dict())

        def update_item(
            self, Key: dict, UpdateExpression: str, ExpressionAttributeValues: dict
        ) -> None:
            update_dict = {HISTORY_KEY: ExpressionAttributeValues[":h"]}

            expression = UpdateExpression.split(", ")
            if len(expression) > 1:
                ttl_key_name = expression[1].replace(" = :t", "")
                update_dict.update({ttl_key_name: ExpressionAttributeValues[":t"]})

            self.items[dict_to_key(Key)] = {"Item": update_dict}

        def delete_item(self, Key: dict) -> None:
            if dict_to_key(Key) in self.items.keys():
                del self.items[dict_to_key(Key)]


class MockBoto3DynamoDBSession:
    """Creates a mock Boto session to return a DynamoDB table for testing
    DynamoDBChatMessageHistory class methods."""

    def resource(
        self, *args: tuple, **kwargs: dict[str, Any]
    ) -> MockDynamoDBChatHistoryTable:
        return MockDynamoDBChatHistoryTable()


@pytest.fixture(scope="module")
def chat_history_config() -> dict:
    return {"key": {"primaryKey": "foo", "secondaryKey": 123}, "ttl": 600}


@pytest.fixture(scope="class")
def ddb_chat_history_with_mock_boto_session(
    chat_history_config: dict,
) -> DynamoDBChatMessageHistory:
    return DynamoDBChatMessageHistory(
        table_name="test_table",
        session_id="test_session",
        boto3_session=MockBoto3DynamoDBSession(),
        key=chat_history_config["key"],
        ttl=chat_history_config["ttl"],
        ttl_key_name=TTL_KEY,
        history_messages_key=HISTORY_KEY,
    )


class TestDynamoDBChatMessageHistory:
    @pytest.mark.requires("botocore")
    def test_add_message(
        self,
        mocker: MockerFixture,
        ddb_chat_history_with_mock_boto_session: DynamoDBChatMessageHistory,
        chat_history_config: dict,
    ) -> None:
        # For verifying the TTL value
        mock_time_1 = 1234567000
        mock_time_2 = 1234568000

        # Get the history class and mock DynamoDB table
        history: DynamoDBChatMessageHistory = ddb_chat_history_with_mock_boto_session
        history_table: MockDynamoDBChatHistoryTable.Table = history.table
        history_item = history_table.get_item(chat_history_config["key"])
        assert history_item == dict()  # Should be empty so far

        # Add the first message
        mocker.patch("time.time", lambda: mock_time_1)
        first_message = HumanMessage(content="new human message")
        history.add_message(message=first_message)
        item_after_human_message = history_table.get_item(chat_history_config["key"])[
            "Item"
        ]
        assert item_after_human_message[HISTORY_KEY] == messages_to_dict(
            [first_message]
        )  # History should only contain the first message
        assert (
            item_after_human_message[TTL_KEY]
            == mock_time_1 + chat_history_config["ttl"]
        )  # TTL should exist

        # Add the second message
        mocker.patch("time.time", lambda: mock_time_2)
        second_message = AIMessage(content="new AI response")
        history.add_message(message=second_message)
        item_after_ai_message = history_table.get_item(chat_history_config["key"])[
            "Item"
        ]
        assert item_after_ai_message[HISTORY_KEY] == messages_to_dict(
            [first_message, second_message]
        )  # Second message should have appended
        assert (
            item_after_ai_message[TTL_KEY] == mock_time_2 + chat_history_config["ttl"]
        )  # TTL should have updated

    @pytest.mark.requires("botocore")
    def test_clear(
        self,
        ddb_chat_history_with_mock_boto_session: DynamoDBChatMessageHistory,
        chat_history_config: dict,
    ) -> None:
        # Get the history class and mock DynamoDB table
        history: DynamoDBChatMessageHistory = ddb_chat_history_with_mock_boto_session
        history_table: MockDynamoDBChatHistoryTable.Table = history.table

        # Use new key so we get a new chat session and add a message to the new session
        new_session_key = {"primaryKey": "bar", "secondaryKey": 456}
        history.key = new_session_key
        history.add_message(
            message=HumanMessage(content="human message for different chat session")
        )

        # Chat history table should now contain both chat sessions
        assert set(history_table.items.keys()) == {
            dict_to_key(chat_history_config["key"]),
            dict_to_key(new_session_key),
        }

        # Reset the key to the original and use the clear method
        history.key = chat_history_config["key"]
        history.clear()

        # Only the original chat session should be removed
        assert set(history_table.items.keys()) == {dict_to_key(new_session_key)}
