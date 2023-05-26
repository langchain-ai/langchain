import logging
from typing import List

from langchain.schema import (
    BaseChatMessageHistory,
    BaseMessage,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)


class DynamoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in AWS DynamoDB.
    This class expects that a DynamoDB table with name `table_name`
    and a partition Key of `SessionId` is present.

    Args:
        table_name: name of the DynamoDB table
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
    """

    def __init__(self, table_name: str, session_id: str):
        import boto3

        client = boto3.resource("dynamodb")
        self.table = client.Table(table_name)
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from DynamoDB"""
        from botocore.exceptions import ClientError

        try:
            response = self.table.get_item(Key={"SessionId": self.session_id})
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("No record found with session id: %s", self.session_id)
            else:
                logger.error(error)

        if response and "Item" in response:
            items = response["Item"]["History"]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in DynamoDB"""
        from botocore.exceptions import ClientError

        messages = messages_to_dict(self.messages)
        _message = _message_to_dict(message)
        messages.append(_message)

        try:
            self.table.put_item(
                Item={"SessionId": self.session_id, "History": messages}
            )
        except ClientError as err:
            logger.error(err)

    def clear(self) -> None:
        """Clear session memory from DynamoDB"""
        from botocore.exceptions import ClientError

        try:
            self.table.delete_item(Key={"SessionId": self.session_id})
        except ClientError as err:
            logger.error(err)
