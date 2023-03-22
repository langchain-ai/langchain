import logging
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError

from langchain.schema import (
    BaseMessage,
    MessageDB,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)


class DynamoDBMessageDB(MessageDB):
    def __init__(self, table_name: str):
        client = boto3.resource("dynamodb")
        self.table = client.Table(table_name)

    def read(
        self, session_id: str, as_dict: bool = False
    ) -> Union[List[BaseMessage], List[Dict[str, Any]]]:
        """Retrieve the messages from DynamoDB"""
        try:
            response = self.table.get_item(Key={"SessionId": session_id})
        except ClientError as error:
            if error.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning("No record found with session id: %s", session_id)
            else:
                logger.error(error)

        if response and "Item" in response:
            items = response["Item"]["History"]
        else:
            items = []

        if as_dict:
            return items

        messages = messages_from_dict(items)
        return messages

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append the message to the record in DynamoDB"""
        messages = self.read(session_id, as_dict=True)
        _message = _message_to_dict(message)
        messages.append(_message)

        try:
            self.table.put_item(Item={"SessionId": session_id, "History": messages})
        except ClientError as err:
            logger.error(err)

    def clear(self, session_id: str) -> None:
        """Clear session memory from DynamoDB"""
        try:
            self.table.delete_item(Key={"SessionId": session_id})
        except ClientError as err:
            logger.error(err)
