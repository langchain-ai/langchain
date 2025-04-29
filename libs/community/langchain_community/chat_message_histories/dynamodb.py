from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

if TYPE_CHECKING:
    from boto3.session import Session


def convert_messages(item: List) -> List:
    if isinstance(item, list):
        return [convert_messages(i) for i in item]
    elif isinstance(item, dict):
        return {k: convert_messages(v) for k, v in item.items()}
    elif isinstance(item, float):
        return Decimal(str(item))
    return item


class DynamoDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in AWS DynamoDB.

    This class expects that a DynamoDB table exists with name `table_name`

    Args:
        table_name: name of the DynamoDB table
        session_id: arbitrary key that is used to store the messages
            of a single chat session.
        endpoint_url: URL of the AWS endpoint to connect to. This argument
            is optional and useful for test purposes, like using Localstack.
            If you plan to use AWS cloud service, you normally don't have to
            worry about setting the endpoint_url.
        primary_key_name: name of the primary key of the DynamoDB table. This argument
            is optional, defaulting to "SessionId".
        key: an optional dictionary with a custom primary and secondary key.
            This argument is optional, but useful when using composite dynamodb keys, or
            isolating records based off of application details such as a user id.
            This may also contain global and local secondary index keys.
        kms_key_id: an optional AWS KMS Key ID, AWS KMS Key ARN, or AWS KMS Alias for
            client-side encryption
        ttl: Optional Time-to-live (TTL) in seconds. Allows you to define a per-item
            expiration timestamp that indicates when an item can be deleted from the
            table. DynamoDB handles deletion of expired items without consuming
            write throughput. To enable this feature on the table, follow the
            [AWS DynamoDB documentation](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/time-to-live-ttl-how-to.html)
        history_size: Maximum number of messages to store. If None then there is no
            limit. If not None then only the latest `history_size` messages are stored.
        history_messages_key: Key for the chat history where the messages
            are stored and updated
        coerce_float_to_decimal: If True, all float values in the messages will be
            converted to Decimal.
    """

    def __init__(
        self,
        table_name: str,
        session_id: str,
        endpoint_url: Optional[str] = None,
        primary_key_name: str = "SessionId",
        key: Optional[Dict[str, str]] = None,
        boto3_session: Optional[Session] = None,
        kms_key_id: Optional[str] = None,
        ttl: Optional[int] = None,
        ttl_key_name: str = "expireAt",
        history_size: Optional[int] = None,
        history_messages_key: Optional[str] = "History",
        *,
        coerce_float_to_decimal: bool = False,
    ):
        if boto3_session:
            client = boto3_session.resource("dynamodb", endpoint_url=endpoint_url)
        else:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "Unable to import boto3, please install with `pip install boto3`."
                ) from e
            if endpoint_url:
                client = boto3.resource("dynamodb", endpoint_url=endpoint_url)
            else:
                client = boto3.resource("dynamodb")
        self.table = client.Table(table_name)
        self.session_id = session_id
        self.key: Dict = key or {primary_key_name: session_id}
        self.ttl = ttl
        self.ttl_key_name = ttl_key_name
        self.history_size = history_size
        self.history_messages_key = history_messages_key
        self.coerce_float_to_decimal = coerce_float_to_decimal

        if kms_key_id:
            try:
                from dynamodb_encryption_sdk.encrypted.table import EncryptedTable
                from dynamodb_encryption_sdk.identifiers import CryptoAction
                from dynamodb_encryption_sdk.material_providers.aws_kms import (
                    AwsKmsCryptographicMaterialsProvider,
                )
                from dynamodb_encryption_sdk.structures import AttributeActions
            except ImportError as e:
                raise ImportError(
                    "Unable to import dynamodb_encryption_sdk, please install with "
                    "`pip install dynamodb-encryption-sdk`."
                ) from e

            actions = AttributeActions(
                default_action=CryptoAction.DO_NOTHING,
                attribute_actions={
                    self.history_messages_key: CryptoAction.ENCRYPT_AND_SIGN
                },
            )
            aws_kms_cmp = AwsKmsCryptographicMaterialsProvider(key_id=kms_key_id)
            self.table = EncryptedTable(
                table=self.table,
                materials_provider=aws_kms_cmp,
                attribute_actions=actions,
                auto_refresh_table_indexes=False,
            )

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from DynamoDB"""
        response = None
        response = self.table.get_item(Key=self.key)

        if response and "Item" in response:
            items = response["Item"][self.history_messages_key]
        else:
            items = []

        messages = messages_from_dict(items)
        return messages

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError(
            "Direct assignment to 'messages' is not allowed."
            " Use the 'add_messages' instead."
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append the message to the record in DynamoDB"""
        existing_messages = messages_to_dict(self.messages)
        existing_messages.extend(messages_to_dict(messages))
        if self.coerce_float_to_decimal:
            existing_messages = convert_messages(existing_messages)

        if self.history_size:
            existing_messages = existing_messages[-self.history_size :]

        if self.ttl:
            import time

            expireAt = int(time.time()) + self.ttl
            self.table.update_item(
                Key={**self.key},
                UpdateExpression=(
                    f"set {self.history_messages_key} = :h, {self.ttl_key_name} = :t"
                ),
                ExpressionAttributeValues={":h": existing_messages, ":t": expireAt},
            )
        else:
            self.table.update_item(
                Key={**self.key},
                UpdateExpression=f"set {self.history_messages_key} = :h",
                ExpressionAttributeValues={":h": existing_messages},
            )

    def clear(self) -> None:
        """Clear session memory from DynamoDB"""
        self.table.delete_item(Key=self.key)
