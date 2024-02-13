from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.stores import ByteStore

if TYPE_CHECKING:
    from boto3.session import Session

logger = logging.getLogger(__name__)

DYNAMODB_MAX_BATCH_SIZE = 100
class DynamoDbStore(ByteStore):
    """ByteStore implementation using AWS DynamoDb as the underlying store.

 This class expects that a DynamoDB table exists with name `table_name`. The
 table must use a compound key, by default the PK is PartitionKey and
 the SK is SortKey

    Args:
        table_name: name of the DynamoDB table
        endpoint_url: URL of the AWS endpoint to connect to. This argument
            is optional and useful for test purposes, like using Localstack.
            If you plan to use AWS cloud service, you normally don't have to
            worry about setting the endpoint_url.
        partition_key_name: name of the partition key of the DynamoDB table.
            This argument is optional, defaulting to "PartitionKey".
        sort_key_name: name of the sort key of the DynamoDB table. This argument
            is optional, defaulting to "SortKey".
        partition_key_length: How many characters of the key to use when generating
            a value for the partition key. A longer length allows for more efficiency
            when calling yield_keys, but is overkill for small datasets where a
            single partition won't exceed the 10GB limit.
            This argument is optional, defaulting to 1.
        kms_key_id: an optional AWS KMS Key ID, AWS KMS Key ARN, or AWS KMS Alias for
            client-side encryption
        ttl: Optional Time-to-live (TTL) in seconds. Allows you to define a per-item
            expiration timestamp that indicates when an item can be deleted from the
            table. DynamoDB handles deletion of expired items without consuming
            write throughput. To enable this feature on the table, follow the
            [AWS DynamoDB documentation](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/time-to-live-ttl-how-to.html)


    Examples:
        Create a DynamoDbStore instance and perform operations on it:

        .. code-block:: python

            # Instantiate the DynamoDbStore with a Dynamodb connection
            from langchain_community.storage import DynamoDbStore

            dynamodb_store = DynamoDbStore(table_name="docs")

            # Set values for keys
            dynamodb_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = dynamodb_store.mget(["key1", "key2"])
            # [b"value1", b"value2"]

            # Delete keys
            dynamodb_store.mdelete(["key1"])

            # Iterate over keys
            for key in dynamodb_store.yield_keys():
                print(key)
    """

    def __init__(
            self,
            table_name: str,
            endpoint_url: Optional[str] = None,
            partition_key_name: str = "PartitionKey",
            sort_key_name: str = "SortKey",
            partition_key_length: int = 1,
            boto3_session: Optional[Session] = None,
            kms_key_id: Optional[str] = None,
            ttl: Optional[int] = None,
            ttl_key_name: str = "expireAt",
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

            client = boto3.resource("dynamodb", endpoint_url=endpoint_url)

        self.client = client
        self.table = client.Table(table_name)
        self.ttl = ttl
        self.ttl_key_name = ttl_key_name
        self.partition_key_name = partition_key_name
        self.sort_key_name = sort_key_name
        self.partition_key_length = partition_key_length

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
                attribute_actions={"History": CryptoAction.ENCRYPT_AND_SIGN},
            )

            aws_kms_cmp = AwsKmsCryptographicMaterialsProvider(key_id=kms_key_id)
            self.table = EncryptedTable(
                table=self.table,
                materials_provider=aws_kms_cmp,
                attribute_actions=actions,
                auto_refresh_table_indexes=False,
            )

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:

        res: OrderedDict[str, Optional[bytes]] = OrderedDict()

        # establish order of responses
        for key in keys:
            res[key] = None

        for i in range(0, len(keys), DYNAMODB_MAX_BATCH_SIZE):
            chunk = keys[i:i + DYNAMODB_MAX_BATCH_SIZE]

            unprocessed_chunk = chunk

            while unprocessed_chunk:
                response = self.client.batch_get_item(
                    RequestItems={
                        self.table.name: {
                            "Keys": [
                                self.make_key_dict(key)
                                for key
                                in unprocessed_chunk
                            ],
                            "AttributesToGet": [self.sort_key_name, "Bytes"]
                        }
                    }
                )
                for item in response["Responses"][self.table.name]:
                    if item and "Bytes" in item:
                        res[item[self.sort_key_name]] = item["Bytes"].value

                if "UnprocessedKeys" in response and len(response["UnprocessedKeys"]):
                    unprocessed_chunk = [
                        key[self.sort_key_name]
                        for key
                        in response["UnprocessedKeys"][self.table.name]["Keys"]
                    ]
                else:
                    unprocessed_chunk = None

        return list(res.values())

    def make_pk(self, key: str) -> str:
        return key[:self.partition_key_length]

    def make_key_dict(self, key: str) -> Dict[str, str]:
        return {
            self.partition_key_name: self.make_pk(key),
            self.sort_key_name: key,
        }

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Append the message to the record in DynamoDB"""
        try:
            from botocore.exceptions import ClientError
        except ImportError as e:
            raise ImportError(
                "Unable to import botocore, please install with `pip install botocore`."
            ) from e

        default_attrs = {}

        if self.ttl:
            import time
            expire_at = int(time.time()) + self.ttl
            default_attrs[self.ttl_key_name] = expire_at

        for key, value in key_value_pairs:
            try:
                self.table.put_item(
                    Item={**self.make_key_dict(key), "Bytes": value, **default_attrs}
                )
            except ClientError as err:
                logger.error(err)
                raise

    def mdelete(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.table.delete_item(
                Key=self.make_key_dict(key)
            )

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Yield keys in the store."""

        next_page = True
        exclusive_start_key: Optional[Dict[str, str]] = None

        while next_page:
            response = None

            query_kwargs = {}
            if exclusive_start_key:
                query_kwargs["ExclusiveStartKey"] = exclusive_start_key

            if prefix:
                partition = self.make_pk(prefix)

                response = self.table.query(
                    **query_kwargs,
                    AttributesToGet=[self.sort_key_name],
                    KeyConditionExpression=
                        "#PK = :partition AND begins_with(#SK, :sort)",
                    ExpressionAttributeNames={
                        "#PK": self.partition_key_name,
                        "#SK": self.sort_key_name
                    },
                    ExpressionAttributeValues={
                        ":partition": partition,
                        ":sort": prefix
                    },
                    Limit=1000,
                )
            else:
                response = self.table.scan(
                    **query_kwargs,
                    AttributesToGet=[self.sort_key_name],
                    Limit=1000,
                )

            for item in response["Items"]:
                yield item[self.sort_key_name]

            exclusive_start_key = response.get("LastEvaluatedKey", None)
            next_page = bool(exclusive_start_key)
