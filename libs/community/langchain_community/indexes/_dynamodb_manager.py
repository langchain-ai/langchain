import logging
import time
from decimal import Decimal
from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union, cast

from langchain_core.indexing import RecordManager

IMPORT_BOTO3_ERROR = "Unable to import boto3, please install with `pip install boto3`."


logger = logging.getLogger(__name__)


DynamoValue = Union[str, int, float, Decimal]

# DynamoDB has a limit of 25 items per batch operation
MAX_BATCH_SIZE = 25

# DynamoDB field names
KEY_FIELD = "index_key"
NAMESPACE_FIELD = "namespace"
GROUP_ID_FIELD = "group_id"
UPDATED_AT_FIELD = "updated_at"


def batched(iterable: Iterable, n: int) -> Iterable[list]:
    """
    Batch data into lists of length n. The last batch may be shorter.
    Drop-in for itertools.batched, which is only available in later Python versions.
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be >= 1")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _to_decimal(value: float) -> Decimal:
    return Decimal(str(value))


class DynamoDBRecordManager(RecordManager):
    """A DynamoDB based implementation of the record manager."""

    def __init__(
        self,
        namespace: str,
        *,
        table_name: str = "langchain_records",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        """Initialize the DynamoDBRecordManager.

        Args:
            namespace: The namespace associated with this record manager.
            table_name: The name of the DynamoDB table to use.
                Default is 'langchain_records'.
            region_name: AWS region name.
                If not provided, will use default from AWS config.
            aws_access_key_id: AWS access key ID.
                If not provided, will use default credentials.
            aws_secret_access_key: AWS secret access key.
                If not provided, will use default credentials.
            endpoint_url: Optional endpoint URL for DynamoDB (useful for local testing).
        """
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "Unable to import boto3, please install with `pip install boto3`."
            ) from e

        super().__init__(namespace=namespace)
        self.table_name = table_name

        self.dynamodb = boto3.resource(
            service_name="dynamodb",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
        )
        self.dynamodb_client = boto3.client(
            service_name="dynamodb",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
        )
        self.table = self.dynamodb.Table(table_name)

    def create_schema(self) -> None:
        """
        User is responsible for creating the table with the following schema:

        TableName=self.table_name
        KeySchema=[
            {"AttributeName": "index_key", "KeyType": "HASH"},
            {"AttributeName": "namespace", "KeyType": "RANGE"},
        ]
        AttributeDefinitions=[
            {"AttributeName": "index_key", "AttributeType": "S"},
            {"AttributeName": "namespace", "AttributeType": "S"},
            {"AttributeName": "group_id", "AttributeType": "S"},
            {"AttributeName": "updated_at", "AttributeType": "N"},
        ]
        GlobalSecondaryIndexes=[
            {
                "IndexName": f"group_id-updated_at-index",
                "KeySchema": [
                    {"AttributeName": "group_id", "KeyType": "HASH"},
                    {"AttributeName": "updated_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ]
        """
        pass

    async def acreate_schema(self) -> None:
        pass

    def get_time(self) -> float:
        # The docstring in the parent class mentions using a monotonic clock,
        # but it seems the official langchain implementations
        # use the database server time,
        # which isn't guaranteed to be monotonic either.
        return time.time()

    async def aget_time(self) -> float:
        # DynamoDB doesn't have native async support
        return self.get_time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        # Validate inputs
        if group_ids is not None and len(group_ids) != len(keys):
            raise ValueError(
                f"Length of keys ({len(keys)}) does not match "
                f"length of group_ids ({len(group_ids)})"
            )

        current_time = self.get_time()
        if time_at_least is not None and current_time < time_at_least:
            raise ValueError(
                f"Server time is behind: current time {current_time} "
                f"is less than required time {time_at_least}"
            )

        # Create a list of (key, group_id) tuples,
        # filtering out duplicates while preserving order
        key_group_pairs: List[tuple[str, Optional[str]]] = []
        seen_keys = set()

        for i, key in enumerate(keys):
            if key not in seen_keys:
                seen_keys.add(key)
                group_id = group_ids[i] if group_ids is not None else None
                key_group_pairs.append((key, group_id))

        # Process in batches to respect DynamoDB limits
        for batch_pairs in batched(key_group_pairs, MAX_BATCH_SIZE):
            with self.table.batch_writer() as batch:
                for key, group_id in batch_pairs:
                    item = {
                        KEY_FIELD: key,
                        NAMESPACE_FIELD: self.namespace,
                        UPDATED_AT_FIELD: _to_decimal(current_time),
                    }

                    if group_id is not None:
                        item[GROUP_ID_FIELD] = group_id

                    batch.put_item(Item=item)

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        # DynamoDB doesn't have native async support, use sync version
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> List[bool]:
        if not keys:
            return []

        results = [False] * len(keys)
        key_to_index = {key: i for i, key in enumerate(keys)}

        # Process in batches to respect DynamoDB limits
        for keys_chunk in batched(keys, MAX_BATCH_SIZE):
            request_items = {
                self.table_name: {
                    "Keys": [
                        {KEY_FIELD: key, NAMESPACE_FIELD: self.namespace}
                        for key in keys_chunk
                    ]
                }
            }

            response = self.dynamodb.batch_get_item(RequestItems=request_items)

            # Update results for found items
            table_responses = response.get("Responses", {})
            if self.table_name in table_responses:
                for item in table_responses[self.table_name]:
                    key = cast(str, item[KEY_FIELD])
                    if key in key_to_index:
                        results[key_to_index[key]] = True

            # Check for unprocessed keys
            unprocessed_keys = response.get("UnprocessedKeys", {})
            if unprocessed_keys:
                unprocessed_items = unprocessed_keys.get(self.table_name, {})
                unprocessed_count = len(unprocessed_items.get("Keys", []))
                if unprocessed_count > 0:
                    total_keys = len(keys_chunk)
                    raise RuntimeError(
                        f"Failed to process {unprocessed_count} "
                        f"out of {total_keys} keys in batch. "
                        "This may indicate throughput limits being reached."
                    )

        return results

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        # DynamoDB doesn't have native async support, use sync version
        return self.exists(keys)

    def _paginate_results(
        self,
        operation_kwargs: Dict[str, Any],
        operation_func: Callable,
        *,
        limit: Optional[int] = None,
    ) -> set[str]:
        keys: set[str] = set()

        # Safety limit to prevent infinite loops
        max_iterations = 1000
        iteration = 0

        response = operation_func(**operation_kwargs)
        for item in response.get("Items", []):
            key = str(item[KEY_FIELD])
            keys.add(key)
            if limit and len(keys) >= limit:
                return keys

        while (
            "LastEvaluatedKey" in response
            and (not limit or len(keys) < limit)
            and iteration < max_iterations
        ):
            operation_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            response = operation_func(**operation_kwargs)
            for item in response.get("Items", []):
                key = str(item[KEY_FIELD])
                keys.add(key)
                if limit and len(keys) >= limit:
                    return keys
            iteration += 1

        if iteration == max_iterations and "LastEvaluatedKey" in response:
            raise RuntimeError(
                f"Operation exceeded maximum number of iterations ({max_iterations}). "
                "This might indicate an unexpectedly large result set."
            )

        return keys

    def _list_keys_by_group_ids(
        self,
        group_ids: Sequence[str],
        *,
        before: Optional[Decimal] = None,
        after: Optional[Decimal] = None,
        limit: Optional[int] = None,
    ) -> set[str]:
        """List keys by querying the group_id-updated_at index."""
        keys: set[str] = set()
        for group_id in group_ids:
            # Build key condition expression (partition key and both time conditions)
            key_condition_parts = ["#group_id = :group_id"]

            # Both time conditions must go in KeyConditionExpression
            # since updated_at is a key
            if before is not None and after is not None:
                key_condition_parts.append("#updated_at BETWEEN :after AND :before")
            elif before is not None:
                key_condition_parts.append("#updated_at < :before")
            elif after is not None:
                key_condition_parts.append("#updated_at > :after")

            # Only filter on namespace
            filter_parts = ["#namespace = :namespace"]

            expr_attr_names: dict[str, Any] = {
                "#group_id": GROUP_ID_FIELD,
                "#namespace": NAMESPACE_FIELD,
            }
            expr_attr_values: dict[str, Any] = {
                ":group_id": group_id,
                ":namespace": self.namespace,
            }

            # Only add updated_at attribute name if we're using it
            if before is not None or after is not None:
                expr_attr_names["#updated_at"] = UPDATED_AT_FIELD

            if before is not None:
                expr_attr_values[":before"] = before
            if after is not None:
                expr_attr_values[":after"] = after

            query_args: dict[str, Any] = {
                "IndexName": "group_id-updated_at-index",
                "KeyConditionExpression": " AND ".join(key_condition_parts),
                "FilterExpression": " AND ".join(filter_parts),
                "ExpressionAttributeNames": expr_attr_names,
                "ExpressionAttributeValues": expr_attr_values,
            }

            if limit is not None:
                query_args["Limit"] = limit

            group_keys = self._paginate_results(
                query_args,
                self.table.query,
                limit=limit,
            )
            keys.update(group_keys)
            if limit and len(keys) >= limit:
                break

        return keys

    def _list_keys_by_scan(
        self,
        *,
        before: Optional[Decimal] = None,
        after: Optional[Decimal] = None,
        limit: Optional[int] = None,
    ) -> set[str]:
        """List keys by scanning the table."""
        filter_parts = ["#namespace = :namespace"]
        if before is not None:
            filter_parts.append("#updated_at < :before")
        if after is not None:
            filter_parts.append("#updated_at > :after")

        expr_attr_names: dict[str, str] = {"#namespace": NAMESPACE_FIELD}
        expr_attr_values: dict[str, Any] = {":namespace": self.namespace}

        if before is not None or after is not None:
            expr_attr_names["#updated_at"] = UPDATED_AT_FIELD
        if before is not None:
            expr_attr_values[":before"] = before
        if after is not None:
            expr_attr_values[":after"] = after

        scan_kwargs: dict[str, Any] = {
            "FilterExpression": " AND ".join(filter_parts),
            "ExpressionAttributeNames": expr_attr_names,
            "ExpressionAttributeValues": expr_attr_values,
        }
        if limit:
            scan_kwargs["Limit"] = limit

        return self._paginate_results(
            scan_kwargs,
            self.table.scan,
            limit=limit,
        )

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        dec_before = _to_decimal(before) if before is not None else None
        dec_after = _to_decimal(after) if after is not None else None

        if group_ids:
            keys = self._list_keys_by_group_ids(
                group_ids,
                before=dec_before,
                after=dec_after,
                limit=limit,
            )
        else:
            keys = self._list_keys_by_scan(
                before=dec_before,
                after=dec_after,
                limit=limit,
            )

        return list(keys)

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        # DynamoDB doesn't have native async support, use sync version
        return self.list_keys(
            before=before,
            after=after,
            group_ids=group_ids,
            limit=limit,
        )

    def delete_keys(self, keys: Sequence[str]) -> None:
        # Process in batches to respect DynamoDB limits
        for keys_chunk in batched(keys, MAX_BATCH_SIZE):
            with self.table.batch_writer() as batch:
                for key in keys_chunk:
                    batch.delete_item(
                        Key={KEY_FIELD: key, NAMESPACE_FIELD: self.namespace}
                    )

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        # DynamoDB doesn't have native async support, use sync version
        self.delete_keys(keys)
