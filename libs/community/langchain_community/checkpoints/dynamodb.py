import logging
import pickle
from typing import Any, Optional

from boto3.session import Session
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint

logger = logging.getLogger(__name__)


class DynamoDbSaver(BaseCheckpointSaver):
    client: Any
    table: Any
    table_name: str
    primary_key_name: str
    checkpoint_field_name: str

    is_setup: bool = Field(False, init=False, repr=False)

    """Checkpoint saver that stores history in AWS DynamoDB.

    This class creates a DynamoDB table with name `table_name` if it does not exist

    Args:
        table_name: name of the DynamoDB table
        endpoint_url: URL of the AWS endpoint to connect to. This argument
            is optional and useful for test purposes, like using Localstack.
            If you plan to use AWS cloud service, you normally don't have to
            worry about setting the endpoint_url.
        primary_key_name: name of the primary key of the DynamoDB table. This argument
            is optional, defaulting to "SessionId".
        checkpoint_field_name: name of the field in the DynamoDB table where to store
            the checkpoint data. This argument is optional, defaulting to "Checkpoint".
        boto3_session: an optional boto3 session to use for the DynamoDB client
        kms_key_id: an optional AWS KMS Key ID, AWS KMS Key ARN, or AWS KMS Alias for
            client-side encryption
    """

    @classmethod
    def from_params(
        cls,
        table_name: str,
        endpoint_url: Optional[str] = None,
        primary_key_name: str = "ThreadId",
        checkpoint_field_name: str = "Checkpoint",
        boto3_session: Optional[Session] = None,
        kms_key_id: Optional[str] = None,
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
        table = client.Table(table_name)

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
            table = EncryptedTable(
                table=table,
                materials_provider=aws_kms_cmp,
                attribute_actions=actions,
                auto_refresh_table_indexes=False,
            )

        return DynamoDbSaver(
            client=client,
            table=table,
            table_name=table_name,
            primary_key_name=primary_key_name,
            checkpoint_field_name=checkpoint_field_name,
        )

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=str,
                name="Thread ID",
                description=None,
                default="",
                is_shared=True,
            ),
        ]

    def setup(self) -> None:
        if self.is_setup:
            return

        # Define table schema
        table_name = self.table_name

        # Check if the table already exists
        existing_tables = self.client.meta.client.list_tables()["TableNames"]
        if table_name in existing_tables:
            self.is_setup = True
            return

        # Create the DynamoDB table with a Global Secondary Index
        try:
            table = self.client.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        "AttributeName": self.primary_key_name,
                        "KeyType": "HASH",  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        "AttributeName": self.primary_key_name,
                        "AttributeType": "S",  # 'S' for string type
                    }
                ],
                ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
            )

            # Wait for the table to be created
            table.meta.client.get_waiter("table_exists").wait(TableName=table_name)
            logger.info(f"Table {table_name} created successfully.")
        except Exception as e:
            logger.error(f"Error creating table: {e}")

        self.is_setup = True

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        self.setup()
        response = self.table.get_item(
            Key={self.primary_key_name: config["configurable"]["thread_id"]}
        )
        item = response.get("Item", None)
        if item and self.checkpoint_field_name in item:
            checkpoint = pickle.loads(item[self.checkpoint_field_name].value)
            return checkpoint
        else:
            return None

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        self.setup()
        self.table.put_item(
            Item={
                self.primary_key_name: config["configurable"]["thread_id"],
                self.checkpoint_field_name: pickle.dumps(checkpoint),
            }
        )

    async def aget(self, config: RunnableConfig) -> Optional[Checkpoint]:
        raise NotImplementedError

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        raise NotImplementedError
