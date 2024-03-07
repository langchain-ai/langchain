import os

import pytest
from moto import mock_aws


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@mock_aws
def test_dynamodb_init(aws_credentials):
    import boto3

    from langchain_community.checkpoints.dynamodb import DynamoDbSaver

    table_name = "test_table"
    primary_key_name = "test_primary_key"
    checkpoint_field_name = "test_checkpoint_field"
    endpoint_url = "https://test.endpoint.url"
    boto3_session = boto3.Session()

    dynamodb = DynamoDbSaver.from_params(
        table_name=table_name,
        primary_key_name=primary_key_name,
        checkpoint_field_name=checkpoint_field_name,
        endpoint_url=endpoint_url,
        boto3_session=boto3_session,
    )

    assert dynamodb.table_name == table_name
    assert dynamodb.primary_key_name == primary_key_name
    assert dynamodb.checkpoint_field_name == checkpoint_field_name
    assert str(dynamodb.client.__class__) == str(
        boto3_session.resource("dynamodb", endpoint_url=endpoint_url).__class__
    )
    assert str(dynamodb.table.__class__) == str(
        dynamodb.client.Table(table_name).__class__
    )

    dynamodb = DynamoDbSaver.from_params(
        table_name=table_name,
        primary_key_name=primary_key_name,
        checkpoint_field_name=checkpoint_field_name,
        endpoint_url=endpoint_url,
    )

    assert dynamodb.table_name == table_name
    assert dynamodb.primary_key_name == primary_key_name
    assert dynamodb.checkpoint_field_name == checkpoint_field_name
    assert str(dynamodb.client.__class__) == str(
        boto3.resource("dynamodb", endpoint_url=endpoint_url).__class__
    )
    assert str(dynamodb.table.__class__) == str(
        dynamodb.client.Table(table_name).__class__
    )

    dynamodb = DynamoDbSaver.from_params(
        table_name=table_name,
        primary_key_name=primary_key_name,
        checkpoint_field_name=checkpoint_field_name,
    )

    assert dynamodb.table_name == table_name
    assert dynamodb.primary_key_name == primary_key_name
    assert dynamodb.checkpoint_field_name == checkpoint_field_name
    assert str(dynamodb.client.__class__) == str(
        boto3.resource("dynamodb", endpoint_url=endpoint_url).__class__
    )
    assert str(dynamodb.table.__class__) == str(
        dynamodb.client.Table(table_name).__class__
    )

    dynamodb = DynamoDbSaver.from_params(
        table_name=table_name,
    )

    assert dynamodb.table_name == table_name
    assert dynamodb.primary_key_name == "ThreadId"
    assert dynamodb.checkpoint_field_name == "Checkpoint"
    assert str(dynamodb.client.__class__) == str(
        boto3.resource("dynamodb", endpoint_url=endpoint_url).__class__
    )
    assert str(dynamodb.table.__class__) == str(
        dynamodb.client.Table(table_name).__class__
    )


@mock_aws
def test_dynamodb_put_get(aws_credentials):
    import boto3

    from langchain_community.checkpoints.dynamodb import DynamoDbSaver

    table_name = "test_table"
    primary_key_name = "test_primary_key"
    checkpoint_field_name = "test_checkpoint_field"
    endpoint_url = "https://test.endpoint.url"
    boto3_session = boto3.Session()

    dynamodb = DynamoDbSaver.from_params(
        table_name=table_name,
        primary_key_name=primary_key_name,
        checkpoint_field_name=checkpoint_field_name,
        endpoint_url=endpoint_url,
        boto3_session=boto3_session,
    )

    patched_client = boto3.resource("dynamodb")
    dynamodb.client = patched_client
    dynamodb.table = patched_client.Table(table_name)

    config = {"configurable": {"thread_id": "test_thread_id"}}
    checkpoint = {
        "v": 1,
        "ts": "2021-01-01T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
    }

    dynamodb.put(config, checkpoint)
    assert dynamodb.get(config) == checkpoint
