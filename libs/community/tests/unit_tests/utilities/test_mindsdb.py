from typing import Any, Dict, Text
from unittest.mock import Mock, patch

import pytest
from langchain_core.pydantic_v1 import SecretStr

from langchain_community.utilities.mindsdb.ai_data_mind.ai_data_mind_wrapper import (
    AIDataMindWrapper,
)
from langchain_community.utilities.mindsdb.base_mind_wrapper import (
    DEFAULT_API_BASE,
    DEFAULT_MODEL,
)

DATASOURCES = ["postgres", "mysql", "mariadb", "clickhouse", "snowflake", "bigquery"]


@pytest.fixture
def datasource_configs() -> Dict[Text, Dict[Text, Any]]:
    return {
        "postgres": {
            "engine": "postgres",
            "description": "dummy description",
            "connection_data": {
                "host": "dummy_host",
                "port": 5432,
                "user": "dummy_user",
                "password": "dummy_password",
                "database": "dummy_database",
                "schema": "dummy_schema",
            },
            "tables": ["dummy_table_1", "dummy_table_2"],
        },
        "mysql": {
            "engine": "mysql",
            "description": "dummy description",
            "connection_data": {
                "host": "dummy_host",
                "port": 3306,
                "user": "dummy_user",
                "password": "dummy_password",
                "database": "dummy_database",
            },
            "tables": ["dummy_table_1", "dummy_table_2"],
        },
        "mariadb": {
            "engine": "mariadb",
            "description": "dummy description",
            "connection_data": {
                "host": "dummy_host",
                "port": 3306,
                "user": "dummy_user",
                "password": "dummy_password",
                "database": "dummy_database",
            },
            "tables": ["dummy_table_1", "dummy_table_2"],
        },
        "clickhouse": {
            "engine": "clickhouse",
            "description": "dummy description",
            "connection_data": {
                "host": "dummy_host",
                "port": 8123,
                "user": "dummy_user",
                "password": "dummy_password",
                "database": "dummy_database",
            },
            "tables": ["dummy_table_1", "dummy_table_2"],
        },
        "snowflake": {
            "engine": "snowflake",
            "description": "dummy description",
            "connection_data": {
                "account": "dummy_account",
                "user": "dummy_user",
                "password": "dummy_password",
                "warehouse": "dummy_warehouse",
                "database": "dummy_database",
                "schema": "dummy_schema",
            },
            "tables": ["dummy_table_1", "dummy_table_2"],
        },
        "bigquery": {
            "engine": "bigquery",
            "description": "dummy description",
            "connection_data": {
                "project_id": "dummy_project_id",
                "dataset": "dummy_dataset",
                "service_account_json": {
                    "type": "service_account",
                    "project_id": "dummy_project_id",
                    "private_key_id": "dummy_private_key_id",
                    "private_key": "dummy_private_key",
                },
            },
            "tables": ["dummy_table_1", "dummy_table_2"],
        },
    }


@pytest.mark.requires("minds")
@pytest.mark.parametrize("datasource_key", DATASOURCES)
@patch("minds.client.Client")
@patch("minds.datasources.DatabaseConfig")
def test_init_with_single_datasource(
    mock_database_config: Mock,
    mock_client: Mock,
    datasource_key: Text,
    datasource_configs: Dict[Text, Dict[Text, Any]],
) -> None:
    datasource_config = datasource_configs[datasource_key]
    ai_data_mind_config = {
        "name": "dummy_mind",
        "minds_api_key": "dummy_key",
        "datasources": [datasource_config],
    }

    mock_client.return_value = Mock(
        minds=Mock(create=Mock(return_value=Mock(name="dummy_mind")))
    )

    mock_database_config.return_value = Mock(
        engine=datasource_config["engine"],
        description=datasource_config["description"],
        connection_data=datasource_config["connection_data"],
        tables=datasource_config["tables"],
    )

    ai_data_mind_wrapper = AIDataMindWrapper(**ai_data_mind_config)

    assert ai_data_mind_wrapper.mind is not None
    assert ai_data_mind_wrapper.datasources == [datasource_config]
    assert ai_data_mind_wrapper.name == "dummy_mind"
    assert ai_data_mind_wrapper.minds_api_base == DEFAULT_API_BASE
    assert ai_data_mind_wrapper.model == DEFAULT_MODEL
    assert isinstance(ai_data_mind_wrapper.minds_api_key, SecretStr)


@pytest.mark.requires("minds")
@pytest.mark.parametrize("datasource_key", DATASOURCES)
@patch("minds.client.Client")
@patch("minds.datasources.DatabaseConfig")
def test_run_with_single_datasource(
    mock_database_config: Mock,
    mock_client: Mock,
    datasource_key: Text,
    datasource_configs: Dict[Text, Dict[Text, Any]],
) -> None:
    datasource_config = datasource_configs[datasource_key]
    ai_data_mind_config = {
        "name": "dummy_mind",
        "minds_api_key": "dummy_key",
        "datasources": [datasource_config],
    }

    mock_client.return_value = Mock(
        minds=Mock(create=Mock(return_value=Mock(name="dummy_mind")))
    )

    mock_database_config.return_value = Mock(
        enine=datasource_config["engine"],
        description=datasource_config["description"],
        connection_data=datasource_config["connection_data"],
        tables=datasource_config["tables"],
    )

    query = "dummy query"

    ai_data_mind_wrapper = AIDataMindWrapper(**ai_data_mind_config)

    ai_data_mind_wrapper.client = Mock(
        create=Mock(return_value=Mock(choices=[Mock(content="dummy response")]))
    )

    ai_data_mind_wrapper.run(query)

    ai_data_mind_wrapper.client.create.assert_called_once_with(
        model=ai_data_mind_wrapper.mind.name,
        messages=[{"role": "user", "content": query}],
        stream=False,
    )


@pytest.mark.requires("minds")
@patch("minds.client.Client")
@patch("minds.datasources.DatabaseConfig")
def test_init_with_multiple_datasources(
    mock_database_config: Mock,
    mock_client: Mock,
    datasource_configs: Dict[Text, Dict[Text, Any]],
) -> None:
    ai_data_mind_config = {
        "name": "dummy_mind",
        "minds_api_key": "dummy_key",
        "datasources": list(datasource_configs.values()),
    }

    mock_client.return_value = Mock(
        minds=Mock(create=Mock(return_value=Mock(name="dummy_mind")))
    )

    mock_return_values = [
        Mock(
            engine=config["engine"],
            description=config["description"],
            connection_data=config["connection_data"],
            tables=config["tables"],
        )
        for config in datasource_configs.values()
    ]

    mock_database_config.side_effect = mock_return_values

    ai_data_mind_wrapper = AIDataMindWrapper(**ai_data_mind_config)

    assert ai_data_mind_wrapper.mind is not None
    assert ai_data_mind_wrapper.datasources == list(datasource_configs.values())
    assert ai_data_mind_wrapper.name == "dummy_mind"
    assert ai_data_mind_wrapper.minds_api_base == DEFAULT_API_BASE
    assert ai_data_mind_wrapper.model == DEFAULT_MODEL
    assert isinstance(ai_data_mind_wrapper.minds_api_key, SecretStr)


@pytest.mark.requires("minds")
@patch("minds.client.Client")
@patch("minds.datasources.DatabaseConfig")
def test_run_with_multiple_datasources(
    mock_database_config: Mock,
    mock_client: Mock,
    datasource_configs: Dict[Text, Dict[Text, Any]],
) -> None:
    ai_data_mind_config = {
        "name": "dummy_mind",
        "minds_api_key": "dummy_key",
        "datasources": list(datasource_configs.values()),
    }

    mock_client.return_value = Mock(
        minds=Mock(create=Mock(return_value=Mock(name="dummy_mind")))
    )

    mock_return_values = [
        Mock(
            engine=config["engine"],
            description=config["description"],
            connection_data=config["connection_data"],
            tables=config["tables"],
        )
        for config in datasource_configs.values()
    ]

    mock_database_config.side_effect = mock_return_values

    query = "dummy query"

    ai_data_mind_wrapper = AIDataMindWrapper(**ai_data_mind_config)

    ai_data_mind_wrapper.client = Mock(
        create=Mock(return_value=Mock(choices=[Mock(content="dummy response")]))
    )

    ai_data_mind_wrapper.run(query)

    ai_data_mind_wrapper.client.create.assert_called_once_with(
        model=ai_data_mind_wrapper.mind.name,
        messages=[{"role": "user", "content": query}],
        stream=False,
    )
