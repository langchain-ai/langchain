import inspect
import json
import sys
from typing import Dict, Literal, Text, Union

from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


def get_supported_data_sources() -> Dict:
    return {
        name.lower().replace("model", ""): cls
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if issubclass(cls, BaseModel) and cls is not BaseModel
    }


class PostgresModel(BaseModel):
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    host: Text = Field(default=None)
    port: int = Field(default=5432)
    database: Text = Field(default=None)
    database_schema: Text = Field(alias="schema", default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.user = get_from_dict_or_env(
            data,
            "user",
            "POSTGRES_USER",
        )
        self.password = convert_to_secret_str(
            get_from_dict_or_env(
                data,
                "password",
                "POSTGRES_PASSWORD",
            )
        )
        self.host = get_from_dict_or_env(
            data,
            "host",
            "POSTGRES_HOST",
        )
        self.port = get_from_dict_or_env(
            data,
            "port",
            "POSTGRES_PORT",
            default=5432,
        )
        self.database = get_from_dict_or_env(
            data,
            "database",
            "POSTGRES_DATABASE",
        )
        self.database_schema = get_from_dict_or_env(
            data,
            "schema",
            "POSTGRES_SCHEMA",
        )

    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()

        # Convert database_schema to schema and remove database_schema.
        base_dict["schema"] = base_dict["database_schema"]
        del base_dict["database_schema"]

        return base_dict


class MySQLModel(BaseModel):
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    host: Text = Field(default=None)
    port: int = Field(default=3306)
    database: Text = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.user = get_from_dict_or_env(
            data,
            "user",
            "MYSQL_USER",
        )
        self.password = convert_to_secret_str(
            get_from_dict_or_env(
                data,
                "password",
                "MYSQL_PASSWORD",
            )
        )
        self.host = get_from_dict_or_env(
            data,
            "host",
            "MYSQL_HOST",
        )
        self.port = get_from_dict_or_env(
            data,
            "port",
            "MYSQL_PORT",
            default=3306,
        )
        self.database = get_from_dict_or_env(
            data,
            "database",
            "MYSQL_DATABASE",
        )

    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()
        return base_dict


class MariaDBModel(MySQLModel):
    def __init__(self, **data):
        super().__init__(**data)
        self.user = get_from_dict_or_env(
            data,
            "user",
            "MARIADB_USER",
        )
        self.password = convert_to_secret_str(
            get_from_dict_or_env(
                data,
                "password",
                "MARIADB_PASSWORD",
            )
        )
        self.host = get_from_dict_or_env(
            data,
            "host",
            "MARIADB_HOST",
        )
        self.port = get_from_dict_or_env(
            data,
            "port",
            "MARIADB_PORT",
            default=3306,
        )
        self.database = get_from_dict_or_env(
            data,
            "database",
            "MARIADB_DATABASE",
        )


class ClickHouseModel(BaseModel):
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    host: Text = Field(default=None)
    port: int = Field(default=8443)
    database: Text = Field(default=None)
    protocol: Literal["native", "http", "https"] = Field(default="http")

    def __init__(self, **data):
        super().__init__(**data)
        self.user = get_from_dict_or_env(
            data,
            "user",
            "CLICKHOUSE_USER",
        )
        self.password = convert_to_secret_str(
            get_from_dict_or_env(
                data,
                "password",
                "CLICKHOUSE_PASSWORD",
            )
        )
        self.host = get_from_dict_or_env(
            data,
            "host",
            "CLICKHOUSE_HOST",
        )
        self.port = get_from_dict_or_env(
            data,
            "port",
            "CLICKHOUSE_PORT",
            default=8443,
        )
        self.database = get_from_dict_or_env(
            data,
            "database",
            "CLICKHOUSE_DATABASE",
        )
        self.protocol = get_from_dict_or_env(
            data,
            "protocol",
            "CLICKHOUSE_PROTOCOL",
            default="http",
        )

    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()
        return base_dict


class SnowflakeModel(BaseModel):
    account: Text = Field(default=None)
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    warehouse: Text = Field(default=None)
    database: Text = Field(default=None)
    database_schema: Text = Field(alias="schema", default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.account = get_from_dict_or_env(
            data,
            "account",
            "SNOWFLAKE_ACCOUNT",
        )
        self.user = get_from_dict_or_env(
            data,
            "user",
            "SNOWFLAKE_USER",
        )
        self.password = convert_to_secret_str(
            get_from_dict_or_env(
                data,
                "password",
                "SNOWFLAKE_PASSWORD",
            )
        )
        self.warehouse = get_from_dict_or_env(
            data,
            "warehouse",
            "SNOWFLAKE_WAREHOUSE",
        )
        self.database = get_from_dict_or_env(
            data,
            "database",
            "SNOWFLAKE_DATABASE",
        )
        self.database_schema = get_from_dict_or_env(
            data,
            "schema",
            "SNOWFLAKE_SCHEMA",
        )

    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()

        # Convert database_schema to schema and remove database_schema.
        base_dict["schema"] = base_dict["database_schema"]
        del base_dict["database_schema"]

        return base_dict


class BigQueryModel(BaseModel):
    project_id: Text = Field(default=None)
    dataset: Text = Field(default=None)
    service_account_json: Union[SecretStr, Dict] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.project_id = get_from_dict_or_env(
            data,
            "project_id",
            "BIGQUERY_PROJECT_ID",
        )
        self.dataset = get_from_dict_or_env(
            data,
            "dataset",
            "BIGQUERY_DATASET",
        )
        service_account_json = get_from_dict_or_env(
            data,
            "service_account_json",
            "BIGQUERY_SERVICE_ACCOUNT_JSON",
        )
        if isinstance(service_account_json, Dict):
            service_account_json = json.dumps(service_account_json)

        self.service_account_json = convert_to_secret_str(service_account_json)

    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret service account json to a dict.
        base_dict["service_account_json"] = json.loads(
            base_dict["service_account_json"].get_secret_value()
        )
        return base_dict
