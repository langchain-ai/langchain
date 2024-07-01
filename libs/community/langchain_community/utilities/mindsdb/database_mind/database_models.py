import sys
import json
import inspect
from typing import Text, Dict, Literal, Union
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


def get_supported_data_sources() -> Dict:
    return {
        name.lower().replace('model', ''): cls
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if issubclass(cls, BaseModel) and cls is not BaseModel
    }


class PostgresModel(BaseModel):
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    host: Text = Field(default=None)
    port: int = Field(default=5432)
    database: Text = Field(default=None)
    database_schema: Text = Field(alias='schema', default=None)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["user"] = get_from_dict_or_env(
            values,
            "user",
            "DATABASE_USER",
        )
        values["password"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "password",
                "DATABASE_PASSWORD",
            )
        )
        values["host"] = get_from_dict_or_env(
            values,
            "host",
            "DATABASE_HOST",
        )
        values["port"] = get_from_dict_or_env(
            values,
            "port",
            "DATABASE_PORT",
            default=5432,
        )
        values["database"] = get_from_dict_or_env(
            values,
            "database",
            "DATABASE_DATABASE",
        )
        values["schema"] = get_from_dict_or_env(
            values,
            "database_schema",
            "DATABASE_SCHEMA",
        )
        del values["database_schema"]

        return values
    
    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()
        return base_dict


class MySQLModel(BaseModel):
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    host: Text = Field(default=None)
    port: int = Field(default=3306)
    database: Text = Field(default=None)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["user"] = get_from_dict_or_env(
            values,
            "user",
            "DATABASE_USER",
        )
        values["password"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "password",
                "DATABASE_PASSWORD",
            )
        )
        values["host"] = get_from_dict_or_env(
            values,
            "host",
            "DATABASE_HOST",
        )
        values["port"] = get_from_dict_or_env(
            values,
            "port",
            "DATABASE_PORT",
            default=5432,
        )
        values["database"] = get_from_dict_or_env(
            values,
            "database",
            "DATABASE_DATABASE",
        )

        return values
    
    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()
        return base_dict


class MariaDBModel(MySQLModel):
    pass


class ClickHouseModel(BaseModel):
    user: Text = Field(default=None)
    password: SecretStr = Field(default=None)
    host: Text = Field(default=None)
    port: int = Field(default=8443)
    database: Text = Field(default=None)
    protocol: Literal['native', 'http', 'https'] = Field(default='http')

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["user"] = get_from_dict_or_env(
            values,
            "user",
            "DATABASE_USER",
        )
        values["password"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "password",
                "DATABASE_PASSWORD",
            )
        )
        values["host"] = get_from_dict_or_env(
            values,
            "host",
            "DATABASE_HOST",
        )
        values["port"] = get_from_dict_or_env(
            values,
            "port",
            "DATABASE_PORT",
            default=8443,
        )
        values["database"] = get_from_dict_or_env(
            values,
            "database",
            "DATABASE_DATABASE",
        )
        values["protocol"] = get_from_dict_or_env(
            values,
            "protocol",
            "DATABASE_PROTOCOL",
            default='http',
        )

        return values
    
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
    schema: Text = Field(default=None)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["account"] = get_from_dict_or_env(
            values,
            "account",
            "DATABASE_ACCOUNT",
        )
        values["user"] = get_from_dict_or_env(
            values,
            "user",
            "DATABASE_USER",
        )
        values["password"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "password",
                "DATABASE_PASSWORD",
            )
        )
        values["warehouse"] = get_from_dict_or_env(
            values,
            "warehouse",
            "DATABASE_WAREHOUSE",
        )
        values["database"] = get_from_dict_or_env(
            values,
            "database",
            "DATABASE_DATABASE",
        )
        values["schema"] = get_from_dict_or_env(
            values,
            "schema",
            "DATABASE_SCHEMA",
        )

        return values
    
    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the secret password to a string.
        base_dict["password"] = base_dict["password"].get_secret_value()
        return base_dict
    

class BigQueryModel(BaseModel):
    project_id: Text = Field(default=None)
    dataset: Text = Field(default=None)
    service_account_json: Union[SecretStr, Dict] = Field(default=None)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["project_id"] = get_from_dict_or_env(
            values,
            "project_id",
            "DATABASE_PROJECT_ID",
        )
        values["dataset"] = get_from_dict_or_env(
            values,
            "dataset",
            "DATABASE_DATASET",
        )
        service_account_json = get_from_dict_or_env(
            values,
            "service_account_json",
            "DATABASE_SERVICE_ACCOUNT_JSON",
        )
        if isinstance(service_account_json, Dict):
            service_account_json_str = json.dumps(service_account_json)

        values["service_account_json"] = convert_to_secret_str(service_account_json_str)

        return values
    
    def dict(self, **kwargs):
        base_dict = super().dict(**kwargs)

        # Convert the  secret service account json to a dict.
        base_dict["service_account_json"] = json.loads(base_dict["service_account_json"].get_secret_value())
        return base_dict