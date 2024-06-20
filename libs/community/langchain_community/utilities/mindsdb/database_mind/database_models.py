from typing import Text, Dict
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


# TODO: Improve this method to generate the mapping dynamically.
def get_supported_data_sources() -> Dict:
    return {
        "postgres": PostgresModel,
        "mysql": MySQLModel,
        "mariadb": MySQLModel
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
