from typing import Text, Dict
from langchain_core.pydantic_v1 import BaseModel, Field


def validate_data_source_connection_args(data_source_type: Text, data_source_connection_args: Dict):
    if data_source_type == 'postgres':
        return PostgresModel(**data_source_connection_args)

    elif data_source_type in ['mysql', 'mariadb']:
        return MySQLModel(**data_source_connection_args)


class PostgresModel(BaseModel):
    user: Text
    password: Text
    host: Text
    port: Text = Field(default=5432)
    database: Text
    database_schema: Text = Field(alias='schema')


class MySQLModel(BaseModel):
    user: Text
    password: Text
    host: Text
    port: Text = Field(default=5432)
    database: Text