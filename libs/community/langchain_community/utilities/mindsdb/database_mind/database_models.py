from typing import Text
from langchain_core.pydantic_v1 import BaseModel, Field


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