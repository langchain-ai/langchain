from typing import Text, Dict
from langchain_core.pydantic_v1 import BaseModel, Field


def validate_data_source_connection_args(data_source_type: Text, data_source_connection_args: Dict) -> None:
    supported_data_sources = get_supported_data_sources()

    model = supported_data_sources[data_source_type]
    model(**data_source_connection_args)


# TODO: Improve this method to generate the mapping dynamically.
def get_supported_data_sources() -> Dict[Text: BaseModel]:
    return {
        "postgres": PostgresModel,
        "mysql": MySQLModel,
        "mariadb": MySQLModel
    }


# TODO: Allow credentials to be specified as environment variables.
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
