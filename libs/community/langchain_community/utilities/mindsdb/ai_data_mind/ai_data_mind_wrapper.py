import secrets
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.utilities.mindsdb.ai_data_mind.database_models import (
    get_supported_datasources,
)
from langchain_community.utilities.mindsdb.base_mind_wrapper import BaseMindWrapper

if TYPE_CHECKING:
    from minds.datasources import DatabaseConfig


class DataSourceConfig(BaseModel):
    engine: Text
    description: Text
    connection_data: Dict[Text, Any]
    tables: Optional[List[Text]] = Field(default=[])
    name: Optional[Text] = Field(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # If a name is not provided, generate a random one.
        if not self.name:
            self.name = f"lc_datasource_{secrets.token_hex(5)}"

        supported_data_sources = get_supported_datasources()
        if self.engine not in supported_data_sources.keys():
            raise ValueError(
                f"Data source engine '{self.engine}' is not supported. "
                f"Supported data source engines are: {supported_data_sources}."
            )

        model_cls = supported_data_sources[self.engine]
        model_obj = model_cls(**self.connection_data)
        self.connection_data = model_obj.dict()

    def to_database_config(self) -> "DatabaseConfig":
        # Validate that the `minds-sdk` package can be imported.
        try:
            from minds.datasources import DatabaseConfig
        except ImportError as e:
            raise ImportError(
                "Could not import minds-sdk python package. "
                "Please install it with `pip install minds-sdk`.",
            ) from e

        return DatabaseConfig(
            name=self.name,
            engine=self.engine,
            description=self.description,
            connection_data=self.connection_data,
            tables=self.tables,
        )


class AIDataMindWrapper(BaseMindWrapper):
    datasources: List[Dict[Text, Any]] = Field(default=None)
    mind: Any = Field(default=None, exclude=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Validate that the `minds-sdk` package can be imported.
        try:
            from minds.client import Client
        except ImportError as e:
            raise ImportError(
                "Could not import minds-sdk python package. "
                "Please install it with `pip install minds-sdk`.",
            ) from e

        # Create the Mind object.
        minds_client = Client(
            self.minds_api_key.get_secret_value(), self.minds_api_base
        )

        # Validate that the correct connection arguments are provided for
        # the chosen data sources.
        datasources = []
        for data_source_config in self.datasources:
            data_source_config_obj = DataSourceConfig(**data_source_config)

            database_config_obj = data_source_config_obj.to_database_config()
            datasource = minds_client.datasources.create(
                database_config_obj, replace=True
            )
            datasources.append(datasource)

        self.mind = minds_client.minds.create(
            name=self.name, model_name=self.model, datasources=datasources, replace=True
        )

    def run(self, query: Text) -> Text:
        completion = self.client.create(
            model=self.mind.name,
            messages=[{"role": "user", "content": query}],
            stream=False,
        )

        return completion.choices[0].message.content
