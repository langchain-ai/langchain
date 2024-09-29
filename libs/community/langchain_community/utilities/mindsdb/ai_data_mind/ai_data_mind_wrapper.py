from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.utilities.mindsdb.ai_data_mind.database_models import (
    get_supported_data_sources,
)
from langchain_community.utilities.mindsdb.base_mind_wrapper import BaseMindWrapper

if TYPE_CHECKING:
    from minds.datasources import DatabaseConfig


class DataSourceConfig(BaseModel):
    type: Text
    description: Text
    connection_args: Dict[Text, Any]
    tables: Optional[List[Text]] = Field(default=[])

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        supported_data_sources = get_supported_data_sources()
        if self.type not in supported_data_sources.keys():
            raise ValueError(
                f"Data source type '{self.type}' is not supported. "
                f"Supported data source types are: {supported_data_sources}."
            )

        model_cls = supported_data_sources[self.type]
        model_obj = model_cls(**self.connection_args)
        self.connection_args = model_obj.dict()

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
            type=self.type,
            description=self.description,
            connection_args=self.connection_args,
            tables=self.tables,
        )


class AIDataMindWrapper(BaseMindWrapper):
    data_source_configs: List[Dict[Text, Any]] = Field(default=None)
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

        # Validate that the correct connection arguments are provided for
        # the chosen data sources.
        data_source_config_objs = []
        for data_source_config in self.data_source_configs:
            data_source_config_obj = DataSourceConfig(**data_source_config)
            data_source_config_objs.append(data_source_config_obj.to_database_config())

        # Create the Mind object.
        minds_client = Client(
            self.minds_api_key.get_secret_value(),
            self.minds_api_base,
        )

        self.mind = minds_client.minds.create(
            name=self.name,
            model_name=self.model,
            datasources=data_source_config_objs,
        )

    def run(self, query: Text) -> Text:
        completion = self.client.create(
            model=self.mind.name,
            messages=[{"role": "user", "content": query}],
            stream=False,
        )

        return completion.choices[0].message.content
