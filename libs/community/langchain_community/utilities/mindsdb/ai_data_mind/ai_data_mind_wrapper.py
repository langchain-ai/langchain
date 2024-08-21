from typing import Text, List, Dict, Any, Optional

from mindsdb_sdk.utils.mind import DatabaseConfig

from langchain_community.utilities.mindsdb import BaseMindWrapper
from langchain_community.utilities.mindsdb.ai_data_mind.database_models import get_supported_data_sources
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator


class DataSourceConfig(BaseModel):
    type: Text
    description: Text
    connection_args: Dict[Text, Any]
    tables: Optional[List[Text]] = Field(default=[])

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        supported_data_sources = get_supported_data_sources()

        if values['type'] not in supported_data_sources.keys():
            raise ValueError(
                f"Data source type '{values['type']}' is not supported. "
                f"Supported data source types are: {supported_data_sources}."
            )

        model_cls = supported_data_sources[values['type']]
        model_obj = model_cls(**values['connection_args'])
        values['connection_args'] = model_obj.dict()

        return values
    
    def to_database_config(self) -> DatabaseConfig:
        return DatabaseConfig(
            type=self.type,
            description=self.description,
            connection_args=self.connection_args,
            tables=self.tables,
        )


class AIDataMindWrapper(BaseMindWrapper):
    data_source_configs: List[Dict[Text, Any]] = Field(default=None)

    mind: Any = Field(default=None, exclude=True)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        # Validate that the `mindsdb_sdk` package can be imported.
        super().validate_environment(values)

        try:
            from mindsdb_sdk.utils.mind import create_mind

        except ImportError as e:
            raise ImportError(
                "Could not import mindsdb_sdk python package. "
                "Please install it with `pip install mindsdb_sdk`.",
            ) from e
        
        # Validate that the correct connection arguments are provided for the chosen data sources.
        data_source_config_objs = []
        for data_source_config in values['data_source_configs']:
            data_source_config_obj = DataSourceConfig(**data_source_config)
            data_source_config_objs.append(data_source_config_obj.to_database_config())

        # Create the MindsDB mind object.
        values['mind'] = create_mind(
            name=values['name'],
            base_url=values['mindsdb_api_base'],
            api_key=values['mindsdb_api_key'].get_secret_value(),
            model=values['model'],
            data_source_configs=data_source_config_objs,
        )

        return values

    def run(self, query: Text) -> Text:
        completion = self.client.create(
            model=self.mind.name,
            messages=[
                {'role': 'user', 'content': query}
            ],
            stream=False
        )

        return completion.choices[0].message.content
