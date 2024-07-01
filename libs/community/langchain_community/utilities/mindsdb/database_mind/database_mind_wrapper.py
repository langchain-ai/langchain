from typing import Text, Dict, Any

from langchain_core.pydantic_v1 import Field, root_validator

from langchain_community.utilities.mindsdb import BaseMindWrapper
from langchain_community.utilities.mindsdb.database_mind.database_models import get_supported_data_sources


class DatabaseMindWrapper(BaseMindWrapper):
    data_source_description: Text
    data_source_type: Text
    data_source_connection_args: Dict

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
        
        # Validate that the data source type is supported.
        supported_data_sources = get_supported_data_sources()

        if values['data_source_type'] not in supported_data_sources.keys():
            raise ValueError(
                f"Data source type '{values['data_source_type']}' is not supported. "
                f"Supported data source types are: {supported_data_sources}."
            )
        
        # Validate that the correct connection arguments are provided for the chosen data source.
        model_cls = supported_data_sources[values['data_source_type']]
        model_obj = model_cls(**values['data_source_connection_args'])
        values['data_source_connection_args'] = model_obj.dict()

        # Create the MindsDB mind object.
        values['mind'] = create_mind(
            name=values['name'],
            description=values['data_source_description'],
            base_url=values['mindsdb_api_base'],
            api_key=values['mindsdb_api_key'].get_secret_value(),
            model=values['model'],
            data_source_type=values['data_source_type'],
            data_source_connection_args=values['data_source_connection_args']
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
