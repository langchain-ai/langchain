# flake8: noqa
"""Tools for interacting with a Cube Semantic Layer."""
import json
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain.utilities.cube import Cube, Query


class BaseCubeTool(BaseModel):
    """Base tool for interacting with a Cube Semantic Layer."""

    cube: Cube = Field(exclude=True)

    class Config(BaseTool.Config):
        pass

def get_format_instructions(parser:PydanticOutputParser) -> str:
        schema = parser.pydantic_object.schema()

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema)

        return """Example Input Format:

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```""".format(schema=schema_str)

class LoadCubeTool(BaseCubeTool, BaseTool):
    """Tool for loading a Cube Semantic Layer."""

    parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Query)

    name: str = "load_cube"
    description: str = f"""
    Input to this tool is a detailed and correct Cube query, query format is JSON, output is a result from the Cube.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    {get_format_instructions(parser).replace("{", "{{").replace("}", "}}")}    
    """

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Load the Cube Semantic Layer."""

        try:
            data = self.cube.load(self.parser.parse(query))
            return json.dumps(data['data'])
        except Exception as e:
            return f"Error: {e}"




class InfoCubeTool(BaseCubeTool, BaseTool):
    """Tool for getting metadata about a Cube Semantic Layer."""

    name: str = "model_cube"
    description: str = """
    Input to this tool is a comma-separated list of models, output is the metadata for those models.
    Be sure that the models actually exist by calling list_models_cube first!

    Example Input: "model1, model2, model3"
    """

    def _run(
            self,
            model_names: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema of the Cube Semantic Layer."""
        return self.cube.get_model_info(model_names.split(", "))


class ListCubeTool(BaseCubeTool, BaseTool):
    """Tool for getting models names."""

    name: str = "list_models_cube"
    description: str = "Input is an empty string, output is a comma separated list of models in the Cube."

    def _run(
            self,
            tool_input: str = "",
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the names of the models."""
        return self.cube.get_usable_models()
