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

class LoadCubeTool(BaseCubeTool, BaseTool):
    """Tool for loading a Cube Semantic Layer."""

    name: str = "load_cube"
    description: str = """
    Input to this tool is a detailed and correct Cube query, query format is JSON, output is a result from the Cube.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    
    The input should be formatted as a JSON instance that conforms to the JSON schema below.

    As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
    the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.
    
    Here is the input schema:
    ```
    {{"properties": {{"measures": {{"title": "measure columns", "type": "array", "items": {{"type": "string"}}}}, "dimensions": {{"title": "dimension columns", "type": "array", "items": {{"type": "string"}}}}, "filters": {{"title": "Filters", "type": "array", "items": {{"$ref": "#/definitions/Filter"}}}}, "timeDimensions": {{"title": "Timedimensions", "type": "array", "items": {{"$ref": "#/definitions/TimeDimension"}}}}, "limit": {{"title": "Limit", "type": "integer"}}, "offset": {{"title": "Offset", "type": "integer"}}, "order": {{"description": "The keys are measures columns or dimensions columns to order by.", "type": "object", "additionalProperties": {{"$ref": "#/definitions/Order"}}}}}}, "definitions": {{"Operator": {{"title": "Operator", "description": "An enumeration.", "enum": ["equals", "notEquals", "contains", "notContains", "startsWith", "endsWith", "gt", "gte", "lt", "lte", "set", "notSet", "inDateRange", "notInDateRange", "beforeDate", "afterDate", "measureFilter"]}}, "Filter": {{"title": "Filter", "type": "object", "properties": {{"member": {{"title": "dimension or measure column", "type": "string"}}, "operator": {{"$ref": "#/definitions/Operator"}}, "values": {{"title": "Values", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["member", "operator", "values"]}}, "Granularity": {{"title": "Granularity", "description": "An enumeration.", "enum": ["second", "minute", "hour", "day", "week", "month", "quarter", "year"]}}, "TimeDimension": {{"title": "TimeDimension", "type": "object", "properties": {{"dimension": {{"title": "dimension column", "type": "string"}}, "dateRange": {{"title": "Daterange", "description": "An array of dates with the following format YYYY-MM-DD or in YYYY-MM-DDTHH:mm:ss.SSS format.", "minItems": 2, "maxItems": 2, "type": "array", "items": {{"anyOf": [{{"type": "string", "format": "date-time"}}, {{"type": "string", "format": "date"}}]}}}}, "granularity": {{"description": "A granularity for a time dimension. If you pass null to the granularity, Cube will only perform filtering by a specified time dimension, without grouping.", "allOf": [{{"$ref": "#/definitions/Granularity"}}]}}}}, "required": ["dimension", "dateRange"]}}, "Order": {{"title": "Order", "description": "An enumeration.", "enum": ["asc", "desc"]}}}}}}
    ```
    """

    def _run(
            self,
            query: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Load the Cube Semantic Layer."""

        try:
            parser = PydanticOutputParser(pydantic_object=Query)

            print("Query: ",parser.parse(query).dict())

            data = self.cube.load(parser.parse(query))
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
