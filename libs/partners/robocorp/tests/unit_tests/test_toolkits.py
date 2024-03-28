"""Test toolkit integration."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_robocorp.toolkits import ActionServerToolkit

from ._fixtures import FakeChatLLMT


def test_initialization() -> None:
    """Test toolkit initialization."""
    ActionServerToolkit(url="http://localhost", llm=FakeChatLLMT())


def test_get_tools_success() -> None:
    # Setup
    toolkit_instance = ActionServerToolkit(
        url="http://example.com", api_key="dummy_key"
    )

    fixture_path = Path(__file__).with_name("_openapi2.fixture.json")

    with patch(
        "langchain_robocorp.toolkits.requests.get"
    ) as mocked_get, fixture_path.open("r") as f:
        data = json.load(f)  # Using json.load directly on the file object
        mocked_response = MagicMock()
        mocked_response.json.return_value = data
        mocked_response.status_code = 200
        mocked_response.headers = {"Content-Type": "application/json"}
        mocked_get.return_value = mocked_response

        # Execute
        tools = toolkit_instance.get_tools()

        # Verify
        assert len(tools) == 5

        tool = tools[2]
        assert tool.name == "add_sheet_rows"
        assert tool.description == (
            "Action to add multiple rows to the Google sheet. "
            "Get the sheets with get_google_spreadsheet_schema if you don't know"
            "\nthe names or data structure.  Make sure the values are in correct"
            """ columns (needs to be ordered the same as in the sample).
Strictly adhere to the schema."""
        )

        openai_func_spec = convert_to_openai_function(tool)

        assert isinstance(
            openai_func_spec, dict
        ), "openai_func_spec should be a dictionary."
        assert set(openai_func_spec.keys()) == {
            "description",
            "name",
            "parameters",
        }, "Top-level keys mismatch."

        assert openai_func_spec["description"] == tool.description
        assert openai_func_spec["name"] == tool.name

        assert isinstance(
            openai_func_spec["parameters"], dict
        ), "Parameters should be a dictionary."

        params = openai_func_spec["parameters"]
        assert set(params.keys()) == {
            "type",
            "properties",
            "required",
        }, "Parameters keys mismatch."
        assert params["type"] == "object", "`type` in parameters should be 'object'."
        assert isinstance(
            params["properties"], dict
        ), "`properties` should be a dictionary."
        assert isinstance(params["required"], list), "`required` should be a list."

        assert set(params["required"]) == {
            "sheet",
            "rows_to_add",
        }, "Required fields mismatch."

        assert set(params["properties"].keys()) == {"sheet", "rows_to_add"}

        desc = "The columns that make up the row"
        expected = {
            "description": "the rows to be added to the sheet",
            "allOf": [
                {
                    "title": "Rows To Add",
                    "type": "object",
                    "properties": {
                        "rows": {
                            "title": "Rows",
                            "description": "The rows that need to be added",
                            "type": "array",
                            "items": {
                                "title": "Row",
                                "type": "object",
                                "properties": {
                                    "columns": {
                                        "title": "Columns",
                                        "description": desc,
                                        "type": "array",
                                        "items": {"type": "string"},
                                    }
                                },
                                "required": ["columns"],
                            },
                        }
                    },
                    "required": ["rows"],
                }
            ],
        }
        assert params["properties"]["rows_to_add"] == expected
