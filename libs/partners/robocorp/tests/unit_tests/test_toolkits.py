"""Test toolkit integration."""
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_robocorp.toolkits import ActionServerToolkit

from ._fixtures import FakeChatLLMT
import json

def test_initialization() -> None:
    """Test toolkit initialization."""
    ActionServerToolkit(url="http://localhost", llm=FakeChatLLMT())


def test_get_tools_success() -> None:
    # Setup
    toolkit_instance = ActionServerToolkit(url="http://example.com", api_key="dummy_key")

    fixture_path = Path(__file__).with_name("_openapi2.fixture.json")

    with patch("langchain_robocorp.toolkits.requests.get") as mocked_get, fixture_path.open("r") as f:
        data = json.load(f)  # Using json.load directly on the file object
        mocked_response = MagicMock()
        mocked_response.json.return_value = data
        mocked_response.status_code = 200
        mocked_response.headers = {'Content-Type': 'application/json'}
        mocked_get.return_value = mocked_response

        # Execute
        tools = toolkit_instance.get_tools()

        # Verify
        assert len(tools) == 5

        assert tools[2].name == 'robocorp_action_server_add_sheet_rows'
        assert tools[2].description == '''Action to add multiple rows to the Google sheet. Get the sheets with get_google_spreadsheet_schema if you don't know
the names or data structure.  Make sure the values are in correct columns (needs to be ordered the same as in the sample).
Strictly adhere to the schema.. The tool must be invoked with a complete sentence starting with "Add Sheet Rows" and additional information on Name of the sheet where the data is added to, the rows to be added to the sheet.'''
        rows_to_add = tools[2].args['rows_to_add']
        print(repr(rows_to_add))
        1/0
