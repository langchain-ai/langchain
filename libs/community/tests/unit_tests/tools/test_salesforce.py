"""Unit tests for Salesforce tools."""

from typing import Any, Dict, Generator, List
from unittest.mock import Mock, patch

import pytest
from pydantic import Field, create_model

try:
    from simple_salesforce import Salesforce
except ImportError:
    class Salesforce:
        pass

from langchain_community.tools.salesforce.tool import (
    BaseSalesforceTool,
    InfoSalesforceTool,
    ListSalesforceTool,
    QuerySalesforceTool,
)


class MockSalesforceObject:
    """Mock of a Salesforce object (like Account, Contact, etc)."""

    def __init__(self, name: str, fields: List[Dict[str, Any]]) -> None:
        self.name = name
        self._fields = fields
        self._records: List[Dict[str, Any]] = []

    def describe(self) -> Dict[str, Any]:
        """Return object metadata."""
        return {
            "name": self.name,
            "label": self.name,
            "fields": self._fields,
            "queryable": True,
        }


class MockSalesforce(Salesforce):
    """Mock implementation of Salesforce client."""

    def __init__(self) -> None:
        """Initialize with some default objects."""
        # Skip actual Salesforce initialization
        self._objects: Dict[str, MockSalesforceObject] = {}
        self.setup_default_objects()

    def setup_default_objects(self) -> None:
        """Set up default Salesforce objects with their fields."""
        account_fields = [
            {
                "name": "Id",
                "type": "id",
                "label": "Account ID",
                "nillable": True,
                "description": "Unique identifier for the Account object",
            },
            {
                "name": "Name",
                "type": "string",
                "label": "Account Name",
                "nillable": False,
                "description": "Name of the account",
            },
        ]

        contact_fields = [
            {
                "name": "Id",
                "type": "id",
                "label": "Contact ID",
                "nillable": True,
                "description": "Unique identifier for the Contact object",
            },
            {
                "name": "FirstName",
                "type": "string",
                "label": "First Name",
                "nillable": True,
                "description": "Contact's first name",
            },
            {
                "name": "LastName",
                "type": "string",
                "label": "Last Name",
                "nillable": False,
                "description": "Contact's last name",
            },
        ]

        self._objects["Account"] = MockSalesforceObject("Account", account_fields)
        self._objects["Contact"] = MockSalesforceObject("Contact", contact_fields)

    def describe(self) -> Dict[str, Any]:
        """Return metadata about all objects."""
        return {
            "sobjects": [
                {
                    "name": name,
                    "label": obj.name,
                    "queryable": True,
                }
                for name, obj in self._objects.items()
            ]
        }

    def query(self, soql: str) -> Dict[str, Any]:
        """Mock SOQL query execution.

        Note: This is a very basic implementation that only supports simple queries.
        For more complex SOQL support, we'd need a proper SOQL parser like\
        simple-mockforce uses.
        """
        # Very basic SOQL parsing - just for demonstration
        try:
            if "FROM Account" in soql:
                return {
                    "records": [
                        {"Id": "001", "Name": "Test Account"},
                        {"Id": "002", "Name": "Another Account"},
                    ],
                    "totalSize": 2,
                    "done": True,
                }
            elif "FROM Contact" in soql:
                return {
                    "records": [
                        {"Id": "003", "FirstName": "John", "LastName": "Doe"},
                        {"Id": "004", "FirstName": "Jane", "LastName": "Smith"},
                    ],
                    "totalSize": 2,
                    "done": True,
                }
            else:
                raise ValueError(f"Unsupported object in SOQL: {soql}")
        except Exception as e:
            raise ValueError(f"Invalid SOQL query: {str(e)}")

    def __getattr__(self, name: str) -> Any:
        """Support dynamic access to Salesforce objects (e.g., sf.Account.describe())"""
        if name in self._objects:
            return self._objects[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


@pytest.fixture
def mock_salesforce() -> MockSalesforce:
    """Create a mock Salesforce instance."""
    return MockSalesforce()


@pytest.fixture(autouse=True)
def patch_salesforce_validation() -> Generator[None, None, None]:
    """Patch the Salesforce validation in the base tool."""
    # Create a new model that accepts our MockSalesforce for sfdc_instance
    new_model = create_model(
        "PatchedBaseSalesforceTool",
        sfdc_instance=(MockSalesforce, Field(exclude=True)),
        __base__=BaseSalesforceTool,
    )

    with patch(
        "langchain_community.tools.salesforce.tool.BaseSalesforceTool", new_model
    ):
        yield


def test_query_salesforce_tool_success(mock_salesforce: MockSalesforce) -> None:
    """Test successful query execution."""
    # Create tool instance
    tool = QuerySalesforceTool(sfdc_instance=mock_salesforce)

    # Execute query
    result = tool.run("SELECT Id, Name FROM Account LIMIT 1")

    # Verify results
    assert isinstance(result, str)
    assert "Test Account" in result


def test_query_salesforce_tool_error(mock_salesforce: MockSalesforce) -> None:
    """Test query execution with error."""
    # Create tool instance
    tool = QuerySalesforceTool(sfdc_instance=mock_salesforce)

    # Execute query with invalid object
    result = tool.run("SELECT Id FROM InvalidObject__c")

    # Verify error handling
    assert "Error" in result
    assert "Invalid SOQL query" in result


def test_info_salesforce_tool_success(mock_salesforce: MockSalesforce) -> None:
    """Test successful object info retrieval."""
    # Create tool instance
    tool = InfoSalesforceTool(sfdc_instance=mock_salesforce)

    # Get object info
    result = tool.run("Account")

    # Verify results
    assert isinstance(result, str)
    assert "Id" in result
    assert "Name" in result
    assert "Account" in result


def test_list_salesforce_tool_success(mock_salesforce: MockSalesforce) -> None:
    """Test successful object listing."""
    # Create tool instance
    tool = ListSalesforceTool(sfdc_instance=mock_salesforce)

    # List objects
    result = tool.run("")

    # Verify results
    assert isinstance(result, str)
    assert "Account" in result
    assert "Contact" in result


def test_list_salesforce_tool_error(mock_salesforce: MockSalesforce) -> None:
    """Test object listing with error."""
    # Create tool instance with a broken describe method
    broken_mock = MockSalesforce()
    # Create a new Mock instance for the describe method
    broken_mock.describe = Mock(side_effect=Exception("API Error"))  # type: ignore

    tool = ListSalesforceTool(sfdc_instance=broken_mock)

    # List objects
    result = tool.run("")

    # Verify error handling
    assert "Error" in result
    assert "API Error" in result
