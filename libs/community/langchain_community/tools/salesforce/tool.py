"""Tools for interacting with Salesforce."""

from typing import Dict, List, Optional, Type, Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from simple_salesforce import Salesforce

from langchain_community.utilities.salesforce import SalesforceAPIWrapper


class BaseSalesforceTool(BaseTool):
    """Base tool for interacting with Salesforce."""

    sfdc_instance: Salesforce = Field(exclude=True)

    @property
    def api_wrapper(self) -> SalesforceAPIWrapper:
        """Get the API wrapper."""
        return SalesforceAPIWrapper(self.sfdc_instance)


class QuerySalesforceInput(BaseModel):
    """Input for Salesforce queries."""
    query: str = Field(
        ..., 
        description="The SOQL query to execute against Salesforce"
    )


class QuerySalesforceTool(BaseSalesforceTool):
    """Tool for querying Salesforce using SOQL."""

    name: str = "salesforce_query"
    description: str = (
        "Execute a SOQL query against Salesforce. "
        "If the query is not correct, an error message will be returned. "
        "If an error is returned, rewrite the query, check the query, and try again."
    )
    args_schema: Type[BaseModel] = QuerySalesforceInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the Salesforce query."""
        return self.api_wrapper.run_no_throw(query)


class InfoSalesforceInput(BaseModel):
    """Input for getting Salesforce object info."""
    object_names: str = Field(
        ...,
        description="Comma-separated list of Salesforce object names to get info about"
    )


class InfoSalesforceTool(BaseSalesforceTool):
    """Tool for getting metadata about Salesforce objects."""

    name: str = "salesforce_object_info"
    description: str = (
        "Get information about one or more Salesforce objects. "
        "Input should be a comma-separated list of object names. "
        "Example: 'Account,Contact,Opportunity'"
    )
    args_schema: Type[BaseModel] = InfoSalesforceInput

    def _run(
        self,
        object_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        object_list = [name.strip() for name in object_names.split(",")]
        return self.api_wrapper.get_object_info_no_throw(object_list)


class ListSalesforceTool(BaseSalesforceTool):
    """Tool for listing available Salesforce objects."""

    name: str = "salesforce_list_objects"
    description: str = (
        "Get a list of available objects in your Salesforce instance. "
        "Input should be an empty string."
    )

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Get a comma-separated list of Salesforce object names."""
        try:
            return ", ".join(self.api_wrapper.get_usable_object_names())
        except Exception as e:
            return f"Error: {str(e)}"