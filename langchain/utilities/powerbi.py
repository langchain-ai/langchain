"""Wrapper around a Power BI endpoint."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import aiohttp
import requests
from aiohttp import ServerTimeoutError
from pydantic import BaseModel, Field, root_validator
from requests.exceptions import Timeout

from langchain.tools.powerbi.prompt import SCHEMA_ERROR_RESPONSE, UNAUTHORIZED_RESPONSE

_LOGGER = logging.getLogger(__name__)

BASE_URL = os.getenv("POWERBI_BASE_URL", "https://api.powerbi.com/v1.0/myorg/datasets/")

if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential


class PowerBIDataset(BaseModel):
    """Create PowerBI engine from dataset ID and credential or token.

    Use either the credential or a supplied token to authenticate.
    If both are supplied the credential is used to generate a token.
    The impersonated_user_name is the UPN of a user to be impersonated.
    If the model is not RLS enabled, this will be ignored.
    """

    dataset_id: str
    table_names: List[str]
    group_id: Optional[str] = None
    credential: Optional[TokenCredential] = None
    token: Optional[str] = None
    impersonated_user_name: Optional[str] = None
    sample_rows_in_table_info: int = Field(default=1, gt=0, le=10)
    aiosession: Optional[aiohttp.ClientSession] = None
    schemas: Dict[str, str] = Field(default_factory=dict, init=False)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator(pre=True, allow_reuse=True)
    def token_or_credential_present(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that at least one of token and credentials is present."""
        if "token" in values or "credential" in values:
            return values
        raise ValueError("Please provide either a credential or a token.")

    @property
    def request_url(self) -> str:
        """Get the request url."""
        if self.group_id:
            return f"{BASE_URL}/{self.group_id}/datasets/{self.dataset_id}/executeQueries"  # noqa: E501 # pylint: disable=C0301
        return f"{BASE_URL}/{self.dataset_id}/executeQueries"  # noqa: E501 # pylint: disable=C0301

    @property
    def headers(self) -> Dict[str, str]:
        """Get the token."""
        from azure.core.exceptions import ClientAuthenticationError

        token = None
        if self.token:
            token = self.token
        if self.credential:
            try:
                token = self.credential.get_token(
                    "https://analysis.windows.net/powerbi/api/.default"
                ).token
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise ClientAuthenticationError(
                    "Could not get a token from the supplied credentials."
                ) from exc
        if not token:
            raise ClientAuthenticationError("No credential or token supplied.")

        return {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
        }

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        return self.table_names

    def get_schemas(self) -> str:
        """Get the available schema's."""
        if self.schemas:
            return ", ".join([f"{key}: {value}" for key, value in self.schemas.items()])
        return "No known schema's yet. Use the schema_powerbi tool first."

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def _get_tables_to_query(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> List[str]:
        """Get the tables names that need to be queried."""
        if table_names is not None:
            if (
                isinstance(table_names, list)
                and len(table_names) > 0
                and table_names[0] != ""
            ):
                return table_names
            if isinstance(table_names, str) and table_names != "":
                return [table_names]
        return self.table_names

    def _get_tables_todo(self, tables_todo: List[str]) -> List[str]:
        for table in tables_todo:
            if table in self.schemas:
                tables_todo.remove(table)
        return tables_todo

    def _get_schema_for_tables(self, table_names: List[str]) -> str:
        """Create a string of the table schemas for the supplied tables."""
        schemas = [
            schema for table, schema in self.schemas.items() if table in table_names
        ]
        return ", ".join(schemas)

    def get_table_info(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> str:
        """Get information about specified tables."""
        tables_requested = self._get_tables_to_query(table_names)
        tables_todo = self._get_tables_todo(tables_requested)
        for table in tables_todo:
            try:
                result = self.run(
                    f"EVALUATE TOPN({self.sample_rows_in_table_info}, {table})"
                )
            except Timeout:
                _LOGGER.warning("Timeout while getting table info for %s", table)
                continue
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if "bad request" in str(exc).lower():
                    return SCHEMA_ERROR_RESPONSE
                if "unauthorized" in str(exc).lower():
                    return UNAUTHORIZED_RESPONSE
                return str(exc)
            self.schemas[table] = json_to_md(result["results"][0]["tables"][0]["rows"])
        return self._get_schema_for_tables(tables_requested)

    async def aget_table_info(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> str:
        """Get information about specified tables."""
        tables_requested = self._get_tables_to_query(table_names)
        tables_todo = self._get_tables_todo(tables_requested)
        for table in tables_todo:
            try:
                result = await self.arun(
                    f"EVALUATE TOPN({self.sample_rows_in_table_info}, {table})"
                )
            except ServerTimeoutError:
                _LOGGER.warning("Timeout while getting table info for %s", table)
                continue
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if "bad request" in str(exc).lower():
                    return SCHEMA_ERROR_RESPONSE
                if "unauthorized" in str(exc).lower():
                    return UNAUTHORIZED_RESPONSE
                return str(exc)
            self.schemas[table] = json_to_md(result["results"][0]["tables"][0]["rows"])
        return self._get_schema_for_tables(tables_requested)

    def run(self, command: str) -> Any:
        """Execute a DAX command and return a json representing the results."""

        result = requests.post(
            self.request_url,
            json={
                "queries": [{"query": command}],
                "impersonatedUserName": self.impersonated_user_name,
                "serializerSettings": {"includeNulls": True},
            },
            headers=self.headers,
            timeout=10,
        )
        result.raise_for_status()
        return result.json()

    async def arun(self, command: str) -> Any:
        """Execute a DAX command and return the result asynchronously."""
        json_content = (
            {
                "queries": [{"query": command}],
                "impersonatedUserName": self.impersonated_user_name,
                "serializerSettings": {"includeNulls": True},
            },
        )
        if self.aiosession:
            async with self.aiosession.post(
                self.request_url, headers=self.headers, json=json_content, timeout=10
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                return response_json
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.request_url, headers=self.headers, json=json_content, timeout=10
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                return response_json


def json_to_md(
    json_contents: List[Dict[str, Union[str, int, float]]],
    table_name: Optional[str] = None,
) -> str:
    """Converts a JSON object to a markdown table."""
    output_md = ""
    headers = json_contents[0].keys()
    for header in headers:
        header.replace("[", ".").replace("]", "")
        if table_name:
            header.replace(f"{table_name}.", "")
        output_md += f"| {header} "
    output_md += "|\n"
    for row in json_contents:
        for value in row.values():
            output_md += f"| {value} "
        output_md += "|\n"
    return output_md
