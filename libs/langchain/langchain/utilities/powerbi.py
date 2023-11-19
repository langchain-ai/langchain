"""Wrapper around a Power BI endpoint."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("POWERBI_BASE_URL", "https://api.powerbi.com/v1.0/myorg")

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
    schemas: Dict[str, str] = Field(default_factory=dict)
    aiosession: Optional[aiohttp.ClientSession] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @validator("table_names", allow_reuse=True)
    def fix_table_names(cls, table_names: List[str]) -> List[str]:
        """Fix the table names."""
        return [fix_table_name(table) for table in table_names]

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
            return f"{BASE_URL}/groups/{self.group_id}/datasets/{self.dataset_id}/executeQueries"  # noqa: E501 # pylint: disable=C0301
        return f"{BASE_URL}/datasets/{self.dataset_id}/executeQueries"  # noqa: E501 # pylint: disable=C0301

    @property
    def headers(self) -> Dict[str, str]:
        """Get the token."""
        if self.token:
            return {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            }
        from azure.core.exceptions import (
            ClientAuthenticationError,  # pylint: disable=import-outside-toplevel
        )

        if self.credential:
            try:
                token = self.credential.get_token(
                    "https://analysis.windows.net/powerbi/api/.default"
                ).token
                return {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + token,
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                raise ClientAuthenticationError(
                    "Could not get a token from the supplied credentials."
                ) from exc
        raise ClientAuthenticationError("No credential or token supplied.")

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
    ) -> Optional[List[str]]:
        """Get the tables names that need to be queried, after checking they exist."""
        if table_names is not None:
            if (
                isinstance(table_names, list)
                and len(table_names) > 0
                and table_names[0] != ""
            ):
                fixed_tables = [fix_table_name(table) for table in table_names]
                non_existing_tables = [
                    table for table in fixed_tables if table not in self.table_names
                ]
                if non_existing_tables:
                    logger.warning(
                        "Table(s) %s not found in dataset.",
                        ", ".join(non_existing_tables),
                    )
                tables = [
                    table for table in fixed_tables if table not in non_existing_tables
                ]
                return tables if tables else None
            if isinstance(table_names, str) and table_names != "":
                if table_names not in self.table_names:
                    logger.warning("Table %s not found in dataset.", table_names)
                    return None
                return [fix_table_name(table_names)]
        return self.table_names

    def _get_tables_todo(self, tables_todo: List[str]) -> List[str]:
        """Get the tables that still need to be queried."""
        return [table for table in tables_todo if table not in self.schemas]

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
        if tables_requested is None:
            return "No (valid) tables requested."
        tables_todo = self._get_tables_todo(tables_requested)
        for table in tables_todo:
            self._get_schema(table)
        return self._get_schema_for_tables(tables_requested)

    async def aget_table_info(
        self, table_names: Optional[Union[List[str], str]] = None
    ) -> str:
        """Get information about specified tables."""
        tables_requested = self._get_tables_to_query(table_names)
        if tables_requested is None:
            return "No (valid) tables requested."
        tables_todo = self._get_tables_todo(tables_requested)
        await asyncio.gather(*[self._aget_schema(table) for table in tables_todo])
        return self._get_schema_for_tables(tables_requested)

    def _get_schema(self, table: str) -> None:
        """Get the schema for a table."""
        try:
            result = self.run(
                f"EVALUATE TOPN({self.sample_rows_in_table_info}, {table})"
            )
            self.schemas[table] = json_to_md(result["results"][0]["tables"][0]["rows"])
        except Timeout:
            logger.warning("Timeout while getting table info for %s", table)
            self.schemas[table] = "unknown"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Error while getting table info for %s: %s", table, exc)
            self.schemas[table] = "unknown"

    async def _aget_schema(self, table: str) -> None:
        """Get the schema for a table."""
        try:
            result = await self.arun(
                f"EVALUATE TOPN({self.sample_rows_in_table_info}, {table})"
            )
            self.schemas[table] = json_to_md(result["results"][0]["tables"][0]["rows"])
        except ServerTimeoutError:
            logger.warning("Timeout while getting table info for %s", table)
            self.schemas[table] = "unknown"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Error while getting table info for %s: %s", table, exc)
            self.schemas[table] = "unknown"

    def _create_json_content(self, command: str) -> dict[str, Any]:
        """Create the json content for the request."""
        return {
            "queries": [{"query": rf"{command}"}],
            "impersonatedUserName": self.impersonated_user_name,
            "serializerSettings": {"includeNulls": True},
        }

    def run(self, command: str) -> Any:
        """Execute a DAX command and return a json representing the results."""
        logger.debug("Running command: %s", command)
        response = requests.post(
            self.request_url,
            json=self._create_json_content(command),
            headers=self.headers,
            timeout=10,
        )
        if response.status_code == 403:
            return (
                "TokenError: Could not login to PowerBI, please check your credentials."
            )
        return response.json()

    async def arun(self, command: str) -> Any:
        """Execute a DAX command and return the result asynchronously."""
        logger.debug("Running command: %s", command)
        if self.aiosession:
            async with self.aiosession.post(
                self.request_url,
                headers=self.headers,
                json=self._create_json_content(command),
                timeout=10,
            ) as response:
                if response.status == 403:
                    return "TokenError: Could not login to PowerBI, please check your credentials."  # noqa: E501
                response_json = await response.json(content_type=response.content_type)
                return response_json
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.request_url,
                headers=self.headers,
                json=self._create_json_content(command),
                timeout=10,
            ) as response:
                if response.status == 403:
                    return "TokenError: Could not login to PowerBI, please check your credentials."  # noqa: E501
                response_json = await response.json(content_type=response.content_type)
                return response_json


def json_to_md(
    json_contents: List[Dict[str, Union[str, int, float]]],
    table_name: Optional[str] = None,
) -> str:
    """Converts a JSON object to a markdown table."""
    if len(json_contents) == 0:
        return ""
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


def fix_table_name(table: str) -> str:
    """Add single quotes around table names that contain spaces."""
    if " " in table and not table.startswith("'") and not table.endswith("'"):
        return f"'{table}'"
    return table
