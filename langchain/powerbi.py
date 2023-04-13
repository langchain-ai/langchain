"""Wrapper around a Power BI endpoint."""
from __future__ import annotations

import logging
from typing import Any, Iterable, List

import requests
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import ChainedTokenCredential
from azure.identity._internal import InteractiveCredential

_LOGGER = logging.getLogger(__name__)


class PowerBIDataset:
    """Power BI Dataset connector."""

    def __init__(
        self,
        group_id: str | None,
        dataset_id: str,
        table_names: list[str],
        credential: ChainedTokenCredential | InteractiveCredential | None = None,
        token: str | None = None,
        impersonated_user_name: str | None = None,
        sample_rows_in_table_info: int = 1,
    ):
        """Create PowerBI engine from dataset ID and credential or token.

        Use either the credential or a supplied token to authenticate. If both are supplied the credential is used to generate a token.
        The impersonated_user_name is the UPN of a user to be impersonated. If the model is not RLS enabled, this will be ignored.
        """
        self._group_id = group_id
        self._dataset_id = dataset_id
        self._table_names = table_names
        self._schemas: dict[str, str] = {}
        if sample_rows_in_table_info < 1:
            raise ValueError("sample_rows_in_table_info must be >= 1")
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._credential = credential
        self._token = token
        self._impersonated_user_name = impersonated_user_name
        self.request_url = f"https://api.powerbi.com/v1.0/myorg/datasets/{self._dataset_id}/executeQueries"  # noqa: E501 # pylint: disable=C0301
        if self._group_id:
            self.request_url = f"https://api.powerbi.com/v1.0/myorg/groups/{self._group_id}/datasets/{self._dataset_id}/executeQueries"  # noqa: E501 # pylint: disable=C0301

    @property
    def token(self) -> str:
        """Get the token."""
        if self._credential:
            return self._credential.get_token(
                "https://analysis.windows.net/powerbi/api/.default"
            ).token
        if self._token:
            return self._token
        raise ClientAuthenticationError("No credential or token supplied.")

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        return self._table_names

    def get_schemas(self) -> str:
        """Get the available schema's."""
        if self._schemas:
            return ", ".join(
                [f"{key}: {value}" for key, value in self._schemas.items()]
            )
        else:
            return "No known schema's yet. Use the schema_powerbi tool first."

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: List[str] | str | None = None) -> str:
        """Get information about specified tables."""
        all_table_names = self._table_names
        if table_names is not None:
            if (
                isinstance(table_names, list)
                and len(table_names) > 0
                and table_names[0] != ""
            ):
                all_table_names = table_names
            elif isinstance(table_names, str) and table_names != "":
                all_table_names = [table_names]

        tables = []
        for table in all_table_names:
            if table in self._schemas:
                tables.append(str(self._schemas[table]))
                continue
            query = f"EVALUATE TOPN({self._sample_rows_in_table_info}, {table})"
            try:
                result = self.run(query)
            except requests.exceptions.Timeout:
                _LOGGER.warning("Timeout while getting table info for %s", table)
                continue
            except requests.exceptions.HTTPError as err:
                _LOGGER.warning(
                    "HTTP error while getting table info for %s: %s", table, err
                )
                return "Error with the connection to PowerBI, please review your authentication credentials."  # noqa: E501 # pylint: disable=C0301
            rows = json_to_md(result["results"][0]["tables"][0]["rows"])
            self._schemas[table] = rows
            tables.append(str(rows))
        return ", ".join(tables)

    def run(self, command: str) -> Any:
        """Execute a DAX command and return a json representing the results."""

        result = requests.post(
            self.request_url,
            json={
                "queries": [{"query": command}],
                "impersonatedUserName": self._impersonated_user_name,
                "serializerSettings": {"includeNulls": True},
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.token,
            },
            timeout=10,
        )
        result.raise_for_status()
        return result.json()


def json_to_md(
    json_contents: list[dict[str, str | int | float]], table_name: str | None = None
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
