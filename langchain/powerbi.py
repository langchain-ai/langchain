"""Wrapper around a Power BI endpoint."""
from __future__ import annotations

import logging
from typing import Any, Iterable, List

import requests
from azure.identity import DefaultAzureCredential

_LOGGER = logging.getLogger(__name__)


class PowerBIDataset:
    """Power BI Dataset connector."""

    def __init__(
        self,
        group_id: str | None,
        dataset_id: str,
        table_names: list[str],
        credential: DefaultAzureCredential,
        sample_rows_in_table_info: int = 1,
    ):
        """Create engine from database URI."""
        self._group_id = group_id
        self._dataset_id = dataset_id
        self._table_names = table_names
        if sample_rows_in_table_info < 1:
            raise ValueError("sample_rows_in_table_info must be >= 1")
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._credential = credential
        self.request_url = f"https://api.powerbi.com/v1.0/myorg/datasets/{self._dataset_id}/executeQueries"
        if self._group_id:
            self.request_url = f"https://api.powerbi.com/v1.0/myorg/groups/{self._group_id}/datasets/{self._dataset_id}/executeQueries"

    @property
    def token(self) -> str:
        """Get the token."""
        return self._credential.get_token(
            "https://analysis.windows.net/powerbi/api/.default"
        ).token

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        return self._table_names

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: List[str] | str | None = None) -> str:
        """Get information about specified tables."""
        all_table_names = self._table_names
        if table_names is not None:
            if isinstance(table_names, list):
                all_table_names = table_names
            else:
                all_table_names = [table_names]

        tables = []
        for table in all_table_names:
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
                return "Error with the connection to PowerBI, please review your authentication credentials."
            rows = result["results"][0]["tables"][0]["rows"]
            tables.append(str(rows))
        return ", ".join(tables)

    def run(self, command: str) -> Any:
        """Execute a DAX command and return a json representing the results."""

        result = requests.post(
            self.request_url,
            json={
                "queries": [{"query": command}],
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
