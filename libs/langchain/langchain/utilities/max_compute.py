from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional

from langchain.utils import get_from_env

if TYPE_CHECKING:
    from odps import ODPS


class MaxComputeAPIWrapper:
    """Interface for querying Alibaba Cloud MaxCompute tables."""

    def __init__(self, client: ODPS):
        """Initialize MaxCompute document loader.

        Args:
            client: odps.ODPS MaxCompute client object.
        """
        self.client = client

    @classmethod
    def from_params(
        cls,
        endpoint: str,
        project: str,
        *,
        access_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ) -> MaxComputeAPIWrapper:
        """Convenience constructor that builds the odsp.ODPS MaxCompute client from
            given parameters.

        Args:
            endpoint: MaxCompute endpoint.
            project: A project is a basic organizational unit of MaxCompute, which is
                similar to a database.
            access_id: MaxCompute access ID. Should be passed in directly or set as the
                environment variable `MAX_COMPUTE_ACCESS_ID`.
            secret_access_key: MaxCompute secret access key. Should be passed in
                directly or set as the environment variable
                `MAX_COMPUTE_SECRET_ACCESS_KEY`.
        """
        try:
            from odps import ODPS
        except ImportError as ex:
            raise ImportError(
                "Could not import pyodps python package. "
                "Please install it with `pip install pyodps` or refer to "
                "https://pyodps.readthedocs.io/."
            ) from ex
        access_id = access_id or get_from_env("access_id", "MAX_COMPUTE_ACCESS_ID")
        secret_access_key = secret_access_key or get_from_env(
            "secret_access_key", "MAX_COMPUTE_SECRET_ACCESS_KEY"
        )
        client = ODPS(
            access_id=access_id,
            secret_access_key=secret_access_key,
            project=project,
            endpoint=endpoint,
        )
        if not client.exist_project(project):
            raise ValueError(f'The project "{project}" does not exist.')

        return cls(client)

    def lazy_query(self, query: str) -> Iterator[dict]:
        # Execute SQL query.
        with self.client.execute_sql(query).open_reader() as reader:
            if reader.count == 0:
                raise ValueError("Table contains no data.")
            for record in reader:
                yield {k: v for k, v in record}

    def query(self, query: str) -> List[dict]:
        return list(self.lazy_query(query))
