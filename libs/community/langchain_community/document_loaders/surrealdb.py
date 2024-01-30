import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class SurrealDBLoader(BaseLoader):
    """Load SurrealDB documents."""

    def __init__(
        self,
        filter_criteria: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        try:
            from surrealdb import Surreal
        except ImportError as e:
            raise ImportError(
                """Cannot import from surrealdb.
                please install with `pip install surrealdb`."""
            ) from e

        self.dburl = kwargs.pop("dburl", "ws://localhost:8000/rpc")

        if self.dburl[0:2] == "ws":
            self.sdb = Surreal(self.dburl)
        else:
            raise ValueError("Only websocket connections are supported at this time.")

        self.filter_criteria = filter_criteria or {}

        if "table" in self.filter_criteria:
            raise ValueError(
                "key `table` is not a valid criteria for `filter_criteria` argument."
            )

        self.ns = kwargs.pop("ns", "langchain")
        self.db = kwargs.pop("db", "database")
        self.table = kwargs.pop("table", "documents")
        self.sdb = Surreal(self.dburl)
        self.kwargs = kwargs

        asyncio.run(self.initialize())

    async def initialize(self) -> None:
        """
        Initialize connection to surrealdb database
        and authenticate if credentials are provided
        """
        await self.sdb.connect()
        if "db_user" in self.kwargs and "db_pass" in self.kwargs:
            user = self.kwargs.get("db_user")
            password = self.kwargs.get("db_pass")
            await self.sdb.signin({"user": user, "pass": password})

        await self.sdb.use(self.ns, self.db)

    def load(self) -> List[Document]:
        async def _load() -> List[Document]:
            await self.initialize()
            return await self.aload()

        return asyncio.run(_load())

    async def aload(self) -> List[Document]:
        """Load data into Document objects."""

        query = "SELECT * FROM type::table($table)"
        if self.filter_criteria is not None and len(self.filter_criteria) > 0:
            query += " WHERE "
            for idx, key in enumerate(self.filter_criteria):
                query += f""" {"AND" if idx > 0 else ""} {key} = ${key}"""

        metadata = {
            "ns": self.ns,
            "db": self.db,
            "table": self.table,
        }
        results = await self.sdb.query(
            query, {"table": self.table, **self.filter_criteria}
        )

        return [
            (
                Document(
                    page_content=json.dumps(result),
                    metadata={"id": result["id"], **result["metadata"], **metadata},
                )
            )
            for result in results[0]["result"]
        ]
