import os
from math import ceil
from typing import Any, Dict, List, Optional


class ArangoGraph:
    """ArangoDB wrapper for graph operations."""

    def __init__(self, db: Any) -> None:
        """Create a new ArangoDB graph wrapper instance."""
        self.set_db(db)
        self.set_schema()

    @property
    def db(self) -> Any:
        return self.__db

    @property
    def schema(self) -> Dict[str, Any]:
        return self.__schema

    def set_db(self, db: Any) -> None:
        from arango.database import Database

        if not isinstance(db, Database):
            msg = "**db** parameter must inherit from arango.database.Database"
            raise TypeError(msg)

        self.__db: Database = db
        self.set_schema()

    def set_schema(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the schema of the ArangoDB Database.
        Auto-generates Schema if **schema** is None.
        """
        self.__schema = self.generate_schema() if schema is None else schema

    def generate_schema(
        self, sample_ratio: float = 0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generates the schema of the ArangoDB Database and returns it
        User can specify a **sample_ratio** (0 to 1) to determine the
        ratio of documents/edges used (in relation to the Collection size)
        to render each Collection Schema.
        """
        if not 0 <= sample_ratio <= 1:
            raise ValueError("**sample_ratio** value must be in between 0 to 1")

        # Stores the Edge Relationships between each ArangoDB Document Collection
        graph_schema: List[Dict[str, Any]] = [
            {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]}
            for g in self.db.graphs()
        ]

        # Stores the schema of every ArangoDB Document/Edge collection
        collection_schema: List[Dict[str, Any]] = []

        for collection in self.db.collections():
            if collection["system"]:
                continue

            # Extract collection name, type, and size
            col_name: str = collection["name"]
            col_type: str = collection["type"]
            col_size: int = self.db.collection(col_name).count()

            # Skip collection if empty
            if col_size == 0:
                continue

            # Set number of ArangoDB documents/edges to retrieve
            limit_amount = ceil(sample_ratio * col_size) or 1

            aql = f"""
                FOR doc in {col_name}
                    LIMIT {limit_amount}
                    RETURN doc
            """

            doc: Dict[str, Any]
            properties: List[Dict[str, str]] = []
            for doc in self.__db.aql.execute(aql):
                for key, value in doc.items():
                    properties.append({"name": key, "type": type(value).__name__})

            collection_schema.append(
                {
                    "collection_name": col_name,
                    "collection_type": col_type,
                    f"{col_type}_properties": properties,
                    f"example_{col_type}": doc,
                }
            )

        return {"Graph Schema": graph_schema, "Collection Schema": collection_schema}

    def query(
        self, query: str, top_k: Optional[int] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Query the ArangoDB database."""
        import itertools

        cursor = self.__db.aql.execute(query, **kwargs)
        return [doc for doc in itertools.islice(cursor, top_k)]

    @classmethod
    def from_db_credentials(
        cls,
        url: Optional[str] = None,
        dbname: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Any:
        """Convenience constructor that builds Arango DB from credentials.

        Args:
            url: Arango DB url. Can be passed in as named arg or set as environment
                var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
            dbname: Arango DB name. Can be passed in as named arg or set as
                environment var ``ARANGODB_DBNAME``. Defaults to "_system".
            username: Can be passed in as named arg or set as environment var
                ``ARANGODB_USERNAME``. Defaults to "root".
            password: Can be passed ni as named arg or set as environment var
                ``ARANGODB_PASSWORD``. Defaults to "".

        Returns:
            An arango.database.StandardDatabase.
        """
        db = get_arangodb_client(
            url=url, dbname=dbname, username=username, password=password
        )
        return cls(db)


def get_arangodb_client(
    url: Optional[str] = None,
    dbname: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """Get the Arango DB client from credentials.

    Args:
        url: Arango DB url. Can be passed in as named arg or set as environment
            var ``ARANGODB_URL``. Defaults to "http://localhost:8529".
        dbname: Arango DB name. Can be passed in as named arg or set as
            environment var ``ARANGODB_DBNAME``. Defaults to "_system".
        username: Can be passed in as named arg or set as environment var
            ``ARANGODB_USERNAME``. Defaults to "root".
        password: Can be passed ni as named arg or set as environment var
            ``ARANGODB_PASSWORD``. Defaults to "".

    Returns:
        An arango.database.StandardDatabase.
    """
    try:
        from arango import ArangoClient
    except ImportError as e:
        raise ImportError(
            "Unable to import arango, please install with `pip install python-arango`."
        ) from e

    _url: str = url or os.environ.get("ARANGODB_URL", "http://localhost:8529")  # type: ignore[assignment]  # noqa: E501
    _dbname: str = dbname or os.environ.get("ARANGODB_DBNAME", "_system")  # type: ignore[assignment]  # noqa: E501
    _username: str = username or os.environ.get("ARANGODB_USERNAME", "root")  # type: ignore[assignment]  # noqa: E501
    _password: str = password or os.environ.get("ARANGODB_PASSWORD", "")  # type: ignore[assignment]  # noqa: E501

    return ArangoClient(_url).db(_dbname, _username, _password, verify=True)
