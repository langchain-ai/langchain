from typing import Any, Dict, List, Optional, Union


class ArangoDBGraph:
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

    def set_schema(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """Set the schema of the ArangoDB Database. Auto-generates Schema if **schema** is None."""
        self.__schema = self.generate_schema() if schema is None else schema

    def generate_schema(self, sample_ratio: float = 0) -> Dict[str, Any]:
        """Generates the schema of the ArangoDB Database and returns it"""
        if not 0 <= sample_ratio <= 1:
            raise ValueError("**sample_ratio** value must be in between 0 to 1")

        graph_schema = [
            {"graph_name": g["name"], "edge_definitions": g["edge_definitions"]}
            for g in self.__db.graphs()
        ]

        collection_schema: List[Dict[str, Union[str, Dict[str, str]]]] = []
        for collection in self.__db.collections():
            if collection["system"]:
                continue

            col_name: str = collection["name"]
            col_type: str = collection["type"]
            col_size: int = self.__db.collection(col_name).count()

            limit_amount = round(sample_ratio * col_size) or 1

            aql = f"""
                FOR doc in {col_name}
                    // SORT RAND() ?
                    LIMIT {limit_amount}
                    RETURN doc
            """

            doc: dict
            properties = {}  # defaultdict(set)
            for doc in self.__db.aql.execute(aql):
                for k, v in doc.items():
                    if k == "_rev":
                        continue

                    properties[k] = type(v).__name__

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
