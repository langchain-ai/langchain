"""Apache Cassandra database wrapper."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

if TYPE_CHECKING:
    from cassandra.cluster import ResultSet, Session

IGNORED_KEYSPACES = [
    "system",
    "system_auth",
    "system_distributed",
    "system_schema",
    "system_traces",
    "system_views",
    "datastax_sla",
    "data_endpoint_auth",
]


class CassandraDatabase:
    """Apache Cassandra® database wrapper."""

    def __init__(
        self,
        session: Optional[Session] = None,
        exclude_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        cassio_init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        _session = self._resolve_session(session, cassio_init_kwargs)
        if not _session:
            raise ValueError("Session not provided and cannot be resolved")
        self._session = _session

        self._exclude_keyspaces = IGNORED_KEYSPACES
        self._exclude_tables = exclude_tables or []
        self._include_tables = include_tables or []

    def run(
        self,
        query: str,
        fetch: str = "all",
        **kwargs: Any,
    ) -> Union[list, Dict[str, Any], ResultSet]:
        """Execute a CQL query and return the results."""
        if fetch == "all":
            return self.fetch_all(query, **kwargs)
        elif fetch == "one":
            return self.fetch_one(query, **kwargs)
        elif fetch == "cursor":
            return self._fetch(query, **kwargs)
        else:
            raise ValueError("Fetch parameter must be either 'one', 'all', or 'cursor'")

    def _fetch(self, query: str, **kwargs: Any) -> ResultSet:
        clean_query = self._validate_cql(query, "SELECT")
        return self._session.execute(clean_query, **kwargs)

    def fetch_all(self, query: str, **kwargs: Any) -> list:
        return list(self._fetch(query, **kwargs))

    def fetch_one(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        result = self._fetch(query, **kwargs)
        return result.one()._asdict() if result else {}

    def get_keyspace_tables(self, keyspace: str) -> List[Table]:
        """Get the Table objects for the specified keyspace."""
        schema = self._resolve_schema([keyspace])
        if keyspace in schema:
            return schema[keyspace]
        else:
            return []

    # This is a more basic string building function that doesn't use a query builder
    # or prepared statements
    # TODO: Refactor to use prepared statements
    def get_table_data(
        self, keyspace: str, table: str, predicate: str, limit: int
    ) -> str:
        """Get data from the specified table in the specified keyspace."""

        query = f"SELECT * FROM {keyspace}.{table}"

        if predicate:
            query += f" WHERE {predicate}"
        if limit:
            query += f" LIMIT {limit}"

        query += ";"

        result = self.fetch_all(query)
        data = "\n".join(str(row) for row in result)
        return data

    def get_context(self) -> Dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        keyspaces = self._fetch_keyspaces()
        return {"keyspaces": ", ".join(keyspaces)}

    def format_keyspace_to_markdown(
        self, keyspace: str, tables: Optional[List[Table]] = None
    ) -> str:
        """
        Generates a markdown representation of the schema for a specific keyspace
        by iterating over all tables within that keyspace and calling their
        as_markdown method.

        Args:
            keyspace: The name of the keyspace to generate markdown documentation for.
            tables: list of tables in the keyspace; it will be resolved if not provided.

        Returns:
            A string containing the markdown representation of the specified
            keyspace schema.
        """
        if not tables:
            tables = self.get_keyspace_tables(keyspace)

        if tables:
            output = f"## Keyspace: {keyspace}\n\n"
            if tables:
                for table in tables:
                    output += table.as_markdown(include_keyspace=False, header_level=3)
                    output += "\n\n"
            else:
                output += "No tables present in keyspace\n\n"

            return output
        else:
            return ""

    def format_schema_to_markdown(self) -> str:
        """
        Generates a markdown representation of the schema for all keyspaces and tables
        within the CassandraDatabase instance. This method utilizes the
        format_keyspace_to_markdown method to create markdown sections for each
        keyspace, assembling them into a comprehensive schema document.

        Iterates through each keyspace in the database, utilizing
        format_keyspace_to_markdown to generate markdown for each keyspace's schema,
        including details of its tables. These sections are concatenated to form a
        single markdown document that represents the schema of the entire database or
        the subset of keyspaces that have been resolved in this instance.

        Returns:
            A markdown string that documents the schema of all resolved keyspaces and
            their tables within this CassandraDatabase instance. This includes keyspace
            names, table names, comments, columns, partition keys, clustering keys,
            and indexes for each table.
        """
        schema = self._resolve_schema()
        output = "# Cassandra Database Schema\n\n"
        for keyspace, tables in schema.items():
            output += f"{self.format_keyspace_to_markdown(keyspace, tables)}\n\n"
        return output

    def _validate_cql(self, cql: str, type: str = "SELECT") -> str:
        """
        Validates a CQL query string for basic formatting and safety checks.
        Ensures that `cql` starts with the specified type (e.g., SELECT) and does
        not contain content that could indicate CQL injection vulnerabilities.

        Args:
            cql: The CQL query string to be validated.
            type: The expected starting keyword of the query, used to verify
                that the query begins with the correct operation type
                (e.g., "SELECT", "UPDATE"). Defaults to "SELECT".

        Returns:
            The trimmed and validated CQL query string without a trailing semicolon.

        Raises:
            ValueError: If the value of `type` is not supported
            DatabaseError: If `cql` is considered unsafe
        """
        SUPPORTED_TYPES = ["SELECT"]
        if type and type.upper() not in SUPPORTED_TYPES:
            raise ValueError(
                f"""Unsupported CQL type: {type}. Supported types: 
                             {SUPPORTED_TYPES}"""
            )

        # Basic sanity checks
        cql_trimmed = cql.strip()
        if not cql_trimmed.upper().startswith(type.upper()):
            raise DatabaseError(f"CQL must start with {type.upper()}.")

        # Allow a trailing semicolon, but remove (it is optional with the Python driver)
        cql_trimmed = cql_trimmed.rstrip(";")

        # Consider content within matching quotes to be "safe"
        # Remove single-quoted strings
        cql_sanitized = re.sub(r"'.*?'", "", cql_trimmed)

        # Remove double-quoted strings
        cql_sanitized = re.sub(r'".*?"', "", cql_sanitized)

        # Find unsafe content in the remaining CQL
        if ";" in cql_sanitized:
            raise DatabaseError(
                """Potentially unsafe CQL, as it contains a ; at a 
                                place other than the end or within quotation marks."""
            )

        # The trimmed query, before modifications
        return cql_trimmed

    def _fetch_keyspaces(self, keyspaces: Optional[List[str]] = None) -> List[str]:
        """
        Fetches a list of keyspace names from the Cassandra database. The list can be
        filtered by a provided list of keyspace names or by excluding predefined
        keyspaces.

        Args:
            keyspaces: A list of keyspace names to specifically include.
                If provided and not empty, the method returns only the keyspaces
                present in this list.
                If not provided or empty, the method returns all keyspaces except those
                specified in the _exclude_keyspaces attribute.

        Returns:
            A list of keyspace names according to the filtering criteria.
        """
        all_keyspaces = self.fetch_all(
            "SELECT keyspace_name FROM system_schema.keyspaces"
        )

        # Filtering keyspaces based on 'keyspace_list' and '_exclude_keyspaces'
        filtered_keyspaces = []
        for ks in all_keyspaces:
            if not isinstance(ks, Dict):
                continue  # Skip if the row is not a dictionary.

            keyspace_name = ks["keyspace_name"]
            if keyspaces and keyspace_name in keyspaces:
                filtered_keyspaces.append(keyspace_name)
            elif not keyspaces and keyspace_name not in self._exclude_keyspaces:
                filtered_keyspaces.append(keyspace_name)

        return filtered_keyspaces

    def _format_keyspace_query(self, query: str, keyspaces: List[str]) -> str:
        # Construct IN clause for CQL query
        keyspace_in_clause = ", ".join([f"'{ks}'" for ks in keyspaces])
        return f"""{query} WHERE keyspace_name IN ({keyspace_in_clause})"""

    def _fetch_tables_data(self, keyspaces: List[str]) -> list:
        """Fetches tables schema data, filtered by a list of keyspaces.
        This method allows for efficiently fetching schema information for multiple
        keyspaces in a single operation, enabling applications to programmatically
        analyze or document the database schema.

        Args:
            keyspaces: A list of keyspace names from which to fetch tables schema data.

        Returns:
            Dictionaries of table details (keyspace name,  table name, and comment).
        """
        tables_query = self._format_keyspace_query(
            "SELECT keyspace_name, table_name, comment  FROM system_schema.tables",
            keyspaces,
        )
        return self.fetch_all(tables_query)

    def _fetch_columns_data(self, keyspaces: List[str]) -> list:
        """Fetches columns schema data, filtered by a list of keyspaces.
        This method allows for efficiently fetching schema information for multiple
        keyspaces in a single operation, enabling applications to programmatically
        analyze or document the database schema.

        Args:
            keyspaces: A list of keyspace names from which to fetch tables schema data.

        Returns:
            Dictionaries of column details (keyspace name, table name, column name,
            type, kind, and position).
        """
        tables_query = self._format_keyspace_query(
            """
                    SELECT keyspace_name, table_name, column_name, type, kind, 
                        clustering_order, position 
                    FROM system_schema.columns
                    """,
            keyspaces,
        )
        return self.fetch_all(tables_query)

    def _fetch_indexes_data(self, keyspaces: List[str]) -> list:
        """Fetches indexes schema data, filtered by a list of keyspaces.
        This method allows for efficiently fetching schema information for multiple
        keyspaces in a single operation, enabling applications to programmatically
        analyze or document the database schema.

        Args:
            keyspaces: A list of keyspace names from which to fetch tables schema data.

        Returns:
            Dictionaries of index details (keyspace name, table name, index name, kind,
            and options).
        """
        tables_query = self._format_keyspace_query(
            """
                    SELECT keyspace_name, table_name, index_name, 
                        kind, options 
                    FROM system_schema.indexes
                    """,
            keyspaces,
        )
        return self.fetch_all(tables_query)

    def _resolve_schema(
        self, keyspaces: Optional[List[str]] = None
    ) -> Dict[str, List[Table]]:
        """
        Efficiently fetches and organizes Cassandra table schema information,
        such as comments, columns, and indexes, into a dictionary mapping keyspace
        names to lists of Table objects.

        Args:
            keyspaces: An optional list of keyspace names from which to fetch tables
                schema data.

        Returns:
            A dictionary with keyspace names as keys and lists of Table objects as
            values, where each Table object is populated with schema details
            appropriate for its keyspace and table name.
        """
        if not keyspaces:
            keyspaces = self._fetch_keyspaces()

        tables_data = self._fetch_tables_data(keyspaces)
        columns_data = self._fetch_columns_data(keyspaces)
        indexes_data = self._fetch_indexes_data(keyspaces)

        keyspace_dict: dict = {}
        for table_data in tables_data:
            keyspace = table_data.keyspace_name
            table_name = table_data.table_name
            comment = table_data.comment

            if self._include_tables and table_name not in self._include_tables:
                continue

            if self._exclude_tables and table_name in self._exclude_tables:
                continue

            # Filter columns and indexes for this table
            table_columns = [
                (c.column_name, c.type)
                for c in columns_data
                if c.keyspace_name == keyspace and c.table_name == table_name
            ]

            partition_keys = [
                c.column_name
                for c in columns_data
                if c.kind == "partition_key"
                and c.keyspace_name == keyspace
                and c.table_name == table_name
            ]

            clustering_keys = [
                (c.column_name, c.clustering_order)
                for c in columns_data
                if c.kind == "clustering"
                and c.keyspace_name == keyspace
                and c.table_name == table_name
            ]

            table_indexes = [
                (c.index_name, c.kind, c.options)
                for c in indexes_data
                if c.keyspace_name == keyspace and c.table_name == table_name
            ]

            table_obj = Table(
                keyspace=keyspace,
                table_name=table_name,
                comment=comment,
                columns=table_columns,
                partition=partition_keys,
                clustering=clustering_keys,
                indexes=table_indexes,
            )

            if keyspace not in keyspace_dict:
                keyspace_dict[keyspace] = []
            keyspace_dict[keyspace].append(table_obj)

        return keyspace_dict

    @staticmethod
    def _resolve_session(
        session: Optional[Session] = None,
        cassio_init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Session]:
        """
        Attempts to resolve and return a Session object for use in database operations.

        This function follows a specific order of precedence to determine the
        appropriate session to use:
        1. `session` parameter if given,
        2. Existing `cassio` session,
        3. A new `cassio` session derived from `cassio_init_kwargs`,
        4. `None`

        Args:
            session: An optional session to use directly.
            cassio_init_kwargs: An optional dictionary of keyword arguments to `cassio`.

        Returns:
            The resolved session object if successful, or `None` if the session
            cannot be resolved.

        Raises:
            ValueError: If `cassio_init_kwargs` is provided but is not a dictionary of
            keyword arguments.
        """

        # Prefer given session
        if session:
            return session

        # If a session is not provided, create one using cassio if available
        # dynamically import cassio to avoid circular imports
        try:
            import cassio.config
        except ImportError:
            raise ValueError(
                "cassio package not found, please install with" " `pip install cassio`"
            )

        # Use pre-existing session on cassio
        s = cassio.config.resolve_session()
        if s:
            return s

        # Try to init and return cassio session
        if cassio_init_kwargs:
            if isinstance(cassio_init_kwargs, dict):
                cassio.init(**cassio_init_kwargs)
                s = cassio.config.check_resolve_session()
                return s
            else:
                raise ValueError("cassio_init_kwargs must be a keyword dictionary")

        # return None if we're not able to resolve
        return None


class DatabaseError(Exception):
    """Exception raised for errors in the database schema.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class Table(BaseModel):
    keyspace: str
    """The keyspace in which the table exists."""

    table_name: str
    """The name of the table."""

    comment: Optional[str] = None
    """The comment associated with the table."""

    columns: List[Tuple[str, str]] = Field(default_factory=list)
    partition: List[str] = Field(default_factory=list)
    clustering: List[Tuple[str, str]] = Field(default_factory=list)
    indexes: List[Tuple[str, str, str]] = Field(default_factory=list)

    class Config:
        frozen = True

    @root_validator(pre=False, skip_on_failure=True)
    def check_required_fields(cls, class_values: dict) -> dict:
        if not class_values["columns"]:
            raise ValueError("non-empty column list for must be provided")
        if not class_values["partition"]:
            raise ValueError("non-empty partition list must be provided")
        return class_values

    @classmethod
    def from_database(
        cls, keyspace: str, table_name: str, db: CassandraDatabase
    ) -> Table:
        columns, partition, clustering = cls._resolve_columns(keyspace, table_name, db)
        return cls(
            keyspace=keyspace,
            table_name=table_name,
            comment=cls._resolve_comment(keyspace, table_name, db),
            columns=columns,
            partition=partition,
            clustering=clustering,
            indexes=cls._resolve_indexes(keyspace, table_name, db),
        )

    def as_markdown(
        self, include_keyspace: bool = True, header_level: Optional[int] = None
    ) -> str:
        """
        Generates a Markdown representation of the Cassandra table schema, allowing for
        customizable header levels for the table name section.

        Args:
            include_keyspace: If True, includes the keyspace in the output.
                Defaults to True.
            header_level: Specifies the markdown header level for the table name.
                If None, the table name is included without a header.
                Defaults to None (no header level).

        Returns:
            A string in Markdown format detailing the table name
            (with optional header level), keyspace (optional), comment, columns,
            partition keys, clustering keys (with optional clustering order),
            and indexes.
        """
        output = ""
        if header_level is not None:
            output += f"{'#' * header_level} "
        output += f"Table Name: {self.table_name}\n"

        if include_keyspace:
            output += f"- Keyspace: {self.keyspace}\n"
        if self.comment:
            output += f"- Comment: {self.comment}\n"

        output += "- Columns\n"
        for column, type in self.columns:
            output += f"  - {column} ({type})\n"

        output += f"- Partition Keys: ({', '.join(self.partition)})\n"
        output += "- Clustering Keys: "
        if self.clustering:
            cluster_list = []
            for column, clustering_order in self.clustering:
                if clustering_order.lower() == "none":
                    cluster_list.append(column)
                else:
                    cluster_list.append(f"{column} {clustering_order}")
            output += f"({', '.join(cluster_list)})\n"

        if self.indexes:
            output += "- Indexes\n"
            for name, kind, options in self.indexes:
                output += f"  - {name} : kind={kind}, options={options}\n"

        return output

    @staticmethod
    def _resolve_comment(
        keyspace: str, table_name: str, db: CassandraDatabase
    ) -> Optional[str]:
        result = db.run(
            f"""SELECT comment 
                FROM system_schema.tables 
                WHERE keyspace_name = '{keyspace}' 
                AND table_name = '{table_name}';""",
            fetch="one",
        )

        if isinstance(result, dict):
            comment = result.get("comment")
            if comment:
                return comment
            else:
                return None  # Default comment if none is found
        else:
            raise ValueError(
                f"""Unexpected result type from db.run: 
                             {type(result).__name__}"""
            )

    @staticmethod
    def _resolve_columns(
        keyspace: str, table_name: str, db: CassandraDatabase
    ) -> Tuple[List[Tuple[str, str]], List[str], List[Tuple[str, str]]]:
        columns = []
        partition_info = []
        cluster_info = []
        results = db.run(
            f"""SELECT column_name, type, kind, clustering_order, position 
                           FROM system_schema.columns 
                           WHERE keyspace_name = '{keyspace}' 
                           AND table_name = '{table_name}';"""
        )
        # Type check to ensure 'results' is a sequence of dictionaries.
        if not isinstance(results, Sequence):
            raise TypeError("Expected a sequence of dictionaries from 'run' method.")

        for row in results:
            if not isinstance(row, Dict):
                continue  # Skip if the row is not a dictionary.

            columns.append((row["column_name"], row["type"]))
            if row["kind"] == "partition_key":
                partition_info.append((row["column_name"], row["position"]))
            elif row["kind"] == "clustering":
                cluster_info.append(
                    (row["column_name"], row["clustering_order"], row["position"])
                )

        partition = [
            column_name for column_name, _ in sorted(partition_info, key=lambda x: x[1])
        ]

        cluster = [
            (column_name, clustering_order)
            for column_name, clustering_order, _ in sorted(
                cluster_info, key=lambda x: x[2]
            )
        ]

        return columns, partition, cluster

    @staticmethod
    def _resolve_indexes(
        keyspace: str, table_name: str, db: CassandraDatabase
    ) -> List[Tuple[str, str, str]]:
        indexes = []
        results = db.run(
            f"""SELECT index_name, kind, options 
                        FROM system_schema.indexes 
                        WHERE keyspace_name = '{keyspace}' 
                        AND table_name = '{table_name}';"""
        )

        # Type check to ensure 'results' is a sequence of dictionaries
        if not isinstance(results, Sequence):
            raise TypeError("Expected a sequence of dictionaries from 'run' method.")

        for row in results:
            if not isinstance(row, Dict):
                continue  # Skip if the row is not a dictionary.

            # Convert 'options' to string if it's not already,
            # assuming it's JSON-like and needs conversion
            index_options = row["options"]
            if not isinstance(index_options, str):
                # Assuming index_options needs to be serialized or simply converted
                index_options = str(index_options)

            indexes.append((row["index_name"], row["kind"], index_options))

        return indexes
