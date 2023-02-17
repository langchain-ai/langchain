"""SQLAlchemy wrapper around a database."""
from __future__ import annotations

import ast
from typing import Any, Iterable, List, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine

_TEMPLATE_PREFIX = """Table data will be described in the following format:

Table 'table name' has columns: {
column1 name: (column1 type, [list of example values for column1]),
column2 name: (column2 type, [list of example values for column2]),
...
}

These are the tables you can use, together with their column information:

"""


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)
        self._all_tables = set(self._inspector.get_table_names(schema=schema))
        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        self._sample_rows_in_table_info = sample_rows_in_table_info

    @classmethod
    def from_uri(cls, database_uri: str, **kwargs: Any) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        return cls(create_engine(database_uri), **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        tables = []
        for table_name in all_table_names:
            columns = []
            if self.dialect in ("sqlite", "duckdb"):
                create_table = self.run(
                    (
                        "SELECT sql FROM sqlite_master WHERE "
                        f"type='table' AND name='{table_name}'"
                    ),
                    fetch="one",
                )
            else:
                create_table = self.run(
                    f"SHOW CREATE TABLE `{table_name}`;",
                )

            for column in self._inspector.get_columns(table_name, schema=self._schema):
                columns.append(column["name"])

            if self._sample_rows_in_table_info:
                if self.dialect in ("sqlite", "duckdb"):
                    select_star = (
                        f"SELECT * FROM '{table_name}' LIMIT "
                        f"{self._sample_rows_in_table_info}"
                    )
                else:
                    select_star = (
                        f"SELECT * FROM `{table_name}` LIMIT "
                        f"{self._sample_rows_in_table_info}"
                    )

                sample_rows = self.run(select_star)

                sample_rows_ls = ast.literal_eval(sample_rows)
                sample_rows_ls = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows_ls)
                )

                columns_str = " ".join(columns)
                sample_rows_str = "\n".join([" ".join(row) for row in sample_rows_ls])

                tables.append(
                    create_table
                    + "\n\n"
                    + select_star
                    + "\n"
                    + columns_str
                    + "\n"
                    + sample_rows_str
                )

            else:
                tables.append(create_table)

        final_str = "\n\n\n".join(tables)
        return final_str

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.exec_driver_sql(command)
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    result = cursor.fetchone()[0]
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                return str(result)
        return ""
