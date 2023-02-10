"""SQLAlchemy wrapper around a database."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 0,
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

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated by Rajkumar et al, 2022 (https://arxiv.org/abs/2204.00498).
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        template = "Table '{table_name}' has columns: {columns}."

        tables = []
        for table_name in all_table_names:
            columns = []
            for column in self._inspector.get_columns(table_name, schema=self._schema):
                columns.append(f"{column['name']} ({str(column['type'])})")
            column_str = ", ".join(columns)
            table_str = template.format(table_name=table_name, columns=column_str)

            if self._sample_rows_in_table_info:
                row_template = (
                    " Here is an example of {n_rows} rows from this table "
                    "(long strings are truncated):\n"
                    "{sample_rows}"
                )
                sample_rows = self.run(
                    f"SELECT * FROM '{table_name}' LIMIT "
                    f"{self._sample_rows_in_table_info}"
                )
                sample_rows = eval(sample_rows)
                if len(sample_rows) > 0:
                    n_rows = len(sample_rows)
                    sample_rows = "\n".join(
                        [" ".join([str(i)[:100] for i in row]) for row in sample_rows]
                    )
                    table_str += row_template.format(
                        n_rows=n_rows, sample_rows=sample_rows
                    )

            tables.append(table_str)
        return "\n".join(tables)

    def run(self, command: str) -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.exec_driver_sql(command)
            if cursor.returns_rows:
                result = cursor.fetchall()
                return str(result)
        return ""
