"""SQLAlchemy wrapper around a database."""
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""

    def __init__(self, engine: Engine):
        """Create engine from database URI."""
        self._engine = engine

    @classmethod
    def from_uri(cls, database_uri: str) -> "SQLDatabase":
        """Construct a SQLAlchemy engine from URI."""
        return cls(create_engine(database_uri))

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        template = "The '{table_name}' table has columns: {columns}."
        tables = []
        inspector = inspect(self._engine)
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append(f"{column['name']} ({str(column['type'])})")
            column_str = ", ".join(columns)
            table_str = template.format(table_name=table_name, columns=column_str)
            tables.append(table_str)
        return "\n".join(tables)

    def run(self, command: str) -> str:
        """Execute a SQL command and return a string of the results."""
        result = self._engine.execute(command).fetchall()
        return str(result)
