"""SQLAlchemy wrapper around a database."""
from sqlalchemy import create_engine, inspect


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""
    def __init__(self, database_uri: str):
        self._engine = create_engine(database_uri)

    @property
    def dialect(self):
        """String representation of dialect to use."""
        return self._engine.dialect.name

    @property
    def table_info(self):
        """Information about all tables in the database."""
        template = "The '{table_name}' table has columns: {columns}."
        tables = []
        inspector = inspect(self._engine)
        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append(f"{column['name']} ({str(column['type'])})")
            tables.append(
                template.format(table_name=table_name, columns=", ".join(columns))
            )
        return "\n".join(tables)

    def run(self, command: str) -> str:
        """Execute a SQL command and return a string of the results."""
        result = self._engine.execute(command).fetchall()
        return str(result)
