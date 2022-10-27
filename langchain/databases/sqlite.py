from langchain.databases.base import Database
import sqlite3

class SQLiteDatabase(Database):


    def __init__(self, database_uri: str):
        self.conn = sqlite3.connect(database_uri)
    @property
    def dialect(self):
        return "SQLite"

    @property
    def table_info(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = cursor.fetchall()
        template = "The '{table_name}' table has columns: {columns}."
        tables = []
        for table in all_tables:
            table_name = table[0]
            columns = []
            for _, col_name, col_type, _, _, _ in self.conn.execute(f"pragma table_info('{table_name}')").fetchall():
                columns.append(f"{col_name} ({col_type})")
            tables.append(template.format(table_name=table_name, columns=", ".join(columns)))
        return "\n".join(tables)

    def run(self, command: str) -> str:
        cursor = self.conn.cursor()
        cursor.execute(command)
        res = cursor.fetchall()
        return str(res)