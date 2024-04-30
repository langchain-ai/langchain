import os
from contextlib import contextmanager, asynccontextmanager

import aioodbc
import pyodbc
from typing_extensions import AsyncGenerator, Generator

MSSQL_USERNAME = os.environ["TEST_MSSQL_USERNAME"]
MSSQL_PASSWORD = os.environ["TEST_MSSQL_PASSWORD"]
MSSQL_SERVER = os.environ["TEST_MSSQL_SERVER"]
MSSQL_DATABASE = os.environ["TEST_MSSQL_DATABASE"]


DSN = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={MSSQL_SERVER};DATABASE={MSSQL_DATABASE};UID={MSSQL_USERNAME};PWD={MSSQL_PASSWORD};Encrypt=no;TrustServerCertificate=yes;"


@asynccontextmanager
async def asyncms_client() -> AsyncGenerator[aioodbc.Connection, None]:
    conn = await aioodbc.connect(dsn=DSN)
    try:
        yield conn
    finally:
        await conn.close()


@contextmanager
def syncms_client() -> Generator[pyodbc.Connection, None, None]:
    conn = pyodbc.connect(DSN)
    try:
        yield conn
    finally:
        conn.close()
