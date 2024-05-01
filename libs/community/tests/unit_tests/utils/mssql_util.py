import os
from contextlib import asynccontextmanager, contextmanager

import aioodbc
import pyodbc
from typing_extensions import AsyncGenerator, Generator

MSSQL_USERNAME = os.environ["TEST_MSSQL_USERNAME"]
MSSQL_PASSWORD = os.environ["TEST_MSSQL_PASSWORD"]
MSSQL_SERVER = os.environ["TEST_MSSQL_SERVER"]
MSSQL_DATABASE = os.environ["TEST_MSSQL_DATABASE"]


DSN = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={MSSQL_SERVER};"
    f"DATABASE={MSSQL_DATABASE};"
    f"UID={MSSQL_USERNAME};"
    f"PWD={MSSQL_PASSWORD};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
)


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
