from __future__ import annotations

import asyncio
from asyncio import InvalidStateError, Task
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from langchain_core.stores import ByteStore

from langchain_community.utilities.cassandra import SetupMode, aexecute_cql

if TYPE_CHECKING:
    from cassandra.cluster import Session
    from cassandra.query import PreparedStatement

CREATE_TABLE_CQL_TEMPLATE = """
    CREATE TABLE IF NOT EXISTS {keyspace}.{table} 
    (row_id TEXT, body_blob BLOB, PRIMARY KEY (row_id));
"""
SELECT_TABLE_CQL_TEMPLATE = (
    """SELECT row_id, body_blob FROM  {keyspace}.{table} WHERE row_id IN ?;"""
)
SELECT_ALL_TABLE_CQL_TEMPLATE = """SELECT row_id, body_blob FROM  {keyspace}.{table};"""
INSERT_TABLE_CQL_TEMPLATE = (
    """INSERT INTO {keyspace}.{table} (row_id, body_blob) VALUES (?, ?);"""
)
DELETE_TABLE_CQL_TEMPLATE = """DELETE FROM {keyspace}.{table} WHERE row_id IN ?;"""


class CassandraByteStore(ByteStore):
    def __init__(
        self,
        table: str,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
    ) -> None:
        if not session or not keyspace:
            try:
                from cassio.config import check_resolve_keyspace, check_resolve_session

                self.keyspace = keyspace or check_resolve_keyspace(keyspace)
                self.session = session or check_resolve_session()
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    "Could not import a recent cassio package."
                    "Please install it with `pip install --upgrade cassio`."
                )
        else:
            self.keyspace = keyspace
            self.session = session
        self.table = table
        self.select_statement = None
        self.insert_statement = None
        self.delete_statement = None

        create_cql = CREATE_TABLE_CQL_TEMPLATE.format(
            keyspace=self.keyspace,
            table=self.table,
        )
        self.db_setup_task: Optional[Task[None]] = None
        if setup_mode == SetupMode.ASYNC:
            self.db_setup_task = asyncio.create_task(
                aexecute_cql(self.session, create_cql)
            )
        else:
            self.session.execute(create_cql)

    def ensure_db_setup(self) -> None:
        if self.db_setup_task:
            try:
                self.db_setup_task.result()
            except InvalidStateError:
                raise ValueError(
                    "Asynchronous setup of the DB not finished. "
                    "NB: AstraDB components sync methods shouldn't be called from the "
                    "event loop. Consider using their async equivalents."
                )

    async def aensure_db_setup(self) -> None:
        if self.db_setup_task:
            await self.db_setup_task

    def get_select_statement(self) -> PreparedStatement:
        if not self.select_statement:
            self.select_statement = self.session.prepare(
                SELECT_TABLE_CQL_TEMPLATE.format(
                    keyspace=self.keyspace, table=self.table
                )
            )
        return self.select_statement

    def get_insert_statement(self) -> PreparedStatement:
        if not self.insert_statement:
            self.insert_statement = self.session.prepare(
                INSERT_TABLE_CQL_TEMPLATE.format(
                    keyspace=self.keyspace, table=self.table
                )
            )
        return self.insert_statement

    def get_delete_statement(self) -> PreparedStatement:
        if not self.delete_statement:
            self.delete_statement = self.session.prepare(
                DELETE_TABLE_CQL_TEMPLATE.format(
                    keyspace=self.keyspace, table=self.table
                )
            )
        return self.delete_statement

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        from cassandra.query import ValueSequence

        self.ensure_db_setup()
        docs_dict = {}
        for row in self.session.execute(
            self.get_select_statement(), [ValueSequence(keys)]
        ):
            docs_dict[row.row_id] = row.body_blob
        return [docs_dict.get(key) for key in keys]

    async def amget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        from cassandra.query import ValueSequence

        await self.aensure_db_setup()
        docs_dict = {}
        for row in await aexecute_cql(
            self.session, self.get_select_statement(), parameters=[ValueSequence(keys)]
        ):
            docs_dict[row.row_id] = row.body_blob
        return [docs_dict.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        self.ensure_db_setup()
        insert_statement = self.get_insert_statement()
        for k, v in key_value_pairs:
            self.session.execute(insert_statement, (k, v))

    async def amset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        await self.aensure_db_setup()
        insert_statement = self.get_insert_statement()
        for k, v in key_value_pairs:
            await aexecute_cql(self.session, insert_statement, parameters=(k, v))

    def mdelete(self, keys: Sequence[str]) -> None:
        from cassandra.query import ValueSequence

        self.ensure_db_setup()
        self.session.execute(self.get_delete_statement(), [ValueSequence(keys)])

    async def amdelete(self, keys: Sequence[str]) -> None:
        from cassandra.query import ValueSequence

        await self.aensure_db_setup()
        await aexecute_cql(
            self.session, self.get_delete_statement(), parameters=[ValueSequence(keys)]
        )

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        self.ensure_db_setup()
        for row in self.session.execute(
            SELECT_ALL_TABLE_CQL_TEMPLATE.format(
                keyspace=self.keyspace, table=self.table
            )
        ):
            key = row.row_id
            if not prefix or key.startswith(prefix):
                yield key

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        await self.aensure_db_setup()
        for row in await aexecute_cql(
            self.session,
            SELECT_ALL_TABLE_CQL_TEMPLATE.format(
                keyspace=self.keyspace, table=self.table
            ),
        ):
            key = row.row_id
            if not prefix or key.startswith(prefix):
                yield key
