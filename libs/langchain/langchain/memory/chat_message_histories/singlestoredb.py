import json
import logging
import re
from typing import (
    Any,
    List,
)

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)


class SingleStoreDBChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a SingleStoreDB database."""

    def __init__(
        self,
        session_id: str,
        *,
        table_name: str = "message_store",
        id_field: str = "id",
        session_id_field: str = "session_id",
        message_field: str = "message",
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: float = 30,
        **kwargs: Any,
    ):
        """Initialize with necessary components.

        Args:


            table_name (str, optional): Specifies the name of the table in use.
                Defaults to "message_store".
            id_field (str, optional): Specifies the name of the id field in the table.
                Defaults to "id".
            session_id_field (str, optional): Specifies the name of the session_id
                field in the table. Defaults to "session_id".
            message_field (str, optional): Specifies the name of the message field
                in the table. Defaults to "message".

            Following arguments pertain to the connection pool:

            pool_size (int, optional): Determines the number of active connections in
                the pool. Defaults to 5.
            max_overflow (int, optional): Determines the maximum number of connections
                allowed beyond the pool_size. Defaults to 10.
            timeout (float, optional): Specifies the maximum wait time in seconds for
                establishing a connection. Defaults to 30.

            Following arguments pertain to the database connection:

            host (str, optional): Specifies the hostname, IP address, or URL for the
                database connection. The default scheme is "mysql".
            user (str, optional): Database username.
            password (str, optional): Database password.
            port (int, optional): Database port. Defaults to 3306 for non-HTTP
                connections, 80 for HTTP connections, and 443 for HTTPS connections.
            database (str, optional): Database name.

            Additional optional arguments provide further customization over the
            database connection:

            pure_python (bool, optional): Toggles the connector mode. If True,
                operates in pure Python mode.
            local_infile (bool, optional): Allows local file uploads.
            charset (str, optional): Specifies the character set for string values.
            ssl_key (str, optional): Specifies the path of the file containing the SSL
                key.
            ssl_cert (str, optional): Specifies the path of the file containing the SSL
                certificate.
            ssl_ca (str, optional): Specifies the path of the file containing the SSL
                certificate authority.
            ssl_cipher (str, optional): Sets the SSL cipher list.
            ssl_disabled (bool, optional): Disables SSL usage.
            ssl_verify_cert (bool, optional): Verifies the server's certificate.
                Automatically enabled if ``ssl_ca`` is specified.
            ssl_verify_identity (bool, optional): Verifies the server's identity.
            conv (dict[int, Callable], optional): A dictionary of data conversion
                functions.
            credential_type (str, optional): Specifies the type of authentication to
                use: auth.PASSWORD, auth.JWT, or auth.BROWSER_SSO.
            autocommit (bool, optional): Enables autocommits.
            results_type (str, optional): Determines the structure of the query results:
                tuples, namedtuples, dicts.
            results_format (str, optional): Deprecated. This option has been renamed to
                results_type.

        Examples:
            Basic Usage:

            .. code-block:: python

                from langchain.memory.chat_message_histories import (
                    SingleStoreDBChatMessageHistory
                )

                message_history = SingleStoreDBChatMessageHistory(
                    session_id="my-session",
                    host="https://user:password@127.0.0.1:3306/database"
                )

            Advanced Usage:

            .. code-block:: python

                from langchain.memory.chat_message_histories import (
                    SingleStoreDBChatMessageHistory
                )

                message_history = SingleStoreDBChatMessageHistory(
                    session_id="my-session",
                    host="127.0.0.1",
                    port=3306,
                    user="user",
                    password="password",
                    database="db",
                    table_name="my_custom_table",
                    pool_size=10,
                    timeout=60,
                )

            Using environment variables:

            .. code-block:: python

                from langchain.memory.chat_message_histories import (
                    SingleStoreDBChatMessageHistory
                )

                os.environ['SINGLESTOREDB_URL'] = 'me:p455w0rd@s2-host.com/my_db'
                message_history = SingleStoreDBChatMessageHistory("my-session")
        """

        self.table_name = self._sanitize_input(table_name)
        self.session_id = self._sanitize_input(session_id)
        self.id_field = self._sanitize_input(id_field)
        self.session_id_field = self._sanitize_input(session_id_field)
        self.message_field = self._sanitize_input(message_field)

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

        # Add connection attributes to the connection kwargs.
        if "conn_attrs" not in self.connection_kwargs:
            self.connection_kwargs["conn_attrs"] = dict()

        self.connection_kwargs["conn_attrs"]["_connector_name"] = "langchain python sdk"
        self.connection_kwargs["conn_attrs"]["_connector_version"] = "1.0.1"

        # Create a connection pool.
        try:
            from sqlalchemy.pool import QueuePool
        except ImportError:
            raise ImportError(
                "Could not import sqlalchemy.pool python package. "
                "Please install it with `pip install singlestoredb`."
            )

        self.connection_pool = QueuePool(
            self._get_connection,
            max_overflow=max_overflow,
            pool_size=pool_size,
            timeout=timeout,
        )
        self.table_created = False

    def _sanitize_input(self, input_str: str) -> str:
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    def _get_connection(self) -> Any:
        try:
            import singlestoredb as s2
        except ImportError:
            raise ImportError(
                "Could not import singlestoredb python package. "
                "Please install it with `pip install singlestoredb`."
            )
        return s2.connect(**self.connection_kwargs)

    def _create_table_if_not_exists(self) -> None:
        """Create table if it doesn't exist."""
        if self.table_created:
            return
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS {}
                    ({} BIGINT PRIMARY KEY AUTO_INCREMENT,
                    {} TEXT NOT NULL,
                    {} JSON NOT NULL);""".format(
                        self.table_name,
                        self.id_field,
                        self.session_id_field,
                        self.message_field,
                    ),
                )
                self.table_created = True
            finally:
                cur.close()
        finally:
            conn.close()

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from SingleStoreDB"""
        self._create_table_if_not_exists()
        conn = self.connection_pool.connect()
        items = []
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """SELECT {} FROM {} WHERE {} = %s""".format(
                        self.message_field,
                        self.table_name,
                        self.session_id_field,
                    ),
                    (self.session_id),
                )
                for row in cur.fetchall():
                    items.append(row[0])
            finally:
                cur.close()
        finally:
            conn.close()
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in SingleStoreDB"""
        self._create_table_if_not_exists()
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """INSERT INTO {} ({}, {}) VALUES (%s, %s)""".format(
                        self.table_name,
                        self.session_id_field,
                        self.message_field,
                    ),
                    (self.session_id, json.dumps(_message_to_dict(message))),
                )
            finally:
                cur.close()
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear session memory from SingleStoreDB"""
        self._create_table_if_not_exists()
        conn = self.connection_pool.connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    """DELETE FROM {} WHERE {} = %s""".format(
                        self.table_name,
                        self.session_id_field,
                    ),
                    (self.session_id),
                )
            finally:
                cur.close()
        finally:
            conn.close()
