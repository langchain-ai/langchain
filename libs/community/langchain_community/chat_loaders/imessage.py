from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

from langchain_core.chat_loaders import BaseChatLoader
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    import sqlite3


def nanoseconds_from_2001_to_datetime(nanoseconds: int) -> datetime:
    """Convert nanoseconds since 2001 to a datetime object.

    Args:
        nanoseconds (int): Nanoseconds since January 1, 2001.

    Returns:
        datetime: Datetime object.
    """
    # Convert nanoseconds to seconds (1 second = 1e9 nanoseconds)
    timestamp_in_seconds = nanoseconds / 1e9

    # The reference date is January 1, 2001, in Unix time
    reference_date_seconds = datetime(2001, 1, 1).timestamp()

    # Calculate the actual timestamp by adding the reference date
    actual_timestamp = reference_date_seconds + timestamp_in_seconds

    # Convert to a datetime object
    return datetime.fromtimestamp(actual_timestamp)


class IMessageChatLoader(BaseChatLoader):
    """Load chat sessions from the `iMessage` chat.db SQLite file.

    It only works on macOS when you have iMessage enabled and have the chat.db file.

    The chat.db file is likely located at ~/Library/Messages/chat.db. However, your
    terminal may not have permission to access this file. To resolve this, you can
    copy the file to a different location, change the permissions of the file, or
    grant full disk access for your terminal emulator
    in System Settings > Security and Privacy > Full Disk Access.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None):
        """
        Initialize the IMessageChatLoader.

        Args:
            path (str or Path, optional): Path to the chat.db SQLite file.
                Defaults to None, in which case the default path
                ~/Library/Messages/chat.db will be used.
        """
        if path is None:
            path = Path.home() / "Library" / "Messages" / "chat.db"
        self.db_path = path if isinstance(path, Path) else Path(path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"File {self.db_path} not found")
        try:
            import sqlite3  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The sqlite3 module is required to load iMessage chats.\n"
                "Please install it with `pip install pysqlite3`"
            ) from e

    @staticmethod
    def _parse_attributed_body(attributed_body: bytes) -> str:
        """
        Parse the attributedBody field of the message table
        for the text content of the message.

        The attributedBody field is a binary blob that contains
        the message content after the byte string b"NSString":

                              5 bytes      1-3 bytes    `len` bytes
        ... | b"NSString" |   preamble   |   `len`   |    contents    | ...

        The 5 preamble bytes are always b"\x01\x94\x84\x01+"

        The size of `len` is either 1 byte or 3 bytes:
        - If the first byte in `len` is b"\x81" then `len` is 3 bytes long.
          So the message length is the 2 bytes after, in little Endian.
        - Otherwise, the size of `len` is 1 byte, and the message length is
          that byte.

        Args:
            attributed_body (bytes): attributedBody field of the message table.
        Return:
            str: Text content of the message.
        """
        content = attributed_body.split(b"NSString")[1][5:]
        length, start = content[0], 1
        if content[0] == 129:
            length, start = int.from_bytes(content[1:3], "little"), 3
        return content[start : start + length].decode("utf-8", errors="ignore")

    @staticmethod
    def _get_session_query(use_chat_handle_table: bool) -> str:
        # Messages sent pre OSX 12 require a join through the chat_handle_join table
        # However, the table doesn't exist if database created with OSX 12 or above.

        joins_w_chat_handle = """
            JOIN chat_handle_join ON
                 chat_message_join.chat_id = chat_handle_join.chat_id
            JOIN handle ON
                 handle.ROWID = chat_handle_join.handle_id"""

        joins_no_chat_handle = """
            JOIN handle ON message.handle_id = handle.ROWID
        """

        joins = joins_w_chat_handle if use_chat_handle_table else joins_no_chat_handle

        return f"""
            SELECT  message.date,
                    handle.id,
                    message.text,
                    message.is_from_me,
                    message.attributedBody
            FROM message
            JOIN chat_message_join ON
                 message.ROWID = chat_message_join.message_id
            {joins}
            WHERE chat_message_join.chat_id = ?
            ORDER BY message.date ASC;
        """

    def _load_single_chat_session(
        self, cursor: "sqlite3.Cursor", use_chat_handle_table: bool, chat_id: int
    ) -> ChatSession:
        """
        Load a single chat session from the iMessage chat.db.

        Args:
            cursor: SQLite cursor object.
            chat_id (int): ID of the chat session to load.

        Returns:
            ChatSession: Loaded chat session.
        """
        results: List[HumanMessage] = []

        query = self._get_session_query(use_chat_handle_table)
        cursor.execute(query, (chat_id,))
        messages = cursor.fetchall()

        for date, sender, text, is_from_me, attributedBody in messages:
            if text:
                content = text
            elif attributedBody:
                content = self._parse_attributed_body(attributedBody)
            else:  # Skip messages with no content
                continue

            results.append(
                HumanMessage(
                    role=sender,
                    content=content,
                    additional_kwargs={
                        "message_time": date,
                        "message_time_as_datetime": nanoseconds_from_2001_to_datetime(
                            date
                        ),
                        "sender": sender,
                        "is_from_me": bool(is_from_me),
                    },
                )
            )

        return ChatSession(messages=results)

    def lazy_load(self) -> Iterator[ChatSession]:
        """
        Lazy load the chat sessions from the iMessage chat.db
        and yield them in the required format.

        Yields:
            ChatSession: Loaded chat session.
        """
        import sqlite3

        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.OperationalError as e:
            raise ValueError(
                f"Could not open iMessage DB file {self.db_path}.\n"
                "Make sure your terminal emulator has disk access to this file.\n"
                "   You can either copy the DB file to an accessible location"
                " or grant full disk access for your terminal emulator."
                "  You can grant full disk access for your terminal emulator"
                " in System Settings > Security and Privacy > Full Disk Access."
            ) from e
        cursor = conn.cursor()

        # See if chat_handle_join table exists:
        query = """SELECT name FROM sqlite_master
                   WHERE type='table' AND name='chat_handle_join';"""

        cursor.execute(query)
        is_chat_handle_join_exists = cursor.fetchone()

        # Fetch the list of chat IDs sorted by time (most recent first)
        query = """SELECT chat_id
        FROM message
        JOIN chat_message_join ON message.ROWID = chat_message_join.message_id
        GROUP BY chat_id
        ORDER BY MAX(date) DESC;"""
        cursor.execute(query)
        chat_ids = [row[0] for row in cursor.fetchall()]

        for chat_id in chat_ids:
            yield self._load_single_chat_session(
                cursor, is_chat_handle_join_exists, chat_id
            )

        conn.close()
