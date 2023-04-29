import json
import logging
from typing import List, Dict, Any

from pydantic import root_validator

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
    MessageLog
)

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_STRING = "postgresql://postgres:mypassword@localhost:5432/"

class PostgresChatMessageHistory(BaseChatMessageHistory):

    connection_string: str = DEFAULT_CONNECTION_STRING
    table_name: str = "message_store"
    connection: Any
    cursor: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        import psycopg
        from psycopg.rows import dict_row

        try:
            values["connection"] = psycopg.connect(values["connection_string"])
            values["cursor"] = values["connection"].cursor(row_factory=dict_row)
        except psycopg.OperationalError as error:
            logger.error(error)
        
        # Create table if not exists
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {values["table_name"]} (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            message_type TEXT NOT NULL,
            extra_variables JSONB 
        );"""
        values["cursor"].execute(create_table_query)
        values["connection"].commit()
        return values

    
    def save_message_log(self, message_log: MessageLog) -> None:
        """Append the message to the record in PostgreSQL"""
        from psycopg import sql

        extra_vars_json = json.dumps(message_log.extra_variables)

        # Create a SQL query using the psycopg2.sql module
        query = sql.SQL("""
            INSERT INTO {} (session_id, created_at, content, role, message_type, extra_variables)
            VALUES (%s, %s, %s, %s, %s, %s)
        """).format(
            sql.Identifier(self.table_name)
        )

        # Create a tuple of values to insert into the query
        values = (
            message_log.session_id,
            message_log.created_at,
            message_log.content,
            message_log.role,
            message_log.message_type,
            extra_vars_json
        )

        self.cursor.execute(
            query, values
        )
        self.connection.commit()

    def load_message_logs(self) -> List[MessageLog]:
        """Retrieve the messages from PostgreSQL"""
        query = f"SELECT * FROM {self.table_name} WHERE session_id = %s;"
        self.cursor.execute(query, (self.session_id,))
        records = []
        for i in self.cursor.fetchall():
            del i["id"]
            records.append(i)
        message_logs = [MessageLog(**i) for i in records]
        return message_logs

    def _clear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = f"DELETE FROM {self.table_name} WHERE session_id = %s;"
        self.cursor.execute(query, (self.session_id,))
        self.connection.commit()

    def __del__(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
