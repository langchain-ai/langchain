# langchain-postgres

The `langchain-postgres` package is an integration package managed by the core LangChain team.

This package contains implementations of core abstractions using `Postgres`.

The package is released under the MIT license. 

Feel free to use the abstraction as provided or else modify them / extend them as appropriate for your own application.

## Installation

```bash
pip install -U langchain-postgres
```

## Usage

### ChatMessageHistory

The chat message history abstraction helps to persist chat message history 
in a postgres table.

PostgresChatMessageHistory is parameterized using a `table_name` and a `session_id`.

The `table_name` is the name of the table in the database where 
the chat messages will be stored.

The `session_id` is a unique identifier for the chat session. It can be assigned
by the caller using `uuid.uuid4()`.

```python
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

# Establish a synchronous connection to the database
# (or use psycopg.AsyncConnection for async)
conn_info = ... # Fill in with your connection info
sync_connection = psycopg.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_schema(sync_connection, table_name)

session_id = str(uuid.uuid4())

# Initialize the chat history manager
chat_history = PostgresChatMessageHistory(
    table_name,
    session_id,
    sync_connection=sync_connection
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
```


### PostgresCheckpoint

An implementation of the `Checkpoint` abstraction in LangGraph using Postgres.


Async Usage: 

```python
from psycopg_pool import AsyncConnectionPool
from langchain_postgres import (
    PostgresCheckpoint, PickleCheckpointSerializer
)

pool = AsyncConnectionPool(
    # Example configuration
    conninfo="postgresql://user:password@localhost:5432/dbname",
    max_size=20,
)

# Uses the pickle module for serialization
# Make sure that you're only de-serializing trusted data
# (e.g., payloads that you have serialized yourself).
# Or implement a custom serializer.
checkpoint = PostgresCheckpoint(
    serializer=PickleCheckpointSerializer(),
    async_connection=pool,
)

# Use the checkpoint object to put, get, list checkpoints, etc.
```

Sync Usage:

```python
from psycopg_pool import ConnectionPool
from langchain_postgres import (
    PostgresCheckpoint, PickleCheckpointSerializer
)

pool = ConnectionPool(
    # Example configuration
    conninfo="postgresql://user:password@localhost:5432/dbname",
    max_size=20,
)

# Uses the pickle module for serialization
# Make sure that you're only de-serializing trusted data
# (e.g., payloads that you have serialized yourself).
# Or implement a custom serializer.
checkpoint = PostgresCheckpoint(
    serializer=PickleCheckpointSerializer(),
    sync_connection=pool,
)

# Use the checkpoint object to put, get, list checkpoints, etc.
```
