# langchain-redis

This package contains the LangChain integration with Redis.

## Installation

```bash
pip install langchain-redis
```

And you should configure credentials by setting the following environment variable:

- `REDIS_URL`: URL for the Redis instance

## Chat Message History

You can use `RedisChatMessageHistory` to store chat message history in a Redis instance:

```python
from langchain_redis import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="session_123",
    redis_url="redis://localhost:6379"
)

history.add_user_message("Hello, AI assistant!")
history.add_ai_message("Hello! How can I assist you today?")

# Get messages
messages = history.messages
```

### Using with key_prefix

You can also use a key prefix to organize your sessions:

```python
from langchain_redis import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="session_123",
    redis_url="redis://localhost:6379",
    key_prefix="chat_app:"
)

history.add_user_message("Hello, AI assistant!")
history.add_ai_message("Hello! How can I assist you today?")

# Messages will be stored under key: "chat_app:session_123"
messages = history.messages
```