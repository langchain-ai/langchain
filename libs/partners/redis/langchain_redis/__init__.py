"""This is the langchain_redis package.

It contains Redis integrations for LangChain.
"""

from langchain_redis.chat_message_histories import RedisChatMessageHistory

__all__ = [
    "RedisChatMessageHistory",
]