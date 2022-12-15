"""Base interface for cache."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from sqlalchemy import Column, String

from sqlalchemy.orm import Session
from sqlalchemy import select, create_engine, insert
from sqlalchemy.ext.declarative import declarative_base

class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        """Update cache based on prompt and llm_string."""


class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: Dict[Tuple[str, str], str] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[(prompt, llm_string)] = return_val



Base = declarative_base()

class LLMCache(Base):
    __tablename__ = 'llm_cache'
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    response = Column(String)


class SQLiteCache(BaseCache):

    def __init__(self, db_name=".langchain.db"):
        self.engine = create_engine(f"sqlite:///{db_name}")
        Base.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        stmt = select(LLMCache.response).where(LLMCache.prompt==prompt).where(LLMCache.llm == llm_string)
        with Session(self.engine) as session:
            for row in session.execute(stmt):
                return row[0]

    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        item = LLMCache(prompt=prompt, llm=llm_string, response=return_val)
        with Session(self.engine) as session, session.begin():
            session.add(item)
