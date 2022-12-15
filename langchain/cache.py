"""Beta Feature: base interface for cache."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from sqlalchemy import Column, String, create_engine, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session


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


class LLMCache(Base):  # type: ignore
    """Table definition for LLM cache."""

    __tablename__ = "llm_cache"
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    response = Column(String)


class SQLiteCache(BaseCache):
    """Implementation of a cache that is backed by SQLite."""

    def __init__(self, database_path: str = ".langchain.db"):
        """Initialize with database path."""
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[str]:
        """Look up based on prompt and llm_string."""
        stmt = (
            select(LLMCache.response)
            .where(LLMCache.prompt == prompt)
            .where(LLMCache.llm == llm_string)
        )
        with Session(self.engine) as session:
            for row in session.execute(stmt):
                return row[0]
        return None

    def update(self, prompt: str, llm_string: str, return_val: str) -> None:
        """Update cache based on prompt and llm_string."""
        item = LLMCache(prompt=prompt, llm=llm_string, response=return_val)
        with Session(self.engine) as session, session.begin():
            session.add(item)
