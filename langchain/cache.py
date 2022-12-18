"""Beta Feature: base interface for cache."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from langchain.schema import Generation

RETURN_VAL_TYPE = List[Generation]


class BaseCache(ABC):
    """Base interface for cache."""

    @abstractmethod
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""

    @abstractmethod
    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""


class InMemoryCache(BaseCache):
    """Cache that stores things in memory."""

    def __init__(self) -> None:
        """Initialize with empty cache."""
        self._cache: Dict[Tuple[str, str], RETURN_VAL_TYPE] = {}

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self._cache.get((prompt, llm_string), None)

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        self._cache[(prompt, llm_string)] = return_val


Base = declarative_base()


class FullLLMCache(Base):  # type: ignore
    """SQLite table for full LLM Cache (all generations)."""

    __tablename__ = "full_llm_cache"
    prompt = Column(String, primary_key=True)
    llm = Column(String, primary_key=True)
    idx = Column(Integer, primary_key=True)
    response = Column(String)


class SQLAlchemyCache(BaseCache):
    """Cache that uses SQAlchemy as a backend."""

    def __init__(self, engine: Engine):
        """Initialize by creating all tables."""
        self.engine = engine
        Base.metadata.create_all(self.engine)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        stmt = (
            select(FullLLMCache.response)
            .where(FullLLMCache.prompt == prompt)
            .where(FullLLMCache.llm == llm_string)
            .order_by(FullLLMCache.idx)
        )
        with Session(self.engine) as session:
            generations = []
            for row in session.execute(stmt):
                generations.append(Generation(text=row[0]))
            if len(generations) > 0:
                return generations
        return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Look up based on prompt and llm_string."""
        for i, generation in enumerate(return_val):
            item = FullLLMCache(
                prompt=prompt, llm=llm_string, response=generation.text, idx=i
            )
            with Session(self.engine) as session, session.begin():
                session.add(item)


class SQLiteCache(SQLAlchemyCache):
    """Cache that uses SQLite as a backend."""

    def __init__(self, database_path: str = ".langchain.db"):
        """Initialize by creating the engine and all tables."""
        engine = create_engine(f"sqlite:///{database_path}")
        super().__init__(engine)
