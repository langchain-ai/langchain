"""Models for the PebbloRetrievalQA chain."""
from typing import Any, List, Optional

from langchain_core.pydantic_v1 import BaseModel


class SemanticEntities(BaseModel):
    """Class for a semantic entity filter."""

    deny: List[str]


class SemanticTopics(BaseModel):
    """Class for a semantic topic filter."""

    deny: List[str]


class SemanticContext(BaseModel):
    """Class for a semantic context."""

    pebblo_semantic_entities: Optional[SemanticEntities] = None
    pebblo_semantic_topics: Optional[SemanticTopics] = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Validate semantic_context
        if (
            self.pebblo_semantic_entities is None
            and self.pebblo_semantic_topics is None
        ):
            raise ValueError(
                "semantic_context must contain 'pebblo_semantic_entities' or "
                "'pebblo_semantic_topics'"
            )


class AuthContext(BaseModel):
    """Class for an authorization context."""

    authorized_identities: List[str]
