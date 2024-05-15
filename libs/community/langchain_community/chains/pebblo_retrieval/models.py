"""Models for the PebbloRetrievalQA chain."""

from typing import Any, List, Optional

from langchain_core.pydantic_v1 import BaseModel


class AuthContext(BaseModel):
    """Class for an authorization context."""

    name: Optional[str] = None
    user_id: str
    user_auth: List[str]
    """List of user authorizations, which may include their User ID and 
    the groups they are part of"""


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


class ChainInput(BaseModel):
    """Input for PebbloRetrievalQA chain."""

    query: str
    auth_context: Optional[AuthContext] = None
    semantic_context: Optional[SemanticContext] = None

    def dict(self, **kwargs: Any) -> dict:
        base_dict = super().dict(**kwargs)
        # Keep auth_context and semantic_context as it is(Pydantic models)
        base_dict["auth_context"] = self.auth_context
        base_dict["semantic_context"] = self.semantic_context
        return base_dict
