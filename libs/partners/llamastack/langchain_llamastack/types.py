"""Data types for LlamaStack safety integration."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SafetyResult(BaseModel):
    """Result from safety check."""

    is_safe: bool
    violations: List[Dict[str, Any]] = []
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
