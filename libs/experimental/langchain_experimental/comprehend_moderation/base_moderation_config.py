from typing import Any, List, Optional

from pydantic import BaseModel


class ModerationPiiConfig(BaseModel):
    """Threshold for PII confidence score, defaults to 0.5 i.e. 50%"""

    threshold: Optional[float] = 0.5
    """
    List of PII Universal Labels. 
    Defaults to `list[]`
    """
    labels: Optional[List[str]] = []
    """Whether to perform redaction of detected PII entities"""
    redact: Optional[bool] = False
    """Redaction mask character in case redact=True, defaults to asterisk (*)"""
    mask_character: Optional[str] = "*"


class ModerationToxicityConfig(BaseModel):
    """Threshold for Toxic label confidence score, defaults to 0.5 i.e. 50%"""

    threshold: Optional[float] = 0.5
    """List of toxic labels, defaults to `list[]`"""
    labels: Optional[List[str]] = []


class ModerationIntentConfig(BaseModel):
    """
    Threshold for Intent classification 
    confidence score, defaults to 0.5 i.e. 50%
    """
    threshold: Optional[float] = 0.5


class BaseModerationConfig(BaseModel):
    """
    Filters applied to the moderation chain, defaults to 
    `[ModerationPiiConfig(), ModerationToxicityConfig(), 
    ModerationIntentConfig()]`
    """

    filters: Optional[List[Any]] = [
        ModerationPiiConfig(),
        ModerationToxicityConfig(),
        ModerationIntentConfig(),
    ]
