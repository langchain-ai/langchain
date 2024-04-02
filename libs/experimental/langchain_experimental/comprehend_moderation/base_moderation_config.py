from typing import List, Union

from pydantic import BaseModel


class ModerationPiiConfig(BaseModel):
    """Configuration for PII moderation filter."""

    threshold: float = 0.5
    """Threshold for PII confidence score, defaults to 0.5 i.e. 50%"""

    labels: List[str] = []
    """
    List of PII Universal Labels. 
    Defaults to `list[]`
    """

    redact: bool = False
    """Whether to perform redaction of detected PII entities"""

    mask_character: str = "*"
    """Redaction mask character in case redact=True, defaults to asterisk (*)"""


class ModerationToxicityConfig(BaseModel):
    """Configuration for Toxicity moderation filter."""

    threshold: float = 0.5
    """Threshold for Toxic label confidence score, defaults to 0.5 i.e. 50%"""

    labels: List[str] = []
    """List of toxic labels, defaults to `list[]`"""


class ModerationPromptSafetyConfig(BaseModel):
    """Configuration for Prompt Safety moderation filter."""

    threshold: float = 0.5
    """
    Threshold for Prompt Safety classification
    confidence score, defaults to 0.5 i.e. 50%
    """


class BaseModerationConfig(BaseModel):
    """Base configuration settings for moderation."""

    filters: List[
        Union[
            ModerationPiiConfig, ModerationToxicityConfig, ModerationPromptSafetyConfig
        ]
    ] = [
        ModerationPiiConfig(),
        ModerationToxicityConfig(),
        ModerationPromptSafetyConfig(),
    ]
    """
    Filters applied to the moderation chain, defaults to
    `[ModerationPiiConfig(), ModerationToxicityConfig(),
    ModerationPromptSafetyConfig()]`
    """
