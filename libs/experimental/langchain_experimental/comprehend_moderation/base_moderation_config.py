from typing import List, Union

from pydantic import BaseModel


class ModerationPiiConfig(BaseModel):
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
    threshold: float = 0.5
    """Threshold for Toxic label confidence score, defaults to 0.5 i.e. 50%"""

    labels: List[str] = []
    """List of toxic labels, defaults to `list[]`"""


class ModerationPromptSafetyConfig(BaseModel):
    threshold: float = 0.5
    """
    Threshold for Prompt Safety classification
    confidence score, defaults to 0.5 i.e. 50%
    """


class BaseModerationConfig(BaseModel):
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
