from enum import Enum


class BaseModerationActions(Enum):
    STOP = 1
    ALLOW = 2


class BaseModerationFilters(str, Enum):
    PII = "pii"
    TOXICITY = "toxicity"
    INTENT = "intent"
