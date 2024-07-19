from enum import Enum


class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""

    MULTI_LINGUAL = "eleven_multilingual_v1"
    MONO_LINGUAL = "eleven_monolingual_v1"
