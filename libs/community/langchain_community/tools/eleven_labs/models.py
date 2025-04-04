from enum import Enum


class ElevenLabsModel(str, Enum):
    """Models available for Eleven Labs Text2Speech."""

    MULTI_LINGUAL = "eleven_multilingual_v2"
    MULTI_LINGUAL_FLASH = "eleven_flash_v2_5"
    MONO_LINGUAL = "eleven_flash_v2"
