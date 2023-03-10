"""Wrappers on top of text to audio model APIs."""

from langchain.audio_models.bananadev import AudioBanana
from langchain.audio_models.whisper import Whisper

__all__ = [
    "AudioBanana",
    "Whisper",
]
