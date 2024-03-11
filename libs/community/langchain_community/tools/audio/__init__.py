from langchain_community.tools.audio.base import AudioTool
from langchain_community.tools.audio.huggingface_text_to_speech_inference import (
    HuggingFaceSupportedAudioFormat,
    HuggingFaceTextToSpeechModelInference,
)

__all__ = [
    "AudioTool",
    "HuggingFaceTextToSpeechModelInference",
    "HuggingFaceSupportedAudioFormat",
]
