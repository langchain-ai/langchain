"""Voice input for Ollama models toolkit"""

from langchain_community.tools.ollama_voice_input.tool import (
    SpeechToText,
    VoiceInputChain,
)

__all__ = ["SpeechToText", "VoiceInputChain"]
