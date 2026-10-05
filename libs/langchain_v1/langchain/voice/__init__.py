"""Create duplex voice agents backed by LangGraph.

!!! warning

    `langchain.voice` is experimental and its API may change.
"""

from langchain.voice.agent import VoiceAgent, create_voice_agent
from langchain.voice.audio import AudioInput, AudioOutput, NullUI, StatusUI
from langchain.voice.conversation import ConversationLayer
from langchain.voice.providers.gemini_live import GeminiLiveConversationLayer
from langchain.voice.providers.openai_realtime import OpenAIRealtimeConversationLayer
from langchain.voice.transport import (
    AudioFormat,
    AudioFrame,
    AudioIOTransport,
    AudioPlayed,
    AudioReceived,
    LocalAudioTransport,
    PlaybackReceipt,
    TransportDisconnected,
    UserSpeechEnded,
    UserSpeechStarted,
    VoiceTransport,
)
from langchain.voice.transports.livekit import LiveKitAudioTransport

__all__ = [
    "AudioFormat",
    "AudioFrame",
    "AudioIOTransport",
    "AudioInput",
    "AudioOutput",
    "AudioPlayed",
    "AudioReceived",
    "ConversationLayer",
    "GeminiLiveConversationLayer",
    "LiveKitAudioTransport",
    "LocalAudioTransport",
    "NullUI",
    "OpenAIRealtimeConversationLayer",
    "PlaybackReceipt",
    "StatusUI",
    "TransportDisconnected",
    "UserSpeechEnded",
    "UserSpeechStarted",
    "VoiceAgent",
    "VoiceTransport",
    "create_voice_agent",
]
