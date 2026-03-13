"""MiniMax model profiles.

This file contains model profile data for MiniMax models.

To update these data, refer to the instructions here:

https://docs.langchain.com/oss/python/langchain/models#updating-or-overwriting-profile-data
"""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "MiniMax-M2.5": {
        "name": "MiniMax M2.5",
        "max_input_tokens": 204800,
        "max_output_tokens": 131072,
        "text_inputs": True,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "structured_output": True,
        "temperature": True,
    },
    "MiniMax-M2.5-highspeed": {
        "name": "MiniMax M2.5 High Speed",
        "max_input_tokens": 204800,
        "max_output_tokens": 131072,
        "text_inputs": True,
        "image_inputs": False,
        "audio_inputs": False,
        "video_inputs": False,
        "text_outputs": True,
        "image_outputs": False,
        "audio_outputs": False,
        "video_outputs": False,
        "reasoning_output": True,
        "tool_calling": True,
        "structured_output": True,
        "temperature": True,
    },
}
