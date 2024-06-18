from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field, root_validator, validator

from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class EdenAiTextToSpeechTool(EdenaiTool):
    """Tool that queries the Eden AI Text to speech API.
    for api reference check edenai documentation:
    https://docs.edenai.co/reference/audio_text_to_speech_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    name: str = "edenai_text_to_speech"
    description = (
        "A wrapper around edenai Services text to speech."
        "Useful for when you need to convert text to speech."
        """the output is a string representing the URL of the audio file,
        or the path to the downloaded wav file """
    )

    language: Optional[str] = "en"
    """
    language of the text passed to the model.
    """

    # optional params see api documentation for more info
    return_type: Literal["url", "wav"] = "url"
    rate: Optional[int]
    pitch: Optional[int]
    volume: Optional[int]
    audio_format: Optional[str]
    sampling_rate: Optional[int]
    voice_models: Dict[str, str] = Field(default_factory=dict)

    voice: Literal["MALE", "FEMALE"]
    """voice option : 'MALE' or 'FEMALE' """

    feature: str = "audio"
    subfeature: str = "text_to_speech"

    @validator("providers")
    def check_only_one_provider_selected(cls, v: List[str]) -> List[str]:
        """
        This tool has no feature to combine providers results.
        Therefore we only allow one provider
        """
        if len(v) > 1:
            raise ValueError(
                "Please select only one provider. "
                "The feature to combine providers results is not available "
                "for this tool."
            )
        return v

    @root_validator
    def check_voice_models_key_is_provider_name(cls, values: dict) -> dict:
        for key in values.get("voice_models", {}).keys():
            if key not in values.get("providers", []):
                raise ValueError(
                    "voice_model should be formatted like this "
                    "{<provider_name>: <its_voice_model>}"
                )
        return values

    def _download_wav(self, url: str, save_path: str) -> None:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            raise ValueError("Error while downloading wav file")

    def _parse_response(self, response: list) -> str:
        result = response[0]
        if self.return_type == "url":
            return result["audio_resource_url"]
        else:
            self._download_wav(result["audio_resource_url"], "audio.wav")
            return "audio.wav"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        all_params = {
            "text": query,
            "language": self.language,
            "option": self.voice,
            "return_type": self.return_type,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "audio_format": self.audio_format,
            "sampling_rate": self.sampling_rate,
            "settings": self.voice_models,
        }

        # filter so we don't send val to api when val is `None
        query_params = {k: v for k, v in all_params.items() if v is not None}

        return self._call_eden_ai(query_params)
