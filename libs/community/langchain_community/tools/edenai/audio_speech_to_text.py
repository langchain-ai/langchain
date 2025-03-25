from __future__ import annotations

import json
import logging
import time
from typing import List, Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, HttpUrl, validator

from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class SpeechToTextInput(BaseModel):
    query: HttpUrl = Field(description="url of the audio to analyze")


class EdenAiSpeechToTextTool(EdenaiTool):  # type: ignore[override, override, override]
    """Tool that queries the Eden AI Speech To Text API.

    for api reference check edenai documentation:
    https://app.edenai.run/bricks/speech/asynchronous-speech-to-text.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings
    """

    name: str = "edenai_speech_to_text"
    description: str = (
        "A wrapper around edenai Services speech to text "
        "Useful for when you have to convert audio to text."
        "Input should be a url to an audio file."
    )
    args_schema: Type[BaseModel] = SpeechToTextInput
    is_async: bool = True

    language: Optional[str] = "en"
    speakers: Optional[int]
    profanity_filter: bool = False
    custom_vocabulary: Optional[List[str]]

    feature: str = "audio"
    subfeature: str = "speech_to_text_async"
    base_url: str = "https://api.edenai.run/v2/audio/speech_to_text_async/"

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

    def _wait_processing(self, url: str) -> requests.Response:
        for _ in range(10):
            time.sleep(1)
            audio_analysis_result = self._get_edenai(url)
            temp = audio_analysis_result.json()
            if temp["status"] == "finished":
                if temp["results"][self.providers[0]]["error"] is not None:
                    raise Exception(
                        f"""EdenAI returned an unexpected response 
                        {temp["results"][self.providers[0]]["error"]}"""
                    )
                else:
                    return audio_analysis_result

        raise Exception("Edenai speech to text job id processing Timed out")

    def _parse_response(self, response: dict) -> str:
        return response["public_id"]

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        all_params = {
            "file_url": query,
            "language": self.language,
            "speakers": self.speakers,
            "profanity_filter": self.profanity_filter,
            "custom_vocabulary": self.custom_vocabulary,
        }

        # filter so we don't send val to api when val is `None
        query_params = {k: v for k, v in all_params.items() if v is not None}

        job_id = self._call_eden_ai(query_params)
        url = self.base_url + job_id
        audio_analysis_result = self._wait_processing(url)
        result = audio_analysis_result.text
        formatted_text = json.loads(result)
        return formatted_text["results"][self.providers[0]]["text"]
