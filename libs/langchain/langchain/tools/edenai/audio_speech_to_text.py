from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

import requests
from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class EdenAiSpeechToText(EdenaiTool):
    """Tool that queries the Eden AI Speech To Text API.

    for api reference check edenai documentation:
    https://app.edenai.run/bricks/speech/asynchronous-speech-to-text.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    edenai_api_key: Optional[str] = None

    name = "edenai_speech_to_text"
    description = (
        "A wrapper around edenai Services speech to text "
        "Useful for when you have to convert audio to text."
        "Input should be a url to an audio file."
    )

    language: Optional[str] = "en"
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

    feature: str = "audio"
    subfeature: str = "speech_to_text_async"
    base_url = "https://api.edenai.run/v2/audio/speech_to_text_async/"

    def _wait_processing(self, url: str) -> requests.Response:
        for _ in range(10):
            time.sleep(1)
            audio_analysis_result = self._get_edenai(url)
            temp = audio_analysis_result.json()
            if temp["status"] == "finished":
                if temp["results"][self.providers[0]]["error"] is not None:
                    raise Exception(
                        f"""EdenAI returned an unexpected response 
                        {temp['results'][self.providers[0]]['error']}"""
                    )
                else:
                    return audio_analysis_result

        raise Exception("Edenai speech to text job id processing Timed out")

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            if self.params is None:
                query_params = {"file_url": query, "language": self.language}
            else:
                query_params = {
                    "file_url": query,
                    "language": self.language,
                    **self.params,
                }
            get_job_result = self._call_eden_ai(query_params)

            job_id = get_job_result.json()["public_id"]

            url = self.base_url + job_id

            audio_analysis_result = self._wait_processing(url)

            result = audio_analysis_result.text
            formatted_text = json.loads(result)

            text = formatted_text["results"][self.providers[0]]["text"]

            return text

        except Exception as e:
            raise RuntimeError(f"Error while running EdenAiExplicitText: {e}")
