from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

from langchain_community.tools.azure_ai_services.utils import (
    detect_file_src_type,
    download_audio_from_url,
)

logger = logging.getLogger(__name__)


class AzureAiServicesSpeechToTextTool(BaseTool):
    """Tool that queries the Azure AI Services Speech to Text API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text?pivots=programming-language-python
    """

    azure_ai_services_key: str = ""  #: :meta private:
    azure_ai_services_region: str = ""  #: :meta private:
    speech_language: str = "en-US"  #: :meta private:
    speech_config: Any  #: :meta private:

    name: str = "azure_ai_services_speech_to_text"
    description: str = (
        "A wrapper around Azure AI Services Speech to Text. "
        "Useful for when you need to transcribe audio to text. "
        "Input should be a url to an audio file."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        azure_ai_services_key = get_from_dict_or_env(
            values, "azure_ai_services_key", "AZURE_AI_SERVICES_KEY"
        )

        azure_ai_services_region = get_from_dict_or_env(
            values, "azure_ai_services_region", "AZURE_AI_SERVICES_REGION"
        )

        try:
            import azure.cognitiveservices.speech as speechsdk

            values["speech_config"] = speechsdk.SpeechConfig(
                subscription=azure_ai_services_key, region=azure_ai_services_region
            )
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-speech is not installed. "
                "Run `pip install azure-cognitiveservices-speech` to install."
            )

        return values

    def _continuous_recognize(self, speech_recognizer: Any) -> str:
        done = False
        text = ""

        def stop_cb(evt: Any) -> None:
            """callback that stop continuous recognition"""
            speech_recognizer.stop_continuous_recognition_async()
            nonlocal done
            done = True

        def retrieve_cb(evt: Any) -> None:
            """callback that retrieves the intermediate recognition results"""
            nonlocal text
            text += evt.result.text

        # retrieve text on recognized events
        speech_recognizer.recognized.connect(retrieve_cb)
        # stop continuous recognition on either session stopped or canceled events
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        # Start continuous speech recognition
        speech_recognizer.start_continuous_recognition_async()
        while not done:
            time.sleep(0.5)
        return text

    def _speech_to_text(self, audio_path: str, speech_language: str) -> str:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            pass

        audio_src_type = detect_file_src_type(audio_path)
        if audio_src_type == "local":
            audio_config = speechsdk.AudioConfig(filename=audio_path)
        elif audio_src_type == "remote":
            tmp_audio_path = download_audio_from_url(audio_path)
            audio_config = speechsdk.AudioConfig(filename=tmp_audio_path)
        else:
            raise ValueError(f"Invalid audio path: {audio_path}")

        self.speech_config.speech_recognition_language = speech_language
        speech_recognizer = speechsdk.SpeechRecognizer(self.speech_config, audio_config)
        return self._continuous_recognize(speech_recognizer)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            text = self._speech_to_text(query, self.speech_language)
            return text
        except Exception as e:
            raise RuntimeError(
                f"Error while running AzureAiServicesSpeechToTextTool: {e}"
            )