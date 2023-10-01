from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import root_validator
from langchain.tools.azure_cognitive_services.utils import (
    detect_file_src_type,
    download_audio_from_url,
)
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class AzureCogsSpeech2TextTool(BaseTool):
    """Tool that queries the Azure Cognitive Services Speech2Text API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-speech-to-text?pivots=programming-language-python
    """

    azure_cogs_key: str = ""  #: :meta private:
    azure_cogs_region: str = ""  #: :meta private:
    speech_language: str = "en-US"  #: :meta private:
    speech_config: Any  #: :meta private:

    name: str = "azure_cognitive_services_speech2text"
    description: str = (
        "A wrapper around Azure Cognitive Services Speech2Text. "
        "Useful for when you need to transcribe audio to text. "
        "Input should be a url to an audio file."
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        azure_cogs_key = get_from_dict_or_env(
            values, "azure_cogs_key", "AZURE_COGS_KEY"
        )

        azure_cogs_region = get_from_dict_or_env(
            values, "azure_cogs_region", "AZURE_COGS_REGION"
        )

        try:
            import azure.cognitiveservices.speech as speechsdk

            values["speech_config"] = speechsdk.SpeechConfig(
                subscription=azure_cogs_key, region=azure_cogs_region
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

    def _speech2text(self, audio_path: str, speech_language: str) -> str:
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
            text = self._speech2text(query, self.speech_language)
            return text
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsSpeech2TextTool: {e}")
