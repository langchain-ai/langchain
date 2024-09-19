from __future__ import annotations

import logging
import tempfile
from typing import Any, Dict, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

logger = logging.getLogger(__name__)


class AzureAiServicesTextToSpeechTool(BaseTool):
    """Tool that queries the Azure AI Services Text to Speech API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
    """

    name: str = "azure_ai_services_text_to_speech"
    description: str = (
        "A wrapper around Azure AI Services Text to Speech API. "
        "Useful for when you need to convert text to speech. "
    )
    return_direct: bool = True

    azure_ai_services_key: str = ""  #: :meta private:
    azure_ai_services_region: str = ""  #: :meta private:
    speech_language: str = "en-US"  #: :meta private:
    speech_config: Any  #: :meta private:

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

    def _text_to_speech(self, text: str, speech_language: str) -> str:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            pass

        self.speech_config.speech_synthesis_language = speech_language
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )
        result = speech_synthesizer.speak_text(text)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            stream = speechsdk.AudioDataStream(result)
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".wav", delete=False
            ) as f:
                stream.save_to_wav_file(f.name)

            return f.name

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.debug(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                raise RuntimeError(
                    f"Speech synthesis error: {cancellation_details.error_details}"
                )

            return "Speech synthesis canceled."

        else:
            return f"Speech synthesis failed: {result.reason}"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            speech_file = self._text_to_speech(query, self.speech_language)
            return speech_file
        except Exception as e:
            raise RuntimeError(
                f"Error while running AzureAiServicesTextToSpeechTool: {e}"
            )
