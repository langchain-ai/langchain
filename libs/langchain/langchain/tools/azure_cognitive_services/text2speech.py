from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import root_validator
from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env


def _import_azure_speech() -> Any:
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError(
            "azure.cognitiveservices.speech is not installed. "
            "Run `pip install azure-cognitiveservices-speech` to install."
        )
    return speechsdk


logger = logging.getLogger(__name__)


class AzureCogsText2SpeechTool(BaseTool):
    """Tool that queries the Azure Cognitive Services Text2Speech API.

    In order to set this up, follow instructions at:
    https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/get-started-text-to-speech?pivots=programming-language-python
    """

    azure_cogs_key: str = ""  #: :meta private:
    azure_cogs_region: str = ""  #: :meta private:
    speech_language: str = "en-US"  #: :meta private:
    speech_config: Any  #: :meta private:

    name: str = "azure_cognitive_services_text2speech"
    description: str = (
        "A wrapper around Azure Cognitive Services Text2Speech. "
        "Useful for when you need to convert text to speech. "
    )

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and endpoint exists in environment."""
        speechsdk = _import_azure_speech()
        azure_cogs_key = get_from_dict_or_env(
            values, "azure_cogs_key", "AZURE_COGS_KEY"
        )

        azure_cogs_region = get_from_dict_or_env(
            values, "azure_cogs_region", "AZURE_COGS_REGION"
        )

        values["speech_config"] = speechsdk.SpeechConfig(
            subscription=azure_cogs_key, region=azure_cogs_region
        )
        return values

    def _text2speech(self, text: str, speech_language: str) -> bytes:
        speechsdk = _import_azure_speech()
        self.speech_config.speech_synthesis_language = speech_language
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )
        result = speech_synthesizer.speak_text(text)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.debug(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                raise RuntimeError(
                    f"Speech synthesis error: {cancellation_details.error_details}"
                )

            raise RuntimeError("Speech synthesis canceled.")

        else:
            raise RuntimeError(f"Speech synthesis failed: {result.reason}")

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            speech = self._text2speech(query, self.speech_language)
            self.play(speech)
            return "Speech has been generated"
        except Exception as e:
            raise RuntimeError(f"Error while running AzureCogsText2SpeechTool: {e}")

    def play(self, speech: bytes) -> None:
        """Play the speech."""
        try:
            from IPython import display
        except ImportError:
            raise ImportError(
                "IPython is not installed. " "Run `pip install ipython` to install."
            )
        audio = display.Audio(speech)
        display.display(audio)
