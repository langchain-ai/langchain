from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any, Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.utilities.vertexai import get_client_info

if TYPE_CHECKING:
    from google.cloud import texttospeech


def _import_google_cloud_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech
    except ImportError as e:
        raise ImportError(
            "Cannot import google.cloud.texttospeech, please install "
            "`pip install google-cloud-texttospeech`."
        ) from e
    return texttospeech


def _encoding_file_extension_map(encoding: texttospeech.AudioEncoding) -> Optional[str]:
    texttospeech = _import_google_cloud_texttospeech()

    ENCODING_FILE_EXTENSION_MAP = {
        texttospeech.AudioEncoding.LINEAR16: ".wav",
        texttospeech.AudioEncoding.MP3: ".mp3",
        texttospeech.AudioEncoding.OGG_OPUS: ".ogg",
        texttospeech.AudioEncoding.MULAW: ".wav",
        texttospeech.AudioEncoding.ALAW: ".wav",
    }
    return ENCODING_FILE_EXTENSION_MAP.get(encoding)


class GoogleCloudTextToSpeechTool(BaseTool):
    """Tool that queries the Google Cloud Text to Speech API.

    In order to set this up, follow instructions at:
    https://cloud.google.com/text-to-speech/docs/before-you-begin
    """

    name: str = "google_cloud_texttospeech"
    description: str = (
        "A wrapper around Google Cloud Text-to-Speech. "
        "Useful for when you need to synthesize audio from text. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )

    _client: Any

    def __init__(self, **kwargs: Any) -> None:
        """Initializes private fields."""
        texttospeech = _import_google_cloud_texttospeech()

        super().__init__(**kwargs)

        self._client = texttospeech.TextToSpeechClient(
            client_info=get_client_info(module="text-to-speech")
        )

    def _run(
        self,
        input_text: str,
        language_code: str = "en-US",
        ssml_gender: Optional[texttospeech.SsmlVoiceGender] = None,
        audio_encoding: Optional[texttospeech.AudioEncoding] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        texttospeech = _import_google_cloud_texttospeech()
        ssml_gender = ssml_gender or texttospeech.SsmlVoiceGender.NEUTRAL
        audio_encoding = audio_encoding or texttospeech.AudioEncoding.MP3

        response = self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=input_text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=language_code, ssml_gender=ssml_gender
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=audio_encoding),
        )

        suffix = _encoding_file_extension_map(audio_encoding)

        with tempfile.NamedTemporaryFile(mode="bx", suffix=suffix, delete=False) as f:
            f.write(response.audio_content)
        return f.name
