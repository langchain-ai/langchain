from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.vertexai import get_client_info

if TYPE_CHECKING:
    from google.cloud.speech_v2 import RecognitionConfig
    from google.protobuf.field_mask_pb2 import FieldMask


@deprecated(
    since="0.0.32",
    removal="0.3.0",
    alternative_import="langchain_google_community.SpeechToTextLoader",
)
class GoogleSpeechToTextLoader(BaseLoader):
    """
    Loader for Google Cloud Speech-to-Text audio transcripts.

    It uses the Google Cloud Speech-to-Text API to transcribe audio files
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    To use, you should have the ``google-cloud-speech`` python package installed.

    Audio files can be specified via a Google Cloud Storage uri or a local file path.

    For a detailed explanation of Google Cloud Speech-to-Text, refer to the product
    documentation.
    https://cloud.google.com/speech-to-text
    """

    def __init__(
        self,
        project_id: str,
        file_path: str,
        location: str = "us-central1",
        recognizer_id: str = "_",
        config: Optional[RecognitionConfig] = None,
        config_mask: Optional[FieldMask] = None,
    ):
        """
        Initializes the GoogleSpeechToTextLoader.

        Args:
            project_id: Google Cloud Project ID.
            file_path: A Google Cloud Storage URI or a local file path.
            location: Speech-to-Text recognizer location.
            recognizer_id: Speech-to-Text recognizer id.
            config: Recognition options and features.
                For more information:
                https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognitionConfig
            config_mask: The list of fields in config that override the values in the
                ``default_recognition_config`` of the recognizer during this
                recognition request.
                For more information:
                https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognizeRequest
        """
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud.speech_v2 import (
                AutoDetectDecodingConfig,
                RecognitionConfig,
                RecognitionFeatures,
                SpeechClient,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-speech python package. "
                "Please install it with `pip install google-cloud-speech`."
            ) from exc

        self.project_id = project_id
        self.file_path = file_path
        self.location = location
        self.recognizer_id = recognizer_id
        # Config must be set in speech recognition request.
        self.config = config or RecognitionConfig(
            auto_decoding_config=AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="chirp",
            features=RecognitionFeatures(
                # Automatic punctuation could be useful for language applications
                enable_automatic_punctuation=True,
            ),
        )
        self.config_mask = config_mask

        self._client = SpeechClient(
            client_info=get_client_info(module="speech-to-text"),
            client_options=(
                ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
                if location != "global"
                else None
            ),
        )
        self._recognizer_path = self._client.recognizer_path(
            project_id, location, recognizer_id
        )

    def load(self) -> List[Document]:
        """Transcribes the audio file and loads the transcript into documents.

        It uses the Google Cloud Speech-to-Text API to transcribe the audio file
        and blocks until the transcription is finished.
        """
        try:
            from google.cloud.speech_v2 import RecognizeRequest
        except ImportError as exc:
            raise ImportError(
                "Could not import google-cloud-speech python package. "
                "Please install it with `pip install google-cloud-speech`."
            ) from exc

        request = RecognizeRequest(
            recognizer=self._recognizer_path,
            config=self.config,
            config_mask=self.config_mask,
        )

        if "gs://" in self.file_path:
            request.uri = self.file_path
        else:
            with open(self.file_path, "rb") as f:
                request.content = f.read()

        response = self._client.recognize(request=request)

        return [
            Document(
                page_content=result.alternatives[0].transcript,
                metadata={
                    "language_code": result.language_code,
                    "result_end_offset": result.result_end_offset,
                },
            )
            for result in response.results
        ]
