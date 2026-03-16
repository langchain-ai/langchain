"""Sarvam AI speech services: Speech-to-Text and Text-to-Speech."""

from __future__ import annotations

import base64
import time
from typing import Any, BinaryIO, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_sarvamcloud.version import __version__  # noqa: F401

_STT_MODES = Literal["transcribe", "translate", "verbatim", "translit", "codemix"]
_STT_MODELS = Literal["saaras:v3", "saarika:v2.5"]
_TTS_MODELS = Literal["bulbul:v3"]


class SarvamSTT(BaseModel):
    """Sarvam AI Speech-to-Text (REST API).

    Transcribes audio files into text across 23 Indian languages. Supports
    multiple transcription modes including translation to English and
    code-mixed output.

    Max audio duration: 30 seconds per request. For longer files use
    `SarvamBatchSTT`.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Supported languages (23): hi-IN, en-IN, ta-IN, te-IN, kn-IN, ml-IN,
        mr-IN, gu-IN, pa-IN, bn-IN, od-IN, as-IN, bodo-IN, sa-IN, ur-IN,
        ks-IN, kok-IN, mai-IN, mni-IN, ne-IN, sat-IN, sd-IN, doi-IN.

    Example:
        ```python
        from langchain_sarvamcloud import SarvamSTT

        stt = SarvamSTT(model="saaras:v3", mode="transcribe")

        with open("audio.wav", "rb") as f:
            result = stt.transcribe(f, language_code="hi-IN")

        print(result["transcript"])
        ```
    """

    model: _STT_MODELS = "saaras:v3"
    """STT model. `saaras:v3` (recommended) or `saarika:v2.5`."""

    mode: _STT_MODES = "transcribe"
    """Transcription mode.

    - `transcribe`: Return text in source language.
    - `translate`: Translate to English.
    - `verbatim`: Exact verbatim transcription.
    - `translit`: Transliterate to Roman script.
    - `codemix`: Code-mixed output.
    """

    language_code: str | None = None
    """BCP-47 language code (e.g. `hi-IN`). If `None`, auto-detected."""

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def transcribe(
        self,
        file: BinaryIO,
        *,
        language_code: str | None = None,
        mode: _STT_MODES | None = None,
        input_audio_codec: str | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file.

        Args:
            file: Binary audio file object (WAV, MP3, AAC, OGG, OPUS,
                FLAC, MP4, AMR, WebM, PCM). Max 30 seconds.
            language_code: BCP-47 language code. Overrides instance default.
            mode: Transcription mode. Overrides instance default.
            input_audio_codec: PCM codec for raw audio files (e.g.
                `pcm_s16le`). Only needed for PCM format.

        Returns:
            Dict with keys: `transcript`, `language_code`,
            `language_probability`, optional `timestamps`.
        """
        kwargs: dict[str, Any] = {
            "file": file,
            "model": self.model,
            "mode": mode or self.mode,
        }
        lang = language_code or self.language_code
        if lang:
            kwargs["language_code"] = lang
        if input_audio_codec:
            kwargs["input_audio_codec"] = input_audio_codec
        response = self._client.speech_to_text.transcribe(**kwargs)
        if not isinstance(response, dict):
            return response.model_dump()
        return response


class SarvamBatchSTT(BaseModel):
    """Sarvam AI Batch Speech-to-Text.

    Processes multiple audio files asynchronously. Supports diarization,
    timestamps, and up to 20 files per job (max 1 hour per file).

    The workflow is:
    1. `create_job()` — returns a `job_id`.
    2. `get_upload_urls(job_id, filenames)` — returns signed S3 URLs.
    3. Upload files directly to the signed URLs (via HTTP PUT).
    4. `start_job(job_id)` — begin processing.
    5. `get_status(job_id)` — poll until `Completed` or `Failed`.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Example:
        ```python
        import httpx
        from langchain_sarvamcloud import SarvamBatchSTT

        batch = SarvamBatchSTT()
        job_id = batch.create_job(model="saaras:v3", with_diarization=True)

        urls = batch.get_upload_urls(job_id, ["audio1.mp3", "audio2.mp3"])
        for name, info in urls.items():
            with open(name, "rb") as f:
                httpx.put(info["url"], content=f.read())

        batch.start_job(job_id)

        status = batch.get_status(job_id)
        while status["job_state"] not in ("Completed", "Failed"):
            import time; time.sleep(5)
            status = batch.get_status(job_id)

        print(status["job_details"])
        ```
    """

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def create_job(
        self,
        *,
        model: _STT_MODELS = "saaras:v3",
        mode: _STT_MODES = "transcribe",
        language_code: str | None = None,
        with_timestamps: bool = False,
        with_diarization: bool = False,
        num_speakers: int | None = None,
        input_audio_codec: str | None = None,
        callback: str | None = None,
    ) -> str:
        """Create a batch STT job.

        Args:
            model: STT model to use.
            mode: Transcription mode.
            language_code: BCP-47 language code. Auto-detected if `None`.
            with_timestamps: Include chunk-level timestamps in output.
            with_diarization: Enable speaker diarization.
            num_speakers: Expected number of speakers (1–8) for diarization.
            input_audio_codec: PCM codec for raw audio files.
            callback: Optional webhook URL for completion notification.

        Returns:
            The `job_id` string for subsequent calls.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "mode": mode,
            "with_timestamps": with_timestamps,
            "with_diarization": with_diarization,
        }
        if language_code:
            kwargs["language_code"] = language_code
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if input_audio_codec:
            kwargs["input_audio_codec"] = input_audio_codec
        if callback:
            kwargs["callback"] = callback

        response = self._client.speech_to_text_job.create_job(**kwargs)
        if not isinstance(response, dict):
            response = response.model_dump()
        return response["job_id"]

    def get_upload_urls(
        self, job_id: str, filenames: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Get signed upload URLs for audio files.

        Args:
            job_id: The batch job ID from `create_job`.
            filenames: List of audio file names to upload.

        Returns:
            Dict mapping each filename to an object with `url` and
            `expires_in` keys.
        """
        response = self._client.speech_to_text_job.get_upload_links(
            job_id=job_id, files=filenames
        )
        if not isinstance(response, dict):
            response = response.model_dump()
        return response.get("upload_urls", {})

    def start_job(self, job_id: str) -> None:
        """Start processing the batch job.

        Must be called after all files have been uploaded to their
        respective signed URLs.

        Args:
            job_id: The batch job ID from `create_job`.
        """
        self._client.speech_to_text_job.start(job_id=job_id)

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get the current status of a batch job.

        Args:
            job_id: The batch job ID from `create_job`.

        Returns:
            Dict with keys: `job_state` (Accepted|Pending|Running|
            Completed|Failed), `job_id`, `total_files`,
            `successful_files_count`, `failed_files_count`, `job_details`.
        """
        response = self._client.speech_to_text_job.get_status(job_id=job_id)
        if not isinstance(response, dict):
            return response.model_dump()
        return response

    def wait_for_completion(
        self, job_id: str, poll_interval: float = 5.0
    ) -> dict[str, Any]:
        """Poll until the batch job reaches a terminal state.

        Args:
            job_id: The batch job ID.
            poll_interval: Seconds to wait between status checks.

        Returns:
            The final status dict when `job_state` is `Completed` or
            `Failed`.
        """
        while True:
            status = self.get_status(job_id)
            if status["job_state"] in ("Completed", "Failed"):
                return status
            time.sleep(poll_interval)


class SarvamTTS(BaseModel):
    """Sarvam AI Text-to-Speech.

    Converts text to speech audio using the Bulbul v3 model. Supports 11
    Indian languages with 30+ voice options. Returns base64-encoded WAV audio.

    Setup:
        Install `langchain-sarvamcloud` and set the environment variable:

        ```bash
        pip install -U langchain-sarvamcloud
        export SARVAM_API_KEY="your-api-key"
        ```

    Supported languages (11): hi-IN, en-IN, ta-IN, te-IN, kn-IN, ml-IN,
        mr-IN, gu-IN, pa-IN, bn-IN, od-IN.

    Example:
        ```python
        import base64
        from langchain_sarvamcloud import SarvamTTS

        tts = SarvamTTS(speaker="shubh", pace=1.0)
        result = tts.synthesize(
            "नमस्ते, मैं सर्वम AI हूँ।",
            target_language_code="hi-IN",
        )

        audio_bytes = base64.b64decode(result["audios"][0])
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)
        ```
    """

    model: _TTS_MODELS = "bulbul:v3"
    """TTS model. Currently only `bulbul:v3` is supported."""

    speaker: str = "shubh"
    """Voice speaker name (lowercase). See Sarvam docs for full list of 30+ speakers."""

    pace: float = 1.0
    """Speech rate. Range: 0.5 (slow) to 2.0 (fast)."""

    speech_sample_rate: int = 24000
    """Audio sample rate in Hz. Options: 8000, 16000, 22050, 24000, 44100, 48000."""

    api_subscription_key: SecretStr | None = Field(default=None)
    """Sarvam API subscription key. Reads from `SARVAM_API_KEY`."""

    _client: Any = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize the Sarvam client."""
        import os  # noqa: PLC0415

        if self.api_subscription_key is None:
            key = os.environ.get("SARVAM_API_KEY")
            if key:
                from pydantic import SecretStr as _SecretStr  # noqa: PLC0415

                self.api_subscription_key = _SecretStr(key)
        try:
            from sarvamai import SarvamAI  # noqa: PLC0415

            key_val = (
                self.api_subscription_key.get_secret_value()
                if self.api_subscription_key
                else None
            )
            self._client = SarvamAI(api_subscription_key=key_val)
        except ImportError as exc:
            msg = (
                "Could not import sarvamai python package. "
                "Please install it with `pip install sarvamai`."
            )
            raise ImportError(msg) from exc
        return self

    def synthesize(
        self,
        text: str,
        *,
        target_language_code: str,
        speaker: str | None = None,
        pace: float | None = None,
        speech_sample_rate: int | None = None,
    ) -> dict[str, Any]:
        """Convert text to speech.

        Args:
            text: Text to synthesize. Maximum 2500 characters.
            target_language_code: BCP-47 language code (e.g. `hi-IN`).
                Must be one of the 11 supported languages.
            speaker: Voice speaker name. Overrides instance default.
            pace: Speech rate (0.5–2.0). Overrides instance default.
            speech_sample_rate: Audio sample rate in Hz. Overrides
                instance default.

        Returns:
            Dict with `audios` key containing a list of base64-encoded
            audio strings and `request_id`.
        """
        kwargs: dict[str, Any] = {
            "text": text,
            "target_language_code": target_language_code,
            "model": self.model,
            "speaker": speaker or self.speaker,
            "pace": pace if pace is not None else self.pace,
            "speech_sample_rate": speech_sample_rate or self.speech_sample_rate,
        }
        response = self._client.text_to_speech.convert(**kwargs)
        if not isinstance(response, dict):
            return response.model_dump()
        return response

    def synthesize_to_bytes(
        self,
        text: str,
        *,
        target_language_code: str,
        speaker: str | None = None,
        pace: float | None = None,
    ) -> bytes:
        """Convert text to speech and return raw audio bytes.

        Args:
            text: Text to synthesize. Maximum 2500 characters.
            target_language_code: BCP-47 language code.
            speaker: Voice speaker name.
            pace: Speech rate (0.5–2.0).

        Returns:
            Raw audio bytes (WAV format).
        """
        result = self.synthesize(
            text,
            target_language_code=target_language_code,
            speaker=speaker,
            pace=pace,
        )
        audios = result.get("audios", [])
        if not audios:
            msg = "No audio returned from Sarvam TTS API."
            raise ValueError(msg)
        return base64.b64decode(audios[0])
