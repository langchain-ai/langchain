from typing import Optional  # ,Union, Any, Dict

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool


class OpenAIText2SpeechTool(BaseTool):  # type: ignore[override]
    """Tool that queries OpenAI Speech API."""

    name: str = "openai_text2speech"
    description: str = (
        "A wrapper around OpenAI Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )

    base_url: str
    api_key: str
    model_id: str
    voice: str
    sample_rate: int

    def __init__(self, model_id: str, voice: str, base_url: str, api_key: str) -> None:
        model_id = model_id
        voice = voice
        base_url = base_url
        api_key = api_key
        sample_rate = 24000

        try:
            import pyaudio
        except ImportError as exc:
            raise ImportError(
                "Could not import pyaudio python package. "
                "Please install it with `pip install pyaudio`."
            ) from exc

        pa = pyaudio.PyAudio()
        try:
            devices_found = pa.get_device_count()
            if devices_found == 0:
                raise Exception("get_device_count() failed.")
        except Exception as exc:
            raise Exception(f"No audio devices found! Error: {exc}")

        try:
            info = pa.get_default_output_device_info()["index"]
            if not pa.is_format_supported(
                sample_rate,
                output_device=info,
                output_channels=1,
                output_format=pyaudio.paInt16,
            ):
                raise Exception("is_format_supported failed.")
        except Exception as exc:
            exc_str = "Default audio device doesn't support "
            exc_str += f"sampleRate={sample_rate}. Error: {exc}"
            raise Exception(exc_str)

        super().__init__(  # type: ignore[call-arg]
            model_id=model_id,
            voice=voice,
            base_url=base_url,
            api_key=api_key,
            sample_rate=sample_rate,
        )

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                f"Please install the `openai` Python package. Error: {e}."
            )

        try:
            out_file_name = "tts-output.mp3"
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            with client.audio.speech.with_streaming_response.create(
                model=self.model_id,
                voice=self.voice,
                input=query,
                response_format="mp3",
            ) as response:
                response.stream_to_file(out_file_name)

            return out_file_name
        except Exception as e:
            raise RuntimeError(f"Error while running OpenAIText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""

        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        _, r = sf.read(speech_file, dtype="float32")

        try:
            from transformers.pipelines.audio_utils import ffmpeg_read
        except ImportError as exc:
            raise ImportError(
                "Could not import ffmpeg_read python package. "
                "Please install it with `pip install torchaudio transformers`."
            ) from exc

        audio_decoded = None
        with open(speech_file, "rb") as f:
            content = f.read()
            audio_decoded = ffmpeg_read(content, r)
        d = np.frombuffer(audio_decoded, dtype=np.float32)

        sd.play(d, r)
        sd.wait()

    def stream_speech(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(f"Please install `openai` Python package. Error: {e}")

        try:
            import pyaudio
        except ImportError as exc:
            raise ImportError(
                "Could not import pyaudio python package. "
                "Please install it with `pip install pyaudio`."
            ) from exc

        try:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            player = pyaudio.PyAudio().open(
                format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True
            )

            with client.audio.speech.with_streaming_response.create(
                model=self.model_id,
                voice=self.voice,
                input=query,
                response_format="pcm",
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    player.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Error while running OpenAIText2SpeechTool: {e}")
