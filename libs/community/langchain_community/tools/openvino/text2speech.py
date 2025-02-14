import tempfile
from enum import Enum
from typing import Any, Dict, Optional, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator

import torch
from dataclasses import dataclass, field
from optimum.intel.openvino import OVModelForCausalLM
from outetts.version.v1.interface import InterfaceHF
from outetts.version.v1.prompt_processor import PromptProcessor
from outetts.version.v1.model import HFModel
from outetts.wav_tokenizer.audio_codec import AudioCodec
from scipy.io import wavfile
import time

@dataclass
class HFModelConfig:
    model_path: str = "OuteAI/OuteTTS-0.2-500M"
    language: str = "en"
    tokenizer_path: str = None
    languages: list = field(default_factory=list)
    verbose: bool = False
    device: str = None
    dtype: torch.dtype = None
    additional_model_config: dict = field(default_factory=dict)
    wavtokenizer_model_path: str = None
    max_seq_length: int = 4096


class OVHFModel(HFModel):
    def __init__(self, model_path, device, load_in_8bit):
        self.device = torch.device("cpu")
        self.model = OVModelForCausalLM.from_pretrained(model_path, device=device, load_in_8bit=load_in_8bit)


class InterfaceOV(InterfaceHF):
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        load_in_8bit: bool = False,
    ) -> None:
        self.device = torch.device("cpu")
        self.audio_codec = AudioCodec(self.device)
        self.prompt_processor = PromptProcessor(model_path, ["en"])
        self.model = OVHFModel(model_path, device, load_in_8bit)
        self.language = "en"
        self.verbose = False
        self.languages = ["en"]
        self._device = torch.device("cpu")
        self.config = HFModelConfig(model_path=model_path, language="en", device=self._device)

class OpenVINOText2SpeechTool(BaseTool):  # type: ignore[override]
    """Tool that queries OpenVINO OVModelForCausalLM.
    """

    name: str = "openvino_text2speech"
    description: str = (
        "A wrapper around OpenVINO Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese, and Hindi. "
    )

    model_id: str
    device: str
    load_in_8bit: bool
    interface: Any

    def __init__(self, model_id: str, device: str, load_in_8bit: bool) -> None:
        model_id = model_id
        device = device
        load_in_8bit = load_in_8bit
        interface = InterfaceOV(model_id, device, load_in_8bit)

        super().__init__(  # type: ignore[call-arg]
            model_id=model_id,
            device=device,
            load_in_8bit=load_in_8bit,
            interface = interface
        )

    def _run(
        self, 
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""


        try:
            start_time = time.time()
            speech = self.interface.generate(text=query, 
                    temperature=0.1, 
                    repetition_penalty=1.1,
                    max_length=4096)
            end_time = time.time()
            print("TTS took: ", end_time - start_time, " (seconds) / ", (end_time-start_time)/60, " (minutes)")

            with tempfile.NamedTemporaryFile(
                mode="bx", suffix=".wav", delete=False
            ) as f:
                wavfile.write(f.name, speech.sr, speech.audio[0].numpy())

            return f.name
        except Exception as e:
            raise RuntimeError(f"Error while running OpenVINOText2SpeechTool: {e}")

    def play(self, speech_file: str) -> None:
        """Play the text as speech."""

        import sounddevice as sd
        import soundfile as sf
        import numpy as np

        _, r = sf.read(speech_file, dtype='float32')

        try:
            from transformers.pipelines.audio_utils import ffmpeg_read
        except ImportError as exc:
            raise ImportError(
                "Could not import ffmpeg-python python package. "
                "Please install it with `pip install ffmpeg-python`."
            ) from exc

        audio_decoded = None
        with open(speech_file, "rb") as f:
            content = f.read()
            audio_decoded = ffmpeg_read(content, r)
        d = np.frombuffer(audio_decoded, dtype=np.float32)

        sd.play(d,r)
        sd.wait()


    def stream_speech(self, query: str) -> None:
        """Stream the text as speech as it is generated.
        Play the text in your speakers."""
        print("todo: stream_speech")
