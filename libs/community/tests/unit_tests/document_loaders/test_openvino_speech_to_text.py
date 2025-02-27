import unittest
from pathlib import Path

from langchain_community.document_loaders.openvino_speech_to_text import (
    OpenVINOSpeechToTextLoader,
)

PARENT_DIR = Path(__file__).parents[2]
EXAMPLES_DIR = PARENT_DIR / "examples"
AUDIO_FILE = EXAMPLES_DIR / "hello_world.m4a"


class TestOpenVINOSpeechToTextLoader(unittest.TestCase):
    file_path: str = AUDIO_FILE
    model_id: str = "distil-whisper/distil-small.en"
    device: str = "cpu"

    def test_invalid_device_npu(self) -> None:
        with self.assertRaises(NotImplementedError) as c:
            loader = OpenVINOSpeechToTextLoader(self.file_path, self.model_id, "npu")

    def test_invalid_device(self) -> None:
        with self.assertRaises(Exception) as c:
            loader = OpenVINOSpeechToTextLoader(self.file_path, self.model_id, "zpu")

    def test_invalid_audio_type(self) -> None:
        with self.assertRaises(NotImplementedError) as c:
            loader = OpenVINOSpeechToTextLoader(
                "invalid_audio.mp4", self.model_id, self.device
            )

    def test_audio_not_found(self) -> None:
        with self.assertRaises(NotImplementedError) as c:
            loader = OpenVINOSpeechToTextLoader(
                "invalid_audio.mp3", self.model_id, self.device
            )
