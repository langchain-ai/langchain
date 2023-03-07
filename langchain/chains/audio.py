from typing import Dict, List

from langchain.audio_models.base import AudioBase
from langchain.chains.base import Chain


class AudioChain(Chain):
    audio_model: AudioBase

    output_key: str = "transcribe"

    @property
    def input_keys(self) -> List[str]:
        return ["audio_file"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        file_path = inputs["audio_file"]
        task = self.output_key
        content = self.audio_model.transcript(file_path, task).strip()
        return {self.output_key: content}
