from typing import Dict, List

from langchain.audio_models.base import AudioBase
from langchain.chains.base import Chain


class AudioChain(Chain):
    audio_model: AudioBase

    output_key: str = "transcript"

    @property
    def input_keys(self) -> List[str]:
        return ["audio_file"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if self.output_key is "translation":
            return {
                self.output_key: self.audio_model.translation(
                    inputs["audio_file"]
                ).strip()
            }

        if self.output_key is "transcript":
            return {
                self.output_key: self.audio_model.transcript(
                    inputs["audio_file"]
                ).strip()
            }
