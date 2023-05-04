import threading
from typing import Optional

from langchain.callbacks.base import BaseCallbackHandler


class VocodeCallbackHandler(BaseCallbackHandler):
    from vocode.turn_based.output_device.base_output_device import BaseOutputDevice
    from vocode.turn_based.synthesizer.base_synthesizer import BaseSynthesizer

    def __init__(
        self,
        synthesizer: BaseSynthesizer,
        output_device: Optional[BaseOutputDevice] = None,
    ) -> None:
        from vocode.turn_based.output_device.speaker_output import SpeakerOutput

        super().__init__()
        self.output_device = output_device or SpeakerOutput.from_default_device()
        self.synthesizer = synthesizer

    def _speak_in_thread(self, text: str) -> None:
        thread = threading.Thread(target=lambda: self._speak(text))
        thread.start()

    def _speak(self, text: str) -> None:
        self.output_device.send_audio(self.synthesizer.synthesize(text))
