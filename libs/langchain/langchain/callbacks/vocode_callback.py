import threading
from typing import TYPE_CHECKING, Optional

from langchain.callbacks.base import BaseCallbackHandler

if TYPE_CHECKING:
    from vocode.turn_based.output_device.base_output_device import BaseOutputDevice
    from vocode.turn_based.synthesizer.base_synthesizer import BaseSynthesizer


class VocodeCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        synthesizer: BaseSynthesizer,
        output_device: Optional[BaseOutputDevice] = None,
    ) -> None:
        from vocode.turn_based.output_device.speaker_output import SpeakerOutput

        super().__init__()
        self.output_device = output_device or SpeakerOutput.from_default_device()
        self.synthesizer = synthesizer

    def speak_in_thread(self, text: str) -> None:
        thread = threading.Thread(target=lambda: self.speak(text))
        thread.start()

    def speak(self, text: str) -> None:
        self.output_device.send_audio(self.synthesizer.synthesize(text))
