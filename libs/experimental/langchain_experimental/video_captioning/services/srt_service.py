from typing import List

from langchain_experimental.video_captioning.models import CaptionModel


class SRTProcessor:
    @staticmethod
    def process(caption_models: List[CaptionModel]) -> str:
        """Generates the full SRT content from a list of caption models."""
        srt_entries = []
        for index, model in enumerate(caption_models, start=1):
            srt_entries.append(model.to_srt_entry(index))

        return "\n".join(srt_entries)
