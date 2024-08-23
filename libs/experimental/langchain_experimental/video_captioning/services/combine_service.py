from typing import Dict, List, Optional, Tuple

from langchain.chains.llm import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForChainRun

from langchain_experimental.video_captioning.models import (
    AudioModel,
    CaptionModel,
    VideoModel,
)
from langchain_experimental.video_captioning.prompts import (
    VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT,
)


class CombineProcessor:
    def __init__(
        self, llm: BaseLanguageModel, verbose: bool = True, char_limit: int = 20
    ):
        self.llm = llm
        self.verbose = verbose

        # Adjust as needed. Be careful adjusting it too low because OpenAI may
        # produce unwanted output
        self._CHAR_LIMIT = char_limit

    def process(
        self,
        video_models: List[VideoModel],
        audio_models: List[AudioModel],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[CaptionModel]:
        caption_models = []
        audio_index = 0

        for video_model in video_models:
            while audio_index < len(audio_models):
                audio_model = audio_models[audio_index]
                overlap_start, overlap_end = self._check_overlap(
                    video_model, audio_model
                )

                if overlap_start == -1:
                    if audio_model.start_time <= video_model.start_time:
                        caption_models.append(
                            CaptionModel.from_audio_model(audio_model)
                        )
                        audio_index += 1
                    else:
                        break
                else:
                    self._handle_overlap(
                        caption_models,
                        video_model,
                        audio_model,
                        overlap_start,
                        overlap_end,
                    )

                    # Update audio model or pop if it's fully used
                    if audio_model.end_time <= overlap_end:
                        audio_index += 1
                    else:
                        audio_model.start_time = overlap_end

            caption_models.append(CaptionModel.from_video_model(video_model))

        # Add remaining audio models
        for i in range(audio_index, len(audio_models)):
            caption_models.append(CaptionModel.from_audio_model(audio_models[i]))

        return caption_models

    @staticmethod
    def _check_overlap(
        video_model: VideoModel, audio_model: AudioModel
    ) -> Tuple[int, int]:
        overlap_start = max(audio_model.start_time, video_model.start_time)
        overlap_end = min(audio_model.end_time, video_model.end_time)
        if overlap_start < overlap_end:
            return overlap_start, overlap_end
        return -1, -1

    def _handle_overlap(
        self,
        caption_models: List[CaptionModel],
        video_model: VideoModel,
        audio_model: AudioModel,
        overlap_start: int,
        overlap_end: int,
    ) -> None:
        # Handle non-overlapping part
        if video_model.start_time < overlap_start:
            caption_models.append(
                CaptionModel.from_video_model(
                    VideoModel(
                        video_model.start_time,
                        overlap_start,
                        video_model.image_description,
                    )
                )
            )
            video_model.start_time = overlap_start

        # Handle the combined caption during overlap
        caption_text = self._validate_and_adjust_description(audio_model, video_model)
        subtitle_text = audio_model.subtitle_text
        caption_models.append(
            CaptionModel.from_video_model(
                VideoModel(overlap_start, overlap_end, caption_text)
            ).add_subtitle_text(subtitle_text)
        )

        # Update video model start time for remaining part
        if video_model.end_time > overlap_end:
            video_model.start_time = overlap_end

    def _validate_and_adjust_description(
        self,
        audio_model: AudioModel,
        video_model: VideoModel,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> str:
        conversation = LLMChain(
            llm=self.llm,
            prompt=VALIDATE_AND_ADJUST_DESCRIPTION_PROMPT,
            verbose=True,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        # Get response from OpenAI using LLMChain
        response: Dict[str, str] = conversation(
            {
                "limit": self._CHAR_LIMIT,
                "subtitle": audio_model.subtitle_text,
                "description": video_model.image_description,
            }
        )

        # Take out the Result: part of the response
        return response["text"].replace("Result:", "").strip()
