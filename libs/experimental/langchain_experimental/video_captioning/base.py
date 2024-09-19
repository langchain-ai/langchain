from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import ConfigDict

from langchain_experimental.video_captioning.services.audio_service import (
    AudioProcessor,
)
from langchain_experimental.video_captioning.services.caption_service import (
    CaptionProcessor,
)
from langchain_experimental.video_captioning.services.combine_service import (
    CombineProcessor,
)
from langchain_experimental.video_captioning.services.image_service import (
    ImageProcessor,
)
from langchain_experimental.video_captioning.services.srt_service import SRTProcessor


class VideoCaptioningChain(Chain):
    """
    Video Captioning Chain.
    """

    llm: BaseLanguageModel
    assemblyai_key: str
    prompt: Optional[PromptTemplate] = None
    verbose: bool = True
    use_logging: Optional[bool] = True
    frame_skip: int = -1
    image_delta_threshold: int = 3000000
    closed_caption_char_limit: int = 20
    closed_caption_similarity_threshold: int = 80
    use_unclustered_video_models: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @property
    def input_keys(self) -> List[str]:
        return ["video_file_path"]

    @property
    def output_keys(self) -> List[str]:
        return ["srt"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        if "video_file_path" not in inputs:
            raise ValueError(
                "Missing 'video_file_path' in inputs for video captioning."
            )
        video_file_path = inputs["video_file_path"]
        nl = "\n"

        run_manager.on_text(
            "Loading processors..." + nl
        ) if self.use_logging and run_manager else None

        audio_processor = AudioProcessor(api_key=self.assemblyai_key)
        image_processor = ImageProcessor(
            frame_skip=self.frame_skip, threshold=self.image_delta_threshold
        )
        caption_processor = CaptionProcessor(
            llm=self.llm,
            verbose=self.verbose,
            similarity_threshold=self.closed_caption_similarity_threshold,
            use_unclustered_models=self.use_unclustered_video_models,
        )
        combine_processor = CombineProcessor(
            llm=self.llm,
            verbose=self.verbose,
            char_limit=self.closed_caption_char_limit,
        )
        srt_processor = SRTProcessor()

        run_manager.on_text(
            "Finished loading processors."
            + nl
            + "Generating subtitles from audio..."
            + nl
        ) if self.use_logging and run_manager else None

        # Get models for speech to text subtitles
        audio_models = audio_processor.process(video_file_path, run_manager)
        run_manager.on_text(
            "Finished generating subtitles:"
            + nl
            + f"{nl.join(str(obj) for obj in audio_models)}"
            + nl
            + "Generating closed captions from video..."
            + nl
        ) if self.use_logging and run_manager else None

        # Get models for image frame description
        image_models = image_processor.process(video_file_path, run_manager)
        run_manager.on_text(
            "Finished generating closed captions:"
            + nl
            + f"{nl.join(str(obj) for obj in image_models)}"
            + nl
            + "Refining closed captions..."
            + nl
        ) if self.use_logging and run_manager else None

        # Get models for video event closed-captions
        video_models = caption_processor.process(image_models, run_manager)
        run_manager.on_text(
            "Finished refining closed captions:"
            + nl
            + f"{nl.join(str(obj) for obj in video_models)}"
            + nl
            + "Combining subtitles with closed captions..."
            + nl
        ) if self.use_logging and run_manager else None

        # Combine the subtitle models with the closed-caption models
        caption_models = combine_processor.process(
            video_models, audio_models, run_manager
        )
        run_manager.on_text(
            "Finished combining subtitles with closed captions:"
            + nl
            + f"{nl.join(str(obj) for obj in caption_models)}"
            + nl
            + "Generating SRT file..."
            + nl
        ) if self.use_logging and run_manager else None

        # Convert the combined model to SRT format
        srt_content = srt_processor.process(caption_models)
        run_manager.on_text(
            "Finished generating srt file." + nl
        ) if self.use_logging and run_manager else None

        return {"srt": srt_content}

    @property
    def _chain_type(self) -> str:
        return "video_captioning_chain"
