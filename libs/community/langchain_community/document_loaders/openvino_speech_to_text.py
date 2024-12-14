from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

import time

class OpenVINOSpeechToTextLoader(BaseLoader):
    """
    Loader for OpenVINO Speech-to-Text audio transcripts.

    It uses the HuggingFace OpenVINO Speech-to-Text to transcribe audio files
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    To use, you should have the ``todo`` python package installed.

    Audio files can be specified via a local file path.

    For a detailed explanation refer to the product
    documentation.
    https://huggingface.co/OpenVINO/distil-whisper-tiny-int4-ov
    """

    def __init__(
        self,
        file_path: str,
        model_id: str,
        device = "CPU",
        return_timestamps = True,
        return_language = "en",
        chunk_length_s = 30,
    ):
        """
        Initializes the OpenVINOSpeechToTextLoader.

        Args:
            file_path: A URI or local file path.
            model_id: Name of the model
            device: Hardware acclerator to utilize for inference
            return_timestamps: Enable text with corresponding timestamps for model
            return_language: Set language for model
            chunk_length: Number of seconds for a chunk
        """
        if device.lower() == "npu":
            raise NotImplementedError("NPU not supported")

        self.file_path = file_path
        self.model_id = model_id
        self.device = device
        self.return_timestamps = return_timestamps
        self.return_language = return_language
        self.chunk_length_s = chunk_length_s

    def load(self) -> List[Document]:
        """Transcribes the audio file and loads the transcript into documents.

        It uses the OpenVINO to transcribe the audio file
        and blocks until the transcription is finished.
        """

        try:
            from transformers import AutoProcessor, pipeline
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        except ImportError as exc:
            raise ImportError(
                "Could not import todo python package. "
                "Please install it with `pip install todo`."
            ) from exc

        try:
            from transformers.pipelines.audio_utils import ffmpeg_read
        except ImportError as exc:
            raise ImportError(
                    "Could not import ffmpeg-python python package. "
                    "Please install it with `pip install ffmpeg-python`."
                    ) from exc

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = OVModelForSpeechSeq2Seq.from_pretrained(self.model_id , 
                load_in_4bit=False,
                export=False)

        model = model.to(self.device)
        model.compile()
        pipe = pipeline("automatic-speech-recognition",
                model=model,
                chunk_length_s=self.chunk_length_s,
                return_language=self.return_language,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor)

        audio_decoded = None
        if "gs://" in self.file_path:
            raise NotImplementedError
        elif ".mp4" in self.file_path:
            raise NotImplementedError
        elif ".mp3" in self.file_path:
            with open(self.file_path, "rb") as f:
                content = f.read()
                audio_decoded = ffmpeg_read(content, 
                    pipe.feature_extractor.sampling_rate)
        else:
            raise NotImplementedError

        audio_info = {
            "raw": audio_decoded,
            "sampling_rate": pipe.feature_extractor.sampling_rate,
            }

        start_time = time.time()
        chunks = pipe(audio_info, 
                return_language=self.return_language, 
                return_timestamps=self.return_timestamps
                )["chunks"]

        result_total_latency = time.time() - start_time
        print("Inf time (seconds/minutes): " , result_total_latency, "/", result_total_latency/60)

        return [
            Document(
                page_content=chunk['text'],
                metadata={
                    "language": chunk['language'],
                    "timestamp": str(chunk['timestamp']),
                    "result_total_latency": str(result_total_latency),
                },
            )
            for chunk in chunks
        ]
