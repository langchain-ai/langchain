import subprocess
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_community.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain_community.document_loaders.assemblyai import TranscriptFormat
from langchain_core.callbacks.manager import CallbackManagerForChainRun

from langchain_experimental.video_captioning.models import AudioModel, BaseModel


class AudioProcessor:
    def __init__(
        self,
        api_key: str,
        output_audio_path: str = "output_audio.mp3",
    ):
        self.output_audio_path = Path(output_audio_path)
        self.api_key = api_key

    def process(
        self,
        video_file_path: str,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> list:
        try:
            self._extract_audio(video_file_path)
            return self._transcribe_audio()
        finally:
            # Cleanup: Delete the MP3 file after processing
            try:
                self.output_audio_path.unlink()
            except FileNotFoundError:
                pass  # File not found, nothing to delete

    def _extract_audio(self, video_file_path: str) -> None:
        # Ensure the directory exists where the output file will be saved
        self.output_audio_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-i",
            video_file_path,
            "-vn",
            "-acodec",
            "mp3",
            self.output_audio_path.as_posix(),
            "-y",  # The '-y' flag overwrites the output file if it exists
        ]

        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

    def _transcribe_audio(self) -> List[BaseModel]:
        if not self.api_key:
            raise ValueError("API key for AssemblyAI is not configured")
        audio_file_path_str = str(self.output_audio_path)
        loader = AssemblyAIAudioTranscriptLoader(
            file_path=audio_file_path_str,
            api_key=self.api_key,
            transcript_format=TranscriptFormat.SUBTITLES_SRT,
        )
        docs = loader.load()
        return self._create_transcript_models(docs)

    @staticmethod
    def _create_transcript_models(docs: List[Document]) -> List[BaseModel]:
        # Assuming docs is a list of Documents with .page_content as the transcript data
        models = []
        for doc in docs:
            models.extend(AudioProcessor._parse_transcript(doc.page_content))
        return models

    @staticmethod
    def _parse_transcript(srt_content: str) -> List[BaseModel]:
        models = []
        entries = srt_content.strip().split("\n\n")  # Split based on double newline

        for entry in entries:
            index, timespan, *subtitle_lines = entry.split("\n")

            # If not a valid entry format, skip
            if len(subtitle_lines) == 0:
                continue

            start_time, end_time = timespan.split(" --> ")
            subtitle_text = " ".join(subtitle_lines).strip()
            models.append(AudioModel.from_srt(start_time, end_time, subtitle_text))

        return models
