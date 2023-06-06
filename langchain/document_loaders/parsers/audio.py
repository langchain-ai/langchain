from typing import Iterator

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document


class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io
        import openai
        from pydub import AudioSegment

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        # Define the duration of each chunk in minutes to meet 25MB OpenAI size limit to Whisper API
        chunk_duration = 20

        # Calculate the chunk duration in milliseconds
        chunk_duration_ms = chunk_duration * 60 * 1000

        # Split the audio into chunk_duration_ms chunks 
        for split_number,i in enumerate(range(0, len(audio), chunk_duration_ms)):
            
            print(f"Transcribing part {split_number}!")

            # Audio chunk 
            chunk = audio[i : i + chunk_duration_ms]
            audio_split = chunk.export(format="mp3").read()
            file_obj = io.BytesIO(audio_split)
            file_obj.name = blob.source + f"_part_{split_number}.mp3"

            # Transcribe
            transcript = openai.Audio.transcribe("whisper-1", file_obj)

            yield Document(
                page_content=transcript.text, metadata={"source": file_obj.name}
            )