"""Module for voice input for Ollama models"""

from typing import Any, Iterator, Literal, Optional, Union

import sounddevice as sd
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field
from scipy.io.wavfile import write

from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.llms import ollama


class SpeechToText(BaseBlobParser):
    """Transcribe and parse audio files for voice input.

    Audio transcription is done with OpenAI Whisper model.

    Note: OpenAI API key must be passed or initialized as an environment variable.

    Args:
        api_key: OpenAI API key
        chunk_duration_threshold: Minimum duration of a chunk in seconds
            NOTE: According to the OpenAI API, the chunk duration should be at least 0.1
            seconds. If the chunk duration is less or equal than the threshold,
            it will be skipped.
        audio_path: Path to the audio file, if a voice input file already present/saved
    """

    def __init__(
        self,
        audio_path: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        chunk_duration_threshold: float = 0.1,
        base_url: Optional[str] = None,
        language: Union[str, None] = None,
        prompt: Union[str, None] = None,
        response_format: Union[
            Literal["json", "text", "srt", "verbose_json", "vtt"], None
        ] = None,
        temperature: Union[float, None] = None,
    ):
        self.parser = OpenAIWhisperParser(
            api_key=api_key,
            base_url=base_url,
            language=language,
            prompt=prompt,
            response_format=response_format,
            chunk_duration_threshold=chunk_duration_threshold,
            temperature=temperature,
        )
        self.audio_blob = Blob(path=audio_path)

    def record_audio(
        self, path: str = "", duration: int = 30, sample_rate: int = 44100
    ) -> None:
        """Record audio and save to a file.

        Args:
            duration (int): Duration of the recording in seconds
            sample_rate (int): Sample rate of the recording (per second)
            path (str): Path to save the audio file

        """
        print("Recording...")  # noqa: T201

        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        # Save the audio data to a WAV file
        output_filename = path + "audio_input.wav"
        write(output_filename, sample_rate, audio_data)

        print(f"Recording complete. File saved as {output_filename}.")  # noqa: T201
        self.audio_blob = Blob(path=output_filename)
        return

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        if not self.audio_blob.path:
            raise ValueError("No audio input has been provided.")
        return self.parser.lazy_parse(blob)


class VoiceInputChain(BaseTool):
    """Transcibe audio inputs and run on ollama models.

    Args:
        stt (speechToText): speechToText object used to transcibe
                            audio input
        base_url (str): base url for where the Ollama model will
                        be hosted
        model (str): Ollama model name
        chain (Runnable): Runnable chain to run on the Ollama model
    """

    base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama model"
    )
    model: str = Field(default="llama2", description="Ollama model name")
    stt: SpeechToText
    chain: Optional[Runnable] = None
    name: str = "VoiceInputChain"
    """ Name of the tool. """
    description: str = "Transcribe audio inputs and run on Ollama models."
    """ Description of the tool. Examples in the docs. """

    llm: Optional[Runnable] = None
    """ Ollama model to run on. """
    text_splitter: Optional[Any] = None
    """ Text splitter to split voice input into chunks. """
    loader: Optional[Any] = None
    """ Loader to load voice input. """
    summarize_chain: Optional[Any] = None
    """ Summarize chain to summarize long voice input. """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.llm = self._initialize_llm(kwargs.get("chain"))
        self.text_splitter = self._initialize_text_splitter()
        self.loader = self._initialize_loader()
        self.summarize_chain = self._initialize_summarize_chain()

    def _initialize_llm(self, chain):
        if chain:
            return chain
        return ollama.Ollama(base_url=self.base_url, model=self.model)

    def _initialize_loader(self):
        return GenericLoader.from_filesystem(
            path=self.stt.audio_blob.path, parser=self.stt
        )

    def _initialize_text_splitter(
        self, chunk_size: int = 4000 * 4, chunk_overlap: int = 50
    ):
        """Initialize text splitter for voice input to improve response
        quality.

        Args:
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks
        """
        # use 4000 * 4 to match context window for llama2
        # (just under 4096 tokens; 4 characters = 1 token)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def _initialize_summarize_chain(self):
        """Initialize load summarize chain for voice input that is
        extremely long.
        """
        # template for mapping prompt on each chunk of text
        map_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
                                The following text is a part of a transcribed voice
                                message or audio file for a voice assistant. Write
                                summary of this chunk of text and make sure to keep
                                all the important points. Here comes the text: 
                                "{text}"
                                """,
        )

        # template for combining summaries from each chunk
        combine_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
                                Write a summary of the following text, which consists
                                of summaries of parts from a transcribed voice message
                                or audio file for a voice assistant. Make sure to keep
                                all the important points. Here comes the text: 
                                "{text}"    
                                """,
        )

        return load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
        )

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Run voice input through a particular Ollama model.
        Note: To edit models further, can create unique chain
        and pass to class instance during initialization.
        """
        prompt = self.loader.load()
        documents = self.text_splitter.transform_documents(prompt)
        response = None
        # check for if prompt is greater than context window,
        # if so map_reduce prompt to make it eligible
        if len(documents) == 1:
            response = self.llm.invoke(documents[0].page_content)
        else:
            prompt = self.summarize_chain.invoke(documents)
            response = self.llm.invoke(prompt["output_text"])
        return response

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Run the voice input through the Ollama model."""
        return self._run(*args, **kwargs)
