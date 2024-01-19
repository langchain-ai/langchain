"""The audio processing pipeline for interactive calls with a chatbot."""

import asyncio

import pywav  # pywav is used instead of built-in wave because of mulaw support
from langchain_community.utilities.nvidia_riva import (
    AudioStream,
    RivaASR,
    RivaAudioEncoding,
    RivaTTS,
)
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_aiplay import ChatNVIDIA

audio_file = "../test.wav"

# read the audio file
wav_file = pywav.WavRead(audio_file)
audio_data = wav_file.getdata()
audio_encoding = RivaAudioEncoding.from_wave_format_code(wav_file.getaudioformat())
sample_rate = wav_file.getsamplerate()
delay_time = 1 / 4
chunk_size = int(sample_rate * delay_time)
delay_time = 1 / 8
num_channels = wav_file.getnumofchannels()
audio_chunks = [
    audio_data[0 + i : chunk_size + i] for i in range(0, len(audio_data), chunk_size)
]
# create the riva asr client
riva_asr = RivaASR(
    url="http://localhost:50051/",  # the location of the Riva ASR server
    encoding=audio_encoding,
    audio_channel_count=num_channels,
    sample_rate_hertz=sample_rate,
    profanity_filter=True,
    enable_automatic_punctuation=True,
    language_code="en-US",
)

# create the prompt template
prompt = PromptTemplate.from_template("{user_input}")

model = ChatNVIDIA(model="mixtral_8x7b")  # type: ignore

# create the riva tts client
riva_tts = RivaTTS(
    url="http://localhost:50051/",  # the location of the Riva TTS server
    output_directory="./scratch",  # location of the output .wav files
    language_code="en-US",
    voice_name="English-US.Female-1",
)

# construct and return the chain
chain = {"user_input": riva_asr} | prompt | model | riva_tts  # type: ignore


async def generate_audio_chunks() -> None:
    """Generates audio chunks from a .wav file
    to mimic streaming."""
    audio_chunks = [
        audio_data[0 + i : chunk_size + i]
        for i in range(0, len(audio_data), chunk_size)
    ]

    input_stream = AudioStream(maxsize=1000)
    # Send bytes into the stream
    for chunk in audio_chunks:
        await input_stream.aput(chunk)
    input_stream.close()

    output_stream = asyncio.Queue()
    while not input_stream.complete:
        async for chunk in chain.astream(input_stream):
            await output_stream.put(chunk)


if __name__ == "__main__":
    asyncio.run(generate_audio_chunks("../test.wav"))  # read in test.wav file
