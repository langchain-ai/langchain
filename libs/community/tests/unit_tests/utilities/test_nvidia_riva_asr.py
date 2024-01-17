"""Unit tests to verify function of the Riva ASR implementation."""
from typing import Generator
from unittest.mock import patch

import pytest
import riva.client.proto.riva_asr_pb2 as rasr
from langchain_community.utilities.nvidia_riva import (
    AudioStream,
    RivaASR,
    RivaAudioEncoding,
)

AUDIO_DATA_MOCK = [
    b"This",
    b"is",
    b"a",
    b"test.",
    b"_",
    b"Hello.",
    b"World",
]
AUDIO_TEXT_MOCK = b" ".join(AUDIO_DATA_MOCK).decode().strip().split("_")

SVC_URI = "not-a-url.asdf:9999"
SVC_USE_SSL = True
CONFIG = {
    "audio_channel_count": 9,
    "profanity_filter": False,
    "enable_automatic_punctuation": False,
    "url": f"{'https' if SVC_USE_SSL else 'http'}://{SVC_URI}",
    "ssl_cert": "/dev/null",
    "encoding": RivaAudioEncoding.ALAW,
    "language_code": "not-a-language",
    "sample_rate_hertz": 5,
}


def response_generator(
    transcript: str = "",
    empty: bool = False,
    final: bool = False,
    alternatives: bool = True,
) -> Generator[rasr.StreamingRecognizeResponse, None, None]:
    """Create a pseudo streaming response."""
    if empty:
        return rasr.StreamingRecognizeResponse()

    if not alternatives:
        return rasr.StreamingRecognizeResponse(
            results=[
                rasr.StreamingRecognitionResult(
                    is_final=final,
                    alternatives=[],
                )
            ]
        )

    return rasr.StreamingRecognizeResponse(
        results=[
            rasr.StreamingRecognitionResult(
                is_final=final,
                alternatives=[
                    rasr.SpeechRecognitionAlternative(transcript=transcript.strip())
                ],
            )
        ]
    )


def streaming_recognize_mock(
    generator: Generator[rasr.StreamingRecognizeRequest, None, None], **_
):
    """A mock function to fake a streaming call to Riva."""
    yield response_generator(empty=True)
    yield response_generator(alternatives=False)

    output_transcript = ""
    for streaming_requests in generator:
        input_bytes = streaming_requests.audio_content.decode()

        final = input_bytes == "_"
        if final:
            input_bytes = ""

        output_transcript += input_bytes + " "
        yield response_generator(final=final, transcript=output_transcript)
        if final:
            output_transcript = ""

    yield response_generator(final=True, transcript=output_transcript)


def riva_asr_stub_init_patch(self, _):
    """Patch for the Riva asr library."""
    self.StreamingRecognize = streaming_recognize_mock


@pytest.fixture
def asr() -> RivaASR:
    """Initialize a copy of the runnable."""
    return RivaASR(**CONFIG)


@pytest.fixture
def stream() -> AudioStream:
    """Initialize and populate a sample audio stream."""
    s = AudioStream()
    for val in AUDIO_DATA_MOCK:
        s.put(val)
    s.close()
    return s


@pytest.mark.requires("riva.client")
def test_init(asr: RivaASR) -> None:
    """Test that ASR accepts valid arguments."""
    for key, expected_val in CONFIG.items():
        assert getattr(asr, key, None) == expected_val


@pytest.mark.requires("riva.client")
def test_init_defaults() -> None:
    """Ensure the runnable can be loaded with no arguments."""
    _ = RivaASR()


@pytest.mark.requires("riva.client")
def test_config(asr: RivaASR) -> None:
    """Verify the Riva config is properly assembled."""
    expected = rasr.StreamingRecognitionConfig(
        interim_results=False,
        config=rasr.RecognitionConfig(
            encoding=CONFIG["encoding"],
            sample_rate_hertz=CONFIG["sample_rate_hertz"],
            audio_channel_count=CONFIG["audio_channel_count"],
            max_alternatives=1,
            profanity_filter=CONFIG["profanity_filter"],
            enable_automatic_punctuation=CONFIG["enable_automatic_punctuation"],
            language_code=CONFIG["language_code"],
        ),
    )
    assert asr.config == expected


@pytest.mark.requires("riva.client")
def test_get_service(asr: RivaASR) -> None:
    """Test generating an asr service class."""
    svc = asr._get_service()
    assert str(svc.auth.ssl_cert) == CONFIG["ssl_cert"]
    assert svc.auth.use_ssl == SVC_USE_SSL
    assert svc.auth.uri == SVC_URI


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_asr_pb2_grpc.RivaSpeechRecognitionStub.__init__",
    riva_asr_stub_init_patch,
)
def test_transcribe(asr, stream) -> None:
    """Test the transcribe method."""
    got_val = asr._transcribe(stream)
    assert got_val == AUDIO_TEXT_MOCK[0].strip()


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_asr_pb2_grpc.RivaSpeechRecognitionStub.__init__",
    riva_asr_stub_init_patch,
)
def test_invoke(asr, stream) -> None:
    """Test the invoke method."""
    got_val = asr.invoke(stream)
    assert got_val == "".join(AUDIO_TEXT_MOCK).strip()


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_asr_pb2_grpc.RivaSpeechRecognitionStub.__init__",
    riva_asr_stub_init_patch,
)
def test_stream(asr, stream) -> None:
    """Test the invoke method."""
    for idx, got_val in enumerate(asr.stream(stream)):
        assert got_val == AUDIO_TEXT_MOCK[idx].strip()


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_asr_pb2_grpc.RivaSpeechRecognitionStub.__init__",
    riva_asr_stub_init_patch,
)
async def test_astream(asr, stream) -> None:
    """Test the invoke method."""
    idx = 0
    async for got_val in asr.astream(stream):
        assert got_val == AUDIO_TEXT_MOCK[idx].strip()
        idx += 1
