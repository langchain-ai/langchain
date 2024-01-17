"""Unit tests to verify function of the Riva TTS implementation."""
from unittest.mock import Mock, patch

import pytest
import riva.client.proto.riva_tts_pb2 as rtts
from langchain_community.utilities.nvidia_riva import RivaAudioEncoding, RivaTTS

AUDIO_TEXT_MOCK = ["This is a test.", "Hello world"]
AUDIO_DATA_MOCK = [s.encode() for s in AUDIO_TEXT_MOCK]

SVC_URI = "not-a-url.asdf:9999"
SVC_USE_SSL = True
CONFIG = {
    "voice_name": "English-Test",
    "output_directory": None,
    "url": f"{'https' if SVC_USE_SSL else 'http'}://{SVC_URI}",
    "ssl_cert": "/dev/null",
    "encoding": RivaAudioEncoding.ALAW,
    "language_code": "not-a-language",
    "sample_rate_hertz": 5,
}

AUDIO_SYNTH_CONFIG = (
    f"[{CONFIG['language_code']},{CONFIG['encoding'].riva_pb2},"
    f"{CONFIG['sample_rate_hertz']},{CONFIG['voice_name']}]"
)


def synthesize_online_mock(request: rtts.SynthesizeSpeechRequest, **_):
    """A mock function to fake a streaming call to Riva."""
    yield rtts.SynthesizeSpeechResponse(
        audio=f"[{request.language_code},{request.encoding},{request.sample_rate_hz},{request.voice_name}]".encode()
    )
    yield rtts.SynthesizeSpeechResponse(audio=request.text.strip().encode())


def riva_tts_stub_init_patch(self, _):
    """Patch for the Riva TTS library."""
    self.SynthesizeOnline = synthesize_online_mock


@pytest.fixture
def tts() -> RivaTTS:
    """Initialize a copy of the runnable."""
    return RivaTTS(**CONFIG)


@pytest.mark.requires("riva.client")
def test_init(tts: RivaTTS) -> None:
    """Test that ASR accepts valid arguments."""
    for key, expected_val in CONFIG.items():
        assert getattr(tts, key, None) == expected_val


@pytest.mark.requires("riva.client")
def test_init_defaults() -> None:
    """Ensure the runnable can be loaded with no arguments."""
    _ = RivaTTS()


@pytest.mark.requires("riva.client")
def test_get_service(tts: RivaTTS) -> None:
    """Test the get service method."""
    svc = tts._get_service()
    assert str(svc.auth.ssl_cert) == CONFIG["ssl_cert"]
    assert svc.auth.use_ssl == SVC_USE_SSL
    assert svc.auth.uri == SVC_URI


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_tts_pb2_grpc.RivaSpeechSynthesisStub.__init__",
    riva_tts_stub_init_patch,
)
def test_invoke(tts: RivaTTS):
    """Test the invoke method."""
    input = " ".join(AUDIO_TEXT_MOCK).strip()
    response = tts.invoke(input)
    expected = (AUDIO_SYNTH_CONFIG + AUDIO_SYNTH_CONFIG.join(AUDIO_TEXT_MOCK)).encode()
    assert response == expected


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_tts_pb2_grpc.RivaSpeechSynthesisStub.__init__",
    riva_tts_stub_init_patch,
)
def test_transform(tts: RivaTTS):
    """Test the transform method."""
    expected = (AUDIO_SYNTH_CONFIG + AUDIO_SYNTH_CONFIG.join(AUDIO_TEXT_MOCK)).encode()
    for idx, response in enumerate(tts.transform(iter(AUDIO_TEXT_MOCK))):
        if idx % 2:
            # odd indices will return the mocked data
            expected = AUDIO_DATA_MOCK[int((idx - 1) / 2)]
        else:
            # even indicies will return the request config
            expected = AUDIO_SYNTH_CONFIG.encode()
        assert response == expected


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_tts_pb2_grpc.RivaSpeechSynthesisStub.__init__",
    riva_tts_stub_init_patch,
)
async def test_atransform(tts: RivaTTS):
    """Test the transform method."""
    expected = (AUDIO_SYNTH_CONFIG + AUDIO_SYNTH_CONFIG.join(AUDIO_TEXT_MOCK)).encode()
    idx = 0

    async def _fake_async_iterable() -> bytes:
        for val in AUDIO_TEXT_MOCK:
            yield val

    async for response in tts.atransform(_fake_async_iterable()):
        if idx % 2:
            # odd indices will return the mocked data
            expected = AUDIO_DATA_MOCK[int((idx - 1) / 2)]
        else:
            # even indicies will return the request config
            expected = AUDIO_SYNTH_CONFIG.encode()
        assert response == expected
        idx += 1
