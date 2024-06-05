"""Unit tests to verify function of the Riva TTS implementation."""

from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, cast
from unittest.mock import patch

import pytest

from langchain_community.utilities.nvidia_riva import RivaAudioEncoding, RivaTTS

if TYPE_CHECKING:
    import riva.client
    import riva.client.proto.riva_tts_pb2 as rtts

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


def synthesize_online_mock(
    request: "rtts.SynthesizeSpeechRequest", **_: Any
) -> Generator["rtts.SynthesizeSpeechResponse", None, None]:
    """A mock function to fake a streaming call to Riva."""
    # pylint: disable-next=import-outside-toplevel
    import riva.client.proto.riva_tts_pb2 as rtts

    yield rtts.SynthesizeSpeechResponse(
        audio=f"[{request.language_code},{request.encoding},{request.sample_rate_hz},{request.voice_name}]".encode()
    )
    yield rtts.SynthesizeSpeechResponse(audio=request.text.strip().encode())


def riva_tts_stub_init_patch(
    self: "riva.client.proto.riva_tts_pb2_grpc.RivaSpeechSynthesisStub", _: Any
) -> None:
    """Patch for the Riva TTS library."""
    self.SynthesizeOnline = synthesize_online_mock


@pytest.fixture
def tts() -> RivaTTS:
    """Initialize a copy of the runnable."""
    return RivaTTS(**CONFIG)  # type: ignore[arg-type]


@pytest.mark.requires("riva.client")
def test_init(tts: RivaTTS) -> None:
    """Test that ASR accepts valid arguments."""
    for key, expected_val in CONFIG.items():
        assert getattr(tts, key, None) == expected_val


@pytest.mark.requires("riva.client")
def test_init_defaults() -> None:
    """Ensure the runnable can be loaded with no arguments."""
    _ = RivaTTS()  # type: ignore[call-arg]


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
def test_invoke(tts: RivaTTS) -> None:
    """Test the invoke method."""
    encoding = cast(RivaAudioEncoding, CONFIG["encoding"]).riva_pb2
    audio_synth_config = (
        f"[{CONFIG['language_code']},"
        f"{encoding},"
        f"{CONFIG['sample_rate_hertz']},"
        f"{CONFIG['voice_name']}]"
    )

    input = " ".join(AUDIO_TEXT_MOCK).strip()
    response = tts.invoke(input)
    expected = (audio_synth_config + audio_synth_config.join(AUDIO_TEXT_MOCK)).encode()
    assert response == expected


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_tts_pb2_grpc.RivaSpeechSynthesisStub.__init__",
    riva_tts_stub_init_patch,
)
def test_transform(tts: RivaTTS) -> None:
    """Test the transform method."""
    encoding = cast(RivaAudioEncoding, CONFIG["encoding"]).riva_pb2
    audio_synth_config = (
        f"[{CONFIG['language_code']},"
        f"{encoding},"
        f"{CONFIG['sample_rate_hertz']},"
        f"{CONFIG['voice_name']}]"
    )
    expected = (audio_synth_config + audio_synth_config.join(AUDIO_TEXT_MOCK)).encode()
    for idx, response in enumerate(tts.transform(iter(AUDIO_TEXT_MOCK))):
        if idx % 2:
            # odd indices will return the mocked data
            expected = AUDIO_DATA_MOCK[int((idx - 1) / 2)]
        else:
            # even indices will return the request config
            expected = audio_synth_config.encode()
        assert response == expected


@pytest.mark.requires("riva.client")
@patch(
    "riva.client.proto.riva_tts_pb2_grpc.RivaSpeechSynthesisStub.__init__",
    riva_tts_stub_init_patch,
)
async def test_atransform(tts: RivaTTS) -> None:
    """Test the transform method."""
    encoding = cast(RivaAudioEncoding, CONFIG["encoding"]).riva_pb2
    audio_synth_config = (
        f"[{CONFIG['language_code']},"
        f"{encoding},"
        f"{CONFIG['sample_rate_hertz']},"
        f"{CONFIG['voice_name']}]"
    )
    expected = (audio_synth_config + audio_synth_config.join(AUDIO_TEXT_MOCK)).encode()
    idx = 0

    async def _fake_async_iterable() -> AsyncGenerator[str, None]:
        for val in AUDIO_TEXT_MOCK:
            yield val

    async for response in tts.atransform(_fake_async_iterable()):
        if idx % 2:
            # odd indices will return the mocked data
            expected = AUDIO_DATA_MOCK[int((idx - 1) / 2)]
        else:
            # even indices will return the request config
            expected = audio_synth_config.encode()
        assert response == expected
        idx += 1
