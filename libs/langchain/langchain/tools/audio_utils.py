import tempfile


def save_audio(audio: bytes) -> str:
    with tempfile.NamedTemporaryFile(mode="bx", suffix=".wav", delete=False) as f:
        f.write(audio)
    return f.name


def load_audio(audio_file: str) -> bytes:
    with open(audio_file, mode="rb") as f:
        audio = f.read()
    return audio
