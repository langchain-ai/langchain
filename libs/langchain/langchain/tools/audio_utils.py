import tempfile
from pathlib import Path


def save_audio(audio: bytes) -> str:
    """Save audio to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(mode="bx", suffix=".wav", delete=False) as f:
        f.write(audio)
    return f.name


def load_audio(audio_file_path: str) -> bytes:
    """Load audio from a file into bytes."""
    if Path(audio_file_path).exists():
        with open(audio_file_path, mode="rb") as f:
            audio = f.read()
        return audio
    raise FileNotFoundError(f"File {audio_file_path} not found.")
