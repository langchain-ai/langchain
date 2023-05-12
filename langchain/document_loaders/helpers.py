"""Document loader helpers."""

from typing import List, NamedTuple, Optional


class FileEncoding(NamedTuple):
    encoding: Optional[str]
    confidence: float
    language: Optional[str]


def detect_file_encodings(file_path: str) -> List[FileEncoding]:
    """Try to detect the file encoding."""
    import chardet

    with open(file_path, "rb") as f:
        rawdata = f.read()
    encodings = chardet.detect_all(rawdata)

    if all(encoding["encoding"] is None for encoding in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")
    return [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]

