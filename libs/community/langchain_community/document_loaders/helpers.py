"""Document loader helpers."""

import concurrent.futures
import logging
from pathlib import Path
from typing import List, NamedTuple, Optional, Union, cast

logger = logging.getLogger(__name__)


class FileEncoding(NamedTuple):
    """File encoding as the NamedTuple."""

    encoding: Optional[str]
    """The encoding of the file."""
    confidence: float
    """The confidence of the encoding."""
    language: Optional[str]
    """The language of the file."""


def detect_file_encodings(
    file_path: Union[str, Path], timeout: int = 5
) -> List[FileEncoding]:
    """Try to detect the file encoding.

    Returns a list of `FileEncoding` tuples with the detected encodings ordered
    by confidence.

    Args:
        file_path: The path to the file to detect the encoding for.
        timeout: The timeout in seconds for the encoding detection.
    """
    import chardet

    file_path = str(file_path)

    def read_and_detect(file_path: str) -> List[dict]:
        with open(file_path, "rb") as f:
            rawdata = f.read()

        detected = cast(List[dict], chardet.detect_all(rawdata))

        encodings_to_try = ["gb18030", "gbk", "gb2312", "big5"]
        detected_encodings = [
            enc["encoding"] for enc in detected if enc["encoding"] is not None
        ]

        for encoding in encodings_to_try:
            if encoding.lower() not in [e.lower() for e in detected_encodings]:
                detected.append(
                    {"encoding": encoding, "confidence": 0.5, "language": "Chinese"}
                )

        if "iso-8859-1" not in [e.lower() for e in detected_encodings]:
            detected.append(
                {"encoding": "iso-8859-1", "confidence": 0.3, "language": None}
            )

        return detected

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(read_and_detect, file_path)
        try:
            encodings = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Timeout reached while detecting encoding for {file_path}"
            )

    if all(encoding["encoding"] is None for encoding in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")

    result = [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]
    result.sort(key=lambda x: x.confidence, reverse=True)

    return result
