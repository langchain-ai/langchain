"""Document loader helpers."""

import concurrent.futures
from typing import List, NamedTuple, Optional, cast


class FileEncoding(NamedTuple):
    encoding: Optional[str]
    confidence: float
    language: Optional[str]


def detect_file_encodings(file_path: str, timeout: int = 5) -> List[FileEncoding]:
    """Try to detect the file encoding.

    Returns a list of `FileEncoding` tuples with the detected encodings ordered
    by confidence.
    """
    import chardet

    def read_and_detect(file_path: str) -> List[dict]:
        with open(file_path, "rb") as f:
            rawdata = f.read()
        return cast(List[dict], chardet.detect_all(rawdata))

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
    return [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]
