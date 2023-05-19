import os
from urllib.parse import urlparse


def detect_file_src_type(file_path: str) -> str:
    """Detect if the file is local or remote."""
    if os.path.isfile(file_path):
        return "local"

    parsed_url = urlparse(file_path)
    if parsed_url.scheme and parsed_url.netloc:
        return "remote"

    return "invalid"