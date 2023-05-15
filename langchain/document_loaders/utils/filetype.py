import os

from typing import IO, Optional


def get_mime_type(
    file_IO: Optional[IO] = None,
    file_path: Optional[str] = None,
) -> str:
    """If file_IO is given, use libmagic to determine file's mime type via buffer.
    If file_path is given, use Python standard lib `mimetypes` to make best guess for
    mime type. If fail (probabaly because file_name has no extension), try libmagic.
    """
    mime_type = None

    # Get mime type via file IO
    if file_IO:
        try:
            import magic

            mime_type = magic.from_buffer(file_IO, mime=True)
        except ImportError:
            pass

    # Get mime type via file name
    if file_path:
        if os.path.isfile(file_path):
            import mimetypes

            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                try:
                    import magic

                    mime_type = magic.from_file(file_path, mime=True)
                except ImportError:
                    pass
        else:
            raise ValueError("File path is not a file.")

    # If no detected mime type, return "UNK" (unknown)
    if mime_type is None:
        mime_type = "UNK"

    return mime_type


def get_extension(file_path: str) -> str:
    """Returns the file's extension (requires file_path), e.g. '.pdf', '.docx', etc."""
    _, extension = os.path.splitext(file_path)

    return extension
