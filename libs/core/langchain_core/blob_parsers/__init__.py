from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.blob_parsers.mime_type import MimeTypeBasedParser


def __getattr__(name):
    if name == "MimeTypeBasedParser":
        from langchain_core.blob_parsers.mime_type import MimeTypeBasedParser

        return MimeTypeBasedParser
    else:
        raise AttributeError(
            f"No {name} attribute in module langchain_core.blob_parsers."
        )


__all__ = ["MimeTypeBasedParser"]
