from typing import Any

from pydantic import Field

from langchain_community.document_loaders import SharePointLoader


class OneDriveLoader(SharePointLoader):
    """
    Load documents from Microsoft OneDrive.
    Uses `SharePointLoader` under the hood.
    """

    drive_id: str = Field(...)
    """The ID of the OneDrive drive to load data from."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["document_library_id"] = kwargs["drive_id"]
        super().__init__(**kwargs)
