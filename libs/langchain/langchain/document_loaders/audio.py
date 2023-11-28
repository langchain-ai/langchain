from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import AzureSpeechServiceParser


class AzureSpeechServiceLoader(GenericLoader):
    @classmethod
    def from_path(
        cls, path: Union[str, Path], **kwargs: Any
    ) -> AzureSpeechServiceLoader:
        path = path if isinstance(path, Path) else Path(path)
        if path.is_file():
            loader_params: dict = {"glob": path.name}
            path = path.parent
        else:
            loader_params = {}
        loader = FileSystemBlobLoader(path, **loader_params)
        parser = AzureSpeechServiceParser(**kwargs)
        return cls(loader, parser)
