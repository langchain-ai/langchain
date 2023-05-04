import json
from pathlib import Path
from typing import Iterable, List, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class TomlLoader(BaseLoader):
    """
    A TOML document loader that inherits from the BaseLoader class.

    This class can be initialized with either a single source file or a source
    directory containing TOML files.
    """

    def __init__(self, source: Union[str, Path]):
        """Initialize the TomlLoader with a source file or directory."""
        self.source = Path(source)

    def load(self) -> List[Document]:
        """Load and return all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterable[Document]:
        """Lazily load the TOML documents from the source file or directory."""
        import tomli

        if self.source.is_file() and self.source.suffix == ".toml":
            files = [self.source]
        elif self.source.is_dir():
            files = list(self.source.glob("**/*.toml"))
        else:
            raise ValueError("Invalid source path or file type")

        for file_path in files:
            with file_path.open("r", encoding="utf-8") as file:
                content = file.read()
                try:
                    data = tomli.loads(content)
                    doc = Document(
                        page_content=json.dumps(data),
                        metadata={"source": str(file_path)},
                    )
                    yield doc
                except tomli.TOMLDecodeError as e:
                    print(f"Error parsing TOML file {file_path}: {e}")
