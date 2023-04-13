import os
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class GitLoader(BaseLoader):
    """Loads files from a local Git repository into a list of documents.
    Currently supports only text files.

    Each document represents one file in the repository. The `path` points to
    the local Git repository, and the `branch` specifies the branch to load
    files from. By default, it loads from the `main` branch.
    """

    def __init__(
        self,
        path: str,
        branch: Optional[str] = "main",
    ):
        self.path = path
        self.branch = branch

    def load(self) -> List[Document]:
        try:
            from git import Blob, Repo
        except ImportError as ex:
            raise ImportError(
                "Could not import git python package. "
                "Please install it with `pip install GitPython`."
            ) from ex

        repo = Repo(self.path)
        repo.git.checkout(self.branch)

        docs: List[Document] = []

        for item in repo.tree().traverse():
            if isinstance(item, Blob):
                file_path = os.path.join(self.path, item.path)
                rel_file_path = os.path.relpath(file_path, self.path)
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                        file_type = os.path.splitext(item.name)[1]

                        # loads only text files
                        if self.bytes2str(content):
                            metadata = {
                                "file_path": rel_file_path,
                                "file_name": item.name,
                                "file_type": file_type,
                            }
                            text_content = content.decode("utf-8", errors="ignore")
                            doc = Document(page_content=text_content, metadata=metadata)
                        else:
                            continue
                        docs.append(doc)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

        return docs

    @staticmethod
    def bytes2str(content: bytes) -> str:
        """Return decoded text from bytes."""
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return None
