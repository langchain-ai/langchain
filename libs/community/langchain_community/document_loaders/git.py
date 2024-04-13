import os
from typing import Callable, Iterator, Optional, List, Union
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class GitLoader(BaseLoader):
    """Load `Git` repository files.

    The Repository can be local on disk available at `repo_path`,
    or remote at `clone_url` that will be cloned to `repo_path`.
    Currently, supports only text files.

    Each document represents one file in the repository. The `path` points to
    the local Git repository, and the `branch` specifies the branch to load
    files from. By default, it loads from the `main` branch.
    """

    def __init__(
            self,
            repo_path: str,
            clone_url: Optional[str] = None,
            branch: Optional[str] = "main",
            file_filter: Optional[Callable[[str], bool]] = None,
            relative_path: Optional[Union[str, List[str]]] = None,
    ):
        """

        Args:
            repo_path: The path to the Git repository.
            clone_url: Optional. The URL to clone the repository from.
            branch: Optional. The branch to load files from. Defaults to `main`.
            file_filter: Optional. A function that takes a file path and returns
              a boolean indicating whether to load the file. Defaults to None.
            relative_path: Optional. A list of relative paths within the repository to load files from.
        """
        self.repo_path = repo_path
        self.clone_url = clone_url
        self.branch = branch
        self.file_filter = file_filter
        self.relative_path = relative_path
        if isinstance(relative_path, str):
            self.relative_path = [relative_path.strip("/")]
        else:
            self.relative_path = [p.strip("/") for p in (relative_path or [])]

            # Validate the relative paths
        for p in self.relative_path:
            if not os.path.isdir(os.path.join(self.repo_path, p)):
                raise ValueError(
                    f"The relative path '{p}' does not exist "
                    f"in the repository at '{self.repo_path}'."
                )

    def lazy_load(self) -> Iterator[Document]:
	    try:
		    from git import Repo
	    except ImportError as ex:
		    raise ImportError(
			    "Could not import git python package. Please install it with `pip install GitPython`.") from ex

	    if not os.path.exists(self.repo_path) and self.clone_url is None:
		    raise ValueError(f"Path {self.repo_path} does not exist")
	    elif self.clone_url:
		    repo = Repo.clone_from(self.clone_url, self.repo_path)
	    else:
		    repo = Repo(self.repo_path)

	    repo.git.checkout(self.branch)

	    for relative_path in self.relative_path:
		    full_path = os.path.join(self.repo_path, relative_path)
		    if not os.path.isdir(full_path):
			    continue  # Skip if the path is not a directory

		    for root, dirs, files in os.walk(full_path):
			    for file_name in files:
				    if self.file_filter and not self.file_filter(file_name):
					    continue

				    file_path = os.path.join(root, file_name)
				    try:
					    with open(file_path, "r", encoding="utf-8") as f:
						    content = f.read()
						    file_type = os.path.splitext(file_name)[1]

						    metadata = {
							    "source": file_path,
							    "file_path": file_path,
							    "file_name": file_name,
							    "file_type": file_type,
						    }
						    yield Document(page_content=content, metadata=metadata)
				    except Exception as e:
					    print(f"Error reading file {file_path}: {e}")
