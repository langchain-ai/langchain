from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from typing_extensions import Self

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

if TYPE_CHECKING:
    from chm import chm


class UnstructuredCHMLoader(UnstructuredFileLoader):
    """Load `CHM` files using `Unstructured`.

    CHM means Microsoft Compiled HTML Help.

    Examples
    --------
    from langchain_community.document_loaders import UnstructuredCHMLoader

    loader = UnstructuredCHMLoader("example.chm")
    docs = loader.load()

    References
    ----------
    https://github.com/dottedmag/pychm
    http://www.jedrea.com/chmlib/
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        mode: str = "single",
        **unstructured_kwargs: Any,
    ):
        """

        Args:
            file_path: The path to the CHM file to load.
            mode: The mode to use when loading the file. Can be one of "single",
                "multi", or "all". Default is "single".
            **unstructured_kwargs: Any kwargs to pass to the unstructured.
        """
        file_path = str(file_path)
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.html import partition_html

        with CHMParser(self.file_path) as f:  # type: ignore[arg-type]
            return [
                partition_html(text=item["content"], **self.unstructured_kwargs)
                for item in f.load_all()
            ]


class CHMParser(object):
    """Microsoft Compiled HTML Help (CHM) Parser."""

    path: str
    file: "chm.CHMFile"

    def __init__(self, path: str):
        from chm import chm

        self.path = path
        self.file = chm.CHMFile()
        self.file.LoadCHM(path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self.file:
            self.file.CloseCHM()

    @property
    def encoding(self) -> str:
        return self.file.GetEncoding().decode("utf-8")

    def index(self) -> List[Dict[str, str]]:
        from urllib.parse import urlparse

        from bs4 import BeautifulSoup

        res = []
        index = self.file.GetTopicsTree().decode(self.encoding)
        soup = BeautifulSoup(index)
        # <OBJECT ..>
        for obj in soup.find_all("object"):
            # <param name="Name" value="<...>">
            # <param name="Local" value="<...>">
            name = ""
            local = ""
            for param in obj.find_all("param"):
                if param["name"] == "Name":
                    name = param["value"]
                if param["name"] == "Local":
                    local = param["value"]
            if not name or not local:
                continue

            local = urlparse(local).path
            if not local.startswith("/"):
                local = "/" + local
            res.append({"name": name, "local": local})

        return res

    def load(self, path: Union[str, bytes]) -> str:
        if isinstance(path, str):
            path = path.encode("utf-8")
        obj = self.file.ResolveObject(path)[1]
        return self.file.RetrieveObject(obj)[1].decode(self.encoding)

    def load_all(self) -> List[Dict[str, str]]:
        res = []
        index = self.index()
        for item in index:
            content = self.load(item["local"])
            res.append(
                {
                    "name": item["name"],
                    "local": item["local"],
                    "content": content,
                }
            )
        return res
