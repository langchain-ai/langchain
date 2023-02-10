"""Loads a PDF with pypdf and chunks at character level."""
from typing import Dict, List, Optional, Tuple


class PagedPDFSplitter:
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200):
        """Initialize with file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(
        self, file_path: str, metadata: Optional[Dict] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Load given path and split into texts and metadatas.

        If given, the metadata given will
        be duplicated and attached to each split along with page number.
        If key is present in metadata, it also has page number
        included (e.g., Foo2012 Pages 3-4).
        """
        import pypdf

        pdfFileObj = open(file_path, "rb")
        pdfReader = pypdf.PdfReader(pdfFileObj)
        splits = []
        split = ""
        pages = []
        metadatas = []
        key = (
            metadata["key"] if metadata is not None and "key" in metadata else file_path
        )
        for i, page in enumerate(pdfReader.pages):
            split += page.extract_text()
            pages.append(str(i + 1))
            if len(split) > self.chunk_size or i == len(pdfReader.pages) - 1:
                splits.append(split[: self.chunk_size])
                # pretty formatting of pages (e.g. 1-3, 4, 5-7)
                pg = "-".join([pages[0], pages[-1]])
                metadatas.append(dict(key=f"{key} pages {pg}", pages=pg))
                if metadata is not None:
                    metadatas[-1].update(metadata)
                split = str(splits[self.chunk_size: self.chunk_overlap])
                pages = [str(i + 1)]
        pdfFileObj.close()
        return splits, metadatas
