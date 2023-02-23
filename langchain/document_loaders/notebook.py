"""Loader that loads .ipynb notebook files."""
import json
from pathlib import Path
from typing import Any, List

import pandas as pd

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def concatenate_cells(cell: dict, include_outputs: bool, max_output_length: int) -> str:
    """Combine cells information in a readable format ready to be used."""

    cell_type = cell["cell_type"]
    source = cell["source"]
    output = cell["outputs"]

    if include_outputs and cell_type == "code" and output:
        output = output[0]["text"]
        min_output = min(max_output_length, len(output))
        return f"{cell_type} cell: {source}\n with output: {output[:min_output]}\n\n"
    else:
        return f"{cell_type} cell: {source}\n\n"


def remove_newlines(x: Any) -> Any:
    if isinstance(x, str):
        return x.replace("\n", "")
    elif isinstance(x, list):
        return [remove_newlines(elem) for elem in x]
    elif isinstance(x, pd.DataFrame):
        return x.applymap(remove_newlines)
    else:
        return x


class NotebookLoader(BaseLoader):
    """Loader that loads .ipynb notebook files."""

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load(
        self,
        include_outputs: bool = False,
        max_output_length: int = 10,
        remove_newline: bool = False,
    ) -> List[Document]:
        """Load documents.
        Select if to include cell outputs by setting include_outputs to True.
        Select how many of the outputs values should be displayed by setting max_output_length
        """
        try:
            import pandas as pd
        except ImportError:
            raise ValueError(
                "pandas is needed for Notebook Loader, "
                "please install with `pip install pandas`"
            )
        p = Path(self.file_path)

        with open(p, encoding="utf8") as f:
            d = json.load(f)

        data = pd.json_normalize(d["cells"])
        filtered_data = data[["cell_type", "source", "outputs"]]
        if remove_newline:
            filtered_data = filtered_data.applymap(remove_newlines)

        text = filtered_data.apply(
            lambda x: concatenate_cells(x, include_outputs, max_output_length), axis=1
        ).str.cat(sep=" ")

        metadata = {"source": str(p)}

        return [Document(page_content=text, metadata=metadata)]
