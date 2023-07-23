"""Loads .ipynb notebook files."""
import json
from pathlib import Path
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


def concatenate_cells(
    cell: dict, include_outputs: bool, max_output_length: int, traceback: bool
) -> str:
    """Combine cells information in a readable format ready to be used.

    Args:
        cell: A dictionary
        include_outputs: Whether to include the outputs of the cell.
        max_output_length: Maximum length of the output to be displayed.
        traceback: Whether to return a traceback of the error.

    Returns:
        A string with the cell information.

    """
    cell_type = cell["cell_type"]
    source = cell["source"]
    output = cell["outputs"]

    if include_outputs and cell_type == "code" and output:
        if "ename" in output[0].keys():
            error_name = output[0]["ename"]
            error_value = output[0]["evalue"]
            if traceback:
                traceback = output[0]["traceback"]
                return (
                    f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',"
                    f" with description '{error_value}'\n"
                    f"and traceback '{traceback}'\n\n"
                )
            else:
                return (
                    f"'{cell_type}' cell: '{source}'\n, gives error '{error_name}',"
                    f"with description '{error_value}'\n\n"
                )
        elif output[0]["output_type"] == "stream":
            output = output[0]["text"]
            min_output = min(max_output_length, len(output))
            return (
                f"'{cell_type}' cell: '{source}'\n with "
                f"output: '{output[:min_output]}'\n\n"
            )
    else:
        return f"'{cell_type}' cell: '{source}'\n\n"

    return ""


def remove_newlines(x: Any) -> Any:
    """Recursively removes newlines, no matter the data structure they are stored in."""
    import pandas as pd

    if isinstance(x, str):
        return x.replace("\n", "")
    elif isinstance(x, list):
        return [remove_newlines(elem) for elem in x]
    elif isinstance(x, pd.DataFrame):
        return x.applymap(remove_newlines)
    else:
        return x


class NotebookLoader(BaseLoader):
    """Loads .ipynb notebook files."""

    def __init__(
        self,
        path: str,
        include_outputs: bool = False,
        max_output_length: int = 10,
        remove_newline: bool = False,
        traceback: bool = False,
    ):
        """Initialize with path.

        Args:
            path: The path to load the notebook from.
            include_outputs: Whether to include the outputs of the cell.
                Defaults to False.
            max_output_length: Maximum length of the output to be displayed.
                Defaults to 10.
            remove_newline: Whether to remove newlines from the notebook.
                Defaults to False.
            traceback: Whether to return a traceback of the error.
                Defaults to False.
        """
        self.file_path = path
        self.include_outputs = include_outputs
        self.max_output_length = max_output_length
        self.remove_newline = remove_newline
        self.traceback = traceback

    def load(
        self,
    ) -> List[Document]:
        """Load documents."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is needed for Notebook Loader, "
                "please install with `pip install pandas`"
            )
        p = Path(self.file_path)

        with open(p, encoding="utf8") as f:
            d = json.load(f)

        data = pd.json_normalize(d["cells"])
        filtered_data = data[["cell_type", "source", "outputs"]]
        if self.remove_newline:
            filtered_data = filtered_data.applymap(remove_newlines)

        text = filtered_data.apply(
            lambda x: concatenate_cells(
                x, self.include_outputs, self.max_output_length, self.traceback
            ),
            axis=1,
        ).str.cat(sep=" ")

        metadata = {"source": str(p)}

        return [Document(page_content=text, metadata=metadata)]
