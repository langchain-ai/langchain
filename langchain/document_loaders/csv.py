from csv import DictReader
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class CSVLoader(BaseLoader):
    """Loads a CSV file into a list of documents.

    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.

    Output Example:
        .. code-block:: txt

            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(self, file_path: str, csv_args: Optional[Dict] = None):
        self.file_path = file_path
        if csv_args is None:
            self.csv_args = {
                "delimiter": ",",
                "quotechar": '"',
            }
        else:
            self.csv_args = csv_args

    def load(self) -> List[Document]:
        docs = []

        with open(self.file_path, newline="") as csvfile:
            csv = DictReader(csvfile, **self.csv_args)  # type: ignore
            for i, row in enumerate(csv):
                docs.append(
                    Document(
                        page_content="\n".join(
                            f"{k.strip()}: {v.strip()}" for k, v in row.items()
                        ),
                        metadata={"source": self.file_path, "row": i},
                    )
                )

        return docs
