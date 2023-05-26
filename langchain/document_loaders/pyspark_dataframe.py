"""Load from a Spark Dataframe object"""
import psutil
import sys
from typing import Any, List, Optional, TYPE_CHECKING, Iterator

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


class PySparkDataFrameLoader(BaseLoader):
    """Load PySpark DataFrames"""

    def __init__(
        self,
        spark_session: Optional[SparkSession] = None,
        df: Optional[Any] = None,
        page_content_column: str = "text",
        fraction_of_memory: int = 2,
    ):
        """Initialize with (any) dataframe object."""
        try:
            from pyspark.sql import SparkSession, DataFrame
        except ImportError:
            raise ValueError(
                "pyspark is not installed. Please install it with `pip install pyspark`"
            )

        self._spark = (
            spark_session if spark_session else SparkSession.builder.getOrCreate()
        )

        if not isinstance(df, DataFrame):
            raise ValueError(
                f"Expected data_frame to be a PySpark DataFrame, got {type(df)}"
            )
        self.df = df
        self.page_content_column = page_content_column
        self.fraction_of_memory = fraction_of_memory

    def load(self) -> List[Document]:
        """Load from the dataframe."""
        # From the Pandas Document Loader source code:
        #
        # For very large dataframes, this needs to yield instead of building a list
        # but that would require chaging return type to a generator for BaseLoader
        # and all its subclasses, which is a bigger refactor. Marking as future TODO.
        result = []

        # This code handles the loading functionality by choosing at most a fraction (1/2) of the driver node's
        # memory availability
        row = self.df.limit(1).collect()[0]
        estimated_row_size = sys.getsizeof(row)
        mem_info = psutil.virtual_memory()
        available_memory = mem_info.available
        max_num_rows = (
            int(available_memory / estimated_row_size) // self.fraction_of_memory
        )
        num_rows = min(max_num_rows, self.df.count())

        rdd_df = self.df.rdd.map(list)
        column_names = self.df.columns
        for row in rdd_df.take(num_rows):
            metadata = {column_names[i]: row[i] for i in range(len(row))}
            text = metadata[self.page_content_column]
            metadata.pop(self.page_content_column)
            result.append(Document(page_content=text, metadata=metadata))

        return result
