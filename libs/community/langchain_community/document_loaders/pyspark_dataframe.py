import itertools
import logging
import sys
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__file__)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class PySparkDataFrameLoader(BaseLoader):
    """Load `PySpark` DataFrames."""

    def __init__(
        self,
        spark_session: Optional["SparkSession"] = None,
        df: Optional[Any] = None,
        page_content_column: str = "text",
        fraction_of_memory: float = 0.1,
    ):
        """Initialize with a Spark DataFrame object.

        Args:
            spark_session: The SparkSession object.
            df: The Spark DataFrame object.
            page_content_column: The name of the column containing the page content.
             Defaults to "text".
            fraction_of_memory: The fraction of memory to use. Defaults to 0.1.
        """
        try:
            from pyspark.sql import DataFrame, SparkSession
        except ImportError:
            raise ImportError(
                "pyspark is not installed. Please install it with `pip install pyspark`"
            )

        self.spark = (
            spark_session if spark_session else SparkSession.builder.getOrCreate()
        )

        if not isinstance(df, DataFrame):
            raise ValueError(
                f"Expected data_frame to be a PySpark DataFrame, got {type(df)}"
            )
        self.df = df
        self.page_content_column = page_content_column
        self.fraction_of_memory = fraction_of_memory
        self.num_rows, self.max_num_rows = self.get_num_rows()
        self.rdd_df = self.df.rdd.map(list)
        self.column_names = self.df.columns

    def get_num_rows(self) -> Tuple[int, int]:
        """Gets the number of "feasible" rows for the DataFrame"""
        try:
            import psutil
        except ImportError as e:
            raise ImportError(
                "psutil not installed. Please install it with `pip install psutil`."
            ) from e
        row = self.df.limit(1).collect()[0]
        estimated_row_size = sys.getsizeof(row)
        mem_info = psutil.virtual_memory()
        available_memory = mem_info.available
        max_num_rows = int(
            (available_memory / estimated_row_size) * self.fraction_of_memory
        )
        return min(max_num_rows, self.df.count()), max_num_rows

    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for document content."""
        for row in self.rdd_df.toLocalIterator():
            metadata = {self.column_names[i]: row[i] for i in range(len(row))}
            text = metadata[self.page_content_column]
            metadata.pop(self.page_content_column)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load from the dataframe."""
        if self.df.count() > self.max_num_rows:
            logger.warning(
                f"The number of DataFrame rows is {self.df.count()}, "
                f"but we will only include the amount "
                f"of rows that can reasonably fit in memory: {self.num_rows}."
            )
        lazy_load_iterator = self.lazy_load()
        return list(itertools.islice(lazy_load_iterator, self.num_rows))
