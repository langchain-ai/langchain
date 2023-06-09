"""Load from Dataframe object"""
from typing import Any, List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class DataFrameLoader(BaseLoader):
    """Load Pandas DataFrames."""

    def __init__(self, data_frame: Any, page_content_column: str = "text"):
        """Initialize with dataframe object."""
        import pandas as pd

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def load(self) -> List[Document]:
        """Load from the dataframe."""
        result = []
        # For very large dataframes, this needs to yield instead of building a list
        # but that would require chaging return type to a generator for BaseLoader
        # and all its subclasses, which is a bigger refactor. Marking as future TODO.
        # This change will allow us to extend this to Spark and Dask dataframes.
        for _, row in self.data_frame.iterrows():
            text = row[self.page_content_column]
            metadata = row.to_dict()
            metadata.pop(self.page_content_column)
            result.append(Document(page_content=text, metadata=metadata))
        return result

   class DataFrameContentLoader(BaseLoader):
    """Load Pandas DataFrames."""
    
    def __init__(self, 
                 data_frame,
                 metadata_column: str = None):
        """Initialize with dataframe object."""
        import pandas as pd
        
        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        self.data_frame = data_frame
        self.metadata_column = metadata_column
            
    def load(self) -> List[Document]:
        """Load from the dataframe."""
        result = []
        for _, row in self.data_frame.iterrows():
            # Extract the metadata column value
            metadata = {self.metadata_column: row[self.metadata_column]}
            
            # Extract all columns except the metadata column and convert to dictionary
            text = row.drop(self.metadata_column).to_dict()
            
            # Convert the text dictionary to a string representation
            text_str = ', '.join(f'{k}: {v}' for k, v in text.items())
            
            # Create a Document object with the text and metadata
            result.append(Document(page_content=text_str, metadata=metadata))
            
        return result
