import copy
import logging
import re
from typing import (
    Any,
    Iterable,
    List,
    Optional,
)

from ai21.models import DocumentType
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import SecretStr
from langchain_text_splitters import TextSplitter

from langchain_ai21.ai21_base import AI21Base

logger = logging.getLogger(__name__)


class AI21SemanticTextSplitter(TextSplitter):
    """Splitting text into coherent and readable units,
    based on distinct topics and lines.
    """

    def __init__(
        self,
        chunk_size: int = 0,
        chunk_overlap: int = 0,
        client: Optional[Any] = None,
        api_key: Optional[SecretStr] = None,
        api_host: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        num_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

        self._segmentation = AI21Base(
            client=client,
            api_key=api_key,
            api_host=api_host,
            timeout_sec=timeout_sec,
            num_retries=num_retries,
        ).client.segmentation

    def split_text(self, source: str) -> List[str]:
        """Split text into multiple components.

        Args:
            source: Specifies the text input for text segmentation
        """
        response = self._segmentation.create(
            source=source, source_type=DocumentType.TEXT
        )

        segments = [segment.segment_text for segment in response.segments]

        if self._chunk_size > 0:
            return self._merge_splits_no_seperator(segments)

        return segments

    def split_text_to_documents(self, source: str) -> List[Document]:
        """Split text into multiple documents.

        Args:
            source: Specifies the text input for text segmentation
        """
        response = self._segmentation.create(
            source=source, source_type=DocumentType.TEXT
        )

        return [
            Document(
                page_content=segment.segment_text,
                metadata={"source_type": segment.segment_type},
            )
            for segment in response.segments
        ]

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []

        for i, text in enumerate(texts):
            normalized_text = self._normalized_text(text)
            index = 0
            previous_chunk_len = 0

            for chunk in self.split_text_to_documents(text):
                # merge metadata from user (if exists) and from segmentation api
                metadata = copy.deepcopy(_metadatas[i])
                metadata.update(chunk.metadata)

                if self._add_start_index:
                    # find the start index of the chunk
                    offset = index + previous_chunk_len - self._chunk_overlap
                    normalized_chunk = self._normalized_text(chunk.page_content)
                    index = normalized_text.find(normalized_chunk, max(0, offset))
                    metadata["start_index"] = index
                    previous_chunk_len = len(normalized_chunk)

                documents.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=metadata,
                    )
                )

        return documents

    def _normalized_text(self, string: str) -> str:
        """Use regular expression to replace sequences of '\n'"""
        return re.sub(r"\s+", "", string)

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """This method overrides the default implementation of TextSplitter"""
        return self._merge_splits_no_seperator(splits)

    def _merge_splits_no_seperator(self, splits: Iterable[str]) -> List[str]:
        """Merge splits into chunks.
        If the segment size is bigger than chunk_size,
        it will be left as is (won't be cut to match to chunk_size).
        If the segment size is smaller than chunk_size,
        it will be merged with the next segment until the chunk_size is reached.
        """
        chunks = []
        current_chunk = ""

        for split in splits:
            split_len = self._length_function(split)

            if split_len > self._chunk_size:
                logger.warning(
                    f"Split of length {split_len}"
                    f"exceeds chunk size {self._chunk_size}."
                )

            if self._length_function(current_chunk) + split_len > self._chunk_size:
                if current_chunk != "":
                    chunks.append(current_chunk)
                    current_chunk = ""

            current_chunk += split

        if current_chunk != "":
            chunks.append(current_chunk)

        return chunks
