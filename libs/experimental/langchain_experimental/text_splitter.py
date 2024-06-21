import copy
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from langchain_community.utils.math import cosine_similarity
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]

BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}

@dataclass
class SentenceData:
    sentence: str
    index: int
    combined_sentence: Optional[str] = None
    combined_sentence_embedding: Optional[NDArray[np.float32]] = None
    distance_to_next: Optional[float] = None

class SentenceProcessor:
    def __init__(self, buffer_size: int = 1):
        self.buffer_size = buffer_size

    def combine_sentences(self, sentences: List[SentenceData]) -> List[SentenceData]:
        """
        Combine sentences based on buffer size.

        Args:
            sentences: List of SentenceData objects.

        Returns:
            List of SentenceData objects with combined sentences.
        """
        for i, sentence in enumerate(sentences):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            combined = [s.sentence for s in sentences[start:end]]
            sentence.combined_sentence = " ".join(combined)
        return sentences

class DistanceCalculator:
    @staticmethod
    def calculate_cosine_distances(sentences: List[SentenceData]) -> Tuple[NDArray[np.float32], List[SentenceData]]:
        """
        Calculate cosine distances between sentences.

        Args:
            sentences: List of SentenceData objects with embeddings.

        Returns:
            Tuple of distances array and updated SentenceData objects.
        """
        embeddings = np.array([s.combined_sentence_embedding for s in sentences if s.combined_sentence_embedding is not None])
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        distances = 1 - similarities.diagonal()

        for i, distance in enumerate(distances):
            sentences[i].distance_to_next = float(distance)

        return distances, sentences

class ThresholdCalculator:
    @staticmethod
    def calculate_threshold(
        distances: NDArray[np.float32],
        threshold_type: BreakpointThresholdType,
        threshold_amount: float
    ) -> float:
        """
        Calculate the threshold based on the specified method.

        Args:
            distances: Array of distances.
            threshold_type: Type of threshold calculation.
            threshold_amount: Amount to use in threshold calculation.

        Returns:
            Calculated threshold value.
        """
        if threshold_type == "percentile":
            return float(np.percentile(distances, threshold_amount))
        elif threshold_type == "standard_deviation":
            return float(np.mean(distances) + threshold_amount * np.std(distances))
        elif threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1
            return float(np.mean(distances) + threshold_amount * iqr)
        elif threshold_type == "gradient":
            distance_gradient = np.gradient(distances)
            return float(np.percentile(distance_gradient, threshold_amount))
        else:
            raise ValueError(f"Unsupported threshold type: {threshold_type}")

    @staticmethod
    def threshold_from_clusters(distances: NDArray[np.float32], number_of_chunks: int) -> float:
        """
        Calculate the threshold based on the number of chunks.

        Args:
            distances: Array of distances.
            number_of_chunks: Desired number of chunks.

        Returns:
            Calculated threshold value.
        """
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0
        x = max(min(number_of_chunks, x1), x2)
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)
        return float(np.percentile(distances, y))

class SemanticChunker(BaseDocumentTransformer):
    """
    Split text based on semantic similarity.

    This class implements a method to split text into chunks based on semantic
    similarity, using embeddings and various thresholding techniques.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
    ):
        self.embeddings = embeddings
        self.sentence_processor = SentenceProcessor(buffer_size)
        self.distance_calculator = DistanceCalculator()
        self.threshold_calculator = ThresholdCalculator()
        self._add_start_index = add_start_index
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.breakpoint_threshold_amount = (
            breakpoint_threshold_amount or BREAKPOINT_DEFAULTS[breakpoint_threshold_type]
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split the input text into chunks based on semantic similarity.

        Args:
            text: Input text to be split.

        Returns:
            List of text chunks.
        """
        sentences = re.split(self.sentence_split_regex, text)
        if len(sentences) <= 1:
            return sentences

        sentence_data = [SentenceData(sentence=s, index=i) for i, s in enumerate(sentences)]
        combined_sentences = self.sentence_processor.combine_sentences(sentence_data)
        
        embeddings = self.embeddings.embed_documents([s.combined_sentence for s in combined_sentences if s.combined_sentence])
        for s, e in zip(combined_sentences, embeddings):
            s.combined_sentence_embedding = e

        distances, updated_sentences = self.distance_calculator.calculate_cosine_distances(combined_sentences)
        
        if self.number_of_chunks is not None:
            threshold = self.threshold_calculator.threshold_from_clusters(distances, self.number_of_chunks)
        else:
            threshold = self.threshold_calculator.calculate_threshold(
                distances, self.breakpoint_threshold_type, self.breakpoint_threshold_amount
            )

        breakpoints = np.where(distances > threshold)[0]
        return self._create_chunks(updated_sentences, breakpoints)

    def _create_chunks(self, sentences: List[SentenceData], breakpoints: NDArray[np.int64]) -> List[str]:
        """
        Create text chunks based on calculated breakpoints.

        Args:
            sentences: List of SentenceData objects.
            breakpoints: Array of breakpoint indices.

        Returns:
            List of text chunks.
        """
        chunks = []
        start = 0
        for end in breakpoints:
            chunk = " ".join(s.sentence for s in sentences[start:end+1])
            chunks.append(chunk)
            start = end + 1
        
        if start < len(sentences):
            chunk = " ".join(s.sentence for s in sentences[start:])
            chunks.append(chunk)
        
        return chunks

    def create_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """
        Create Document objects from a list of texts.

        Args:
            texts: List of input texts.
            metadatas: Optional list of metadata dictionaries.

        Returns:
            List of Document objects.
        """
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for text, metadata in zip(texts, _metadatas):
            index = -1
            for chunk in self.split_text(text):
                doc_metadata = copy.deepcopy(metadata)
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    doc_metadata["start_index"] = index
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Split a collection of Document objects.

        Args:
            documents: Iterable of Document objects.

        Returns:
            List of split Document objects.
        """
        texts = []
        metadatas = []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """
        Transform a sequence of documents by splitting them.

        Args:
            documents: Sequence of Document objects.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence of transformed Document objects.
        """
        return self.split_documents(list(documents))