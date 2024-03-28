"""Experimental **text splitter** based on semantic similarity."""
import copy
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from langchain_community.utils.math import (
    cosine_similarity,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings


def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    """Combine sentences based on buffer size.

    Args:
        sentences: List of sentences to combine.
        buffer_size: Number of sentences to combine. Defaults to 1.

    Returns:
        List of sentences with combined sentences.
    """

    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative
            # (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    """Calculate cosine distances between sentences.

    Args:
        sentences: List of sentences to calculate distances for.

    Returns:
        Tuple of distances and sentences.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


def split_string(input_str: str, limit: int, sep: str = " ") -> List[str]:
    # Split the input string into words
    words = input_str.split()

    # Check if any single word exceeds the limit, which is not allowed
    if max(map(len, words)) > limit:
        raise ValueError(
            "A single word exceeds the limit, making splitting impossible."
        )

    # Create the result list, the current part being constructed, and remaining words
    res = []  # List to store the final result of split parts
    part = words[0]  # Start the first part with the first word
    others = words[1:]  # Remaining words to process

    # Iterate through the remaining words
    for word in others:
        # Check if adding the next word exceeds the limit for the current part
        if len(sep) + len(word) > limit - len(part):
            # If it does, add the current part to the result list and start a new part
            res.append(part)
            part = word
        else:
            # Otherwise, add the word to the current part
            part += sep + word

    # After the loop, add the last part to the result list if it's not empty
    if part:
        res.append(part)

    return res


def add_chunk(
    sentences: List[dict],
    start_index: int,
    end_index: int,
    chunks: List[str],
    max_chunk_size: Optional[int] = None,
):
    """Adds sentences as a chunk if total length does not exceed max_chunk_size."""
    if not max_chunk_size:
        combined_text = " ".join(
            [d["sentence"] for d in sentences[start_index:end_index]]
        )
        chunks.append(combined_text)
    else:
        group = []
        chunk_size = 0
        for i in range(start_index, end_index):
            sentence_length = len(sentences[i]["sentence"])
            if chunk_size + sentence_length > max_chunk_size:
                if group:  # Ensures that the group is not empty
                    chunks.append(" ".join([d["sentence"] for d in group]))
                group = [sentences[i]]  # Start a new group with the current sentence
                chunk_size = sentence_length  # Resets chunk size
            else:
                group.append(sentences[i])
                chunk_size += sentence_length

        # Adds the last group if it is not empty
        if group:
            chunks.append(" ".join([d["sentence"] for d in group]))


BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile"]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
}


class SemanticChunker(BaseDocumentTransformer):
    """Split the text based on semantic similarity.

    Taken from Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

    All credits to him.

    At a high level, this splits into sentences, then groups into groups of 3
    sentences, and then merges one that are similar in the embedding space.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ):
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.max_chunk_size = max_chunk_size
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount

    def _calculate_breakpoint_threshold(self, distances: List[float]) -> float:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            )
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(distances) + self.breakpoint_threshold_amount * iqr
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """
        Calculate the threshold based on the number of chunks.
        Inverse of percentile method.
        """
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        # Linear interpolation formula
        y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""

        if self.max_chunk_size:
            # Preparing a new list to store the results
            new_single_sentences_list = []

            for sentence in single_sentences_list:
                # Check whether the sentence exceeds the maximum authorised size
                if len(sentence) >= self.max_chunk_size:
                    # Dividing the sentence into sub-parts
                    sentences = split_string(sentence, self.max_chunk_size, " ")
                    # Extension of the new list by sub-parts
                    new_single_sentences_list.extend(sentences)
                else:
                    # Add the original sentence if it does not exceed the maximum size
                    new_single_sentences_list.append(sentence)

            # Replacing the original list with the new one
            single_sentences_list = new_single_sentences_list

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return calculate_cosine_distances(sentences)

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        # Splitting the essay on '.', '?', and '!'
        single_sentences_list = re.split(r"(?<=[.?!])\s+", text)

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(
                distances
            )

        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
        ]

        chunks: List[str] = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            add_chunk(
                sentences, start_index, end_index + 1, chunks, self.max_chunk_size
            )

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            add_chunk(
                sentences, start_index, len(sentences), chunks, self.max_chunk_size
            )

        return chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))
