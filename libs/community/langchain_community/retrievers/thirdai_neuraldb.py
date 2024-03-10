from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class NeuralDBRetriever(BaseRetriever):
    """Document retriever that uses ThirdAI's NeuralDB."""

    thirdai_key: SecretStr
    """ThirdAI API Key"""

    db: Any = None  #: :meta private:
    """NeuralDB instance"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        underscore_attrs_are_private = True

    @staticmethod
    def _verify_thirdai_library(thirdai_key: Optional[str] = None) -> None:
        try:
            from thirdai import licensing

            importlib.util.find_spec("thirdai.neural_db")

            licensing.activate(thirdai_key or os.getenv("THIRDAI_KEY"))
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import thirdai python package and neuraldb dependencies. "
                "Please install it with `pip install thirdai[neural_db]`."
            )

    @classmethod
    def from_scratch(
        cls,
        thirdai_key: Optional[str] = None,
        **model_kwargs: dict,
    ) -> NeuralDBRetriever:
        """
        Create a NeuralDBRetriever from scratch.

        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.retrievers import NeuralDBRetriever

                retriever = NeuralDBRetriever.from_scratch(
                    thirdai_key="your-thirdai-key",
                )

                retriever.insert([
                    "/path/to/doc.pdf",
                    "/path/to/doc.docx",
                    "/path/to/doc.csv",
                ])

                documents = retriever.get_relevant_documents("AI-driven music therapy")
        """
        NeuralDBRetriever._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb

        return cls(thirdai_key=thirdai_key, db=ndb.NeuralDB(**model_kwargs))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        thirdai_key: Optional[str] = None,
    ) -> NeuralDBRetriever:
        """
        Create a NeuralDBRetriever with a base model from a saved checkpoint

        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.retrievers import NeuralDBRetriever

                retriever = NeuralDBRetriever.from_checkpoint(
                    checkpoint="/path/to/checkpoint.ndb",
                    thirdai_key="your-thirdai-key",
                )

                retriever.insert([
                    "/path/to/doc.pdf",
                    "/path/to/doc.docx",
                    "/path/to/doc.csv",
                ])

                documents = retriever.get_relevant_documents("AI-driven music therapy")
        """
        NeuralDBRetriever._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb

        return cls(thirdai_key=thirdai_key, db=ndb.NeuralDB.from_checkpoint(checkpoint))

    @root_validator()
    def validate_environments(cls, values: Dict) -> Dict:
        """Validate ThirdAI environment variables."""
        values["thirdai_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "thirdai_key",
                "THIRDAI_KEY",
            )
        )
        return values

    def insert(
        self,
        sources: List[Any],
        train: bool = True,
        fast_mode: bool = True,
        **kwargs: dict,
    ) -> None:
        """Inserts files / document sources into the retriever.

        Args:
            train: When True this means that the underlying model in the
            NeuralDB will undergo unsupervised pretraining on the inserted files.
            Defaults to True.
            fast_mode: Much faster insertion with a slight drop in performance.
            Defaults to True.
        """
        sources = self._preprocess_sources(sources)
        self.db.insert(
            sources=sources,
            train=train,
            fast_approximation=fast_mode,
            **kwargs,
        )

    def _preprocess_sources(self, sources: list) -> list:
        """Checks if the provided sources are string paths. If they are, convert
        to NeuralDB document objects.

        Args:
            sources: list of either string paths to PDF, DOCX or CSV files, or
            NeuralDB document objects.
        """
        from thirdai import neural_db as ndb

        if not sources:
            return sources
        preprocessed_sources = []
        for doc in sources:
            if not isinstance(doc, str):
                preprocessed_sources.append(doc)
            else:
                if doc.lower().endswith(".pdf"):
                    preprocessed_sources.append(ndb.PDF(doc))
                elif doc.lower().endswith(".docx"):
                    preprocessed_sources.append(ndb.DOCX(doc))
                elif doc.lower().endswith(".csv"):
                    preprocessed_sources.append(ndb.CSV(doc))
                else:
                    raise RuntimeError(
                        f"Could not automatically load {doc}. Only files "
                        "with .pdf, .docx, or .csv extensions can be loaded "
                        "automatically. For other formats, please use the "
                        "appropriate document object from the ThirdAI library."
                    )
        return preprocessed_sources

    def upvote(self, query: str, document_id: int) -> None:
        """The retriever upweights the score of a document for a specific query.
        This is useful for fine-tuning the retriever to user behavior.

        Args:
            query: text to associate with `document_id`
            document_id: id of the document to associate query with.
        """
        self.db.text_to_result(query, document_id)

    def upvote_batch(self, query_id_pairs: List[Tuple[str, int]]) -> None:
        """Given a batch of (query, document id) pairs, the retriever upweights
        the scores of the document for the corresponding queries.
        This is useful for fine-tuning the retriever to user behavior.

        Args:
            query_id_pairs: list of (query, document id) pairs. For each pair in
            this list, the model will upweight the document id for the query.
        """
        self.db.text_to_result_batch(query_id_pairs)

    def associate(self, source: str, target: str) -> None:
        """The retriever associates a source phrase with a target phrase.
        When the retriever sees the source phrase, it will also consider results
        that are relevant to the target phrase.

        Args:
            source: text to associate to `target`.
            target: text to associate `source` to.
        """
        self.db.associate(source, target)

    def associate_batch(self, text_pairs: List[Tuple[str, str]]) -> None:
        """Given a batch of (source, target) pairs, the retriever associates
        each source phrase with the corresponding target phrase.

        Args:
            text_pairs: list of (source, target) text pairs. For each pair in
            this list, the source will be associated with the target.
        """
        self.db.associate_batch(text_pairs)

    def _get_relevant_documents(
        self, query: str, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Retrieve {top_k} contexts with your retriever for a given query

        Args:
            query: Query to submit to the model
            top_k: The max number of context results to retrieve. Defaults to 10.
        """
        try:
            if "top_k" not in kwargs:
                kwargs["top_k"] = 10
            references = self.db.search(query=query, **kwargs)
            return [
                Document(
                    page_content=ref.text,
                    metadata={
                        "id": ref.id,
                        "upvote_ids": ref.upvote_ids,
                        "source": ref.source,
                        "metadata": ref.metadata,
                        "score": ref.score,
                        "context": ref.context(1),
                    },
                )
                for ref in references
            ]
        except Exception as e:
            raise ValueError(f"Error while retrieving documents: {e}") from e

    def save(self, path: str) -> None:
        """Saves a NeuralDB instance to disk. Can be loaded into memory by
        calling NeuralDB.from_checkpoint(path)

        Args:
            path: path on disk to save the NeuralDB instance to.
        """
        self.db.save(path)
