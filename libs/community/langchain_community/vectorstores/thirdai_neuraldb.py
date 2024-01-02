import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.vectorstores import VectorStore


class NeuralDBVectorStore(VectorStore):
    """Vectorstore that uses ThirdAI's NeuralDB."""

    db: Any = None  #: :meta private:
    """NeuralDB instance"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        underscore_attrs_are_private = True

    @staticmethod
    def _verify_thirdai_library(thirdai_key: Optional[str] = None):
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
        **model_kwargs,
    ):
        """
        Create a NeuralDBVectorStore from scratch.

        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import NeuralDBVectorStore

                vectorstore = NeuralDBVectorStore.from_scratch(
                    thirdai_key="your-thirdai-key",
                )

                vectorstore.insert([
                    "/path/to/doc.pdf",
                    "/path/to/doc.docx",
                    "/path/to/doc.csv",
                ])

                documents = vectorstore.similarity_search("AI-driven music therapy")
        """
        NeuralDBVectorStore._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb

        return cls(db=ndb.NeuralDB(**model_kwargs))

    @classmethod
    def from_bazaar(
        cls,
        base: str,
        bazaar_cache: Optional[str] = None,
        thirdai_key: Optional[str] = None,
    ):
        """
        Create a NeuralDBVectorStore with a base model from the ThirdAI
        model bazaar.

        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import NeuralDBVectorStore

                vectorstore = NeuralDBVectorStore.from_bazaar(
                    base="General QnA",
                    thirdai_key="your-thirdai-key",
                )

                vectorstore.insert([
                    "/path/to/doc.pdf",
                    "/path/to/doc.docx",
                    "/path/to/doc.csv",
                ])

                documents = vectorstore.similarity_search("AI-driven music therapy")
        """
        NeuralDBVectorStore._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb

        cache = bazaar_cache or str(Path(os.getcwd()) / "model_bazaar")
        if not os.path.exists(cache):
            os.mkdir(cache)
        model_bazaar = ndb.Bazaar(cache)
        model_bazaar.fetch()
        return cls(db=model_bazaar.get_model(base))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        thirdai_key: Optional[str] = None,
    ):
        """
        Create a NeuralDBVectorStore with a base model from a saved checkpoint

        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import NeuralDBVectorStore

                vectorstore = NeuralDBVectorStore.from_checkpoint(
                    checkpoint="/path/to/checkpoint.ndb",
                    thirdai_key="your-thirdai-key",
                )

                vectorstore.insert([
                    "/path/to/doc.pdf",
                    "/path/to/doc.docx",
                    "/path/to/doc.csv",
                ])

                documents = vectorstore.similarity_search("AI-driven music therapy")
        """
        NeuralDBVectorStore._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb

        return cls(db=ndb.NeuralDB.from_checkpoint(checkpoint))

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "NeuralDBVectorStore":
        """Return VectorStore initialized from texts and embeddings."""
        model_kwargs = {}
        if "thirdai_key" in kwargs:
            model_kwargs["thirdai_key"] = kwargs["thirdai_key"]
            del kwargs["thirdai_key"]
        vectorstore = cls.from_scratch(**model_kwargs)
        vectorstore.add_texts(texts, metadatas, **kwargs)
        return vectorstore

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        import pandas as pd
        from thirdai import neural_db as ndb

        df = pd.DataFrame({"texts": texts})
        if metadatas:
            df = pd.concat([df, pd.DataFrame.from_records(metadatas)], axis=1)
        temp = tempfile.NamedTemporaryFile("w", delete=False, delete_on_close=False)
        df.to_csv(temp)
        source_id = self.insert([ndb.CSV(temp.name)], **kwargs)[0]
        offset = self.db._savable_state.documents.get_source_by_id(source_id)[1]
        return [str(offset + i) for i in range(len(texts))]

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
        **kwargs,
    ):
        """Inserts files / document sources into the vectorstore.

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

    def _preprocess_sources(self, sources):
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

    def upvote(self, query: str, document_id: Union[int, str]):
        """The vectorstore upweights the score of a document for a specific query.
        This is useful for fine-tuning the vectorstore to user behavior.

        Args:
            query: text to associate with `document_id`
            document_id: id of the document to associate query with.
        """
        self.db.text_to_result(query, int(document_id))

    def upvote_batch(self, query_id_pairs: List[Tuple[str, int]]):
        """Given a batch of (query, document id) pairs, the vectorstore upweights
        the scores of the document for the corresponding queries.
        This is useful for fine-tuning the vectorstore to user behavior.

        Args:
            query_id_pairs: list of (query, document id) pairs. For each pair in
            this list, the model will upweight the document id for the query.
        """
        self.db.text_to_result_batch(
            [(query, int(doc_id)) for query, doc_id in query_id_pairs]
        )

    def associate(self, source: str, target: str):
        """The vectorstore associates a source phrase with a target phrase.
        When the vectorstore sees the source phrase, it will also consider results
        that are relevant to the target phrase.

        Args:
            source: text to associate to `target`.
            target: text to associate `source` to.
        """
        self.db.associate(source, target)

    def associate_batch(self, text_pairs: List[Tuple[str, str]]):
        """Given a batch of (source, target) pairs, the vectorstore associates
        each source phrase with the corresponding target phrase.

        Args:
            text_pairs: list of (source, target) text pairs. For each pair in
            this list, the source will be associated with the target.
        """
        self.db.associate_batch(text_pairs)

    def similarity_search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Document]:
        """Retrieve {k} contexts with for a given query

        Args:
            query: Query to submit to the model
            k: The max number of context results to retrieve. Defaults to 10.
        """
        try:
            references = self.db.search(query=query, top_k=k, **kwargs)
            return [
                Document(
                    page_content=ref.text,
                    metadata={
                        "id": ref.id,
                        "upvote_ids": ref.upvote_ids,
                        "text": ref.text,
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

    def save(self, path: str):
        """Saves a NeuralDB instance to disk. Can be loaded into memory by
        calling NeuralDB.from_checkpoint(path)

        Args:
            path: path on disk to save the NeuralDB instance to.
        """
        self.db.save(path)
