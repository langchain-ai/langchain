import importlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict


class NeuralDBVectorStore(VectorStore):
    """Vectorstore that uses ThirdAI's NeuralDB.

    To use, you should have the ``thirdai[neural_db]`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import NeuralDBVectorStore
            from thirdai import neural_db as ndb

            db = ndb.NeuralDB()
            vectorstore = NeuralDBVectorStore(db=db)
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    db: Any = None  #: :meta private:
    """NeuralDB instance"""

    model_config = ConfigDict(
        extra="forbid",
    )

    @staticmethod
    def _verify_thirdai_library(thirdai_key: Optional[str] = None):  # type: ignore[no-untyped-def]
        try:
            from thirdai import licensing

            importlib.util.find_spec("thirdai.neural_db")

            licensing.activate(thirdai_key or os.getenv("THIRDAI_KEY"))
        except ImportError:
            raise ImportError(
                "Could not import thirdai python package and neuraldb dependencies. "
                "Please install it with `pip install thirdai[neural_db]`."
            )

    @classmethod
    def from_scratch(  # type: ignore[no-untyped-def, no-untyped-def]
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

        return cls(db=ndb.NeuralDB(**model_kwargs))  # type: ignore[call-arg]

    @classmethod
    def from_checkpoint(  # type: ignore[no-untyped-def]
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

        return cls(db=ndb.NeuralDB.from_checkpoint(checkpoint))  # type: ignore[call-arg]

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
        temp = tempfile.NamedTemporaryFile("w", delete=False, delete_on_close=False)  # type: ignore[call-overload]
        df.to_csv(temp)
        source_id = self.insert([ndb.CSV(temp.name)], **kwargs)[0]
        offset = self.db._savable_state.documents.get_source_by_id(source_id)[1]
        return [str(offset + i) for i in range(len(texts))]  # type: ignore[arg-type]

    def insert(  # type: ignore[no-untyped-def, no-untyped-def]
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

    def _preprocess_sources(self, sources):  # type: ignore[no-untyped-def]
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

    def upvote(self, query: str, document_id: Union[int, str]):  # type: ignore[no-untyped-def]
        """The vectorstore upweights the score of a document for a specific query.
        This is useful for fine-tuning the vectorstore to user behavior.

        Args:
            query: text to associate with `document_id`
            document_id: id of the document to associate query with.
        """
        self.db.text_to_result(query, int(document_id))

    def upvote_batch(self, query_id_pairs: List[Tuple[str, int]]):  # type: ignore[no-untyped-def]
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

    def associate(self, source: str, target: str):  # type: ignore[no-untyped-def]
        """The vectorstore associates a source phrase with a target phrase.
        When the vectorstore sees the source phrase, it will also consider results
        that are relevant to the target phrase.

        Args:
            source: text to associate to `target`.
            target: text to associate `source` to.
        """
        self.db.associate(source, target)

    def associate_batch(self, text_pairs: List[Tuple[str, str]]):  # type: ignore[no-untyped-def]
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

    def save(self, path: str):  # type: ignore[no-untyped-def]
        """Saves a NeuralDB instance to disk. Can be loaded into memory by
        calling NeuralDB.from_checkpoint(path)

        Args:
            path: path on disk to save the NeuralDB instance to.
        """
        self.db.save(path)


class NeuralDBClientVectorStore(VectorStore):
    """Vectorstore that uses ThirdAI's NeuralDB Enterprise Python Client for NeuralDBs.

    To use, you should have the ``thirdai[neural_db]`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.vectorstores import NeuralDBClientVectorStore
            from thirdai.neural_db import ModelBazaar, NeuralDBClient

            bazaar = ModelBazaar(base_url="http://{NEURAL_DB_ENTERPRISE_IP}/api/")
            bazaar.log_in(email="user@thirdai.com", password="1234")

            ndb_client = NeuralDBClient(
                deployment_identifier="user/model-0:user/deployment-0",
                base_url="http://{NEURAL_DB_ENTERPRISE_IP}/api/",
                bazaar=bazaar
            )
            vectorstore = NeuralDBClientVectorStore(db=ndb_client)
            retriever = vectorstore.as_retriever(search_kwargs={'k':5})

    """

    def __init__(self, db: Any) -> None:
        self.db = db

    db: Any = None  #: :meta private:
    """NeuralDB Client instance"""

    model_config = ConfigDict(
        extra="forbid",
    )

    def similarity_search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[Document]:
        """Retrieve {k} contexts with for a given query

        Args:
            query: Query to submit to the model
            k: The max number of context results to retrieve. Defaults to 10.
        """
        try:
            references = self.db.search(query=query, top_k=k, **kwargs)["references"]
            return [
                Document(
                    page_content=ref["text"],
                    metadata={
                        "id": ref["id"],
                        "source": ref["source"],
                        "metadata": ref["metadata"],
                        "score": ref["source"],
                        "context": ref["context"],
                    },
                )
                for ref in references
            ]
        except Exception as e:
            raise ValueError(f"Error while retrieving documents: {e}") from e

    def insert(self, documents: List[Dict[str, Any]]):  # type: ignore[no-untyped-def, no-untyped-def]
        """
        Inserts documents into the VectorStore and return the corresponding Sources.

        Args:
            documents (List[Dict[str, Any]]): A list of dictionaries that
            represent documents to be inserted to the VectorStores.
            The document dictionaries must be in the following format:
            {"document_type": "DOCUMENT_TYPE", **kwargs} where "DOCUMENT_TYPE"
            is one of the following:
            "PDF", "CSV", "DOCX", "URL", "SentenceLevelPDF", "SentenceLevelDOCX",
            "Unstructured", "InMemoryText".
            The kwargs for each document type are shown below:

            class PDF(Document):
                document_type: Literal["PDF"]
                path: str
                metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False
                version: str = "v1"
                chunk_size: int = 100
                stride: int = 40
                emphasize_first_words: int = 0
                ignore_header_footer: bool = True
                ignore_nonstandard_orientation: bool = True

            class CSV(Document):
                document_type: Literal["CSV"]
                path: str
                id_column: Optional[str] = None
                strong_columns: Optional[List[str]] = None
                weak_columns: Optional[List[str]] = None
                reference_columns: Optional[List[str]] = None
                save_extra_info: bool = True
                metadata: Optional[dict[str, Any]] = None
                has_offset: bool = False
                on_disk: bool = False

            class DOCX(Document):
                document_type: Literal["DOCX"]
                path: str
                metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False

            class URL(Document):
                document_type: Literal["URL"]
                url: str
                save_extra_info: bool = True
                title_is_strong: bool = False
                metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False

            class SentenceLevelPDF(Document):
                document_type: Literal["SentenceLevelPDF"]
                path: str
                metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False

            class SentenceLevelDOCX(Document):
                document_type: Literal["SentenceLevelDOCX"]
                path: str
                metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False

            class Unstructured(Document):
                document_type: Literal["Unstructured"]
                path: str
                save_extra_info: bool = True
                metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False

            class InMemoryText(Document):
                document_type: Literal["InMemoryText"]
                name: str
                texts: list[str]
                metadatas: Optional[list[dict[str, Any]]] = None
                global_metadata: Optional[dict[str, Any]] = None
                on_disk: bool = False

            For Document types with the arg "path", ensure that
            the path exists on your local machine.
        """
        return self.db.insert(documents)

    def remove_documents(self, source_ids: List[str]):  # type: ignore[no-untyped-def]
        """
        Deletes documents from the VectorStore using source ids.

        Args:
            files (List[str]): A list of source ids to delete from the VectorStore.
        """

        self.db.delete(source_ids)
