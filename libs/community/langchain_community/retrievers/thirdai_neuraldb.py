import os
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, SecretStr, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

# TODO How do I add thirdai dependencies?

class NeuralDBRetriever(BaseRetriever):
    """Document retriever that uses ThirdAI's NeuralDB.
    """

    db: Any = None  #: :meta private:
    """NeuralDB instance"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        underscore_attrs_are_private = True
    
    @staticmethod
    def _verify_thirdai_library(thirdai_key: Optional[str] = None):
        try:
            from thirdai import licensing, neural_db
            licensing.activate(thirdai_key or os.getenv("THIRDAI_KEY"))
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import thirdai python package and neuraldb dependencies. "
                "Please install it with `pip install thirdai[neural_db]`."
            )
        
    @classmethod
    def from_bazaar(
        cls,
        base: str,
        bazaar_cache: Optional[str] = None,
        thirdai_key: Optional[str] = None,
    ):
        """
        Create a NeuralDBRetriever with a base model from the ThirdAI
        model bazaar.
        
        To use, set the ``THIRDAI_KEY`` environment variable with your ThirdAI 
        API key, or pass ``thirdai_key`` as a named parameter.

        Example:
            .. code-block:: python

                from langchain_community.retrievers import NeuralDBRetriever

                retriever = NeuralDBRetriever.from_bazaar(
                    base="General QnA",
                    thirdai_key="your-thirdai-key",
                )
                
                retriever.insert(["/path/to/doc.pdf", "/path/to/doc.docx", "/path/to/doc.csv"])

                documents = retriever.get_relevant_documents("AI-driven music therapy")
        """
        NeuralDBRetriever._verify_thirdai_library(thirdai_key)
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
                
                retriever.insert(["/path/to/doc.pdf", "/path/to/doc.docx", "/path/to/doc.csv"])

                documents = retriever.get_relevant_documents("AI-driven music therapy")
        """
        NeuralDBRetriever._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb
        return cls(db=ndb.NeuralDB.from_checkpoint(checkpoint))
    
    @classmethod
    def from_scratch(
        cls,
        thirdai_key: Optional[str] = None,
        **model_kwargs,
    ):
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
                
                retriever.insert(["/path/to/doc.pdf", "/path/to/doc.docx", "/path/to/doc.csv"])

                documents = retriever.get_relevant_documents("AI-driven music therapy")
        """
        NeuralDBRetriever._verify_thirdai_library(thirdai_key)
        from thirdai import neural_db as ndb
        return cls(db=ndb.NeuralDB(**model_kwargs))

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
        sources = self._preprocess_sources(sources)
        self.db.insert(
            sources=sources,
            train=train,
            fast_approximation=fast_mode,
            **kwargs,
        )
    
    def _preprocess_sources(self, sources):
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

    def upvote(self, text: str, document_id: int):
        self.db.text_to_result(text, document_id)

    def upvote_batch(self, text_id_pairs: List[Tuple[str, int]]):
        self.db.text_to_result_batch(text_id_pairs)
    
    def associate(self, source: str, target: str):
        self.db.associate(source, target)
    
    def associate_batch(self, text_pairs: List[Tuple[str, str]]):
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
            references = self.db.search(
                query=query,
                top_k=kwargs.get("top_k", 10),
                **kwargs)
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
        self.db.save(path)
        
