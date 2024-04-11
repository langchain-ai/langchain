# System imports
from itertools import cycle, islice, repeat
import logging
from typing import Any, Iterable, List, Optional, Tuple, Type

# Third-party imports
import numpy as np

# Local imports
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

ENGINE = "HNSW"
METRIC = "L2"
DESCRIPTOR_SET = "langchain"

class ApertureDB(VectorStore):
    def __init__(self, 
                 embeddings:Embeddings, 
                 descriptor_set:str=DESCRIPTOR_SET, 
                 dimensions:Optional[int]=None, 
                 engine:str=ENGINE,
                 metric:str=METRIC,
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        self.embeddings = embeddings
        self.descriptor_set = descriptor_set
        self.dimensions = dimensions
        self.engine = engine
        self.metric = metric

        try:
            import aperturedb
        except ImportError:
            self.logger.exception("ApertureDB is not installed. Please install it using 'pip install aperturedb'")
            raise

        self.connection = aperturedb.Util.create_connector()
        try:
            self.connection.status()
            assert self.connection.last_query_ok(), self.connection.last_error()
        except Exception:
            self.logger.exception(f"Failed to connect to ApertureDB")
            raise
        self.utils = aperturedb.Utils.Utils(self.connection)

        self._find_or_add_descriptor_set(descriptor_set, dimensions)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Selects the relevance score function based on the metric.
        
        This method is defined in order to make the base class implementation of similarity_search_with_relevance_scores work.
        """
        if self.metric == "L2":
            return VectorStore._euclidean_relevance_score_fn
        elif self.metric == "IP":
            return VectorStore._max_inner_product_relevance_score_fn
        elif self.metric == "CS":
            return VectorStore._cosine_relevance_score_fn
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _find_or_add_descriptor_set(self, descriptor_set:str, dimensions:Optional[int]=None):
        """Checks if the descriptor set exists, if not, creates it"""
        r,b = self.connection.query([{ "FindDescriptorSet": { "with_name": descriptor_set } }])
        assert self.connection.last_query_ok(), self.connection.last_error()
        # TODO: Could check that dimensions, engine and metric are the same
        if r[0]["FindDescriptorSet"]["returned"] == 0:
            if self.dimensions is None:
                self.dimensions = len(self.embeddings.embed_query("test"))
            success = self.utils.add_descriptor_set(descriptor_set, self.dimensions, engine=self.engine, metric=[self.metric])
            assert success, self.connection.last_error()
            self.utils.create_entity_index("_Descriptor", "_uniqueid")

    def add_texts(self, texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if metadatas is not None:
            assert len(texts) == len(metadatas), "Length of texts and metadatas should be the same"

        embeddings = self.embeddings.embed_documents(texts)
        if metadatas is None:
            metadatas = repeat({})
        refs = cycle(range(1,100000))
        data = zip(texts, embeddings, metadatas, refs) 

        commands = []
        blobs = []        
        for text, embedding, metadata, ref in data:
            commands.append({
                "AddDescriptor": {
                    "set": self.descriptor_set,
                    "_ref": ref,
                    "results": {
                       "list": ["_uniqueid"]
                    }
                }
            })
            blobs.append(embedding.tobytes())                
        
            commands.append({
                "AddBlob": {
                    "_ref": ref,
                    "properties": metadata,
                    "connect": {
                        "class": "has_descriptor",
                        "ref": ref
                    }
                }
            })
            blobs.append(text.encode())

        status, responses, blobs = aperturedb.ParallelQuery.execute_batch(q=commands, blobs=blobs, db=self.connection, commands_per_query=2, blobs_per_query=2)
        assert status == 0, responses
        unique_ids = [r["AddDescriptor"]["results"]["_uniqueid"] for r in responses[::2]]
        return unique_ids
    
    def delete(self, ids: List[str]) -> Optional[bool]:
        refs = cycle(range(1,100000))
        commands = []
        for id, ref in zip(ids, refs):
            commands.extend([{ 
                "FindDescriptor": {
                    "_ref": ref,
                    "set": self.descriptor_set,
                    "unique": True,
                    "constraints": {
                        "_uniqueid": [ "==", id ]
                    }
                }
            }, {
                "DeleteDescriptor": {
                    "_ref": ref,
                    "id": id
                }
            }, {
                "DeleteBlob": {
                    "is_connected_to": {
                        "ref": ref
                    }
                }
            }])

        status, responses, blobs = aperturedb.ParallelQuery.execute_batch(q=commands, blobs=blobs, db=self.connection, commands_per_query=3, blobs_per_query=0)
        assert status == 0, responses
        results = [r["DeleteDescriptor"]["results"]["count"] == 1 for r in responses[1::3]]
        return results
    
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding = self.embeddings.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_with_score(
        self, query: str, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embeddings.embed_query(query)
        self._similarity_search_with_score_by_vector(embedding, **kwargs)

    def _similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        query = [ {
            "FindDescriptor": {
                "set": self.descriptor_set,
                "k_neighbors": k,
                "_ref": 1,
                "results": { "list": ["_uniqueid"] },
                "distances": True,
            }
        }, {
            "FindBlob": {
                "is_connected_to": {
                    "ref": 1
                }
                "results": { "all_properties": True },
                "blobs": True,
            }
        }]   
        blobs_in = [np.array(embedding).tobytes()]
        responses, blobs_out = self.connection.query(query, blobs_in) 
        assert self.connection.last_query_ok(), self.connection.last_error()
        results = []
        for i in range(0, len(responses), 2):
            unique_ids = responses[i]["FindDescriptor"]["results"]["_uniqueid"]
            metadata = responses[i+1]["FindBlob"]["entities"][0]
            distance = metadata["_distance"]
            blob_index = metadata["_blob_index"]
            text = blobs_out[blob_index].decode()
            # consider filtering internal properties out
            results.append((Document(page_content=text, _uniqueid=unique_ids, **metadata), distance))
        return results
    
    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        results = self._similarity_search_with_score_by_vector(embedding, k)
        return [document for document, score in results]    

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls: Type[ApertureDB],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> ApertureDB:
        store = cls(embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store