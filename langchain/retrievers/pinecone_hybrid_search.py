"""Taken from: https://www.pinecone.io/learn/hybrid-search-intro/"""
from collections import Counter
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Extra

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document


def build_dict(input_batch: List[List[int]]) -> List[Dict]:
    # store a batch of sparse embeddings
    sparse_emb = []
    # iterate through input batch
    for token_ids in input_batch:
        indices = []
        values = []
        # convert the input_ids list to a dictionary of key to frequency values
        d = dict(Counter(token_ids))
        for idx in d:
            indices.append(idx)
            values.append(d[idx])
        sparse_emb.append({"indices": indices, "values": values})
    # return sparse_emb list
    return sparse_emb


def create_index(
    contexts: List[str], index: Any, embeddings: Embeddings, tokenizer: Any
) -> None:
    batch_size = 32
    _iterator = range(0, len(contexts), batch_size)
    try:
        from tqdm.auto import tqdm

        _iterator = tqdm(_iterator)
    except ImportError:
        pass

    for i in _iterator:
        # find end of batch
        i_end = min(i + batch_size, len(contexts))
        # extract batch
        context_batch = contexts[i:i_end]
        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]
        # add context passages as metadata
        meta = [{"context": context} for context in context_batch]
        # create dense vectors
        dense_embeds = embeddings.embed_documents(context_batch)
        # create sparse vectors
        sparse_embeds = generate_sparse_vectors(context_batch, tokenizer)
        for s in sparse_embeds:
            s["values"] = [float(s1) for s1 in s["values"]]

        vectors = []
        # loop through the data and create dictionaries for upserts
        for _id, sparse, dense, metadata in zip(ids, sparse_embeds, dense_embeds, meta):
            vectors.append(
                {
                    "id": _id,
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata,
                }
            )

        # upload the documents to the new hybrid index
        index.upsert(vectors)


def generate_sparse_vectors(context_batch: List[str], tokenizer: Any) -> List[Dict]:
    # create batch of input_ids
    inputs = tokenizer(
        context_batch,
        padding=True,
        truncation=True,
        max_length=512,  # special_tokens=False
    )["input_ids"]
    # create sparse dictionaries
    sparse_embeds = build_dict(inputs)
    return sparse_embeds


def hybrid_scale(
    dense: List[float], sparse: Dict, alpha: float
) -> Tuple[List[float], Dict]:
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


class PineconeHybridSearchRetriever(BaseRetriever, BaseModel):
    embeddings: Embeddings
    index: Any
    tokenizer: Any
    top_k: int = 4
    alpha: float = 0.5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_texts(self, texts: List[str]) -> None:
        create_index(texts, self.index, self.embeddings, self.tokenizer)

    def get_relevant_documents(self, query: str) -> List[Document]:
        sparse_vec = generate_sparse_vectors([query], self.tokenizer)[0]
        # convert the question into a dense vector
        dense_vec = self.embeddings.embed_query(query)
        # scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        # query pinecone with the query parameters
        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
        )
        final_result = []
        for res in result["matches"]:
            final_result.append(Document(page_content=res["metadata"]["context"]))
        # return search results as json
        return final_result

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
