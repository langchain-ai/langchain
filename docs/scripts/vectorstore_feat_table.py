import inspect
import sys
from pathlib import Path

from langchain_astradb import AstraDBVectorStore
from langchain_chroma import Chroma
from langchain_community import vectorstores
from langchain_core.vectorstores import VectorStore
from langchain_couchbase import CouchbaseSearchVectorStore
from langchain_milvus import Milvus
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_pinecone import PineconeVectorStore
from langchain_postgres import PGVectorStore
from langchain_qdrant import QdrantVectorStore

vectorstore_list = [
    "FAISS",
    "ElasticsearchStore",
    "PGVector",
    "Redis",
    "Clickhouse",
    "InMemoryVectorStore",
]

from_partners = [
    ("PGVectorStore", PGVectorStore),
    ("Chroma", Chroma),
    ("AstraDBVectorStore", AstraDBVectorStore),
    ("QdrantVectorStore", QdrantVectorStore),
    ("PineconeVectorStore", PineconeVectorStore),
    ("Milvus", Milvus),
    ("MongoDBAtlasVectorSearch", MongoDBAtlasVectorSearch),
    ("CouchbaseSearchVectorStore", CouchbaseSearchVectorStore),
]

VECTORSTORE_TEMPLATE = """\
---
sidebar_position: 1
sidebar_class_name: hidden
keywords: [compatibility]
custom_edit_url:
---

# Vectorstores

## Features 

The table below lists the features for some of our most popular vector stores.

{table}

"""


def get_vectorstore_table():
    vectorstore_feat_table = {
        "PGVectorStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": True,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "FAISS": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "ElasticsearchStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "PGVector": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "Redis": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "Clickhouse": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "InMemoryVectorStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "Chroma": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "AstraDBVectorStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "openGauss": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": False,
            "Passes Standard Tests": True,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "QdrantVectorStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "PineconeVectorStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "Milvus": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "MongoDBAtlasVectorSearch": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": True,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
        "CouchbaseSearchVectorStore": {
            "Delete by ID": True,
            "Filtering": True,
            "similarity_search_by_vector": True,
            "similarity_search_with_score": True,
            "asearch": True,
            "Passes Standard Tests": False,
            "Multi Tenancy": False,
            "Local/Cloud": "Local",
            "IDs in add Documents": True,
        },
    }
    for vs in vectorstore_list + from_partners:
        if isinstance(vs, tuple):
            cls = vs[1]
            vs_name = vs[0]
        else:
            cls = getattr(vectorstores, vs)
            vs_name = vs
        for feat in (
            "similarity_search_with_score",
            "similarity_search_by_vector",
            "asearch",
        ):
            feat, name = feat, feat
            if getattr(cls, feat) != getattr(VectorStore, feat):
                vectorstore_feat_table[vs_name][name] = True
            else:
                vectorstore_feat_table[vs_name][name] = False

        if "filter" not in [
            key
            for key, _ in inspect.signature(
                getattr(cls, "similarity_search")
            ).parameters.items()
        ]:
            vectorstore_feat_table[vs_name]["Filtering"] = False

    header = [
        "Vectorstore",
        "Delete by ID",
        "Filtering",
        "similarity_search_by_vector",
        "similarity_search_with_score",
        "asearch",
        "Passes Standard Tests",
        "Multi Tenancy",
        "Local/Cloud",
        "IDs in add Documents",
    ]
    title = [
        "Vectorstore",
        "Delete by ID",
        "Filtering",
        "Search by Vector",
        "Search with score",
        "Async",
        "Passes Standard Tests",
        "Multi Tenancy",
        "Local/Cloud",
        "IDs in add Documents",
    ]
    rows = [title, [":-"] + [":-:"] * (len(title) - 1)]
    for vs, feats in sorted(vectorstore_feat_table.items()):
        rows += [
            [vs, "✅"]
            + [
                ("✅" if feats.get(h) else "❌") if h != "Local/Cloud" else feats.get(h)
                for h in header[1:]
            ]
        ]
    return "\n".join(["|".join(row) for row in rows])


if __name__ == "__main__":
    output_dir = Path(sys.argv[1])
    output_integrations_dir = output_dir / "integrations"
    output_integrations_dir_vectorstore = output_integrations_dir / "vectorstores"
    output_integrations_dir_vectorstore.mkdir(parents=True, exist_ok=True)

    vectorstore_page = VECTORSTORE_TEMPLATE.format(table=get_vectorstore_table())
    with open(output_integrations_dir / "vectorstores" / "index.mdx", "w") as f:
        f.write(vectorstore_page)
