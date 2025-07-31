# LangChain Coherence Integration

This package integrates Oracle Coherence as a vector store in LangChain.

## Installation

```bash
pip install langchain_coherence
```

## Usage

Before using LangChain's CoherenceVectorStore you must ensure that a Coherence server ([Coherence CE](https://github.com/oracle/coherence) 25.03+ or [Oracle Coherence](https://www.oracle.com/java/coherence/) 14.1.2+) is running 

For local development, we recommend using the Coherence CE container image:
```aiignore
docker run -d -p 1408:1408 ghcr.io/oracle/coherence-ce:25.03.2
```

### Adding and retrieving Documents

```python
import asyncio

from langchain_coherence import CoherenceVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from coherence import NamedMap, Session

async def do_run():
    session: Session = await Session.create()
    try:
        named_map: NamedMap[str, Document] = await session.get_map("my-map")
        embedding :Embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-l6-v2")
        # this embedding generates vectors of dimension 384
        cvs :CoherenceVectorStore = await CoherenceVectorStore.create(
            named_map,embedding)
        d1 :Document = Document(id="1", page_content="apple")
        d2 :Document = Document(id="2", page_content="orange")
        documents = [d1, d2]
        await cvs.aadd_documents(documents)
    
        ids = [doc.id for doc in documents]
        l = await cvs.aget_by_ids(ids)
        assert len(l) == len(ids)
        print("====")
        for e in l:
            print(e)
    finally:
        await session.close()

asyncio.run(do_run())
```
### SimilaritySearch on Documents

```python
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from coherence import NamedMap, Session
from langchain_coherence import CoherenceVectorStore

def test_data():
    d1 :Document = Document(id="1", page_content="apple")
    d2 :Document = Document(id="2", page_content="orange")
    d3 :Document = Document(id="3", page_content="tiger")
    d4 :Document = Document(id="4", page_content="cat")
    d5 :Document = Document(id="5", page_content="dog")
    d6 :Document = Document(id="6", page_content="fox")
    d7 :Document = Document(id="7", page_content="pear")
    d8 :Document = Document(id="8", page_content="banana")
    d9 :Document = Document(id="9", page_content="plum")
    d10 :Document = Document(id="10", page_content="lion")

    documents = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]
    return documents

async def test_asimilarity_search():
    documents = test_data()
    session: Session = await Session.create()
    try:
        named_map: NamedMap[str, Document] = await session.get_map("my-map")
        embedding :Embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-l6-v2")
        # this embedding generates vectors of dimension 384
        cvs :CoherenceVectorStore = await CoherenceVectorStore.create(
                                            named_map,embedding)
        await cvs.aadd_documents(documents)
        ids = [doc.id for doc in documents]
        l = await cvs.aget_by_ids(ids)
        assert len(l) == 10

        result = await cvs.asimilarity_search("fruit")
        assert len(result) == 4
        print("====")
        for e in result:
            print(e)
    finally:
        await session.close()
```
