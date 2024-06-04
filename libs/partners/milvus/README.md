# langchain-milvus

This is a library integration with [Milvus](https://milvus.io/) and [Zilliz Cloud](https://zilliz.com/cloud).

## Installation

```bash
pip install -U langchain-milvus
```

## Milvus vector database

See a [usage example](https://python.langchain.com/v0.2/docs/integrations/vectorstores/milvus/)

```python
from langchain_milvus import Milvus
```

## Milvus hybrid search

See a [usage example](https://python.langchain.com/v0.2/docs/integrations/retrievers/milvus_hybrid_search/).

```python
from langchain_milvus import MilvusCollectionHybridSearchRetriever
```


## Zilliz Cloud vector database

See a [usage example](https://python.langchain.com/v0.2/docs/integrations/vectorstores/zilliz/).

```python
from langchain_milvus import Zilliz
```

## Zilliz Cloud Pipeline Retriever

See a [usage example](https://python.langchain.com/v0.2/docs/integrations/retrievers/zilliz_cloud_pipeline/).

```python
from langchain_milvus import ZillizCloudPipelineRetriever
```