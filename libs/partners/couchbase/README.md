# langchain-couchbase

This package contains the LangChain integration with Couchbase

## Installation

```bash
pip install -U langchain-couchbase
```

## Usage

The `CouchbaseVectorStore` class exposes the connection to the Couchbase vector store.

```python
from langchain_couchbase.vectorstores import CouchbaseVectorStore

from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
from datetime import timedelta

auth = PasswordAuthenticator(username, password)
options = ClusterOptions(auth)
connect_string = "couchbases://localhost"
cluster = Cluster(connect_string, options)

# Wait until the cluster is ready for use.
cluster.wait_until_ready(timedelta(seconds=5))

embeddings = OpenAIEmbeddings()

vectorstore = CouchbaseVectorStore(
    cluster=cluster,
    bucket_name="",
    scope_name="",
    collection_name="",
    embedding=embeddings,
    index_name="vector-search-index",
)

```
