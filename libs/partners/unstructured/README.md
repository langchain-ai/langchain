# langchain-unstructured

This package contains the LangChain integration with Unstructured

## Installation

```bash
pip install -U langchain-unstructured
```

And you should configure credentials by setting the following environment variables:

```bash
export UNSTRUCTURED_API_KEY="your-api-key"
```

## Loaders

Partition and load files using either the `unstructured-client` sdk and the
Unstructured API or locally using the `unstructured` library.

API:
To partition via the Unstructured API `pip install unstructured-client` and set
`partition_via_api=True` and define `api_key`. If you are running the unstructured API
locally, you can change the API rule by defining `url` when you initialize the
loader. The hosted Unstructured API requires an API key. See the links below to
learn more about our API offerings and get an API key.

Local:
By default the file loader uses the Unstructured `partition` function and will
automatically detect the file type.

In addition to document specific partition parameters, Unstructured has a rich set
of "chunking" parameters for post-processing elements into more useful text segments
for uses cases such as Retrieval Augmented Generation (RAG). You can pass additional
Unstructured kwargs to the loader to configure different unstructured settings.

Setup:
```bash
    pip install -U langchain-unstructured
    pip install -U unstructured-client
    export UNSTRUCTURED_API_KEY="your-api-key"
```

Instantiate:
```python
from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    file_path = ["example.pdf", "fake.pdf"],
    api_key=UNSTRUCTURED_API_KEY,
    partition_via_api=True,
    chunking_strategy="by_title",
    strategy="fast",
)
```

Load:
```python
docs = loader.load()

print(docs[0].page_content[:100])
print(docs[0].metadata)
```

References
----------
https://docs.unstructured.io/api-reference/api-services/sdk
https://docs.unstructured.io/api-reference/api-services/overview
https://docs.unstructured.io/open-source/core-functionality/partitioning
https://docs.unstructured.io/open-source/core-functionality/chunking
