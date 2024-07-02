# langchain-mixedbreadai

This package contains the LangChain integration with MixedbreadAI

## Installation

```bash
pip install -U langchain-mixedbreadai
```

And you should configure credentials by setting the following environment variables:

* Get a Mixedbread AI api key and set the environment variable `MXBAI_API_KEY` to it or use it as a SDK argument.

## Embeddings

`MixedbreadAIEmbeddings` class exposes embeddings from MixedbreadAI.

```python
from langchain_mixedbreadai import MixedbreadAIEmbeddings

embeddings = MixedbreadAIEmbeddings(mxbai_api_key="a legit baker id")
embeddings.embed_query("What is the meaning of life?")
```

## Rerank

`MixedbreadAIRerank` class exposes reranker from MixedbreadAI.

```python
from langchain_mixedbreadai import MixedbreadAIRerank

reranker = MixedbreadAIRerank(mxbai_api_key="a legit baker id", top_n=3)
reranker.rerank(
    query="Who is the author of To Kill a Mockingbird?",
    documents=[
    "To Kill a Mockingbird is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
    "The novel Moby-Dick was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
    "Harper Lee, an American novelist widely known for her novel To Kill a Mockingbird, was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
    "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
    "The Harry Potter series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
    "The Great Gatsby, a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
  ]
)
```


