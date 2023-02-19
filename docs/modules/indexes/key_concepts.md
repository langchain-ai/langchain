# Key Concepts

## Text Splitter
This class is responsible for splitting long pieces of text into smaller components.
It contains different ways for splitting text (on characters, using Spacy, etc)
as well as different ways for measuring length (token based, character based, etc).

## Embeddings
These classes are very similar to the LLM classes in that they are wrappers around models, 
but rather than return a string they return an embedding (list of floats). These are particularly useful when 
implementing semantic search functionality. They expose separate methods for embedding queries versus embedding documents.

## Vectorstores
These are datastores that store embeddings of documents in vector form.
They expose a method for passing in a string and finding similar documents.


## CombineDocuments Chains
These are a subset of chains designed to work with documents. There are two pieces to consider:

1. The underlying chain method (eg, how the documents are combined)
2. Use cases for these types of chains.

For the first, please see [this documentation](combine_docs.md) for more detailed information on the types of chains LangChain supports.
For the second, please see the Use Cases section for more information on [question answering](/use_cases/question_answering.md), 
[question answering with sources](/use_cases/qa_with_sources.md), and [summarization](/use_cases/summarization.md).

