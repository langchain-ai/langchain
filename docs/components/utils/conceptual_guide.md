# Conceptual Guide

## Embeddings
These classes are very similar to the LLM classes in that they are wrappers around models, 
but rather than return a string they return an embedding (list of floats). These are particularly useful when 
implementing semantic search functionality. They expose separate methods for embedding queries versus embedding documents.

## Vectorstores
These are datastores that store documents. They expose a method for passing in a string and finding similar documents.
