# HyDE

Hypothetical Document Embeddings (HyDE) are a method to improve retrieval.
To do this, a hypothetical document is generated for an incoming query.
That document is then embedded, and that embedding is used to look up real documents similar to that hypothetical document.
The idea behind this is that the hypothetical document may be closer in the embedding space than the query.
For a more detailed description, read the full paper [here](https://arxiv.org/abs/2212.10496).

For this example, we use a simple RAG architecture, although you can easily use this technique in other more complicated architectures.
