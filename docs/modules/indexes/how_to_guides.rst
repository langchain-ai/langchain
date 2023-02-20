How To Guides
====================================


Utils
-----

There are a lot of different utilities that LangChain provides integrations for
These guides go over how to use them.
The utilities here are all utilities that make it easier to work with documents.

`Text Splitters <./examples/textsplitter.html>`_: A walkthrough of how to split large documents up into smaller, more manageable pieces of text.

`VectorStores <./examples/vectorstores.html>`_: A walkthrough of the vectorstore abstraction that LangChain supports.

`Embeddings <./examples/embeddings.html>`_: A walkthrough of embedding functionalities, and different types of embeddings, that LangChain supports.

`HyDE <./examples/hyde.html>`_: How to use Hypothetical Document Embeddings, a novel way of constructing embeddings for document retrieval systems.

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Utils
   :name: utils
   :hidden:

   examples/*


Vectorstores
------------


Vectorstores are one of the most important components of building indexes.
In the below guides, we cover different types of vectorstores and how to use them.

`Chroma <./vectorstore_examples/chroma.html>`_: A walkthrough of how to use the Chroma vectorstore wrapper.

`FAISS <./vectorstore_examples/faiss.html>`_: A walkthrough of how to use the FAISS vectorstore wrapper.

`Elastic Search <./vectorstore_examples/elasticsearch.html>`_: A walkthrough of how to use the ElasticSearch wrapper.

`Milvus <./vectorstore_examples/milvus.html>`_: A walkthrough of how to use the Milvus vectorstore wrapper.

`Pinecone <./vectorstore_examples/pinecone.html>`_: A walkthrough of how to use the Pinecone vectorstore wrapper.

`Qdrant <./vectorstore_examples/qdrant.html>`_: A walkthrough of how to use the Qdrant vectorstore wrapper.

`Weaviate <./vectorstore_examples/weaviate.html>`_: A walkthrough of how to use the Weaviate vectorstore wrapper.


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Vectorstores
   :name: vectorstores
   :hidden:

   vectorstore_examples/*


Chains
------

The examples here are all end-to-end chains that use indexes or utils covered above.

`Question Answering <./chain_examples/question_answering.html>`_: A walkthrough of how to use LangChain for question answering over specific documents.

`Question Answering with Sources <./chain_examples/qa_with_sources.html>`_: A walkthrough of how to use LangChain for question answering (with sources) over specific documents.

`Summarization <./chain_examples/summarize.html>`_: A walkthrough of how to use LangChain for summarization over specific documents.

`Vector DB Text Generation <./chain_examples/vector_db_text_generation.html>`_: A walkthrough of how to use LangChain for text generation over a vector database.

`Vector DB Question Answering <./chain_examples/vector_db_qa.html>`_: A walkthrough of how to use LangChain for question answering over a vector database.

`Vector DB Question Answering with Sources <./chain_examples/vector_db_qa_with_sources.html>`_: A walkthrough of how to use LangChain for question answering (with sources) over a vector database.

`Graph Question Answering <./chain_examples/graph_qa.html>`_: A walkthrough of how to use LangChain for question answering (with sources) over a graph database.

`Chat Vector DB <./chain_examples/chat_vector_db.html>`_: A walkthrough of how to use LangChain as a chatbot over a vector database.

`Analyze Document <./chain_examples/analyze_document.html>`_: A walkthrough of how to use LangChain to analyze long documents.


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: With Chains
   :name: chains
   :hidden:

   ./chain_examples/*