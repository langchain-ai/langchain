# Data I/O

Many LLM applications require user-specific data that is not part of the model's training set. LangChain gives you the 
building blocks to connect, manage, and interface with your data via:

- [Document loaders](#document-loaders): Load documents from many different sources

- [Document transformers](#document-transformers): Split documents, drop redundant documents, and more

- [Vector stores](#vector-stores): Store and search over embedded data

- [Retrievers](#retrievers): Query your data


## [Document loaders](./data_io/document_loaders.html)

Use document loaders to load data from a source as a `Document`. A `Document` is a piece of text 
and associated metadata. For example, there are document loaders for loading a simple `.txt` file, for loading the text 
contents of any web page, or even for loading a transcript of a YouTube video.

## [Document transformers](./data_io/text_splitters.html)

Once you've loaded documents, you'll often want to transform them to better suit your application. The simplest example
is you may want to split a long document into smaller chunks that can fit into your model's context window. LangChain
has a number of built-in document transformers that make it easy to split, filter, and otherwise manipulate documents.

## [Vector stores](./data_io/vectorstores.html)

One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding
vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 
'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search 
for you.

With slightly less jargon: each source text is converted to a list of numbers. When you get a question you convert that 
to a list of numbers, too, and then return the texts whose list of numbers is most similar to the question list of 
numbers. There's a number of common ways to define "most similar". For example, you can use the cosine of the angle 
between the question and source text vectors.

## [Retrievers](./data_io/retrievers.html)

A retriever is an interface that returns documents given an unstructured query. It is more general than a vector store.
A retriever does not need to be able to store documents, only to return (or retrieve) it. Vector stores can be used
as the backbone of a retriever, but there are other types of retrievers as well.