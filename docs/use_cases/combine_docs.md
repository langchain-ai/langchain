# Data Augmented Generation

## Overview

Language models are trained on large amounts of unstructured data, which makes them fantastic at general purpose text generation. However, there are many instances where you may want the language model to generate text based not on generic data but rather on specific data. Some common examples of this include:

- Summarization of a specific piece of text (a website, a private document, etc.)
- Question answering over a specific piece of text (a website, a private document, etc.)
- Question answering over multiple pieces of text (multiple websites, multiple private documents, etc.)
- Using the results of some external call to an API (results from a SQL query, etc.)

All of these examples are instances when you do not want the LLM to generate text based solely on the data it was trained over, but rather you want it to incorporate other external data in some way. At a high level, this process can be broken down into two steps:

1. Fetching: Fetching the relevant data to include.
2. Augmenting: Passing the data in as context to the LLM.

This guide is intended to provide an overview of how to do this. This includes an overview of the literature, as well as common tools, abstractions and chains for doing this.

## Related Literature
There are a lot of related papers in this area. Most of them are focused on end-to-end methods that optimize the fetching of the relevant data as well as passing it in as context. These are a few of the papers that are particularly relevant:

**[RAG](https://arxiv.org/abs/2005.11401):** Retrieval Augmented Generation. 
This paper introduces RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever.

**[REALM](https://arxiv.org/abs/2002.08909):** Retrieval-Augmented Language Model Pre-Training. 
To capture knowledge in a more modular and interpretable way, this paper augments language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference.

**[HayStack](https://haystack.deepset.ai/):** This is not a paper, but rather an open source library aimed at semantic search, question answering, summarization, and document ranking for a wide range of NLP applications. The underpinnings of this library are focused on the same `fetching` and `augmenting` concepts discussed here, and incorporate some methods in the above papers.

These papers/open-source projects are centered around retrieval of documents, which is important for question-answering tasks over a large corpus of documents (which is how they are evaluated). However, we use the terminology of `Data Augmented Generation` to highlight that retrieval from some document store is only one possible way of fetching relevant data to include. Other methods to fetch relevant data could involve hitting an API, querying a database, or just working with user provided data (eg a specific document that they want to summarize).

Let's now deep dive on the two steps involved: fetching and augmenting.

## Fetching
There are many ways to fetch relevant data to pass in as context to a LM, and these methods largely depend
on the use case.

**User provided:** In some cases, the user may provide the relevant data, and no algorithm for fetching is needed.
An example of this is for summarization of specific documents: the user will provide the document to be summarized,
and task the language model with summarizing it.

**Document Retrieval:** One of the more common use cases involves fetching relevant documents or pieces of text from
a large corpus of data. A common example of this is question answering over a private collection of documents.

**API Querying:** Another common way to fetch data is from an API query. One example of this is WebGPT like system,
where you first query Google (or another search API) for relevant information, and then those results are used in
the generation step. Another example could be querying a structured database (like SQL) and then using a language model
to synthesize those results.

There are two big issues to deal with in fetching:

1. Fetching small enough pieces of information
2. Not fetching too many pieces of information (e.g. fetching only the most relevant pieces)

### Text Splitting
One big issue with all of these methods is how to make sure you are working with pieces of text that are not too large.
This is important because most language models have a context length, and so you cannot (yet) just pass a 
large document in as context. Therefore, it is important to not only fetch relevant data but also make sure it is in
small enough chunks.

LangChain provides some utilities to help with splitting up larger pieces of data. This comes in the form of the TextSplitter class.
The class takes in a document and splits it up into chunks, with several parameters that control the
size of the chunks as well as the overlap in the chunks (important for maintaining context).
See [this walkthrough](../modules/utils/combine_docs_examples/textsplitter.ipynb) for more information.

### Relevant Documents
A second large issue related fetching data is to make sure you are not fetching too many documents, and are only fetching
the documents that are relevant to the query/question at hand. There are a few ways to deal with this.

One concrete example of this is vector stores for document retrieval, often used for semantic search or question answering.
With this method, larger documents are split up into
smaller chunks and then each chunk of text is passed to an embedding function which creates an embedding for that piece of text.
Those are embeddings are then stored in a database. When a new search query or question comes in, an embedding is
created for that query/question and then documents with embeddings most similar to that embedding are fetched. 
Examples of vector database companies include [Pinecone](https://www.pinecone.io/) and [Weaviate](https://weaviate.io/).

Although this is perhaps the most common way of document retrieval, people are starting to think about alternative
data structures and indexing techniques specifically for working with language models. For a leading example of this,
check out [GPT Index](https://github.com/jerryjliu/gpt_index) - a collection of data structures created by and optimized
for language models.

## Augmenting
So you've fetched your relevant data - now what? How do you pass them to the language model in a format it can understand?
For a detailed overview of the different ways of doing so, and the tradeoffs between them, please see 
[this documentation](../modules/chains/combine_docs.md)

## Use Cases
LangChain supports the above three methods of augmenting LLMs with external data.
These methods can be used to underpin several common use cases, and they are discussed below.
For all three of these use cases, all three methods are supported.
It is important to note that a large part of these implementations is the prompts
that are used. We provide default prompts for all three use cases, but these can be configured.
This is in case you discover a prompt that works better for your specific application.

- [Question-Answering](question_answering.md)
- [Summarization](summarization.md)
