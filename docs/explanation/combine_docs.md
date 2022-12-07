# Context Guided Generation

## Overview

Language models are trained on large amounts of unstructured data,
which makes them really good at general purpose text generation.
However, when you want the language model to base its reply on some specific
piece of text, it is useful to guide its generation by including those pieces
of text in the context.
There are several common use cases that do this: question answering, question answering with sources, and summarization are the ones covered in LangChain.
In

There are two components to this:

1. Retrieval: given a user question/user input, how do you find the pieces of information most relevant to include. 
2. Context Guided Generation: Given that you retrieved some pieces of information, how do you now pass them as context into the LLM?

This mainly focuses on the second question (passing information as context to the LLM).
However, before diving deep there let's talk about retrieval for a bit.

## Retrieval

There are a few different ways to do retrieval. No one way is best in all contexts; different problems may require different solutions.
The most extreme is just to retrieve ALL the documents. This is good because you use all the information present, but does not scale well when the underlying document corpus is large.
A more common approach is to do some sort semantic similarity search for relevant documents based on the user input.
The most common way to do this is probably by creating embeddings of the document corpus, and then taking the user input, embedding that, and finding documents close in the embedding space.
However, vector databases are not the only approach to this - for a novel indexing strategy check out GPTIndex (TODO link)

## Generation

Now that we've covered how to do retrieval, lets do a deep dive on how to incorporate the fetched documents as context.
Note that I use "document" here to really mean any piece of text - it could be a page, a chapter, a comment, a slack conversation, etc.

At a high level, LangChain has a base `CombineDocuments` chain which takes in a list of documents, other user input, and returns a string.
There are three chains that stem from `CombineDocuments`, which we cover here.
Note that there is not necessarily
one "best" way to do this - they all have their own pros and cons. We will cover these
chains and the pros/cons in this document. Also note that these chains
are NOT tied to a specific prompt - although we provide basic prompts
for all methods for an easy quick start, you can edit/modify/improve these prompts as you see fit.

The chains we cover are: `stuff`, `map_reduce`, and `refine`.

### `StuffDocumentsChain`
The `StuffDocumentsChain` simply puts all the relevant documents into a single prompt, along with the user input,
and passes it the LLM.

**Pros:** Only makes a single call to the LLM. When generating text, the LLM has access to all the documents at once.
**Cons:** Most LLMs have a context length, and for large documents (or many documents) this will not work as it will result in a prompt larger than the context length.

### `MapReduceDocumentsChain`
The `MapReduceDocumentsChain` first calls an LLM with each individual document, and then makes a final call combining the results of those calls.

**Pros:** Can scale to larger (and more documents) than `StuffDocumentsChain`. The calls to the LLM on individual documents are independent and could therefor be paralellized.
**Cons:** Requires many more calls to the LLM than `StuffDocumentsChain`. Loses some information by the final combining call.

### `RefineDocumentsChain`
The `RefineDocumentsChain` loops over the documents. On the first document, it makes a standard LLM call. In subsequent calls, it passes in the result of the previous call and a new document, and asks the LLM to refine that answer based on the new document.

**Pros:** Can perhaps pull in more relevant context, and may be less lossy than `RefineDocumentsChain`.
**Cons:** Requires many more calls to the LLM than `StuffDocumentsChain`. The calls are also NOT independent, meaning they cannot be parralized like `RefineDocumentsChain`.

## Use Cases

As mentioned before, there are three common use cases that LangChain supports.
Please the examples for more information on how to work with them.

- `Question-Answering With Sources <../examples/chains/qa_with_sources.ipynb>`_
- `Question-Answering <../examples/chains/question_answering.ipynb>`_
- `Summarization <../examples/chains/summarize.ipynb>`_
