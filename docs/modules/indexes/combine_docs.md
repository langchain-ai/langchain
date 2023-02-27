# CombineDocuments Chains
CombineDocuments chains are useful for when you need to run a language over multiple documents.
Common use cases for this include question answering, question answering with sources, summarization, and more.
For more information on specific use cases as well as different methods for **fetching** these documents, please see 
[this overview](/use_cases/combine_docs.md).

This documentation now picks up from after you've fetched your documents - now what?
How do you pass them to the language model in a format it can understand?
There are a few different methods, or chains, for doing so. LangChain supports four of the more common ones - and
we are actively looking to include more, so if you have any ideas please reach out! Note that there is not
one best method - the decision of which one to use is often very context specific. In order from simplest to
most complex:

## Stuffing
Stuffing is the simplest method, whereby you simply stuff all the related data into the prompt as context
to pass to the language model. This is implemented in LangChain as the `StuffDocumentsChain`.

**Pros:** Only makes a single call to the LLM. When generating text, the LLM has access to all the data at once.

**Cons:** Most LLMs have a context length, and for large documents (or many documents) this will not work as it will result in a prompt larger than the context length.

The main downside of this method is that it only works one smaller pieces of data. Once you are working
with many pieces of data, this approach is no longer feasible. The next two approaches are designed to help deal with that.

## Map Reduce
This method involves an initial prompt on each chunk of data (for summarization tasks, this 
could be a summary of that chunk; for question-answering tasks, it could be an answer based solely on that chunk).
Then a different prompt is run to combine all the initial outputs. This is implemented in the LangChain as the `MapReduceDocumentsChain`.

**Pros:** Can scale to larger documents (and more documents) than `StuffDocumentsChain`. The calls to the LLM on individual documents are independent and can therefore be parallelized.

**Cons:** Requires many more calls to the LLM than `StuffDocumentsChain`. Loses some information during the final combining call.

## Refine
This method involves an initial prompt on the first chunk of data, generating some output.
For the remaining documents, that output is passed in, along with the next document, 
asking the LLM to refine the output based on the new document. 

**Pros:** Can pull in more relevant context, and may be less lossy than `MapReduceDocumentsChain`.

**Cons:** Requires many more calls to the LLM than `StuffDocumentsChain`. The calls are also NOT independent, meaning they cannot be paralleled like `MapReduceDocumentsChain`. There is also some potential dependencies on the ordering of the documents.


## Map-Rerank
This method involves running an initial prompt on each chunk of data, that not only tries to complete a
task but also gives a score for how certain it is in its answer. The responses are then
ranked according to this score, and the highest score is returned.

**Pros:** Similar pros as `MapReduceDocumentsChain`. Compared to `MapReduceDocumentsChain`, it requires fewer calls.

**Cons:** Cannot combine information between documents. This means it is most useful when you expect there to be a single simple answer in a single document.
