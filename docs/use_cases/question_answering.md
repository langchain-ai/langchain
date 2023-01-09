# Question Answering

Question answering involves fetching multiple documents, and then asking a question of them.
The LLM response will contain the answer to your question, based on the content of the documents.

The recommended way to get started using a question answering chain is:

```python
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)
```

The following resources exist:
- [Question Answering Notebook](/modules/chains/combine_docs_examples/question_answering.ipynb): A notebook walking through how to accomplish this task.
- [VectorDB Question Answering Notebook](/modules/chains/combine_docs_examples/vector_db_qa.ipynb): A notebook walking through how to do question answering over a vector database. This can often be useful for when you have a LOT of documents, and you don't want to pass them all to the LLM, but rather first want to do some semantic search over embeddings.

### Adding in sources

There is also a variant of this, where in addition to responding with the answer the language model will also cite its sources (eg which of the documents passed in it used).

The recommended way to get started using a question answering with sources chain is:

```python
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
chain = load_qa_with_sources_chain(llm, chain_type="stuff")
chain({"input_documents": docs, "question": query}, return_only_outputs=True)
```

The following resources exist:
- [QA With Sources Notebook](/modules/chains/combine_docs_examples/qa_with_sources.ipynb): A notebook walking through how to accomplish this task.
- [VectorDB QA With Sources Notebook](/modules/chains/combine_docs_examples/vector_db_qa_with_sources.ipynb): A notebook walking through how to do question answering with sources over a vector database. This can often be useful for when you have a LOT of documents, and you don't want to pass them all to the LLM, but rather first want to do some semantic search over embeddings.

### Additional Related Resources

Additional related resources include:
- [Utilities for working with Documents](/modules/utils/how_to_guides.rst): Guides on how to use several of the utilities which will prove helpful for this task, including Text Splitters (for splitting up long documents) and Embeddings & Vectorstores (useful for the above Vector DB example).
- [CombineDocuments Chains](/modules/chains/combine_docs.md): A conceptual overview of specific types of chains by which you can accomplish this task.
- [Data Augmented Generation](combine_docs.md): An overview of data augmented generation, which is the general concept of combining external data with LLMs (of which this is a subset).
