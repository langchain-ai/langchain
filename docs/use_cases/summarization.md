# Summarization

Summarization involves creating a smaller summary of multiple longer documents.
This can be useful for distilling long documents into the core pieces of information.

The recommended way to get started using a summarization chain is:

```python
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

The following resources exist:
- [Summarization Notebook](../modules/indexes/chain_examples/summarize.ipynb): A notebook walking through how to accomplish this task.

Additional related resources include:
- [Utilities for working with Documents](../modules/utils/how_to_guides.rst): Guides on how to use several of the utilities which will prove helpful for this task, including Text Splitters (for splitting up long documents).
- [CombineDocuments Chains](../modules/indexes/combine_docs.md): A conceptual overview of specific types of chains by which you can accomplish this task.
- [Data Augmented Generation](./combine_docs.md): An overview of data augmented generation, which is the general concept of combining external data with LLMs (of which this is a subset).
