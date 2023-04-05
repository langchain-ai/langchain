# Summarization

> [Conceptual Guide](https://docs.langchain.com/docs/use-cases/summarization)


Summarization involves creating a smaller summary of multiple longer documents.
This can be useful for distilling long documents into the core pieces of information.

The recommended way to get started using a summarization chain is:

```python
from langchain.chains.summarize import load_summarize_chain
chain = load_summarize_chain(llm, chain_type="map_reduce")
chain.run(docs)
```

The following resources exist:
- [Summarization Notebook](../modules/chains/index_examples/summarize.ipynb): A notebook walking through how to accomplish this task.

Additional related resources include:
- [Utilities for working with Documents](../reference/utils.rst): Guides on how to use several of the utilities which will prove helpful for this task, including Text Splitters (for splitting up long documents).
