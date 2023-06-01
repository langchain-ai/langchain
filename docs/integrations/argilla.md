# Argilla

![Argilla - Open-source data platform for LLMs](https://argilla.io/og.png)

>[Argilla](https://argilla.io/) is an open-source data curation platform for LLMs.
> Using Argilla, everyone can build robust language models through faster data curation 
> using both human and machine feedback. We provide support for each step in the MLOps cycle, 
> from data labeling to model monitoring.

## Installation and Setup

First, you'll need to install the  `argilla` Python package as follows:

```bash
pip install argilla --upgrade
```

If you already have an Argilla Server running, then you're good to go; but if
you don't, follow the next steps to install it.

If you don't you can refer to [Argilla - 🚀 Quickstart](https://docs.argilla.io/en/latest/getting_started/quickstart.html#Running-Argilla-Quickstart) to deploy Argilla either on HuggingFace Spaces, locally, or on a server.

## Tracking

See a [usage example of `ArgillaCallbackHandler`](../modules/callbacks/examples/examples/argilla.ipynb).

```python
from langchain.callbacks import ArgillaCallbackHandler
```
