# SWI-Prolog

SWI-Prolog offers a comprehensive free Prolog environment.

## Installation and Setup

Install lanchain-prolog using pip:
```bash
pip install langchain-prolog
```

## Tools

The `PrologTool` class allows the generation of langchain tools that use Prolog rules to generate answers.

```python
from langchain_prolog import PrologConfig, PrologTool
```

See a [usage example](/docs/integrations/tools/prolog_tool).

See the same guide for usage examples of `PrologRunnable`, which allows the generation
of LangChain runnables that use Prolog rules to generate answers.
