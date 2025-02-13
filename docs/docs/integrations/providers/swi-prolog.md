# SWI-Prolog

SWI-Prolog offers a comprehensive free Prolog environment.

## Installation and Setup

Install lanchain-prolog using pip:
```bash
pip install langchain-prolog
```
## Runnables

The PrologRunnable class allows the generation of langchain runnables that use Prolog rules to generate answers

```python
from langchain_prolog import PrologConfig, PrologRunnable
```

See a [usage example](../tools/prolog_runnable.ipynb).

## Tools

The PrologTool class allows the generation of langchain tools that use Prolog rules to generate answers.

```python
from langchain_prolog import PrologConfig, PrologTool
```

See a [usage example](../tools/prolog_tool.ipynb).

## API reference
