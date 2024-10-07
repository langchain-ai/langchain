# Migrating

## 🚨Breaking Changes for select chains (SQLDatabase) on 7/28/23

In an effort to make `langchain` leaner and safer, we are moving select chains to `langchain_experimental`.
This migration has already started, but we are remaining backwards compatible until 7/28.
On that date, we will remove functionality from `langchain`.
Read more about the motivation and the progress [here](https://github.com/langchain-ai/langchain/discussions/8043).

### Migrating to `langchain_experimental`

We are moving any experimental components of LangChain, or components with vulnerability issues, into `langchain_experimental`.
This guide covers how to migrate.

### Installation

Previously:

```bash
pip install -U langchain
```

Now (only if you want to access things in experimental):

```bash
pip install -U langchain langchain_experimental
```

### Things in `langchain.experimental`

Previously:

```bash
from langchain.experimental import ...
```

Now:

```bash
from langchain_experimental import ...
```

### PALChain

Previously:

```bash
from langchain.chains import PALChain
```

Now:

```bash
from langchain_experimental.pal_chain import PALChain
```

### SQLDatabaseChain

Previously:

```bash
from langchain.chains import SQLDatabaseChain
```

Now:

```bash
from langchain_experimental.sql import SQLDatabaseChain
```

Alternatively, if you are interested in using the query generation part of the SQL chain, you can check out this link: [`SQL question-answering tutorial`](https://python.langchain.com/v0.2/docs/tutorials/sql_qa/#convert-question-to-sql-query)

```bash
from langchain.chains import create_sql_query_chain
```

### `load_prompt` for Python files

Note: this only applies if you want to load Python files as prompts.
If you want to load json/yaml files, no change is needed.

Previously:

```bash
from langchain.prompts import load_prompt
```

Now:

```bash
from langchain_experimental.prompts import load_prompt
```
