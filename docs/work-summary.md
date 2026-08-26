# Historical LangChain v0.0.64 Work Summary

## Scope and baseline

This note covers the historical main `langchain` package tag `v0.0.64`
(`08400f5542`), not the later `langchain-experimental==0.0.64` release. The
repository does not have a `v0.0.63` tag, so the commit that set the package
version to 0.0.63
(`2a54e73fec`) is used as the exclusive baseline. The range contains the six
subsequent commits through `v0.0.64`:

```text
2a54e73fec..08400f5542
```

The range changes 15 files, with 519 insertions and 20 deletions. It combines
several user-facing improvements with release automation and the final version
bump.

## User-visible changes

### More tolerant MRKL final-answer parsing

The MRKL agent parser no longer requires a space after `Final Answer:`. It now
strips surrounding whitespace from the extracted answer, so both
`Final Answer: 1994` and a newline-delimited form such as
`Final Answer:\n1994` produce the same result. A regression test covers the new
newline form while the existing single-line and multiline cases remain in
place.

### Smaller and more focused SQL prompts

`SQLDatabaseChain` gained a `top_k` setting, whose implementation default is
`5`. The SQL-generation prompt asks the model to apply a `LIMIT` clause unless
the user requests a specific result count. This is a prompt-level instruction;
the database execution layer does not independently enforce the limit.

The release also introduced `SQLDatabaseSequentialChain` for databases with
many tables. It first asks an LLM to select relevant table names, then passes
only those tables' schema information to the existing SQL chain. Supporting
changes include:

- public `SQLDatabase.get_table_names()` and
  `SQLDatabase.get_table_info(table_names=...)` methods;
- validation that rejects requested tables that are not present;
- an optional internal `table_names_to_use` input on `SQLDatabaseChain`;
- `CommaSeparatedListOutputParser` for the table-selection response; and
- a more useful `SequentialChain` missing-input error that also reports the
  variables already known to the chain.

The SQLite notebook already contained the custom-prompt example; this release
adds sections demonstrating `top_k` and `SQLDatabaseSequentialChain`.

### Clearer API URL generation

The API-chain URL prompt now explicitly asks for a complete URL that returns as
little data as possible while retaining what is needed to answer the question.
The prompt also separates the question and URL markers more clearly. A new
notebook demonstrates the chain with OpenMeteo and TMDB, and the utility-chain
guide links to that example.

### Wikipedia source metadata

Wikipedia lookups now attach the resolved page URL to returned documents as
`metadata["page"]`. Consumers can therefore retain the source URL alongside the
page content instead of reconstructing it later.

## Historical usage

These examples reflect the API at `v0.0.64`, not the current LangChain API.

```python
db_chain = SQLDatabaseChain(
    llm=llm,
    database=db,
    top_k=3,
    verbose=True,
)

from langchain.chains.sql_database.base import SQLDatabaseSequentialChain

table_aware_chain = SQLDatabaseSequentialChain.from_llm(
    llm,
    db,
    verbose=True,
)
```

```python
chain = APIChain.from_llm_and_api_docs(
    llm,
    open_meteo_docs.OPEN_METEO_DOCS,
    verbose=True,
)
```

## Release and packaging

The release added a GitHub Actions workflow for publishing the package. The
workflow is configured for closed pull requests targeting `master` and touching
`pyproject.toml`; its release job runs only when the pull request was merged and
carried the `release` label. It builds with Poetry, uploads `dist/*` to a
generated-notes GitHub release tagged from the Poetry version, and publishes
the artifacts to PyPI using `PYPI_API_TOKEN`.

The final commit changed the Poetry package version from `0.0.63` to `0.0.64`.
Separately, the release-workflow commit reflowed the README contribution
paragraph without altering documented behavior.

## Changed-file map

| Outcome | Files in the release range |
| --- | --- |
| MRKL parsing | `langchain/agents/mrkl/base.py`, `tests/unit_tests/agents/test_mrkl.py` |
| Wikipedia metadata | `langchain/docstore/wikipedia.py` |
| SQL selection and limits | `langchain/chains/sequential.py`, `langchain/chains/sql_database/base.py`, `langchain/chains/sql_database/prompt.py`, `langchain/prompts/base.py`, `langchain/sql_database.py`, `docs/modules/chains/examples/sqlite.ipynb` |
| API-chain prompting and examples | `langchain/chains/api/prompt.py`, `docs/modules/chains/examples/api.ipynb`, `docs/modules/chains/utility_how_to.rst` |
| Release and packaging | `.github/workflows/release.yml`, `pyproject.toml`, `README.md` |

## Verification evidence

The scope and file counts can be reproduced from the immutable historical
references:

```bash
git log --reverse --oneline 2a54e73fec..v0.0.64
git diff --shortstat 2a54e73fec..v0.0.64
git diff --name-status 2a54e73fec..v0.0.64
```

A clean archive of `v0.0.64` was also tested in an isolated uv environment:

```bash
uv run --python 3.11 --no-project \
  --with 'pytest>=7.2,<8' \
  --with 'pydantic>=1,<2' \
  --with 'SQLAlchemy>=1,<2' \
  --with 'requests>=2,<3' \
  --with 'PyYAML>=6,<7' \
  --with 'numpy>=1,<2' \
  pytest tests/unit_tests/agents/test_mrkl.py -q
```

The result was seven passing tests under Python 3.11.15 and pytest 7.4.4, with
one SQLAlchemy 2.0 deprecation warning. The newline-delimited sample returned
`("Search", "NBA")` at the exclusive baseline and
`("Final Answer", "1994")` at `v0.0.64`, demonstrating that the added test
exercises the changed behavior.

MRKL was the only area to receive a new regression test, through
`test_get_final_answer_new_line`. Existing API and SQL tests remained, but this
release added no focused tests for API URL-prompt semantics, `top_k`, SQL table
selection, or Wikipedia metadata. A full historical `git diff --check` is not
clean because `.github/workflows/release.yml` contains trailing whitespace
after the block scalar on its `run: |` line. This note records the historical
state rather than rewriting the immutable tag.

## Known limitations

- The SQLite notebook says the `top_k` default is `10`, while the implementation
  default is `5`.
- `top_k` relies on the model following a prompt instruction; it is not a hard
  row cap.
- `CommaSeparatedListOutputParser` splits only on the exact delimiter `", "`,
  so differently formatted model output can produce incorrect table names.
- `SQLDatabaseSequentialChain` was not exported from the package-level
  `__init__.py`; the historical example imports it from the implementation
  module.
- Table selection only reduces the schema text shown to the model; it does not
  restrict which database tables the generated SQL can access.
- `SQLDatabaseSequentialChain.from_llm()` does not forward `top_k` when it
  constructs its internal `SQLDatabaseChain`, so that chain retains the default
  of `5` unless it is modified or constructed separately.
- A successful Wikipedia lookup calls `wikipedia.page(search)` twice: once for
  content and once for the URL.
- Both API examples use `OpenAI()` and therefore require OpenAI credentials and
  network access. The TMDB notebook also sets `TMDB_BEARER_TOKEN` to an empty
  placeholder that the user must replace.
