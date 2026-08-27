# Historical LangChain v0.0.64 to v0.0.100 Work Summary

## Scope and baseline

This note has two related scopes for the historical main `langchain` package.
The first part records what entered the `v0.0.64` tag (`08400f5542`). The
second uses that tag as an exclusive baseline and follows the package through
`v0.0.100` (`499e76b199`). Neither scope refers to the later
`langchain-experimental==0.0.64` release.

The repository does not have a `v0.0.63` tag, so the commit that set the
package version to 0.0.63 (`2a54e73fec`) is used as the exclusive baseline for
the first part. The range contains the six subsequent commits through
`v0.0.64`:

```text
2a54e73fec..08400f5542
```

The range changes 15 files, with 519 insertions and 20 deletions. It combines
several user-facing improvements with release automation and the final version
bump.

## v0.0.64 user-visible changes

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

## v0.0.64 historical usage

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

## v0.0.64 release and packaging

The release added a GitHub Actions workflow for publishing the package. The
workflow is configured for closed pull requests targeting `master` and touching
`pyproject.toml`; its release job runs only when the pull request was merged and
carried the `release` label. It builds with Poetry, uploads `dist/*` to a
generated-notes GitHub release tagged from the Poetry version, and publishes
the artifacts to PyPI using `PYPI_API_TOKEN`.

The final commit changed the Poetry package version from `0.0.63` to `0.0.64`.
Separately, the release-workflow commit reflowed the README contribution
paragraph without altering documented behavior.

## v0.0.64 changed-file map

| Outcome | Files in the release range |
| --- | --- |
| MRKL parsing | `langchain/agents/mrkl/base.py`, `tests/unit_tests/agents/test_mrkl.py` |
| Wikipedia metadata | `langchain/docstore/wikipedia.py` |
| SQL selection and limits | `langchain/chains/sequential.py`, `langchain/chains/sql_database/base.py`, `langchain/chains/sql_database/prompt.py`, `langchain/prompts/base.py`, `langchain/sql_database.py`, `docs/modules/chains/examples/sqlite.ipynb` |
| API-chain prompting and examples | `langchain/chains/api/prompt.py`, `docs/modules/chains/examples/api.ipynb`, `docs/modules/chains/utility_how_to.rst` |
| Release and packaging | `.github/workflows/release.yml`, `pyproject.toml`, `README.md` |

## v0.0.64 verification evidence

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

## v0.0.64 known limitations

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

## Evolution from v0.0.64 to v0.0.100

### Comparison scope and scale

The second historical range is:

```text
v0.0.64..v0.0.100
```

`v0.0.64` was committed on 2023-01-15 and `v0.0.100` on 2023-03-02. The
exclusive range contains 361 commits and changes 561 files, with 49,246
insertions and 3,816 deletions. The `langchain` package grows from 149 to 300
Python files, while the test tree grows from 72 to 110 Python files. This is a
framework expansion rather than a narrowly scoped patch release.

The comparison below describes the net state at `v0.0.100`. Individual
releases inside the range sometimes introduced behavior that a later release
revised or reverted. For example, the `v0.0.64` SQL prompt explicitly required
a `LIMIT` clause, while `v0.0.100` retained the requested result bound without
requiring that dialect-specific spelling.

### Executive summary

By `v0.0.100`, LangChain had evolved in several connected directions:

- LLM, chain, agent, tool, and callback contracts gained explicit asynchronous
  paths, while OpenAI integrations added retries and callback-driven streaming
  inside generation. Many concrete implementations still reported async
  operation as unsupported.
- Prompts, selected output parsers, LLMs, chains, and agents gained broader
  typed serialization and loading support, including LangChainHub dispatch.
- Tools became callback-aware runtime objects, and toolkits packaged related
  tools behind SQL, JSON, OpenAPI, Pandas, CSV, Python, and vector-store agent
  factories.
- New conversational retrieval, constitutional, graph QA, document-analysis,
  and summarization-checking chains broadened the application layer.
- A formal document-loader package and vector-store indexing helpers created a
  recognizable ingestion pipeline from external sources to searchable
  documents.
- Model, embedding, search, and vector-store integrations expanded rapidly,
  increasing both capability and the optional dependency surface.
- Callbacks moved beyond verbose console output toward a shared observability
  layer for tracing, streamed tokens, errors, agent actions, and OpenAI usage.

### Abstraction evolution

The central architectural change is that a component increasingly became more
than a callable implementation. By `v0.0.100`, the emerging model was:

```text
component = typed configuration + execution protocol + lifecycle events
            + optional persistence
```

| Abstraction | At `v0.0.64` | By `v0.0.100` | Why it matters |
| --- | --- | --- | --- |
| LLM execution | `BaseLLM.generate()` and synchronous `_generate()`/`_call()` dominated; OpenAI exposed a separate beta `stream()` generator | `agenerate()`, `_agenerate()`, and `_acall()` complemented synchronous execution; OpenAI added retries and integrated token streaming with generation callbacks | Providers could participate in non-blocking applications and callback-driven output, but custom base subclasses acquired new protocol obligations |
| Chain execution | `Chain.__call__()` combined input preparation, memory, execution, callbacks, and output preparation | `prep_inputs()` and `prep_outputs()` separated lifecycle stages; `acall()`/`arun()` and saving were added | Chain execution became easier to observe and extend, while sync and async implementations still remained separate |
| Callbacks | Handlers primarily supported verbose execution events | Sync and async managers covered LLM, chain, tool, agent, error, and streamed-token events; tracing and OpenAI usage handlers were added | Observability became a cross-cutting execution plane instead of console logging embedded in components |
| Serialization | Saving and loading existed for selected prompts and LLMs | `_type` discriminators plus `dict()`, `save()`, and loader registries covered prompts, one registered output-parser type, selected chains, and selected agents | Supported component configuration could be treated as data, versioned, loaded from files, or resolved through the Hub |
| Tools | `Tool` was a dataclass holding a name, callable, description, and `return_direct` flag | `BaseTool` owned `run()`/`arun()`, callbacks, error handling, validation, and metadata; `@tool` adapted functions | Tool execution stopped being a special case implemented inside `AgentExecutor` |
| Agents | The planner/executor split already existed, but executors invoked raw tool functions directly | Agents declared allowed tools and supported `aplan()`; executors handled async dispatch, invalid tools, stopping, and callbacks, while `save_agent()` persisted only the underlying agent | Planning, loop control, and tool execution gained clearer responsibilities |
| Prompts | Prompt templates primarily formatted complete input dictionaries | Partial variables, template validation, serialization for `RegexParser`, and additional few-shot selectors were added | Prompts became bindable, typed configuration components rather than only format strings |
| Data ingestion | Documents, splitters, embeddings, and vector stores existed without a common loader layer | `BaseLoader.load()`/`load_and_split()`, `TextSplitter.split_documents()`, `VectorStore.add_documents()`, vector search, and MMR formed a staged pipeline | Many external sources could converge on the same `Document` and vector-store contracts |
| Results | `Generation` and `LLMResult` were `NamedTuple` values | Both became JSON-capable dataclasses | Results became extensible domain objects, although tuple-based consumer code could break |

### The emerging execution lifecycle

At `v0.0.64`, a chain's public call path performed most work in one method. By
`v0.0.100`, the intended lifecycle was more explicit:

```text
prepare and validate inputs
        ↓
emit start callback
        ↓
run synchronous or asynchronous implementation
        ↓
emit end or error callback
        ↓
validate outputs and persist memory context
```

LLMs, tools, and agents adopted parallel versions of this lifecycle. This made
tracing and streaming possible across nested components. Async and streaming
coverage was not uniform: base fallbacks and many concrete tools still raised
`NotImplementedError`, while streaming remained provider-specific rather than
a general chain protocol. The lifecycle was also repeated across each
abstraction instead of being expressed once as a common runnable interface.

### Tool and agent boundaries

The `v0.0.64` `Tool` was passive metadata around a single string function. By
`v0.0.100`, `Tool` inherited from the new `BaseTool` active runtime component.
A tool now owned invocation, asynchronous invocation, lifecycle callbacks,
error reporting, and direct-return metadata. `AgentExecutor` still interpreted
that metadata to decide whether to end its loop. The `@tool` decorator
preserved a simple function-authoring path by adapting a documented function
into that richer object.

`Agent` remained responsible for turning intermediate steps into an
`AgentAction` or `AgentFinish`. `AgentExecutor` increasingly became the loop
runtime: it validated allowed tool names, dispatched sync or async calls,
handled invalid model-selected tools, applied iteration limits, and emitted
callbacks. `BaseToolkit` then abstracted a related collection of tools, while
agent factory functions captured domain-specific assembly policy.

The protocol was still single-action and string-oriented. `AgentAction` held
one string `tool_input`, and `BaseTool` accepted one string argument. Tools
that needed structured or multiple values had to encode and parse them inside
that string.

### Serialization as typed configuration

Prompts, `RegexParser`, LLMs, selected chains, and selected agents increasingly
emitted a dictionary containing a `_type` discriminator. Loader registries used
that field to select a concrete implementation from YAML, JSON, or Hub content.
This separated reusable configuration from construction code and made the
supported subset of a component graph portable within the registry.

The mechanism was intentionally incomplete. Callback managers were excluded
from serialized chain configuration, a chain with memory could not be saved,
`AgentExecutor` could save only its underlying agent, and only registered
loader types could be reconstructed. Hub references could be pinned explicitly
but otherwise followed the then-default branch. Historical Python prompt files
were loaded with `exec()`, so untrusted files crossed a code-execution boundary.
The mechanism should therefore be understood as controlled configuration
loading, not arbitrary Python object serialization.

### Ingestion and retrieval pipeline

The new loader layer made the following pipeline visible:

```text
external source → BaseLoader → Document → TextSplitter
                → Embeddings → VectorStore/Index → chain or agent
```

Loader implementations covered local files, directories, web pages, PDF,
Word, PowerPoint, images, notebooks, Google Drive, YouTube, S3, Obsidian,
Evernote, Telegram, and other sources. Vector-store support expanded with
Chroma, Qdrant, Milvus, OpenSearch, Atlas, and DeepLake, while existing FAISS
and Pinecone implementations gained persistence or search controls.

The common vector-store API gained document insertion, similarity search by
vector, and maximal marginal relevance by vector. A general retriever protocol
had not yet separated retrieval policy from storage, so retrieval chains still
depended directly on vector stores.

The new pipeline was eager and synchronous. It had no lazy loader protocol,
incremental record manager, deduplication, or cleanup lifecycle, and optional
vector operations still defaulted to `NotImplementedError` when a backend did
not implement them.

### Model abstraction remained transitional

`OpenAIChat` introduced historical ChatGPT support, and `v0.0.100` allowed the
regular `OpenAI` constructor to route `gpt-3.5-turbo` model names to that
implementation. However, `OpenAIChat` still implemented the string-oriented
`BaseLLM` contract: it converted one string prompt into a user message and
returned generated text. Messages and chat models had not yet become separate
first-class abstractions.

The result schema was transitional too: `Generation` and `LLMResult` became
JSON-capable dataclasses, while `AgentAction` and `AgentFinish` remained
`NamedTuple` values.

Similarly, there was not yet one common `Runnable` protocol for prompts, LLMs,
chains, tools, and retrievers. `v0.0.100` is best understood as the transitional
release range in which the need for a unified execution abstraction became
clear.

### Integration inventory at v0.0.100

- **Models and embeddings:** Anthropic, Aleph Alpha, DeepInfra, GooseAI,
  CerebriumAI, Petals, ForefrontAI, Writer, Banana, Modal, StochasticAI,
  Hugging Face Endpoint, and self-hosted implementations joined the model
  layer. OpenAI and Cohere embeddings gained batching or normalized return
  behavior.
- **Chains and memory:** `ChatVectorDBChain`, `ConstitutionalChain`,
  `GraphQAChain`, `AnalyzeDocumentChain`, and
  `LLMSummarizationCheckerChain` were added. Entity and knowledge-graph memory
  expanded state beyond a plain conversation transcript.
- **Agents and tools:** SQL, JSON, OpenAPI, Pandas, CSV, Python, and
  vector-store toolkits standardized common agent assemblies.
- **Prompts and text splitting:** Partial prompt variables, string-based prompt
  construction, template validation, n-gram example selection, token-aware
  splitting, Markdown splitting, and Python-code splitting were added.
- **Packaging:** `dataclasses-json`, `tenacity`, and `aiohttp` became required
  runtime dependencies. Numerous provider-specific optional dependencies and
  a `langchain-server` console entry point were added, while Ruff began
  replacing the earlier Flake8 and isort lint configuration.

## Migration review map

These paths belong to the historical root package and should be inspected with
`git show v0.0.100:<path>` or a focused tag diff rather than through the modern
monorepo layout.

| Concern | Primary historical files | What to review |
| --- | --- | --- |
| Dependency and installation behavior | `pyproject.toml`, `poetry.lock` | Required versus optional dependencies, Python/platform constraints, extras, and the server entry point |
| Shared data contracts | `langchain/schema.py` | Loss of tuple semantics when `Generation` and `LLMResult` became dataclasses |
| LLM extension contract | `langchain/llms/base.py`, `langchain/llms/openai.py` | Async requirements, caching, retries, streaming, callback behavior, and historical chat routing |
| Chain extension contract | `langchain/chains/base.py`, `langchain/chains/loading.py` | Input/output lifecycle, memory interaction, async calls, `_chain_type`, and serialized loading |
| Agent runtime | `langchain/agents/agent.py`, `langchain/agents/loading.py` | Allowed tools, iteration stopping, async execution, invalid tools, return shapes, and persistence |
| Tool runtime | `langchain/agents/tools.py`, `langchain/tools/base.py` | Required descriptions, Pydantic validation, `run()`/`arun()`, callbacks, and function adaptation |
| Callback extension contract | `langchain/callbacks/base.py`, `langchain/callbacks/tracers/base.py` | New abstract events, sync/async handlers, verbose filtering, streaming, and tracing |
| Prompt extension contract | `langchain/prompts/base.py`, `langchain/prompts/prompt.py`, `langchain/prompts/loading.py` | `_prompt_type`, partial variables, parser serialization, validation, and file formats |
| Retrieval | `langchain/vectorstores/base.py`, `langchain/vectorstores/faiss.py`, `langchain/indexes/vectorstore.py` | Document insertion, vector queries, MMR, persistence, and index construction |
| Ingestion | `langchain/document_loaders/base.py`, `langchain/document_loaders/__init__.py`, `langchain/text_splitter.py` | Loader-to-document contract, source-specific dependencies, splitting, and metadata preservation |
| SQL behavior | `langchain/sql_database.py`, `langchain/chains/sql_database/base.py`, `langchain/chains/sql_database/prompt.py` | Table introspection, custom table information, returned intermediate data, dialect handling, and prompt-only result limits |
| Public imports | `langchain/__init__.py` and subsystem `__init__.py` files | New exports, compatibility aliases, moved symbols, and serialized type registries |

### Migration-sensitive behavior

- `AgentExecutor.max_iterations` changed from no default cap to `15`.
- Custom callback handlers needed the new `on_llm_new_token()` and
  `on_agent_action()` events and had to account for changed callback arguments,
  including `on_tool_start()` receiving an input string instead of an
  `AgentAction`. `BaseCallbackHandler` also changed from a Pydantic model with
  ignore fields into a plain ABC with ignore properties, while custom callback
  managers needed the new `set_handlers()` method.
- Direct `BaseLLM` subclasses needed `_agenerate()`; custom
  `BasePromptTemplate` subclasses needed `_prompt_type` for the expanded
  contracts. Custom `Agent` subclasses needed the new `_agent_type` property.
- `Tool` moved from a permissive dataclass toward the Pydantic-based
  `BaseTool`; its compatibility constructor required a description.
- Tuple unpacking or indexing of `Generation` and `LLMResult` no longer matched
  their dataclass representation.
- HyDE moved from the embeddings namespace into chains. Its public embeddings
  shim forwarded `from_llm()` but did not preserve every construction path.
  SerpAPI implementation moved into utilities; the old module re-exported
  `SerpAPIWrapper`, while `SerpAPIChain` survived only as a top-level alias.
  Direct implementation-module imports therefore required review.
- `SQLDatabase.__init__()` inserted `metadata` before the previous positional
  table filters, so old positional calls could bind to the wrong parameter.
  Its table information also began including three sample rows by default,
  changing token usage, database reads, and the data exposed to the model.
- SQL `top_k` remained a model instruction rather than an enforced database
  row limit. The prompt stopped insisting on the literal `LIMIT` syntax to
  support other SQL dialects.
- `BaseLLM`, `Chain`, and `Agent` overrode Pydantic's `dict()` with typed
  configuration representations. Code expecting a complete runtime model dump
  needed to distinguish configuration from live state.
- Python, PAL, bash, SQL, and request/OpenAPI tools crossed code-execution,
  database, or network trust boundaries and required explicit input and
  credential controls.
- Historical `.py` prompt files and unpinned Hub references carried the
  code-execution and deployment risks described in the serialization section.

## v0.0.100 comparison verification evidence

The immutable tag identities and range statistics can be reproduced locally:

```bash
git show -s --format='%H %ad %s' --date=iso-strict v0.0.64
git show -s --format='%H %ad %s' --date=iso-strict v0.0.100
git rev-list --count v0.0.64..v0.0.100
git diff --shortstat v0.0.64..v0.0.100
git diff --name-status v0.0.64..v0.0.100
git ls-tree -r --name-only v0.0.64 langchain | rg '\.py$' | wc -l
git ls-tree -r --name-only v0.0.100 langchain | rg '\.py$' | wc -l
git ls-tree -r --name-only v0.0.64 tests | rg '\.py$' | wc -l
git ls-tree -r --name-only v0.0.100 tests | rg '\.py$' | wc -l
```

The resulting endpoints are `08400f554215e98f886d4f4a1b98d6029c4b2a02`
and `499e76b1996787f714a020917a58a4be0d2896ac`; the count is 361 commits and the
short statistic is 561 files changed, 49,246 insertions, and 3,816 deletions.
Focused contract inspection can be reproduced with commands such as:

```bash
git diff v0.0.64..v0.0.100 -- \
  langchain/schema.py \
  langchain/callbacks/base.py \
  langchain/llms/base.py \
  langchain/chains/base.py \
  langchain/agents/agent.py \
  langchain/agents/tools.py \
  langchain/tools/base.py \
  langchain/prompts/base.py \
  langchain/vectorstores/base.py \
  pyproject.toml
```

There is no curated changelog in either tag. GitHub's generated release notes
for the intervening releases provide a useful index, while the immutable
[aggregate comparison](https://github.com/langchain-ai/langchain/compare/v0.0.64...v0.0.100)
and focused source diffs are the authority for net behavior.

This comparison was verified through source, export, dependency, and commit
history inspection. It did not execute the entire historical unit or
integration suite. The focused `v0.0.64` MRKL test described earlier remains
the only runtime verification recorded in this note.

## v0.0.100 comparison limitations

- The range crosses 36 rapid pre-1.0 releases, so public extension interfaces
  changed without the compatibility expectations of a stable major version.
- Generated release-note titles are terse and sometimes describe branch names
  instead of behavior. Net source state takes precedence over those titles.
- Some behavior was introduced and later revised inside the range; this note
  describes the `v0.0.100` endpoint rather than every intermediate state.
- Notebooks and documentation account for a large portion of the insertions
  and should not be treated as API contracts.
- Historical integrations depend on old provider SDKs, credentials, external
  services, and platform-specific optional packages. Examples are
  illustrative, not recommendations for a modern deployment.
- The absence of a common runnable, general retriever, and first-class chat
  model at `v0.0.100` is an architectural observation about these tags, not a
  statement about current LangChain.
- Async methods, vector operations, and serialization loaders described an
  intended framework contract, but support varied by concrete implementation;
  their presence did not guarantee an end-to-end capability.

## Compact recall notes

The shortest useful mental model is:

```text
v0.0.64: composable synchronous LLM components
v0.0.100: typed components with sync/async contracts, callbacks, and loading
next architectural pressure: unify those parallel protocols
```

## Review questions

When reviewing claims or migrating historical code, ask:

1. Is the claim a net difference between the two endpoint tags, rather than a
   transient change from an intermediate release?
2. Does custom code subclass an expanded base contract such as `BaseLLM`,
   `BasePromptTemplate`, `BaseCallbackHandler`, `Chain`, or `BaseTool`?
3. Does it rely on tuple behavior, an old import path, an uncapped agent loop,
   or a particular callback method signature?
4. Is a configured dependency available on the intended Python version and
   platform?
5. Are network, database, filesystem, and code-execution trust boundaries
   explicit for loaders, tools, and agent-selected actions?
6. Is the behavior supported by a runtime test, or only by source and release
   history evidence?
