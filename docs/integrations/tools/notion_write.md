# Notion Write Toolkit

The **LangChain Notion Write Toolkit** is an external package that equips LangChain agents with *first-class* read/write access to Notion via the official Notion API. It targets scenarios where an assistant needs to collate research, prepare status reports, or log decisions directly in Notion without human copy/paste steps.

> Package name: [`langchain-notion-tools`](https://pypi.org/project/langchain-notion-tools/)
> 
> Maintainer: [Dinesh Kumar Kummara](https://github.com/dineshkumarkummara)
> 
> License: MIT (same as LangChain core)

---

## Why another Notion integration?

Existing community integrations focus on **searching** Notion. Product teams asked for the inverse—creating/updating pages as part of an agent workflow. The toolkit provides:

- A **NotionWriteTool** that understands LangChain’s tool schema and safely appends or replaces Notion block content.
- A matching **NotionSearchTool** for retrieval so the same toolkit can both read and write.
- A **NotionToolkit** factory that shares client configuration, retries, and rate-limit handling across both tools.
- CLI utilities (`notion-write`, `notion-search`) to debug payloads outside the agent loop.
- Strong input validation (Pydantic + JSON Schema) to prevent malformed Notion API calls.
- Built‑in sanitisation that strips unsupported block types, trims oversized payloads, and redacts secrets from logs.

---

## Installation

```bash
pip install langchain-notion-tools
```

The package depends only on `langchain-core`, `pydantic>=2`, and the official `notion-client`. No extra vendor SDKs are required.

---

## Authenticating with Notion

The toolkit reads credentials from environment variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `NOTION_API_TOKEN` | ✅ | Integration token generated in Notion’s developer console. |
| `NOTION_DEFAULT_PARENT_PAGE_ID` | Optional | Page/database ID used for create operations when the agent omits a parent. |

```bash
export NOTION_API_TOKEN="secret_xyz"
export NOTION_DEFAULT_PARENT_PAGE_ID="a1b2c3d4e5f6"
```

Tokens are redacted automatically when the toolkit logs requests.

---

## Toolkit contents

```python
from langchain_notion_tools import create_toolkit

toolkit = create_toolkit()
toolkit.tools  # -> [NotionSearchTool(...), NotionWriteTool(...)]
```

| Tool | Description |
| --- | --- |
| `NotionSearchTool` | Normalises Notion search results (pages or databases) with extracted titles, previews, and parent IDs. Supports async usage. |
| `NotionWriteTool` | Creates pages, appends/overwrites blocks, or updates page properties. Validates payloads and supports dry-run summaries so the agent can confirm before writing. |

Each tool exposes `.args_schema` JSON schemas suitable for `AgentType.OPENAI_FUNCTIONS` and other structured-output LLM modes.

---

## End-to-end agent example

```python
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_notion_tools import create_toolkit

# Share a single Notion client between search + write tools.
toolkit = create_toolkit()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent(
    toolkit.tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

prompt = (
    "Find the engineering status board, summarise blockers, "
    "and create a page called 'LLM Launch Update' with action items."
)
result = agent.invoke({"input": prompt})

print(result["output"])
```

The agent will:

1. Call `notion_search` to locate the relevant status board.
2. Collate the data via the LLM.
3. Invoke `notion_write` to create or append to a Notion page.
4. Return a human-readable summary.

---

## CLI debugging (optional)

Before wiring an agent, you can test payloads using the installed CLIs:

```bash
# Preview the blocks that would be sent to Notion
notion-write   --title "Sprint Review"   --parent-page "$NOTION_DEFAULT_PARENT_PAGE_ID"   --blocks-from-text "## Summary
- Release green
- Ship follow-up doc"   --dry-run

# Retrieve the raw results of a search query
notion-search --query "roadmap" | jq .
```

CLI and tool schemas share the same validation layer, so failures match what an agent would see.

---

## Example block payload logging

![Notion page creation demo](./notion-write-demo.png)

The screenshot shows the toolkit appending a bulleted list with a success summary returned to the caller.

---

## Advanced configuration

- **Custom settings**: `create_toolkit(settings=NotionClientSettings(...))` allows you to tweak Notion client timeouts, retries, or provide a pre-built sync/async client bundle.
- **Dry-run summaries**: pass `is_dry_run=True` to `NotionWriteTool` to inspect a natural-language summary instead of committing changes immediately.
- **Property updates only**: supply `update={"page_id": "...", "mode": "append"}` with `properties={...}` to edit select properties without touching blocks.
- **JSON Schema validation**: `.args_schema.schema()` returns the exact JSON schema you can expose through agent tool registries or OpenAI function-calling metadata.

### Supported block helpers

The package ships helpers that mirror the Notion REST block types. Each helper returns a valid block payload that `NotionWriteTool` can accept directly.

| Helper | Notion type | Notes |
| --- | --- | --- |
| `paragraph(text)` | `paragraph` | Arbitrary rich text content. |
| `heading_1/2/3(text)` | `heading_1/2/3` | Automatically trims surrounding whitespace. |
| `bulleted_list_item(text)` | `bulleted_list_item` | Used by the Markdown-to-Notion converter for `- ` lines. |
| `numbered_list_item(text)` | `numbered_list_item` | Supports incremental numbering. |
| `to_do(text, checked=False)` | `to_do` | Agents can toggle follow-up tasks. |
| `toggle(text, children=None)` | `toggle` | Allows nested bullet hierarchies. |
| `callout(text, icon=None)` | `callout` | Icon payload fully supported. |
| `quote(text)` | `quote` | Useful for quoting previous decisions. |
| `code(text, language="plain text")` | `code` | Automatically strips external hyperlinks for safety. |

The toolkit enforces two global guards before calling the Notion API:

1. `MAX_BLOCKS` (default 100) — prevents agents from accidentally flooding a page.
2. `MAX_TOTAL_TEXT_LENGTH` (default 20,000 characters) — keeps payloads within Notion’s limits.

If a request exceeds either limit, a `NotionConfigurationError` is raised before the API call is sent.

### Error handling philosophy

- Upstream HTTP errors are mapped to `ToolException` so agent executors can surface the failure reason.
- Notion API error codes (e.g., rate limits) are appended to the exception message for easier debugging.
- Token redaction strips the integration token from stack traces and logs.
- All errors include actionable summaries, e.g., *“Update page blocks failed (code conflict_resolution_failed) [status 409]”*.

### Roadmap & community contributions

The maintainer welcomes PRs that:

- Add more block helper variants (synced blocks, databases).
- Expose additional CLI options for property templates.
- Provide recipe notebooks that chain Notion updates with other LangChain integrations (e.g., summarising Slack threads into Notion).

See the upstream [`CONTRIBUTING.md`](https://github.com/dineshkumarkummara/langchain-notion-tool/blob/main/CONTRIBUTING.md) for the test matrix and coding standards.

---

## Resources

- [GitHub repository](https://github.com/dineshkumarkummara/langchain-notion-tool) – source, CLI entry points, and examples.
- [PyPI package](https://pypi.org/project/langchain-notion-tools/) – release artifacts and changelog.
- [Full documentation site](https://dineshkumarkummara.github.io/langchain-notion-tool/) – quickstart, troubleshooting, JSON Schemas, and CLI walkthroughs.

If you need additional recipe coverage (e.g., orchestrating multi-step Notion updates) open an issue or PR in the upstream repository above. The maintainer is actively looking for feedback to expand block helper coverage and database workflows.
