# LangChain v1 examples

Small, self-contained, runnable examples that demonstrate the core build surface of
the actively maintained `langchain` package (v1) — agents, tools, structured output,
memory, and middleware. Each file is a copy-paste starting point.

## Prerequisites

- The `langchain_v1` environment is set up:

  ```bash
  cd libs/langchain_v1 && uv sync
  ```

- An OpenAI API key exported in your shell:

  ```bash
  export OPENAI_API_KEY="sk-..."
  ```

## Running

Run each script from `libs/langchain_v1` so it uses that package's environment. The
`--with langchain-openai` flag supplies the OpenAI integration **ephemerally** — it is
not a dependency of `langchain`, so `pyproject.toml` and `uv.lock` stay untouched.

```bash
cd libs/langchain_v1
uv run --with langchain-openai python ../../examples/01_react_agent.py
uv run --with langchain-openai python ../../examples/02_structured_extractor.py
uv run --with langchain-openai python ../../examples/03_memory_chatbot.py   # interactive
uv run --with langchain-openai python ../../examples/04_planner_agent.py
```

## What each example shows

| File | Demonstrates |
|------|--------------|
| `01_react_agent.py` | ReAct loop: the agent calls `@tool` functions (`get_weather`, `calculator`) to answer a question. |
| `02_structured_extractor.py` | `response_format` → a validated Pydantic object pulled from free text. |
| `03_memory_chatbot.py` | `checkpointer` + a fixed `thread_id` → multi-turn memory in an interactive chat. |
| `04_planner_agent.py` | `TodoListMiddleware` → the agent decomposes a goal into a todo plan. |

## Notes

- All four use `openai:gpt-4o-mini` to keep cost low. Change the `MODEL` constant in
  any file to use a different model — e.g. another OpenAI model, or an Anthropic model
  with `langchain-anthropic` installed and `ANTHROPIC_API_KEY` set (and `--with
  langchain-anthropic` on the run command).
- API keys are read from the environment at runtime — never hardcode them.
- `03_memory_chatbot.py` is interactive: send a message, then a follow-up that refers
  back to it (e.g. "what's my name?") to see the memory work. Type `exit` to quit.
