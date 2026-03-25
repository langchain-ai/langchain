# Mistakes & Lessons Learned

## Format
Each entry: what happened, why it happened, how to avoid it.

---

### 2026-03-25 — Anthropic SDK doesn't work with OpenRouter
**What:** The original `llm_router.py` used `anthropic.Anthropic(base_url="https://openrouter.ai/api/v1")` which returned 404 errors.
**Why:** OpenRouter uses the OpenAI chat completions API format (`/chat/completions`), not Anthropic's messages API format (`/messages`). The Anthropic SDK sends to the wrong endpoint.
**Fix/Avoid:** Use the OpenAI SDK (`openai.OpenAI`) when routing through OpenRouter. Keep provider-specific code behind an `if _use_openrouter()` check.

---

### 2026-03-25 — OpenRouter model IDs differ from Anthropic native IDs
**What:** Calls to OpenRouter with `claude-haiku-4-5-20251001` failed with "not a valid model ID".
**Why:** OpenRouter uses its own model ID format: `anthropic/claude-haiku-4.5` instead of Anthropic's `claude-haiku-4-5-20251001`.
**Fix/Avoid:** Maintain separate model ID lookup tables per provider (`_ANTHROPIC_MODELS` vs `_OPENROUTER_MODELS`). Use an internal tier name (haiku/sonnet/opus) and resolve to provider-specific IDs at call time.

---

### 2026-03-25 — Railway deployment failed with no config
**What:** Railway couldn't build or deploy the repo — it failed silently.
**Why:** No Dockerfile, Procfile, or railway.toml existed. Nixpacks couldn't determine how to build a Python monorepo with multiple `pyproject.toml` files.
**Fix/Avoid:** Always create deployment config (`Dockerfile` + `railway.toml`) before pushing to Railway. Use `.dockerignore` to exclude the monorepo `libs/` directory.

---

### 2026-03-25 — exec_sql chicken-and-egg problem
**What:** Couldn't create Supabase tables via the REST API because the `exec_sql` RPC function didn't exist yet, and you need `exec_sql` to run CREATE TABLE statements.
**Why:** Supabase REST API only exposes existing tables/functions. You can't run arbitrary SQL through it without a server-side function.
**Fix/Avoid:** Run the initial migration SQL directly in the Supabase Dashboard SQL Editor. The migration creates the `exec_sql` function first, then all tables. After that, `supabase_tools.py` can use `exec_sql` for future schema changes.
