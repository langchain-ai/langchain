# Threat Model: langchain-ai/langchain

> Generated: 2026-03-27 | Commit: 494b760028 | Scope: /workspace/langchain (full monorepo) | Visibility: Open Source | Mode: deep

For vulnerability reporting, see [SECURITY.md](SECURITY.md) if present; otherwise report via [GitHub Security Advisories](https://github.com/langchain-ai/langchain/security/advisories/new).

---

## Scope

### In Scope

- `libs/core/` — `langchain-core` v1.2.22: base abstractions, serialization, messages, prompts, tools, runnables, callbacks, SSRF protection
- `libs/langchain_v1/` — `langchain` v1.2.13: active implementation layer, agent middleware, execution policies, file search middleware
- `libs/langchain/` — `langchain-classic` v1.0.3: legacy package (no new features, retained for compatibility)
- `libs/text-splitters/` — `langchain-text-splitters` v1.1.1: document chunking utilities
- `libs/partners/` — 15 partner integration packages (OpenAI, Anthropic, Groq, Mistral, Fireworks, Ollama, DeepSeek, xAI, Perplexity, OpenRouter, Exa, Chroma, Qdrant, Nomic, HuggingFace)
- `libs/model-profiles/` — `langchain-model-profiles`: model configuration metadata

### Out of Scope

- `libs/standard-tests/` — shared test harnesses; not shipped as attack surface
- User application code that imports from langchain — users control their own deployment, model selection, custom tools, and callbacks
- LLM model behavior — the project cannot guarantee model safety across all models users may select
- Deployment infrastructure — users control their own hosting, network topology, and secrets management outside of what langchain ships
- External repos (`langchain-ai/langchain-google`, `langchain-ai/langchain-aws`, etc.) — separate threat models apply
- LangSmith, LangGraph — separate products and repositories

### Assumptions

1. The project is used as a library/framework — users control their own application code, model selection, and deployment infrastructure.
2. API keys are sourced from environment variables or passed explicitly; the framework does not store them persistently.
3. Users are responsible for validating that serialized LangChain objects (passed to `loads()`/`load()`) come from trusted sources.
4. The `langchain-core` serialization allowlist (`allowed_objects='core'`) is the default and correct choice for untrusted data.
5. Agent execution policies (HostExecutionPolicy, DockerExecutionPolicy, CodexSandboxExecutionPolicy) are selected by deployers; the project ships all three, each with documented security guarantees.

---

## System Overview

LangChain is a Python framework for building LLM-powered applications. It provides base abstractions (messages, prompts, tools, runnables, callbacks), a plugin system for integrating with external LLM providers, and utilities for agent orchestration including tool execution, file search, and subprocess isolation. Users import from langchain packages to compose pipelines; the framework itself does not serve HTTP traffic or store user data — it is a library that processes data on behalf of user applications.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Application                          │
│                                                                  │
│  User Code ──────────► langchain-core ──────► Partner SDK       │
│      │                 (C1: messages,          (C4: OpenAI,      │
│      │                  prompts, tools,         Anthropic, etc.) │
│      │                  serialization)               │           │
│      │                      │                        ▼           │
│      │               ─ ─ ─ ─│─ ─ ─ ─ ─ ─    External LLM API  │
│      │               TB1    │ TB2                               │
│      │                      │                                    │
│      └──► Agent Middleware (C5)                                  │
│               │                                                  │
│          ─ ─ ─│─ ─ ─ TB3                                        │
│               ▼                                                  │
│          OS Subprocess / Docker / Codex Sandbox                  │
│                                                                  │
│      File Search Middleware (C6)                                 │
│          ├── glob_search ──► filesystem (within root_path)       │
│          └── grep_search ──► filesystem (within root_path)       │
│               ─ ─ ─ TB4 ─ ─                                     │
└──────────────────────────────────────────────────────────────────┘
```

> Note: Trust boundaries TB1–TB4 are overlaid on the diagram with dashed lines. TB5 (serialization API) is between User Code and langchain-core's `loads()`/`load()` functions.

---

## Components

| ID  | Component | Description | Trust Level | Default? | Entry Points |
|-----|-----------|-------------|-------------|----------|--------------|
| C1  | langchain-core | Base abstractions: messages, prompts, tools, runnables, callbacks, serialization, SSRF protection | framework-controlled | Yes | `langchain_core.load.load:load`, `langchain_core.load.load:loads`, `langchain_core.messages.utils:messages_from_dict`, `langchain_core.prompts.chat:ChatPromptTemplate.format_messages` |
| C2  | langchain (langchain_v1) | Active implementation layer: agents, agent middleware, chains, document utilities | framework-controlled | Yes | `langchain.agents.middleware._execution:HostExecutionPolicy.spawn`, `langchain.agents.middleware.file_search:FilesystemFileSearchMiddleware` |
| C3  | langchain-classic | Legacy package, no new features; retained for backward compatibility | framework-controlled | Yes (if installed) | Deprecated API surface |
| C4  | Partner Integrations | 15 partner packages for external LLM/vector-store APIs; each wraps a provider SDK | external | No (opt-in per provider) | Per-partner constructors: `ChatOpenAI()`, `ChatAnthropic()`, `ChatGroq()`, etc. |
| C5  | Agent Execution Policies | Subprocess execution with configurable isolation: HostExecutionPolicy, DockerExecutionPolicy, CodexSandboxExecutionPolicy | user-controlled | No (opt-in; must be configured) | `_execution:HostExecutionPolicy.spawn`, `_execution:DockerExecutionPolicy.spawn`, `_execution:CodexSandboxExecutionPolicy.spawn` |
| C6  | File Search Middleware | Glob and grep search over local filesystem within a user-configured root_path | user-controlled | No (opt-in) | `file_search:FilesystemFileSearchMiddleware.glob_search`, `file_search:FilesystemFileSearchMiddleware.grep_search` |
| C7  | Serialization System | Allowlist-based JSON (de)serialization for LangChain objects; blocks jinja2 by default | framework-controlled | Yes (when load/loads called) | `langchain_core.load.load:loads`, `langchain_core.load.load:load`, `langchain_core.load.dump:dumpd`, `langchain_core.load.dump:dumps` |
| C8  | Prompt Templates | Template rendering for LLM prompts; supports f-string (safe) and jinja2 (blocked in deserialization) | user-controlled | Yes (when prompts used) | `langchain_core.prompts.loading:load_prompt` (deprecated since 1.2.21), `langchain_core.prompts.chat:ChatPromptTemplate.from_template` |
| C9  | SSRF Protection | URL validation utility blocking private IPs, localhost, and cloud metadata endpoints | framework-controlled | No (called explicitly by components that fetch URLs) | `langchain_core._security._ssrf_protection:validate_safe_url`, `langchain_core._security._ssrf_protection:is_safe_url` |
| C10 | Model Profiles | Model configuration metadata (capabilities, context windows, etc.) | framework-controlled | No | `langchain_model_profiles.cli:main` |

---

## Data Classification

Classifies all sensitive data types found in the codebase with their sensitivity levels, storage locations, and retention policies.

| ID  | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|-----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | API credentials | `openai_api_key`, `anthropic_api_key`, `groq_api_key`, `mistral_api_key`, `fireworks_api_key`, `openrouter_api_key`, `xai_api_key`, `perplexity_api_key` | Critical | In-memory (Pydantic `SecretStr`), OS environment variables | Yes (SecretStr; masked in `repr`/logs) | Process lifetime; sourced from env vars at instantiation | All |
| DC2 | LLM conversation data | `HumanMessage.content`, `AIMessage.content`, `ToolMessage.content`, prompt template variables | High | In-memory (transient); persistence is user-application responsibility | N/A (not persisted by framework) | Transient (garbage-collected) | GDPR, CCPA (when containing PII) |
| DC3 | Serialized LangChain objects | JSON payloads passed to `loads()`/`load()`; includes `kwargs` for any allowed class | High | User-application storage (not framework responsibility) | N/A (framework does not store) | User-controlled | N/A |
| DC4 | OS environment variables | Arbitrary `os.environ` values accessible via `secrets_from_env=True` in `Reviver` | Critical | Host OS environment | N/A | Process lifetime | All (secrets may include credentials, tokens, API keys) |
| DC5 | Filesystem paths and content | `FilesystemFileSearchMiddleware.root_path`, matched file paths and content returned by grep/glob | Medium | Host filesystem (read-only by middleware) | N/A | Not stored by middleware | N/A |

### Data Classification Details

#### DC1: API credentials

- **Fields**: `openai_api_key` (`ChatOpenAI`), `anthropic_api_key` (`ChatAnthropic`), `groq_api_key` (`ChatGroq`), `mistral_api_key` (`ChatMistralAI`), `fireworks_api_key` (`ChatFireworks`), `openrouter_api_key` (`ChatOpenRouter`), `xai_api_key` (`ChatXAI`), `perplexity_api_key` (`ChatPerplexity`)
- **Storage**: In-memory only, wrapped in Pydantic `SecretStr`. Sourced from environment variables via `langchain_core.utils.utils:secret_from_env` at instantiation. Not persisted to disk or transmitted outside of the intended API call.
- **Access**: Read by partner SDK constructors at instantiation. The `SecretStr` wrapper prevents accidental logging via `repr`/`str`.
- **Encryption**: In-memory only; no at-rest encryption needed (not persisted). In-transit via HTTPS to each provider API.
- **Retention**: Process lifetime. Credentials released when the model object is garbage-collected.
- **Logging exposure**: Protected by `SecretStr`; direct access requires `.get_secret_value()`. Risk exists if users log message contents that embed credentials (user responsibility).
- **Cross-border**: Transmitted to respective provider APIs over HTTPS; users choose which provider and thus which jurisdiction.
- **Gaps**: None identified in framework code. Risk is user configuration (e.g., committing credentials to source code or leaking in application logs).

#### DC4: OS environment variables

- **Fields**: Any `os.environ` key named in a serialized payload's `secret` field, when `secrets_from_env=True` is passed to `loads()`/`load()`.
- **Storage**: Host OS environment.
- **Access**: `langchain_core.load.load:Reviver.__call__` reads `os.environ[key]` directly when `secrets_from_env=True`. An attacker who controls the serialized payload can name any environment variable.
- **Encryption**: N/A (environment variables are plaintext at the OS level).
- **Retention**: Process lifetime.
- **Logging exposure**: Values returned as deserialized constructor kwargs; exposure depends on user logging.
- **Gaps**: **Critical gap**: When `secrets_from_env=True`, any environment variable can be read by a crafted payload. This is documented (see T1) but opt-in. Default is `False`.

---

## Trust Boundaries

| ID  | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|-----|----------|-------------|-------------------|---------------------------|
| TB1 | User application ↔ langchain-core | Entry point for all user-provided inputs to the framework | Parsing logic, type validation (Pydantic), default configuration values, tool argument schemas | Model selection, custom tool implementations, custom callback handlers, application-level input sanitization, deployment topology |
| TB2 | Framework ↔ external LLM provider API | HTTPS API boundary with external model providers | Request formatting, API key management via `SecretStr`, SSRF protection for image URLs (`validate_safe_url`) | Model behavior and output content, API availability, provider-side authentication failures |
| TB3 | Framework ↔ agent subprocess | Process boundary for agent tool execution | Execution policy selection (Host/Docker/Codex), process group creation, resource limits (CPU/memory via prlimit), environment variable filtering, workspace path | Commands executing inside the subprocess (no isolation in HostExecutionPolicy); container-level isolation only in DockerExecutionPolicy |
| TB4 | Framework ↔ filesystem | Filesystem access via file search and prompt loading | Path traversal prevention (`file_search:FilesystemFileSearchMiddleware._validate_and_resolve_path`, `loading:_validate_path`), root_path confinement, `..`/`~` blocking | Content of files within root_path; symbolic link targets (see T7) |
| TB5 | User application ↔ serialization API | JSON deserialization entry point via `loads()`/`load()` | Namespace allowlist (`DEFAULT_NAMESPACES`), class path allowlist (`allowed_objects='core'` default), jinja2 blocking (`default_init_validator`), `__lc_escaped__` injection protection | Trustworthiness of the serialized payload; whether `secrets_from_env=True` is used; whether `allowed_objects='all'` is used |

### Boundary Details

#### TB1: User application ↔ langchain-core

- **Inside**: Pydantic model validation on all public classes. Default configurations (e.g., `template_format='f-string'`, `allowed_objects='core'`, `secrets_from_env=False`). Core parsing logic for messages (`langchain_core.messages.utils:messages_from_dict`), prompts, and tools.
- **Outside**: What users pass as tool implementations, callback handlers, and model configurations. Users may register tools that perform arbitrary operations; the framework validates tool argument schemas but not tool behavior.
- **Crossing mechanism**: Python function calls to public API methods.

#### TB2: Framework ↔ external LLM provider API

- **Inside**: HTTPS connections via provider SDKs. API key transmission via `SecretStr.get_secret_value()`. SSRF validation for image URLs via `langchain_core._security._ssrf_protection:validate_safe_url` (called in `langchain_openai.chat_models.base:_url_to_size`).
- **Outside**: LLM response content is untrusted — it may contain prompt injection, malicious tool call arguments, or unexpected structured data. The framework passes LLM-generated tool call arguments to user-configured tools without additional sanitization beyond Pydantic schema validation.
- **Crossing mechanism**: HTTPS requests to provider API endpoints.

#### TB3: Framework ↔ agent subprocess

- **Inside**: `_execution:_launch_subprocess` constructs the `subprocess.Popen` call without `shell=True`. `DockerExecutionPolicy._build_command` sets `--network none` by default. `HostExecutionPolicy` applies `resource.prlimit` limits where available.
- **Outside**: Commands executing inside the subprocess are outside the framework's control. `HostExecutionPolicy` offers no filesystem or network sandboxing.
- **Crossing mechanism**: `subprocess.Popen` with explicit argument list (no shell=True).

#### TB4: Framework ↔ filesystem

- **Inside**: `file_search:FilesystemFileSearchMiddleware._validate_and_resolve_path` rejects `..` and `~`, resolves to an absolute path, and verifies the result is within `root_path` using `Path.relative_to()`. `loading:_validate_path` rejects absolute paths and `..` components.
- **Outside**: Symbolic links are resolved by `Path.resolve()` before the `relative_to` check — a symlink within `root_path` pointing outside can be followed (see T8).
- **Crossing mechanism**: Python `Path.glob()`, `Path.rglob()`, `Path.read_text()`, `subprocess.run(["rg", ...])`.

#### TB5: User application ↔ serialization API

- **Inside**: `load.py:Reviver.__init__` builds the class path allowlist. `Reviver.__call__` enforces namespace validation (`DEFAULT_NAMESPACES`), allowlist check, `Serializable` subclass check, and calls `init_validator`. `_validation:_is_escaped_dict` prevents user data dicts from being treated as LC objects.
- **Outside**: The content of the JSON payload; whether the caller passes trusted or untrusted data; whether the caller enables `secrets_from_env=True` or broadens `allowed_objects`.
- **Crossing mechanism**: `json.loads(text)` + `Reviver` object hook.

---

## Data Flows

| ID  | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|-----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | User application | C7 Serialization (`loads`/`load`) | JSON serialized LC object payload | DC3 | TB5 | Python function call |
| DF2 | User application | C8 Prompt Templates → C1 langchain-core | User input, prompt template variables | DC2 | TB1 | Python function call |
| DF3 | C1 langchain-core via C4 Partner SDKs | External LLM provider API | Messages (DC2) + API credentials (DC1) | DC1, DC2 | TB2 | HTTPS |
| DF4 | External LLM provider API via C4 | C1 langchain-core | LLM response content, tool call arguments | DC2 | TB2 | HTTPS |
| DF5 | C2 langchain_v1 Agent Middleware | OS subprocess (C5 execution policy) | Shell commands, environment variables | DC1 (env passthrough), DC5 | TB3 | `subprocess.Popen` |
| DF6 | User application | C6 File Search Middleware → filesystem | Glob/grep patterns, file paths and content | DC5 | TB4 | Python/ripgrep |
| DF7 | User application (messages with image_url) | C9 SSRF Protection → external HTTP | Image URL, fetched image content | DC2 | TB2 | HTTPS via `httpx` |
| DF8 | User application | C8 `load_prompt` → filesystem | Prompt config file path, template file path | DC5 | TB4 | Python file I/O |

### Flow Details

#### DF1: User application → Serialization API (`loads`/`load`)

- **Data**: JSON string or dict representing serialized LangChain objects. Sensitivity depends on whether it contains `secrets` fields (DC3/DC4).
- **Validation**: `Reviver.__call__` enforces namespace allowlist, class path allowlist, jinja2 blocking via `default_init_validator`. `_is_escaped_dict` prevents user data injection.
- **Trust assumption**: Caller ensures the payload comes from a trusted source. If `secrets_from_env=True`, caller trusts the payload completely.

#### DF3: Framework → External LLM provider

- **Data**: `BaseMessage` list (conversation history), tool schemas, API key.
- **Validation**: API key is `SecretStr`; `.get_secret_value()` called only at API call time. Request formatted by provider SDK.
- **Trust assumption**: The LLM provider API is trusted; responses in DF4 are not.

#### DF4: External LLM provider → Framework (LLM output)

- **Data**: LLM-generated text, tool call names, and tool call arguments.
- **Validation**: Tool call arguments are validated against Pydantic schemas defined in `BaseTool`. Tool names are matched against the registered tool list. No sanitization of free-form text responses.
- **Trust assumption**: LLM output is **untrusted** — equivalent to user input. Tool call argument schemas provide partial validation but cannot prevent adversarial content in text fields.

#### DF5: Agent Middleware → OS subprocess

- **Data**: Shell commands (list form, no shell=True), environment variables (from `env` mapping passed by caller).
- **Validation**: `HostExecutionPolicy.spawn` applies optional CPU/memory limits. `DockerExecutionPolicy.spawn` adds `--network none` by default. No command-content validation — commands are passed as-is.
- **Trust assumption**: Commands are constructed by trusted framework/user code. In agent scenarios, command construction may be influenced by LLM output.

#### DF6: File Search Middleware → filesystem

- **Data**: User-supplied glob pattern and base path; returns file paths and content.
- **Validation**: `_validate_and_resolve_path` rejects `..` and `~`, checks result is within `root_path`. Regex patterns compiled before use (validation only; no ReDoS protection).
- **Trust assumption**: `root_path` is set by the deployer to a safe directory. File content returned may contain sensitive data.

---

## Threats

| ID  | Data Flow | Classification | Threat | Boundary | Severity | Status | Validation | Code Reference |
|-----|-----------|----------------|--------|----------|----------|--------|------------|----------------|
| T1  | DF1 | DC4 | Arbitrary OS environment variable exfiltration via crafted serialized payload when `secrets_from_env=True` | TB5 | High | Mitigated (default `False`; documented warning) | Verified | `langchain_core/load/load.py:Reviver.__call__` |
| T2  | DF1 | DC3 | Side effects in allowed class `__init__` during deserialization (network calls, file I/O) when using `allowed_objects='all'` | TB5 | Medium | Accepted (documented in module docstring) | Likely | `langchain_core/load/load.py:loads` |
| T3  | DF4 → DF2 | DC2 | Prompt injection via LLM-generated tool call arguments influencing subsequent LLM context | TB2 | Medium | Unmitigated (inherent to agentic frameworks; user responsibility) | Unverified | `langchain_core/tools/base.py:BaseTool.invoke` |
| T4  | DF5 | DC1 | API credentials leaked into subprocess environment via `env` dict passthrough in agent execution | TB3 | High | Unmitigated (framework passes caller-supplied `env` dict; no credential filtering) | Likely | `langchain/agents/middleware/_execution.py:_launch_subprocess` |
| T5  | DF7 | — | SSRF via DNS rebinding bypass in `ChatOpenAI.get_num_tokens_from_messages` image URL token counting | TB2 | Medium | Partially mitigated (`validate_safe_url` added; DNS rebinding is residual risk) | Likely | `langchain_openai/chat_models/base.py:_url_to_size`, `langchain_core/_security/_ssrf_protection.py:validate_safe_url` |
| T6  | DF5 | — | Unrestricted host filesystem and network access when `HostExecutionPolicy` is used for agent subprocess execution | TB3 | Medium | Accepted (documented: "best suited for trusted or single-tenant environments") | Verified | `langchain/agents/middleware/_execution.py:HostExecutionPolicy.spawn` |
| T7  | DF6 | DC5 | Path traversal via symbolic link outside `root_path` in `FilesystemFileSearchMiddleware` | TB4 | Medium | Unmitigated (`.resolve()` follows symlinks before `relative_to` check; symlink pointing outside root passes validation) | Likely | `langchain/agents/middleware/file_search.py:FilesystemFileSearchMiddleware._validate_and_resolve_path` |
| T8  | DF6 | — | ReDoS (Regular Expression Denial of Service) via malicious regex pattern in `grep_search` | TB4 | Low | Unmitigated (no timeout or complexity limit on user-supplied regex) | Likely | `langchain/agents/middleware/file_search.py:FilesystemFileSearchMiddleware.grep_search` |

### Threat Details

#### T1: Arbitrary OS environment variable exfiltration via `secrets_from_env=True`

- **Flow**: DF1 (User app → `loads()`/`load()`)
- **Description**: When `secrets_from_env=True` is passed to `loads()`/`load()`, a crafted serialized payload can name any OS environment variable in its `secret` fields (e.g., `{"lc":1,"type":"secret","id":["AWS_SECRET_ACCESS_KEY"]}`). The `Reviver.__call__` method reads that key from `os.environ` and injects it as a constructor `kwarg` for the deserialized object.
- **Preconditions**: (1) User passes `secrets_from_env=True` to `loads()`/`load()`; AND (2) user passes attacker-controlled serialized data. Both conditions must be true simultaneously.
- **Mitigations**: Default is `secrets_from_env=False`. The module docstring and `loads()` docstring explicitly warn: "A crafted payload can name arbitrary environment variables in its `secret` fields, so enabling this on untrusted data can leak sensitive values."
- **Residual risk**: Users who misread the warning and enable `secrets_from_env=True` on user-provided inputs remain vulnerable. The framework cannot prevent user misconfiguration.
- **Historical context**: GHSA-c67j-w6g6-q2cm covers this pattern.

#### T2: Side effects in allowed class `__init__` during deserialization

- **Flow**: DF1 (User app → `loads()`/`load()`)
- **Description**: When `allowed_objects='all'` is used, the allowlist includes partner integrations such as `ChatOpenAI`. If `ChatOpenAI` (or any other allowed class) performs side effects during `__init__` (e.g., API validation calls, network probes), those side effects trigger on deserializing a crafted payload. The allowlist prevents instantiation of classes outside the list, but does not sandbox `__init__` of allowed classes.
- **Preconditions**: (1) User uses `allowed_objects='all'`; AND (2) user passes attacker-controlled serialized data.
- **Mitigations**: Default is `allowed_objects='core'`, which limits to core langchain-core types (messages, documents, prompts) that have no network side effects.
- **Residual risk**: Users who switch to `'all'` on untrusted data may trigger unintended network calls. The recommendation is to use `'core'` or an explicit class list.

#### T3: Prompt injection via LLM-generated tool call arguments

- **Flow**: DF4 → DF2 (LLM response → tool execution → subsequent LLM context)
- **Description**: In agentic workflows, LLM-generated tool call arguments are validated against Pydantic schemas but free-text fields are not sanitized. A malicious instruction in a retrieved document, tool output, or environment variable can cause the LLM to emit tool calls designed to exfiltrate data, modify state, or influence subsequent agent behavior. This is the standard prompt injection escalation path for agentic frameworks.
- **Preconditions**: An agent is processing untrusted external content (web pages, documents, tool outputs) that includes adversarial instructions; the model follows those instructions.
- **Mitigations**: None at the framework level — this is inherent to LLM-based agents. Tool argument schema validation (`BaseTool.invoke` via Pydantic) prevents type errors but not semantic manipulation.
- **Residual risk**: This threat is partially mitigated by model alignment and guardrails at the application layer. Framework responsibility ends at providing correct tool argument schemas.

#### T4: API credentials leaked into subprocess environment

- **Flow**: DF5 (Agent Middleware → OS subprocess)
- **Description**: `_launch_subprocess` in `_execution.py` accepts an `env` mapping that is passed directly to `subprocess.Popen`. If the caller includes API keys, secrets, or other credentials in the `env` mapping, they are inherited by the subprocess and accessible to any command it runs, including malicious ones. In agent scenarios, a compromised subprocess could read and exfiltrate these credentials.
- **Preconditions**: (1) User configures agent middleware to pass credentials in the `env` dict; AND (2) subprocess commands are influenced by untrusted input (LLM output, user input).
- **Mitigations**: None at the framework level — environment construction is caller responsibility. The framework does not add, remove, or filter env vars.
- **Residual risk**: High if users pass API keys in the `env` dict. Recommendation: pass only the minimum necessary env vars; do not pass `os.environ.copy()`.

#### T5: SSRF via DNS rebinding in image URL token counting

- **Flow**: DF7 (`ChatOpenAI.get_num_tokens_from_messages` → `_url_to_size` → `httpx.get`)
- **Description**: `_url_to_size` calls `validate_safe_url` to check if the image URL resolves to a private IP or cloud metadata endpoint. However, `validate_safe_url` performs DNS resolution via `socket.getaddrinfo` at validation time, and then `httpx.get(image_source)` performs a second DNS resolution at request time. An attacker with control over a domain's DNS records can set a short TTL, return a safe public IP during validation, and switch to a private IP (e.g., 169.254.169.254) for the actual request (DNS rebinding).
- **Preconditions**: (1) `pillow` and `httpx` are installed; (2) a vision-enabled OpenAI model is used; (3) messages contain `image_url` with attacker-controlled domains; (4) attacker controls the domain's DNS with short TTL.
- **Mitigations**: `validate_safe_url` with `allow_private=False` is called before `httpx.get`. Cloud metadata IPs are always blocked. Fails closed on DNS errors. Response timeout (5s) and size limit (50 MB) are enforced.
- **Residual risk**: DNS rebinding bypass. This is a known limitation of pre-request DNS validation. A complete fix requires pinning the resolved IP at validation time and passing it to `httpx` (or using `allow_fetching_images=False`).
- **Historical context**: GHSA-2g6r-c272-w58r; SSRF protection was added post-advisory.

#### T6: Unrestricted host access via `HostExecutionPolicy`

- **Flow**: DF5 (Agent Middleware → OS subprocess)
- **Description**: `HostExecutionPolicy` runs commands directly on the host OS with the same user privileges as the Python process. There is no filesystem sandboxing, network restriction, or syscall filtering. Optional CPU and memory limits can be configured but are not set by default.
- **Preconditions**: User configures agent middleware with `HostExecutionPolicy` (the simplest policy).
- **Mitigations**: The docstring explicitly states this policy is "best suited for trusted or single-tenant environments (CI jobs, developer workstations, pre-sandboxed containers)." `DockerExecutionPolicy` and `CodexSandboxExecutionPolicy` provide stronger isolation.
- **Residual risk**: By design. Users choosing `HostExecutionPolicy` accept the risk. Framework responsibility is accurate documentation of the trust model.

#### T7: Symlink path traversal in `FilesystemFileSearchMiddleware`

- **Flow**: DF6 (User app → File Search Middleware → filesystem)
- **Description**: `_validate_and_resolve_path` calls `Path(root_path / relative).resolve()` to canonicalize the path, then checks that the resolved path is within `root_path` via `Path.relative_to(self.root_path)`. `Path.resolve()` follows symbolic links. If a symlink exists inside `root_path` pointing to a directory outside `root_path` (e.g., `/etc`), the resolved path will be outside `root_path`, causing `relative_to` to raise `ValueError` — which is caught and re-raised as a path traversal error. **However**, if `root_path` itself was resolved via `Path(root_path).resolve()` at init time (line 128), and the symlink target is *outside the original root but still passes the resolved root check*, files can be read. More critically, `base_full.glob(pattern)` in `glob_search` iterates over all paths yielded by glob, but glob itself can follow symlinks — each matched symlink file may point outside the intended scope.
- **Preconditions**: (1) A symlink inside `root_path` points outside; (2) the operator did not mount `root_path` without symlink following (e.g., no `nofollow` mount option).
- **Mitigations**: `_validate_and_resolve_path` resolves and checks the base path. File read via `_python_search` calls `file_path.read_text()` — if `file_path` is a symlink, it follows.
- **Residual risk**: Symlinks within `root_path` that point outside are followed during glob/rglob iteration. Recommendation: set `root_path` to a filesystem mount that does not contain symlinks to sensitive areas, or add `file_path.is_symlink()` check before reading.

#### T8: ReDoS via user-supplied regex in `grep_search`

- **Flow**: DF6 (User app → File Search Middleware → filesystem)
- **Description**: `grep_search` compiles the user-supplied `pattern` with `re.compile(pattern)` to validate it, then uses `regex.search(line)` in `_python_search` for each line of each file. A catastrophically backtracking regex (e.g., `(a+)+$`) against a large file can consume CPU for seconds or minutes, blocking the event loop or thread.
- **Preconditions**: (1) Python fallback is used (ripgrep unavailable or disabled); (2) user or LLM supplies a malicious regex pattern.
- **Mitigations**: Ripgrep is preferred and does not suffer from Python ReDoS. Ripgrep is tried first (`use_ripgrep=True` default); Python fallback only if ripgrep is unavailable.
- **Residual risk**: If ripgrep is not installed, ReDoS is possible. Python 3.13+ includes a `timeout` parameter for `re.search`; earlier versions do not.

---

## Input Source Coverage

Maps each input source category to its data flows, threats, and validation. Open source responsibility column reflects that users control many input paths.

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| User direct input (tool definitions, model configs) | DF2 | T3 | `BaseTool.invoke` Pydantic schema | User | Users responsible for trusted tool implementations |
| Serialized payloads (`loads`/`load`) | DF1 | T1, T2 | `Reviver.__call__`: namespace + allowlist + jinja2 blocker | Project (framework controls allowlist defaults) | `secrets_from_env=True` with untrusted data; `allowed_objects='all'` with untrusted data |
| LLM output (tool call args, content) | DF4 | T3 | `BaseTool.invoke` Pydantic schema for args; no content sanitization | User/shared | No semantic sanitization of LLM free-text output |
| URL-fetched content (image_url) | DF7 | T5 | `_ssrf_protection:validate_safe_url` | Project (framework fetches the URL) | DNS rebinding bypass; only used in token counting |
| Configuration (env vars, SecretStr) | DF3, DF5 | T1, T4 | `SecretStr` wrapper; no env filtering in subprocess | Shared | `secrets_from_env=True` bypass; subprocess env passthrough |
| Filesystem paths (file search, load_prompt) | DF6, DF8 | T7, (T4 patched) | `_validate_and_resolve_path`; `_validate_path` | Project (framework validates paths) | Symlink following in glob; load_prompt deprecated |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to arbitrary code execution | The project does not control model selection, prompt construction, or what tools users register. A user who registers a `PythonREPL` tool (not shipped in this repo) and uses a jailbreakable model accepts the risk. | Providing correct tool argument schemas (`BaseTool.invoke`); documenting that tool behavior is user responsibility |
| Jinja2 SSTI via runtime `PromptTemplate(template_format='jinja2')` | Users explicitly request jinja2 at runtime; this is a deliberate configuration choice. Jinja2 is only blocked in the deserialization path. | Blocking jinja2 during deserialization (`load.py:default_init_validator`); documented in `load_prompt` deprecation notice |
| API key leakage via user application logs | The project wraps API keys in `SecretStr` to prevent accidental logging by the framework itself. User logging behavior is outside the project's control. | `SecretStr` wrapping in all partner integrations; `secret_from_env` helper |
| Malicious custom callback handler execution | Callback handlers are user-provided code. A malicious callback can do anything the Python process allows. | Providing a well-defined `BaseCallbackHandler` interface |
| Model output containing harmful content | The project does not control model behavior, alignment, or safety filtering. | Correctly forwarding model responses without modification |
| Supply chain attacks on third-party provider SDKs | The project depends on `openai`, `anthropic`, `groq`, etc. SDKs. Compromise of those packages is outside the project's control. | Pinning dependency versions in `pyproject.toml` and `uv.lock` per package |
| Exfiltration via tool calls in agentic workflows | An agent equipped with network-capable tools (user-registered) can exfiltrate data if prompted to do so. Tool capabilities are user-controlled. | Not providing dangerous default tools (no default PythonREPL, shell tool, or HTTP fetch tool in core) |
| Docker container escape | Depends on host Docker daemon security, kernel version, and container configuration. `DockerExecutionPolicy` is a best-effort isolation layer. | `DockerExecutionPolicy` defaults (`--network none`, `--rm`); documentation of security requirements for the host daemon |

### Rationale

**Prompt injection as out-of-scope**: LangChain is a framework; users choose which models and tools to attach. The framework provides correct Pydantic schemas for tool arguments (`langchain_core/tools/base.py:BaseTool.invoke`) and validates argument types, but cannot prevent a model from being manipulated into misusing legitimate tools. This is consistent with the industry-wide understanding that prompt injection is an application-layer concern when deploying LLM agents.

**Runtime Jinja2 as out-of-scope**: The project explicitly blocks jinja2 during *deserialization* (`load.py:_block_jinja2_templates`) because deserializing untrusted data with jinja2 enabled is a known RCE vector (GHSA-6qv9-48xg-fc7f). However, a user who explicitly instantiates `PromptTemplate(template_format='jinja2')` has made a deliberate, visible choice. The framework cannot protect against all deliberate user choices without removing legitimate functionality.

**Supply chain threats**: All partner integrations pin their dependencies via `uv.lock` files under each partner package (`libs/partners/openai/uv.lock`, etc.). Monitoring for dependency compromise is a deployer responsibility.

---

## Investigated and Dismissed

Threats investigated during flaw validation that were found to be non-exploitable or already patched.

| ID  | Original Threat | Investigation | Evidence | Conclusion |
|-----|----------------|---------------|----------|------------|
| D1  | Jinja2 SSTI via deserialized `PromptTemplate` (CVE path: GHSA-6qv9-48xg-fc7f) | Traced full deserialization path: `loads()` → `Reviver.__call__()` → `init_validator(mapping_key, kwargs)` → `default_init_validator` → `_block_jinja2_templates`. Checked whether `init_validator=None` could be passed. | `langchain_core/load/load.py:_block_jinja2_templates`; `langchain_core/load/load.py:default_init_validator`; default `init_validator=default_init_validator` in function signature | The jinja2 check fires before `cls(**kwargs)` is called. Overriding with `init_validator=None` removes the check but requires the caller to explicitly opt out. Non-exploitable with default settings. |
| D2  | Path traversal in `load_prompt()` via `template_path` field (GHSA-qh6h-p6c9-ff54) | Reviewed `loading.py:_load_template`, `_validate_path`, `_load_few_shot_prompt`. Both `load_prompt` and `load_prompt_from_config` are deprecated since v1.2.21 with `allow_dangerous_paths=False` default. `_validate_path` rejects absolute paths and `..` components. | `langchain_core/prompts/loading.py:_validate_path`; `langchain_core/prompts/loading.py:load_prompt` (deprecated since 1.2.21) | Patched in v1.2.21. Current code raises `ValueError` for absolute paths and `..` traversal by default. Functions marked deprecated with removal target 2.0.0. Not exploitable in current version with default settings. |

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-27 | langster-threat-model (deep mode, commit 494b760028) | Initial threat model — 10 components, 5 data classifications, 5 trust boundaries, 8 data flows, 8 threats, 7 out-of-scope patterns, 2 investigated and dismissed |
