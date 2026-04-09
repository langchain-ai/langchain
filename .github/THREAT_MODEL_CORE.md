# Threat Model: langchain-core

> Generated: 2026-04-08 | Commit: d3e60f5c03 | Scope: libs/core/ (langchain-core v1.2.27) | Visibility: Open Source | Mode: deep

> **Disclaimer:** This threat model is automatically generated to help developers and security researchers understand where trust is placed in this system and where boundaries exist. It is experimental, subject to change, and not an authoritative security reference — findings should be validated before acting on them. The analysis may be incomplete or contain inaccuracies. We welcome suggestions and corrections to improve this document.

For vulnerability reporting, see [GitHub Security Advisories](https://github.com/langchain-ai/langchain/security/advisories/new).

See also: the [langchain_v1 threat model](THREAT_MODEL_V1.md) for the agent middleware layer.

---

## Scope

### In Scope

- `libs/core/langchain_core/load/` — Serialization/deserialization system (`loads`, `load`, `dumpd`, `dumps`, `Reviver`, allowlists, secret handling)
- `libs/core/langchain_core/_security/` — SSRF protection (`validate_safe_url`, `is_safe_url`, `SSRFProtected*` annotated types)
- `libs/core/langchain_core/prompts/` — Prompt templates, template formatting, deprecated prompt loading from files
- `libs/core/langchain_core/tools/` — Tool base classes, argument validation, schema generation
- `libs/core/langchain_core/output_parsers/` — JSON, XML, Pydantic output parsers
- `libs/core/langchain_core/runnables/` — Composable pipeline primitives (`RunnableLambda`, `RunnableSequence`, etc.)
- `libs/core/langchain_core/callbacks/` — Callback manager, handler invocation
- `libs/core/langchain_core/messages/` — Message types, content blocks, message utilities
- `libs/core/langchain_core/language_models/` — Abstract base classes for LLMs and chat models
- `libs/core/langchain_core/utils/` — Environment variable access, formatting, function schema extraction

### Out of Scope

- `libs/langchain_v1/` — Agent middleware, execution policies, file search middleware (separate package; see [THREAT_MODEL_V1.md](THREAT_MODEL_V1.md))
- `libs/partners/` — Partner integration packages (separate packages, each with their own threat surface)
- `libs/text-splitters/` — Document chunking (separate package)
- `libs/standard-tests/` — Test harnesses; not shipped as attack surface
- `tests/` — Unit and integration tests (read during analysis for understanding; not threat-modeled)
- User application code, model selection, custom tools, custom callbacks — user-controlled
- LLM model behavior — the project cannot guarantee model safety across all models users may select
- Deployment infrastructure — users control hosting, network topology, and secrets management
- LangSmith, LangGraph — separate products and repositories

### Assumptions

1. The project is used as a library/framework — users control their own application code, model selection, and deployment infrastructure.
2. API keys are sourced from environment variables or passed explicitly; the framework does not store them persistently.
3. Users are responsible for validating that serialized LangChain objects (passed to `loads()`/`load()`) come from trusted sources.
4. The `langchain-core` serialization allowlist (`allowed_objects='core'`) is the default and correct choice for untrusted data.
5. `defusedxml` is not a required dependency of langchain-core; users who need `XMLOutputParser` must install it separately or accept reduced XML security.
6. Jinja2 template format is blocked in deserialization and file-based prompt loading but available at runtime construction with `SandboxedEnvironment` — users who opt in accept the residual sandbox bypass risk.

---

## System Overview

`langchain-core` is the foundational Python library for the LangChain ecosystem. It provides base abstractions for building LLM-powered applications: messages, prompts, tools, runnables (composable pipelines), callbacks, output parsers, serialization, and SSRF protection. It does not serve HTTP traffic, store user data persistently, or communicate with external services directly — it is a library that processes data on behalf of user applications. Concrete LLM provider integrations live in separate partner packages.

### Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                        User Application                           │
│                                                                   │
│  User Code ────┬──► Prompt Templates (C3) ──► Messages (C8)      │
│                │         │                                        │
│                │    Template vars                                 │
│                │                                                  │
│                ├──► Tools Framework (C4) ──► Tool execution       │
│                │    (arg schema validation)    (user-defined)     │
│                │                                                  │
│                ├──► Runnables (C6) ──► Composition pipeline       │
│                │                                                  │
│                ├──► Callbacks (C7) ──► User callback handlers     │
│                │                                                  │
│                └──► Output Parsers (C5) ◄── LLM output text      │
│                                                                   │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ TB1 ─ ─ ─ ─ │
│                                                                   │
│  Serialization System (C1)                                        │
│    loads()/load() ──► Reviver ──► importlib + cls(**kwargs)       │
│         │               │                                         │
│    ─ ─ ─│─ ─ ─ TB2 ─ ─ │                                         │
│         │               ├──► Allowlist check                      │
│         │               ├──► Namespace validation                 │
│         │               ├──► Jinja2 blocking                      │
│         │               └──► secrets_from_env ──► os.environ      │
│                                                                   │
│  SSRF Protection (C2)                                             │
│    validate_safe_url() ──► DNS resolution ──► IP validation       │
│    ─ ─ ─ ─ ─ ─ ─ ─ ─ TB6 ─ ─ ─ ─ ─ ─ ─                        │
│                                                                   │
│  Deprecated Prompt Loading (C10)                                  │
│    load_prompt() ──► _validate_path() ──► filesystem              │
│    ─ ─ ─ ─ ─ ─ ─ TB5 ─ ─ ─ ─ ─ ─ ─                             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Components

| ID | Component | Description | Trust Level | Default? | Entry Points |
|----|-----------|-------------|-------------|----------|--------------|
| C1 | Serialization System | Allowlist-based JSON (de)serialization for LangChain objects; blocks jinja2, validates namespaces, handles secrets | framework-controlled | Yes (when `load`/`loads` called) | `langchain_core/load/load.py:loads`, `langchain_core/load/load.py:load`, `langchain_core/load/dump.py:dumpd`, `langchain_core/load/dump.py:dumps` |
| C2 | SSRF Protection | URL validation utility blocking private IPs, localhost, and cloud metadata endpoints | framework-controlled | No (explicit opt-in via `validate_safe_url` or `SSRFProtected*` Pydantic types) | `langchain_core/_security/_ssrf_protection.py:validate_safe_url`, `langchain_core/_security/_ssrf_protection.py:is_safe_url` |
| C3 | Prompt Templates | Template rendering for LLM prompts; supports f-string (default, safe) and mustache; jinja2 allowed at runtime with SandboxedEnvironment | framework-controlled | Yes | `langchain_core/prompts/prompt.py:PromptTemplate.from_template`, `langchain_core/prompts/chat.py:ChatPromptTemplate.from_template`, `langchain_core/prompts/chat.py:ChatPromptTemplate.format_messages` |
| C4 | Tools Framework | Base tool classes with Pydantic schema validation for arguments; user-defined tool functions execute without sandboxing | framework-controlled (validation) / user-controlled (execution) | Yes | `langchain_core/tools/base.py:BaseTool.invoke`, `langchain_core/tools/base.py:BaseTool.ainvoke`, `langchain_core/tools/convert.py:tool` (decorator) |
| C5 | Output Parsers | Parse LLM text output into structured formats (JSON, XML, Pydantic models) | framework-controlled | Yes | `langchain_core/output_parsers/json.py:JsonOutputParser.parse_result`, `langchain_core/output_parsers/xml.py:XMLOutputParser.parse`, `langchain_core/output_parsers/pydantic.py:PydanticOutputParser.parse_result` |
| C6 | Runnables | Composable pipeline primitives; wraps arbitrary user functions via `RunnableLambda` | framework-controlled (composition) / user-controlled (lambda bodies) | Yes | `langchain_core/runnables/base.py:Runnable.invoke`, `langchain_core/runnables/base.py:RunnableLambda` |
| C7 | Callbacks System | Invokes user-provided callback handlers with run data (inputs, outputs, errors, metadata) | framework-controlled (invocation) / user-controlled (handler code) | Yes | `langchain_core/callbacks/manager.py:CallbackManager`, `langchain_core/callbacks/manager.py:handle_event` |
| C8 | Messages | Pydantic-validated message types (Human, AI, System, Tool) and content blocks (text, image, audio, file) | framework-controlled | Yes | `langchain_core/messages/utils.py:convert_to_messages`, `langchain_core/messages/utils.py:messages_from_dict` |
| C9 | Language Model Abstractions | Abstract base classes for chat models and LLMs; define interfaces for partner implementations | framework-controlled | Yes | `langchain_core/language_models/chat_models.py:BaseChatModel.invoke` |
| C10 | Prompt Loading (deprecated) | File-based prompt loading with path validation; deprecated since v1.2.21, removal target v2.0.0 | framework-controlled | Yes (when `load_prompt` called) | `langchain_core/prompts/loading.py:load_prompt`, `langchain_core/prompts/loading.py:_validate_path` |
| C11 | Utility Layer | Environment variable access, string formatting, function schema extraction | framework-controlled | Yes | `langchain_core/utils/env.py:get_from_dict_or_env`, `langchain_core/utils/formatting.py:StrictFormatter` |

---

## Data Classification

Classifies all sensitive data types found in the codebase with their sensitivity levels, storage locations, and retention policies.

| ID | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | API/service credentials | `SecretStr` fields in partner model constructors (e.g., `openai_api_key`), `langchain_core/load/serializable.py:Serializable.lc_secrets` property values | Critical | In-memory (`SecretStr`); OS environment variables | Yes (`SecretStr` masks in repr/logs) | Process lifetime | All |
| DC2 | LLM conversation data | `HumanMessage.content`, `AIMessage.content`, `SystemMessage.content`, `ToolMessage.content`, prompt template variables | High | In-memory (transient) | N/A (not persisted by framework) | Transient (garbage-collected) | GDPR, CCPA (when containing PII) |
| DC3 | Serialized LangChain objects | JSON payloads to `loads()`/`load()`; includes `kwargs` for any allowed class, `secret` type entries | High | User-application storage (not framework responsibility) | N/A (framework does not store) | User-controlled | N/A |
| DC4 | OS environment variables | Arbitrary `os.environ` values accessible via `secrets_from_env=True` in `Reviver` | Critical | Host OS environment | N/A | Process lifetime | All (secrets may include credentials, tokens, database URLs) |
| DC5 | Prompt template content | Template strings, template file contents, interpolated variables | Medium | In-memory; filesystem (for file-based loading) | N/A | Process lifetime | N/A |
| DC6 | Tool call arguments | LLM-generated function call arguments passed to `BaseTool.invoke` | High | In-memory (transient) | N/A | Transient | N/A (may contain user PII depending on tool) |
| DC7 | Callback/tracer data | Run inputs, outputs, errors, metadata, tags passed to callback handlers | Medium | In-memory; LangSmith (if tracer enabled, user-configured) | N/A (framework does not persist) | User-controlled | GDPR, CCPA (when containing PII) |

### Data Classification Details

#### DC1: API/service credentials

- **Fields**: API key fields across all partner integrations inheriting from `langchain_core.load.serializable.py:Serializable.lc_secrets`. Common pattern: `{field_name: "ENV_VAR_NAME"}`.
- **Storage**: In-memory only, wrapped in Pydantic `SecretStr`. Sourced from environment variables via `langchain_core/utils/utils.py:secret_from_env` at instantiation.
- **Access**: Read by partner SDK constructors at instantiation. `SecretStr` wrapper prevents accidental logging via `repr`/`str`.
- **Encryption**: In-memory only; no at-rest encryption needed (not persisted). In-transit via HTTPS to each provider API.
- **Retention**: Process lifetime. Credentials released when the model object is garbage-collected.
- **Logging exposure**: Protected by `SecretStr`; direct access requires `.get_secret_value()`. Risk exists if users log message contents that embed credentials (user responsibility).
- **Cross-border**: Transmitted to respective provider APIs over HTTPS; users choose which provider and thus which jurisdiction.
- **Gaps**: None identified in framework code.

#### DC2: LLM conversation data

- **Fields**: `HumanMessage.content`, `AIMessage.content`, `SystemMessage.content`, `ToolMessage.content`, `ChatMessage.content`, prompt template input variables (arbitrary user-supplied kwargs).
- **Storage**: In-memory only within langchain-core. No persistence layer in core — persistence is user-application responsibility (e.g., chat history databases, LangSmith tracing).
- **Access**: Read by prompt templates (`langchain_core/prompts/chat.py:ChatPromptTemplate.format_messages`), output parsers, callback handlers, and tracers. Passed as kwargs through runnables.
- **Encryption**: N/A (not persisted by framework; in-transit encryption depends on user's tracing/logging configuration).
- **Retention**: Transient — garbage-collected when message objects go out of scope. No framework-level caching.
- **Logging exposure**: Message content is passed to callback handlers via `langchain_core/callbacks/manager.py:handle_event`. If a user registers a logging callback (e.g., `StdOutCallbackHandler`), message content appears in logs.
- **Gaps**: No framework-level PII detection or redaction. Users who pass PII in messages are responsible for downstream handling.

#### DC3: Serialized LangChain objects

- **Fields**: JSON payloads to `loads()`/`load()` containing `{"lc": 1, "type": "constructor", "id": [...], "kwargs": {...}}` structures, including `secret` type entries (`{"lc": 1, "type": "secret", "id": [env_var_name]}`).
- **Storage**: Not stored by langchain-core. Users provide serialized payloads from their own storage (databases, files, APIs).
- **Access**: Consumed by `langchain_core/load/load.py:Reviver.__call__` during deserialization. The `kwargs` within become constructor arguments for instantiated classes.
- **Encryption**: N/A (framework does not store; encryption of serialized data at rest is user responsibility).
- **Retention**: Transient during deserialization; final objects retained per user's reference management.
- **Logging exposure**: Serialized payloads may contain secret references that, if logged before deserialization, reveal environment variable names (not values). Post-deserialization, values depend on whether `secrets_from_env=True`.
- **Gaps**: No integrity validation (signing/MAC) on serialized payloads. The allowlist prevents arbitrary class instantiation but does not verify payload authenticity.

#### DC6: Tool call arguments

- **Fields**: LLM-generated `ToolCall.args` dict — keys and values determined by the LLM based on the tool's Pydantic schema. Passed to `langchain_core/tools/base.py:BaseTool._parse_input` then to user-defined `_run()`.
- **Storage**: In-memory only. Passed through the tool invocation chain; not persisted by the framework.
- **Access**: Validated by Pydantic in `BaseTool._parse_input`, then passed as `**kwargs` to user-defined tool functions via `langchain_core/tools/structured.py:StructuredTool._run`. Also passed to callback handlers via `langchain_core/callbacks/manager.py:CallbackManager.on_tool_start`.
- **Encryption**: N/A (in-memory only).
- **Retention**: Transient.
- **Logging exposure**: Tool arguments are passed to `on_tool_start` callback handlers. Default `StdOutCallbackHandler` prints tool inputs.
- **Gaps**: Pydantic validates types but not semantic content of string fields. An LLM can generate tool arguments containing adversarial content (prompt injection payloads, exfiltration URLs) that pass type validation.

#### DC4: OS environment variables

- **Fields**: Any `os.environ` key named in a serialized payload's `secret` field, when `secrets_from_env=True` is passed to `loads()`/`load()`.
- **Storage**: Host OS environment.
- **Access**: `langchain_core/load/load.py:Reviver.__call__` (line 417) reads `os.environ[key]` directly when `secrets_from_env=True`. An attacker who controls the serialized payload can name any environment variable — there is no allowlist on variable names.
- **Encryption**: N/A (environment variables are plaintext at the OS level).
- **Retention**: Process lifetime.
- **Logging exposure**: Values returned as deserialized constructor kwargs; exposure depends on user logging.
- **Gaps**: **Critical gap**: When `secrets_from_env=True`, any environment variable can be read by a crafted payload. Default is `False`. The escape mechanism (`langchain_core/load/_validation.py:_is_escaped_dict`) prevents injection through the normal serialization round-trip, but direct `loads()` of attacker-controlled JSON bypasses this protection.

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | User application ↔ langchain-core public API | Entry point for all user-provided inputs to the framework | Pydantic model validation on all public classes, default configurations (`template_format='f-string'`, `allowed_objects='core'`, `secrets_from_env=False`), tool argument schema validation | Model selection, custom tool implementations, custom callback handlers, application-level input sanitization, deployment topology |
| TB2 | Untrusted payload ↔ serialization engine | JSON deserialization entry point via `loads()`/`load()` | Namespace allowlist (`DEFAULT_NAMESPACES`), class path allowlist (`allowed_objects='core'` default), jinja2 blocking (`langchain_core/load/load.py:default_init_validator`), Bedrock SSRF blocking (`langchain_core/load/validators.py:_bedrock_validator`), `__lc_escaped__` injection protection (`langchain_core/load/_validation.py:_is_escaped_dict`), `Serializable` subclass enforcement | Trustworthiness of the serialized payload; whether `secrets_from_env=True` is used; whether `allowed_objects='all'` is used |
| TB3 | LLM output ↔ output parsers / tool invocation | Boundary where untrusted LLM-generated content enters framework processing | Pydantic schema validation for tool arguments (`langchain_core/tools/base.py:BaseTool._parse_input`), JSON/XML structural parsing, `defusedxml` for XML parsing (when installed) | LLM response content, model behavior, semantic meaning of tool arguments |
| TB4 | Framework ↔ user-provided callbacks/tools | Boundary where user-authored code is invoked by the framework | Callback invocation protocol (`langchain_core/callbacks/manager.py:handle_event`), tool argument schema validation, exception handling around callback calls | What callback/tool code does, side effects, network calls, file I/O performed by user code |
| TB5 | Framework ↔ filesystem | File access via deprecated `load_prompt` and `_load_template` | Path traversal prevention (`langchain_core/prompts/loading.py:_validate_path`): rejects absolute paths and `..` components; file type restriction to `.txt`; symlink resolution before suffix check | Content of files on disk; filesystem permissions; symbolic link targets outside validated paths |
| TB6 | URL input ↔ SSRF protection | URL validation before external HTTP requests | Private IP range blocking (RFC 1918), cloud metadata endpoint blocking (AWS/GCP/Azure/Alibaba/Oracle), localhost blocking, DNS resolution validation (`langchain_core/_security/_ssrf_protection.py:validate_safe_url`) | DNS infrastructure behavior (rebinding); whether calling code actually uses `validate_safe_url` before fetching |

### Boundary Details

#### TB1: User application ↔ langchain-core public API

- **Inside**: Pydantic model validation on all public classes. Default configurations: `template_format='f-string'`, `allowed_objects='core'`, `secrets_from_env=False`. `StrictFormatter` (`langchain_core/utils/formatting.py:StrictFormatter`) blocks positional arguments. Template variable validation (`langchain_core/prompts/string.py:get_template_variables`) blocks attribute access (`.`) and indexing (`[`, `]`) in f-string variables.
- **Outside**: What users pass as tool implementations, callback handlers, and model configurations. Users may register tools that perform arbitrary operations; the framework validates tool argument schemas but not tool behavior.
- **Crossing mechanism**: Python function calls to public API methods.

#### TB2: Untrusted payload ↔ serialization engine

- **Inside**: `langchain_core/load/load.py:Reviver.__init__` builds the class path allowlist. `langchain_core/load/load.py:Reviver.__call__` enforces: (1) namespace validation against `DEFAULT_NAMESPACES`, (2) allowlist check against `allowed_class_paths`, (3) `DISALLOW_LOAD_FROM_PATH` blocks for `langchain_community` and `langchain`, (4) class-specific validators via `CLASS_INIT_VALIDATORS`, (5) general init validator (jinja2 blocking), (6) `Serializable` subclass check, (7) `importlib.import_module()` with validated namespace. `langchain_core/load/_validation.py:_is_escaped_dict` prevents user data dicts from being treated as LC objects.
- **Outside**: The content of the JSON payload; whether the caller passes trusted or untrusted data; whether the caller enables `secrets_from_env=True` or broadens `allowed_objects`.
- **Crossing mechanism**: `json.loads(text)` + `Reviver` object hook.

#### TB3: LLM output ↔ output parsers / tool invocation

- **Inside**: `langchain_core/tools/base.py:BaseTool._parse_input` validates tool arguments against Pydantic schemas. `langchain_core/output_parsers/json.py:JsonOutputParser.parse_result` uses `json.loads()` (safe). `langchain_core/output_parsers/xml.py:XMLOutputParser.parse` uses `defusedxml` by default. Tool names matched against registered tool list.
- **Outside**: LLM response content is untrusted — it may contain prompt injection, malicious tool call arguments, or unexpected structured data. No semantic sanitization of free-form text.
- **Crossing mechanism**: Python function calls from LLM response processing to parser/tool invocation.

#### TB4: Framework ↔ user-provided callbacks/tools

- **Inside**: `langchain_core/callbacks/manager.py:handle_event` invokes handler methods with exception handling. `langchain_core/tools/base.py:BaseTool.run` passes validated arguments to user-defined `_run()`.
- **Outside**: What callback/tool code does — arbitrary Python execution, side effects, network calls.
- **Crossing mechanism**: Python method calls to user-provided handler/tool instances.

#### TB5: Framework ↔ filesystem

- **Inside**: `langchain_core/prompts/loading.py:_validate_path` rejects absolute paths (line 30) and `..` directory traversal (line 38). `langchain_core/prompts/loading.py:_load_template` resolves symlinks before checking file suffix (line 101), restricts to `.txt` files only (line 103).
- **Outside**: Filesystem contents within validated paths; OS-level file permissions.
- **Crossing mechanism**: Python `Path.read_text()`, `json.load()`, `yaml.safe_load()`.

#### TB6: URL input ↔ SSRF protection

- **Inside**: `langchain_core/_security/_ssrf_protection.py:validate_safe_url` validates URL scheme (http/https only), resolves DNS via `socket.getaddrinfo()`, checks each resolved IP against private ranges (RFC 1918), cloud metadata IPs/hostnames (169.254.169.254, metadata.google.internal, etc.), and localhost. Cloud metadata is ALWAYS blocked, even with `allow_private=True`. Fails closed on DNS errors.
- **Outside**: DNS infrastructure behavior; whether downstream code actually calls `validate_safe_url` before making HTTP requests.
- **Crossing mechanism**: URL string passed in, validated string returned.

---

## Data Flows

| ID | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | User application | C1 Serialization (`loads`/`load`) | JSON serialized LC object payload (may contain secret refs for DC1) | DC1, DC3 | TB2 | Python function call |
| DF2 | User application | C3 Prompt Templates → C8 Messages | Prompt template strings and variables, producing formatted messages | DC5, DC2 | TB1 | Python function call |
| DF3 | User application | C10 `load_prompt` → filesystem | Config file path, template file path | DC5 | TB5 | Python file I/O |
| DF4 | User application / partner code | C2 SSRF Protection | URL string for validation | — | TB6 | Python function call |
| DF5 | LLM output (C8 Messages via partner) | C5 Output Parsers | LLM-generated text (JSON, XML, structured) | DC2, DC6 | TB3 | Python function call |
| DF6 | LLM output (C8 Messages via partner) | C4 Tools Framework (`BaseTool.invoke`) | Tool call arguments (name, args dict) | DC6 | TB3, TB4 | Python function call |
| DF7 | C6 Runnables / C4 Tools | C7 Callbacks (`CallbackManager`) | Run data: inputs, outputs, errors, metadata | DC7, DC2 | TB4 | Python function call |
| DF8 | C1 Serialized payload (secret type) | OS environment (`os.environ`) | Environment variable name from payload | DC4 | TB2 | `os.environ[key]` |
| DF9 | C1 Serialized payload (constructor type) | Python runtime (`importlib`) | Module path, class name, kwargs | DC3 | TB2 | `importlib.import_module()` |
| DF10 | User application | C6 Runnables (`RunnableLambda`) | Arbitrary user function + input data | DC2 | TB4 | Python function call |

### Flow Details

#### DF1: User application → Serialization API (`loads`/`load`)

- **Data**: JSON string or dict representing serialized LangChain objects. Sensitivity depends on whether it contains `secret` fields (DC3/DC4).
- **Validation**: `langchain_core/load/load.py:Reviver.__call__` enforces namespace allowlist, class path allowlist, jinja2 blocking, Bedrock endpoint blocking, `Serializable` subclass check, and `__lc_escaped__` injection protection.
- **Trust assumption**: Caller ensures the payload comes from a trusted source. If `secrets_from_env=True`, caller trusts the payload completely with access to all OS environment variables.

#### DF5: LLM output → Output Parsers

- **Data**: LLM-generated text intended to be parsed as JSON, XML, or Pydantic-structured data.
- **Validation**: `langchain_core/output_parsers/json.py:JsonOutputParser` uses `json.loads()` (safe). `langchain_core/output_parsers/xml.py:XMLOutputParser` defaults to `defusedxml` but falls back to standard library if not installed. `langchain_core/output_parsers/pydantic.py:PydanticOutputParser` validates against user-defined Pydantic schema.
- **Trust assumption**: LLM output is untrusted. Parsers extract structure but do not sanitize semantic content.

#### DF6: LLM output → Tool invocation

- **Data**: Tool call arguments — function name and arguments dict generated by the LLM.
- **Validation**: Tool names matched against registered tool list. Arguments validated via `langchain_core/tools/base.py:BaseTool._parse_input` using Pydantic schema. Type validation only — no semantic sanitization of string field contents.
- **Trust assumption**: LLM output is untrusted. Schema validation prevents type errors but not adversarial content in text fields.

#### DF8: Serialized secret → `os.environ`

- **Data**: Environment variable name extracted from `{"lc": 1, "type": "secret", "id": ["VAR_NAME"]}` in deserialized payload.
- **Validation**: None on variable name — any `os.environ` key can be read.
- **Trust assumption**: Only activated when `secrets_from_env=True` (default `False`). Caller trusts the payload not to name sensitive environment variables.

#### DF9: Serialized constructor → `importlib`

- **Data**: Module path and class name from `{"lc": 1, "type": "constructor", "id": ["namespace", ..., "ClassName"]}`.
- **Validation**: Namespace validated against `DEFAULT_NAMESPACES` (line 456-462, 480-482). Class path checked against allowlist. Imported class must be `Serializable` subclass. Init validators run before instantiation.
- **Trust assumption**: Allowlist constrains which classes can be instantiated. Side effects in allowed classes' `__init__` are accepted risk.

---

## Threats

| ID | Data Flow | Classification | Threat | Boundary | Severity | Validation | Code Reference |
|----|-----------|----------------|--------|----------|----------|------------|----------------|
| T1 | DF8 | DC4 | Arbitrary OS environment variable exfiltration via crafted serialized payload when `secrets_from_env=True` | TB2 | High | Verified | `langchain_core/load/load.py:Reviver.__call__` (line 417) |
| T2 | DF9 | DC3 | Side effects in allowed class `__init__` during deserialization when using `allowed_objects='all'` | TB2 | Medium | Likely | `langchain_core/load/load.py:Reviver.__call__` (line 506) |
| T3 | DF5 | DC2 | XML entity expansion (DTD bomb) via `XMLOutputParser` when `defusedxml` not installed and `parser='xml'` | TB3 | Medium | Verified | `langchain_core/output_parsers/xml.py:XMLOutputParser.parse` (line 246) |
| T4 | DF4 | — | DNS rebinding SSRF bypass in `validate_safe_url` due to TOCTOU between DNS validation and downstream HTTP request | TB6 | Medium | Likely | `langchain_core/_security/_ssrf_protection.py:validate_safe_url` (lines 251-280) |
| T5 | DF6 | DC6 | Prompt injection via LLM-generated tool call arguments influencing subsequent LLM context in agentic workflows | TB3 | Medium | Unverified | `langchain_core/tools/base.py:BaseTool.invoke` |
| T6 | DF2 | DC5 | Jinja2 sandbox escape via runtime `PromptTemplate(template_format='jinja2')` using SandboxedEnvironment bypass | TB1 | Medium | Unverified | `langchain_core/prompts/string.py:jinja2_formatter` (line 71) |

### Threat Details

#### T1: Arbitrary OS environment variable exfiltration via `secrets_from_env=True`

- **Flow**: DF8 (Serialized payload → `os.environ` via `Reviver.__call__`)
- **Description**: When `secrets_from_env=True` is passed to `loads()`/`load()`, a crafted serialized payload can name any OS environment variable in its `secret` fields (e.g., `{"lc":1,"type":"secret","id":["AWS_SECRET_ACCESS_KEY"]}`). The `Reviver.__call__` method reads that key from `os.environ` and returns it as a constructor `kwarg`. There is no allowlist or validation on which environment variable names can be read. The escape mechanism (`__lc_escaped__`) prevents injection through the normal `dumpd`/`dumps` round-trip, but direct `loads()` of attacker-controlled JSON bypasses this protection entirely.
- **Preconditions**: (1) User passes `secrets_from_env=True` to `loads()`/`load()`; AND (2) user passes attacker-controlled serialized data that did not originate from `dumpd()`/`dumps()`. Both conditions must be true simultaneously.
- **Historical context**: GHSA-c67j-w6g6-q2cm covers this pattern.

#### T2: Side effects in allowed class `__init__` during deserialization

- **Flow**: DF9 (Serialized constructor → `importlib` → `cls(**kwargs)`)
- **Description**: When `allowed_objects='all'` is used, the allowlist includes partner integrations such as `ChatOpenAI`. If an allowed class performs side effects during `__init__` (e.g., API validation calls, network probes), those side effects trigger on deserializing a crafted payload. The allowlist prevents instantiation of classes outside the list, but does not sandbox `__init__` of allowed classes.
- **Preconditions**: (1) User uses `allowed_objects='all'`; AND (2) user passes attacker-controlled serialized data. Default `allowed_objects='core'` limits to core langchain-core types (messages, documents, prompts) that have no network side effects.

#### T3: XML entity expansion via `XMLOutputParser`

- **Flow**: DF5 (LLM output → `XMLOutputParser.parse`)
- **Description**: `XMLOutputParser` defaults to `parser="defusedxml"` but `defusedxml` is not a required dependency of langchain-core. If `defusedxml` is not installed, users encounter an `ImportError` that steers them toward setting `parser="xml"`. With `parser="xml"`, the standard library `xml.etree.ElementTree.fromstring()` processes internal DTD entity declarations, allowing expansion up to ~300KB from a small input (limited by libexpat's built-in amplification limit in Python 3.9.8+/3.10.1+). External entity resolution (classic XXE file read) is blocked by modern expat defaults. A reduced DTD bomb (5 levels or fewer) succeeds silently; 6+ levels are blocked by libexpat.
- **Preconditions**: (1) `defusedxml` is not installed; AND (2) user sets `parser="xml"` or LLM output containing DTD declarations reaches the parser; AND (3) non-streaming `parse()` path is used (streaming parser accidentally strips DTD preamble).

#### T4: DNS rebinding SSRF bypass in `validate_safe_url`

- **Flow**: DF4 (URL → `validate_safe_url` → downstream HTTP request)
- **Description**: `validate_safe_url` performs DNS resolution via `socket.getaddrinfo` at validation time and validates each resolved IP. However, the calling code typically performs a second DNS resolution when making the actual HTTP request (e.g., via `httpx.get()`). An attacker with DNS control can set a short TTL, return a public IP during validation, and switch to a private IP (169.254.169.254) for the actual request. Cloud metadata IPs are always blocked at validation time, but the TOCTOU window between validation and request remains.
- **Preconditions**: (1) Calling code uses `validate_safe_url` but does not pin the resolved IP; AND (2) attacker controls a domain's DNS with short TTL; AND (3) the URL reaches an HTTP client that re-resolves DNS.
- **Historical context**: GHSA-2g6r-c272-w58r; SSRF protection was added post-advisory.

#### T5: Prompt injection via LLM-generated tool call arguments

- **Flow**: DF6 (LLM output → `BaseTool.invoke`)
- **Description**: In agentic workflows, LLM-generated tool call arguments are validated against Pydantic schemas by `BaseTool._parse_input`, but free-text string fields are not sanitized. A malicious instruction in a retrieved document, tool output, or environment variable can cause the LLM to emit tool calls designed to exfiltrate data or manipulate downstream behavior. Pydantic validates types but not semantic content.
- **Preconditions**: An agent processes untrusted external content containing adversarial instructions; the model follows those instructions; a tool with side effects is registered.

#### T6: Jinja2 sandbox escape via runtime `PromptTemplate`

- **Flow**: DF2 (User app → `PromptTemplate` with `template_format='jinja2'`)
- **Description**: While jinja2 is blocked in deserialization (`_block_jinja2_templates`) and file-based prompt loading, it is available at runtime construction via `PromptTemplate(template_format='jinja2')`. The framework uses Jinja2's `SandboxedEnvironment` (`langchain_core/prompts/string.py:jinja2_formatter`, line 71), which blocks dunder attribute access but allows regular attribute/method calls. The docstring explicitly warns this is "best-effort" sandboxing, not a security guarantee. Known sandbox bypass techniques exist for `SandboxedEnvironment`.
- **Preconditions**: (1) User explicitly sets `template_format='jinja2'`; AND (2) user passes attacker-controlled template content; AND (3) a `SandboxedEnvironment` bypass is achievable in the deployed Jinja2 version.

### Chain Analysis

**T1 + T2 combined**: If an attacker controls a serialized payload and the user enables both `secrets_from_env=True` and `allowed_objects='all'`, the attacker can both exfiltrate arbitrary environment variables (T1) and trigger network side effects from allowed class constructors (T2). The exfiltrated credentials could then be sent to an attacker-controlled endpoint via a side-effecting `__init__`. However, both `secrets_from_env=True` and `allowed_objects='all'` must be explicitly enabled by the user — the default configuration prevents both attacks.

No other threat chains identified within langchain-core alone. Cross-package chains (e.g., core serialization + partner init side effects) may exist but are outside this document's scope.

---

## Input Source Coverage

Maps each input source category to its data flows, threats, and validation. The "Responsibility" column reflects that users control many input paths in this open source library.

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| Serialized payloads (`loads`/`load`) | DF1, DF8, DF9 | T1, T2 | `langchain_core/load/load.py:Reviver.__call__`: namespace + allowlist + jinja2 blocker + Bedrock validator + escape protection | Project (framework controls allowlist defaults) | `secrets_from_env=True` with untrusted data; `allowed_objects='all'` with untrusted data |
| User direct input (prompts, tool defs) | DF2, DF10 | T5, T6 | `langchain_core/utils/formatting.py:StrictFormatter` (blocks positional args); `langchain_core/prompts/string.py:get_template_variables` (blocks `.` and `[]` in f-string vars); Pydantic schema validation for tools | User | Users responsible for template content trust and tool implementation safety |
| LLM output (tool calls, structured) | DF5, DF6 | T3, T5 | `langchain_core/tools/base.py:BaseTool._parse_input` (Pydantic schema); `langchain_core/output_parsers/xml.py:XMLOutputParser.parse` (defusedxml default) | User/shared | No semantic sanitization of free-text; XML DTD not blocked without defusedxml |
| URL-sourced content | DF4 | T4 | `langchain_core/_security/_ssrf_protection.py:validate_safe_url` | Project (framework provides validation utility) | DNS rebinding TOCTOU; validation is opt-in, not automatic |
| Configuration (env vars) | DF8 | T1 | `SecretStr` wrapper for credentials | Shared | `secrets_from_env=True` reads arbitrary env vars |
| Filesystem paths (prompt loading) | DF3 | — | `langchain_core/prompts/loading.py:_validate_path` | Project (framework validates paths) | Deprecated; symlink resolution before suffix check mitigates bypass |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Prompt injection leading to arbitrary code execution via user-registered tools | The project does not control which tools users register. A user who registers a code execution tool and uses a jailbreakable model accepts the risk. | Providing correct tool argument schemas (`langchain_core/tools/base.py:BaseTool._parse_input`); validating argument types via Pydantic |
| API key leakage via user application logs | The project wraps API keys in `SecretStr` to prevent accidental logging by the framework itself. User logging behavior is outside the project's control. | `SecretStr` wrapping; `langchain_core/load/serializable.py:Serializable.lc_secrets` property; `langchain_core/utils/utils.py:secret_from_env` helper |
| Malicious custom callback handler execution | Callback handlers are user-provided code. A malicious callback can do anything the Python process allows. | Providing a well-defined `BaseCallbackHandler` interface; exception handling in `langchain_core/callbacks/manager.py:handle_event` |
| Model output containing harmful content | The project does not control model behavior, alignment, or safety filtering. | Correctly forwarding model responses without modification; providing output parser framework for structured validation |
| Supply chain attacks on dependencies (Pydantic, PyYAML, tenacity, jsonpatch) | The project depends on these packages. Compromise of those packages is outside the project's control. | Pinning dependency versions in `pyproject.toml` and `uv.lock` |
| Exfiltration via tool calls in agentic workflows | An agent equipped with network-capable tools (user-registered) can exfiltrate data if prompted to do so. Tool capabilities are user-controlled. | Not providing dangerous default tools (no PythonREPL, shell, or HTTP fetch tool in langchain-core) |
| Arbitrary code execution via `RunnableLambda` with user functions | `RunnableLambda` wraps arbitrary Python callables. The wrapped function can do anything. | Providing composition primitives (`langchain_core/runnables/base.py:RunnableLambda`); users control what functions they wrap |
| YAML deserialization attacks via prompt loading | `langchain_core/prompts/loading.py:_load_examples` uses `yaml.safe_load()` (not `yaml.load()`), preventing unsafe YAML deserialization. | Using `yaml.safe_load()` exclusively (`langchain_core/prompts/loading.py:_load_examples`, line 121) |

### Rationale

**Prompt injection as out-of-scope**: langchain-core is a library; users choose which models and tools to compose. The framework provides correct Pydantic schemas for tool arguments (`langchain_core/tools/base.py:BaseTool._parse_input`) and validates argument types, but cannot prevent a model from being manipulated into misusing legitimate tools. This is consistent with the industry-wide understanding that prompt injection is an application-layer concern when deploying LLM agents.

**Runtime Jinja2 as a boundary case**: The project blocks jinja2 during *deserialization* (`langchain_core/load/load.py:_block_jinja2_templates`) and *file-based prompt loading* (`langchain_core/prompts/loading.py:_load_prompt`) because these are the paths where untrusted data is most likely to arrive. Runtime construction via `PromptTemplate(template_format='jinja2')` is a deliberate user choice — the framework uses `SandboxedEnvironment` and warns in docstrings that this is best-effort. This is classified as T6 (in-scope, Medium) rather than out-of-scope because the framework does provide the jinja2 execution path.

**Callback data exposure**: Callback handlers receive run inputs, outputs, and metadata via `langchain_core/callbacks/manager.py:handle_event`. This data may include user PII. However, the framework's callback system is designed to pass this data — it is the feature, not a bug. Users who register callback handlers accept that those handlers receive run data.

---

## Investigated and Dismissed

Threats investigated during flaw validation that were found to be non-exploitable in the current version.

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|----------------|---------------|----------|------------|
| D1 | Jinja2 SSTI via deserialized `PromptTemplate` (CVE path: GHSA-6qv9-48xg-fc7f) | Traced full deserialization path: `loads()` → `Reviver.__call__()` → `init_validator` → `default_init_validator` → `_block_jinja2_templates`. Checked whether `init_validator=None` could be passed to bypass. | `langchain_core/load/load.py:_block_jinja2_templates` (line 177); `langchain_core/load/load.py:default_init_validator` (line 208); default `init_validator=default_init_validator` in function signatures | Jinja2 check fires before `cls(**kwargs)` is called. Overriding with `init_validator=None` removes the check but requires the caller to explicitly opt out. Non-exploitable with default settings. |
| D2 | Path traversal in `load_prompt()` via `template_path` (GHSA-qh6h-p6c9-ff54) | Reviewed `langchain_core/prompts/loading.py:_load_template`, `_validate_path`. Both `load_prompt` and `load_prompt_from_config` deprecated since v1.2.21 with `allow_dangerous_paths=False` default. | `langchain_core/prompts/loading.py:_validate_path` (line 21); `langchain_core/prompts/loading.py:load_prompt` (deprecated since 1.2.21); `_load_template` resolves symlinks at line 101 before suffix check | Patched in v1.2.21. Current code raises `ValueError` for absolute paths and `..` traversal by default. Symlink resolution happens before suffix validation. Not exploitable with default settings. |
| D3 | F-string template injection via attribute access (e.g., `{input.__class__}`) | Reviewed `langchain_core/prompts/string.py:get_template_variables` and `langchain_core/utils/formatting.py:StrictFormatter`. | `langchain_core/prompts/string.py:get_template_variables` (lines 284-306): blocks variables containing `.`, `[`, `]`, and all-digit names. `langchain_core/utils/formatting.py:StrictFormatter.vformat` (lines 23-48): rejects positional arguments. | F-string attribute access, indexing, and positional arguments are all blocked. Not exploitable. |
| D4 | XXE (external entity file read) via `XMLOutputParser` with `parser='xml'` | Tested standard library `xml.etree.ElementTree.fromstring()` with `<!ENTITY xxe SYSTEM "file:///etc/passwd">` payload. | Modern Python expat (3.9.8+/3.10.1+) does not resolve `SYSTEM` external entities in `fromstring()`. Returns `ParseError: undefined entity`. | External entity resolution is blocked by default in modern expat. Not exploitable for file read. Internal entity expansion (T3) remains a separate, verified concern. |

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-08 | langster-threat-model (deep mode, commit d3e60f5c03) | Initial langchain-core focused threat model — 11 components, 7 data classifications (2 Critical, 3 High, 1 Medium, 1 Low; details for all Critical/High entries), 6 trust boundaries, 10 data flows, 6 threats (1 High verified, 5 Medium), 8 out-of-scope patterns, 4 investigated and dismissed. Initial langchain-core focused threat model. |
