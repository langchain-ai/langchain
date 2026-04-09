# Threat Model: langchain (langchain_v1)

> Generated: 2026-04-08 | Commit: d3e60f5c03 | Scope: libs/langchain_v1/ (langchain v1.2.15) | Visibility: Open Source | Mode: deep

> **Disclaimer:** This threat model is automatically generated to help developers and security researchers understand where trust is placed in this system and where boundaries exist. It is experimental, subject to change, and not an authoritative security reference -- findings should be validated before acting on them. The analysis may be incomplete or contain inaccuracies. We welcome suggestions and corrections to improve this document.

For vulnerability reporting, see [GitHub Security Advisories](https://github.com/langchain-ai/langchain/security/advisories/new).

See also: the [langchain-core threat model](THREAT_MODEL_CORE.md) for the base abstractions layer.

---

## Scope

### In Scope

- `libs/langchain_v1/langchain/agents/` -- Agent factory (`create_agent`), agent middleware framework, middleware types
- `libs/langchain_v1/langchain/agents/middleware/` -- All shipped middleware: `ShellToolMiddleware`, `FilesystemFileSearchMiddleware`, `PIIMiddleware`, `HumanInTheLoopMiddleware`, `ContextEditingMiddleware`, `SummarizationMiddleware`, `LLMToolEmulator`, `TodoListMiddleware`, `LLMToolSelectorMiddleware`, `ToolCallLimitMiddleware`, `ModelCallLimitMiddleware`, `ModelFallbackMiddleware`, `ModelRetryMiddleware`, `ToolRetryMiddleware`
- `libs/langchain_v1/langchain/agents/middleware/_execution.py` -- Execution policies: `HostExecutionPolicy`, `DockerExecutionPolicy`, `CodexSandboxExecutionPolicy`
- `libs/langchain_v1/langchain/agents/middleware/_redaction.py` -- PII detection and redaction engine
- `libs/langchain_v1/langchain/chat_models/base.py` -- `init_chat_model` factory with dynamic `importlib` loading
- `libs/langchain_v1/langchain/embeddings/base.py` -- `init_embeddings` factory with dynamic `importlib` loading
- `libs/langchain_v1/langchain/agents/structured_output.py` -- Structured output strategies (ToolStrategy, ProviderStrategy)
- `libs/langchain_v1/langchain/tools/tool_node.py` -- Tool node re-exports from LangGraph

### Out of Scope

- `libs/core/` -- langchain-core base abstractions (separate threat model at `.github/THREAT_MODEL_CORE.md`)
- `libs/partners/` -- Partner integration packages (separate per-partner threat surface)
- `libs/langchain/` -- langchain-classic legacy package
- `libs/text-splitters/` -- Document chunking utilities
- `libs/standard-tests/` -- Test harnesses; not shipped attack surface
- `tests/` -- Unit and integration tests (read during analysis; not threat-modeled)
- User application code, model selection, custom tools, custom callbacks -- user-controlled
- LLM model behavior -- the project cannot guarantee model safety across all models users may select
- LangGraph internals -- separate product and repository; langchain_v1 depends on LangGraph but does not own its code
- Deployment infrastructure -- users control hosting, network topology, and secrets management

### Assumptions

1. The project is used as a library/framework -- users control their own application code, model selection, and deployment infrastructure.
2. `ShellToolMiddleware` is an opt-in middleware that grants the agent explicit shell access by design; users who add it accept that the agent can execute arbitrary commands within the configured execution policy.
3. `FilesystemFileSearchMiddleware` is an opt-in middleware; the `root_path` is set by the deployer to confine filesystem access.
4. `HumanInTheLoopMiddleware` assumes the interrupt/resume boundary (LangGraph `interrupt()`) is trusted infrastructure; the human reviewer is a trusted party.
5. API keys are managed by partner integrations in langchain-core via `SecretStr`; langchain_v1 does not directly handle credentials.
6. The `create_agent` function delegates to LangGraph for graph compilation and execution; LangGraph's own security properties are inherited, not verified here.

---

## System Overview

`langchain` (v1.2.15, published as the `langchain` PyPI package from `libs/langchain_v1/`) is the actively maintained implementation layer of the LangChain Python ecosystem. It provides `create_agent` -- a high-level factory for building LLM-powered tool-calling agents -- along with a composable middleware system that intercepts and modifies agent behavior at model call, tool call, and lifecycle boundaries. Key shipped middleware includes shell command execution, filesystem search, PII redaction, human-in-the-loop approval, context window management, and rate limiting. The package depends on `langchain-core` (base abstractions) and `langgraph` (graph execution engine).

### Architecture Diagram

```
+----------------------------------------------------------------------+
|                          User Application                            |
|                                                                      |
|  User Code ---> create_agent(model, tools, middleware)               |
|                      |                                               |
|                      v                                               |
|               LangGraph StateGraph                                   |
|                      |                                               |
|          +-----------+-----------+                                   |
|          |                       |                                   |
|          v                       v                                   |
|    [model node]            [tools node]                              |
|    Middleware hooks:       ToolNode dispatch:                         |
|    before_model            wrap_tool_call                            |
|    wrap_model_call         tool execution                            |
|    after_model                   |                                   |
|          |              +--------+--------+                          |
|          |              |        |        |                          |
| - - - - -|- - - - - - -|- - - - | - - - -|- - - - - TB1 - - - - -  |
|          v              v        v        v                          |
|    External LLM    ShellSession  FS     HITL                         |
|    Provider API    (C2 via C3)  Search  interrupt                    |
|                         |       (C4)    (C5)                         |
|                  - - - -|- - - TB3 - - - - - -                       |
|                         v                                            |
|                   OS subprocess                                      |
|                   (HostExec / Docker / Codex)                        |
+----------------------------------------------------------------------+
```

> Trust boundaries TB1-TB5 are described in the Trust Boundaries section below.

---

## Components

| ID | Component | Description | Trust Level | Default? | Entry Points |
|----|-----------|-------------|-------------|----------|--------------|
| C1 | Agent Factory | `create_agent` -- assembles a LangGraph `StateGraph` from model, tools, and middleware; composes middleware hooks into chained handlers | framework-controlled | Yes (when `create_agent` called) | `factory.py:create_agent` |
| C2 | Shell Tool Middleware | Persistent interactive bash session with configurable execution policies; writes LLM-generated commands to bash stdin | framework-controlled (shell infra) / user-controlled (execution policy selection) | No (opt-in middleware) | `shell_tool.py:ShellToolMiddleware.__init__`, `shell_tool.py:ShellSession.execute` |
| C3 | Execution Policies | `HostExecutionPolicy` (bare subprocess), `DockerExecutionPolicy` (container isolation), `CodexSandboxExecutionPolicy` (Codex sandbox) | user-controlled (policy selection) | No (opt-in; `HostExecutionPolicy` is default when `ShellToolMiddleware` is used without specifying a policy) | `_execution.py:HostExecutionPolicy.spawn`, `_execution.py:DockerExecutionPolicy.spawn`, `_execution.py:CodexSandboxExecutionPolicy.spawn` |
| C4 | File Search Middleware | Glob and grep search over local filesystem within a user-configured `root_path`; uses ripgrep with Python fallback | user-controlled (root_path, patterns) | No (opt-in middleware) | `file_search.py:FilesystemFileSearchMiddleware.__init__` (creates `glob_search` and `grep_search` tools) |
| C5 | Human-in-the-Loop Middleware | Interrupts agent execution for human review of tool calls; supports approve/edit/reject decisions | framework-controlled (interrupt mechanism) / user-controlled (decision content) | No (opt-in middleware) | `human_in_the_loop.py:HumanInTheLoopMiddleware.after_model` |
| C6 | PII Middleware | Detects and redacts PII (email, credit card, IP, MAC, URL) in message content using regex-based detectors; supports redact/mask/hash/block strategies | framework-controlled | No (opt-in middleware) | `pii.py:PIIMiddleware.before_model`, `pii.py:PIIMiddleware.after_model` |
| C7 | Context Editing Middleware | Prunes tool use history from conversation when token limits are exceeded; operates on deep copies | framework-controlled | No (opt-in middleware) | `context_editing.py:ContextEditingMiddleware.wrap_model_call` |
| C8 | Summarization Middleware | Summarizes older conversation history when token/message limits are approached; replaces old messages with a summary | framework-controlled | No (opt-in middleware) | `summarization.py:SummarizationMiddleware.before_model` |
| C9 | Chat Model Factory | `init_chat_model` -- dynamic provider loading via `importlib.import_module` from a hardcoded provider registry | framework-controlled | Yes (when string model names used) | `chat_models/base.py:init_chat_model` |
| C10 | Embeddings Factory | `init_embeddings` -- dynamic provider loading via `importlib.import_module` from a hardcoded provider registry | framework-controlled | Yes (when string model names used) | `embeddings/base.py:init_embeddings` |
| C11 | Middleware Type System | Base `AgentMiddleware` class, `ModelRequest`/`ModelResponse`/`ToolCallRequest` data types, hook decorators, state schemas | framework-controlled | Yes | `types.py:AgentMiddleware`, `types.py:ModelRequest`, `types.py:AgentState` |
| C12 | Structured Output | `ToolStrategy`, `ProviderStrategy`, `AutoStrategy` for enforcing structured LLM responses; Pydantic-based parsing | framework-controlled | No (opt-in via `response_format`) | `structured_output.py:ToolStrategy`, `structured_output.py:ProviderStrategy` |

---

## Data Classification

Classifies all sensitive data types found in the codebase with their sensitivity levels, storage locations, and retention policies.

| ID | PII Category | Specific Fields | Sensitivity | Storage Location(s) | Encrypted at Rest | Retention | Regulatory |
|----|-------------|----------------|-------------|---------------------|-------------------|-----------|------------|
| DC1 | Shell commands and output | `_ShellToolInput.command`, `CommandExecutionResult.output` | High | In-memory (transient); written to bash stdin pipe | N/A | Process lifetime; output returned to LLM | N/A (may contain arbitrary data) |
| DC2 | Filesystem paths and content | `FilesystemFileSearchMiddleware.root_path`, glob/grep results including file content | Medium | Host filesystem (read-only by middleware); in-memory results | N/A | Transient | N/A (depends on file content) |
| DC3 | LLM conversation state | `AgentState.messages` (HumanMessage, AIMessage, ToolMessage content) | High | In-memory; LangGraph checkpointer (if configured) | N/A (framework does not persist) | Checkpointer-dependent | GDPR, CCPA (when containing PII) |
| DC4 | HITL decision payloads | `HITLRequest`, `HITLResponse`, `EditDecision.edited_action` (tool name + args) | Medium | In-memory; LangGraph interrupt/resume state | N/A | Transient | N/A |
| DC5 | PII detection results | `PIIMatch.value` (matched PII content), redacted output | High | In-memory (transient) | N/A | Transient | GDPR, CCPA |
| DC6 | Subprocess environment | `env` dict passed to execution policies; may contain API keys or secrets | Critical | OS process environment; Docker `-e` flags | N/A | Process lifetime | All |
| DC7 | Agent execution metadata | Tool call counts (`ToolCallLimitState`), model call counts (`ModelCallLimitState`), conversation summaries | Low | LangGraph state (checkpointer-dependent) | N/A | Checkpointer-dependent | N/A |

### Data Classification Details

#### DC1: Shell commands and output

- **Fields**: `_ShellToolInput.command` (LLM-generated string), `CommandExecutionResult.output` (shell stdout/stderr).
- **Storage**: In-memory only within langchain_v1. Commands are written directly to bash stdin; output is collected via pipe reader threads and returned as `ToolMessage` content.
- **Access**: Read by `shell_tool.py:ShellToolMiddleware._run_shell_tool` (dispatches command); `shell_tool.py:ShellSession.execute` (writes to stdin). Output read by `_collect_output`.
- **Encryption**: N/A (in-memory, piped to subprocess).
- **Retention**: Transient -- garbage-collected when `ToolMessage` goes out of scope or conversation is pruned.
- **Logging exposure**: `shell_tool.py:ShellToolMiddleware._run_shell_tool` logs the raw command string at INFO level. Output is logged only if operator configures verbose logging.
- **Gaps**: Commands are logged in plaintext. If commands contain secrets (e.g., `export API_KEY=...`), they appear in application logs. Redaction rules apply to output only, not to command input.

#### DC6: Subprocess environment

- **Fields**: `env` dict passed to `ShellToolMiddleware.__init__`, forwarded to `BaseExecutionPolicy.spawn`.
- **Storage**: OS process environment for `HostExecutionPolicy`; Docker `-e K=V` flags for `DockerExecutionPolicy`.
- **Access**: `_execution.py:HostExecutionPolicy.spawn` passes `env` to `subprocess.Popen`. `_execution.py:DockerExecutionPolicy.spawn` iterates env as `-e` flags. No filtering or sanitization of keys or values.
- **Encryption**: N/A (environment variables are plaintext).
- **Retention**: Process lifetime of the subprocess.
- **Logging exposure**: Not logged by default. However, commands executed within the shell can read and exfiltrate env vars (e.g., `env`, `printenv`).
- **Gaps**: **Critical**: If the operator passes API keys or secrets in the `env` dict, any command executing in the shell can read them. The framework does not filter, warn, or redact environment variable content. For `DockerExecutionPolicy`, `_execution.py:DockerExecutionPolicy.spawn` also copies `os.environ` for the Docker CLI process itself (the host process running `docker run`).

---

## Trust Boundaries

| ID | Boundary | Description | Controls (Inside) | Does NOT Control (Outside) |
|----|----------|-------------|-------------------|---------------------------|
| TB1 | User application / deployer <-> agent framework | Configuration boundary where the deployer selects model, tools, middleware, and policies | Middleware composition, execution policy enforcement, tool registration, structured output validation, model provider loading from hardcoded registry | Which middleware the user enables, what tools the user registers, what execution policy the user selects, what `root_path` or `env` the user configures |
| TB2 | Framework <-> external LLM provider API | HTTPS API boundary; inherited from langchain-core partner integrations | Request formatting via `init_chat_model` (C9); model is bound via provider registry; API key handling delegated to partner packages | Model behavior, LLM response content, tool call argument semantics |
| TB3 | Framework <-> shell subprocess | Process boundary between the Python agent and the bash shell session | Execution policy selection (`_execution.py`), command timeout enforcement, output line/byte truncation, process group management, output redaction (post-execution) | Content of commands written to bash stdin (no validation); behavior of executed commands; filesystem/network access within the policy's isolation scope |
| TB4 | Framework <-> filesystem (file search) | Filesystem access boundary via `FilesystemFileSearchMiddleware` | Path traversal prevention (`file_search.py:FilesystemFileSearchMiddleware._validate_and_resolve_path`): `..` and `~` blocking, `resolve()` + `relative_to()` containment check on user-supplied base path; file size limits; ripgrep subprocess with no `--follow` flag | Content of files within `root_path`; symbolic link targets discovered during glob/rglob traversal (per-file containment not checked in Python fallback); filesystem permissions |
| TB5 | Framework <-> human reviewer (HITL) | LangGraph `interrupt()` boundary where agent execution pauses for human decision | Interrupt trigger (which tools require review), decision type gating (`allowed_decisions`), decision count validation | Content of human edit decisions (tool name and args are unconstrained); whether the edited tool name exists in the agent's tool registry; schema validity of edited args |

### Boundary Details

#### TB1: User application / deployer <-> agent framework

- **Inside**: `factory.py:create_agent` composes the middleware stack, binds tools to `ToolNode`, validates no duplicate middleware. `chat_models/base.py:init_chat_model` loads providers only from `_BUILTIN_PROVIDERS` hardcoded registry via `importlib.import_module`. `embeddings/base.py:init_embeddings` uses the same pattern.
- **Outside**: All middleware is opt-in. The deployer chooses which middleware to enable and how to configure it. Dangerous middleware (`ShellToolMiddleware`) with a permissive default policy (`HostExecutionPolicy`) is the deployer's explicit choice.
- **Crossing mechanism**: Python function calls to `create_agent` and middleware constructors.

#### TB3: Framework <-> shell subprocess

- **Inside**: `_execution.py:_launch_subprocess` uses `subprocess.Popen` with list arguments (no `shell=True`). `HostExecutionPolicy` optionally applies CPU/memory `prlimit`. `DockerExecutionPolicy` adds `--network none`, `--rm`, optional `--read-only`, workspace bind-mount. `shell_tool.py:ShellSession` enforces command timeout with session restart, output truncation via `max_output_lines`/`max_output_bytes`.
- **Outside**: Commands written to bash stdin are not validated, escaped, filtered, or allowlisted. The bash process interprets all shell metacharacters (`;`, `&&`, `||`, `|`, `$()`, backticks, redirects). `HostExecutionPolicy` provides no filesystem or network sandboxing. Output redaction via PII rules is post-execution only.
- **Crossing mechanism**: `shell_tool.py:ShellSession.execute` writes command string to `self._stdin` (pipe to bash process).

#### TB4: Framework <-> filesystem (file search)

- **Inside**: `file_search.py:FilesystemFileSearchMiddleware._validate_and_resolve_path` resolves the user-supplied path with `Path.resolve()` (follows symlinks), then checks `resolved.relative_to(self.root_path)`. The `root_path` itself is resolved at init time. `..` and `~` are blocked in the raw path string. Ripgrep subprocess uses `--` to prevent flag injection and does not pass `--follow` (no symlink following).
- **Outside**: When the Python fallback (`_python_search`) is active, `Path.rglob("*")` follows directory symlinks by default. Individual files discovered by rglob are not re-validated through `_validate_and_resolve_path`. `file_path.read_text()` follows symlinks to read content of files whose targets may be outside `root_path`.
- **Crossing mechanism**: Python `Path.glob()`, `Path.rglob()`, `Path.read_text()`, `subprocess.run(["rg", ...])`.

#### TB5: Framework <-> human reviewer (HITL)

- **Inside**: `human_in_the_loop.py:HumanInTheLoopMiddleware.after_model` checks tool calls against `self.interrupt_on`, builds `HITLRequest` with `ActionRequest` and `ReviewConfig`, calls `langgraph.types.interrupt()`. Validates that `len(decisions) == len(interrupted_tool_calls)`. Validates decision type is in `allowed_decisions`.
- **Outside**: The `EditDecision.edited_action` allows the human to set any `name` (string) and any `args` (dict). No validation checks the edited name against the agent's registered tool list. The `args_schema` field in `InterruptOnConfig` is declared but never read or enforced. The policy lookup for edit processing uses the *original* tool name's config, not the edited tool name's config.
- **Crossing mechanism**: LangGraph `interrupt()` suspend/resume.

---

## Data Flows

| ID | Source | Destination | Data Type | Classification | Crosses Boundary | Protocol |
|----|--------|-------------|-----------|----------------|------------------|----------|
| DF1 | User application | C1 Agent Factory (`create_agent`) | Model config, tools, middleware, system prompt | -- | TB1 | Python function call |
| DF2 | C1 Agent Factory -> C9/C10 | External LLM provider (via partner SDK) | Messages (DC3), API credentials | DC3 | TB2 | HTTPS (via partner SDK) |
| DF3 | External LLM provider | C1 Agent Factory (model node) | LLM response, tool call arguments | DC3 | TB2 | HTTPS (via partner SDK) |
| DF4 | C1 Agent Factory (model node) | C2 Shell Tool Middleware -> C3 Execution Policy | LLM-generated shell command string, env dict | DC1, DC6 | TB3 | Python -> stdin pipe |
| DF5 | C3 Execution Policy (bash process) | C2 Shell Tool Middleware | Command stdout/stderr, exit code | DC1 | TB3 | stdout/stderr pipe |
| DF6 | C1 Agent Factory (model node) | C4 File Search Middleware -> filesystem | Glob/grep patterns, base path | DC2 | TB4 | Python/ripgrep |
| DF7 | Filesystem | C4 File Search Middleware | File paths, file content (grep results) | DC2 | TB4 | Python file I/O, ripgrep JSON |
| DF8 | C1 Agent Factory (after_model hook) | C5 HITL Middleware -> human reviewer | HITLRequest (tool calls for review) | DC4 | TB5 | LangGraph interrupt |
| DF9 | Human reviewer | C5 HITL Middleware -> C1 Agent Factory | HITLResponse (approve/edit/reject decisions) | DC4 | TB5 | LangGraph resume |
| DF10 | C6 PII Middleware | Agent state (messages) | Redacted message content, PIIMatch results | DC5 | -- | In-memory state update |
| DF11 | C7/C8 Context/Summarization Middleware | Agent state (messages) | Pruned/summarized conversation history | DC3, DC7 | -- | In-memory state update |

### Flow Details

#### DF4: LLM-generated command -> Shell subprocess

- **Data**: Raw command string from `_ShellToolInput.command`; env dict from middleware configuration.
- **Validation**: `_ShellToolInput.validate_payload` checks mutual exclusion of `command`/`restart` only. `shell_tool.py:ShellToolMiddleware._run_shell_tool` checks `not command or not isinstance(command, str)`. **No content validation, escaping, allowlisting, or denylisting.** The string is written verbatim to bash stdin.
- **Trust assumption**: The command is generated by the LLM and is therefore **untrusted**. The execution policy is the sole isolation mechanism.

#### DF7: Filesystem -> File Search Middleware (Python fallback)

- **Data**: File paths discovered by `Path.rglob("*")`, file content read by `Path.read_text()`.
- **Validation**: Base path is validated via `_validate_and_resolve_path`. Individual files from rglob are **not** validated -- their path strings are children of the validated base, but symlink targets may be outside `root_path`.
- **Trust assumption**: Files within `root_path` are assumed safe to read. Symlinks within `root_path` pointing outside are not expected by the middleware.

#### DF9: Human reviewer -> HITL Middleware

- **Data**: `HITLResponse` containing `Decision` objects. `EditDecision` carries `edited_action` with `name` (str) and `args` (dict).
- **Validation**: Decision count is validated. Decision type is checked against `allowed_decisions`. **No validation of edited tool name or args content.**
- **Trust assumption**: The human reviewer is a trusted party. However, the middleware does not distinguish between a legitimate human edit and a compromised/malicious client submitting the resume payload.

---

## Threats

| ID | Data Flow | Classification | Threat | Boundary | Severity | Validation | Code Reference |
|----|-----------|----------------|--------|----------|----------|------------|----------------|
| T1 | DF4 | DC1, DC6 | Unrestricted shell command execution via `HostExecutionPolicy` default -- LLM-generated commands are written verbatim to bash stdin with no validation, escaping, or sandboxing | TB3 | High | Verified | `shell_tool.py:ShellSession.execute`, `shell_tool.py:ShellToolMiddleware._run_shell_tool`, `_execution.py:HostExecutionPolicy.spawn` |
| T2 | DF4 | DC6 | Environment variable exfiltration from shell subprocess -- commands can read all env vars passed to the subprocess; operator-supplied secrets in `env` dict are accessible | TB3 | Medium | Verified | `_execution.py:HostExecutionPolicy.spawn`, `shell_tool.py:ShellToolMiddleware.__init__` |
| T3 | DF7 | DC2 | Symlink-following file read outside `root_path` in Python fallback search -- `_python_search` uses `rglob("*")` which follows symlinks; `read_text()` reads content without per-file containment check | TB4 | Medium | Verified | `file_search.py:FilesystemFileSearchMiddleware._python_search` |
| T4 | DF6, DF7 | DC2 | Filesystem structure disclosure via symlink following in `glob_search` -- `Path.glob()` follows directory symlinks, disclosing filenames and metadata outside `root_path` | TB4 | Low | Verified | `file_search.py:FilesystemFileSearchMiddleware.__init__` (glob_search closure) |
| T5 | DF9 | DC4 | HITL edit decision allows arbitrary tool redirection -- edited tool name and args are not validated against the agent's registered tool list or any schema | TB5 | Medium | Verified | `human_in_the_loop.py:HumanInTheLoopMiddleware._process_decision`, `human_in_the_loop.py:HumanInTheLoopMiddleware.after_model` |
| T6 | DF6 | -- | ReDoS via user/LLM-supplied regex in `grep_search` Python fallback and custom PII detectors -- no timeout or complexity limit on regex patterns | TB4 | Low | Likely | `file_search.py:FilesystemFileSearchMiddleware.__init__` (grep_search closure), `_redaction.py:resolve_detector` |
| T7 | DF4 | DC1 | Shell command logging in plaintext -- `_run_shell_tool` logs raw command at INFO level; commands containing secrets appear in application logs | TB3 | Low | Verified | `shell_tool.py:ShellToolMiddleware._run_shell_tool` |
| T8 | DF3 -> DF4 | DC1 | Prompt injection escalation via shell tool -- LLM processes untrusted content (retrieved documents, tool outputs) that instructs it to execute malicious shell commands | TB2, TB3 | High | Unverified | `shell_tool.py:ShellSession.execute` (sink), `factory.py:create_agent` (agent loop) |

### Threat Details

#### T1: Unrestricted shell command execution via `HostExecutionPolicy`

- **Flow**: DF4 (LLM tool call -> `_run_shell_tool` -> `ShellSession.execute` -> bash stdin)
- **Description**: When `ShellToolMiddleware` is used with the default `HostExecutionPolicy`, LLM-generated commands are written verbatim to bash stdin. The complete validation surface is: (1) `_ShellToolInput.validate_payload` checks mutual exclusion of `command`/`restart`; (2) `_run_shell_tool` checks `not command or not isinstance(command, str)`. No content inspection occurs. Shell metacharacters (`;`, `&&`, `||`, `|`, `$()`, backticks, redirects, embedded newlines) are passed directly to bash. The bash binary is launched as `/bin/bash` with no restricted-mode flags (`-r`). `HostExecutionPolicy` provides no filesystem or network sandboxing; only optional CPU/memory `prlimit` limits (off by default).
- **Preconditions**: (1) User enables `ShellToolMiddleware` (opt-in); (2) user uses `HostExecutionPolicy` (default when no policy specified); (3) the LLM generates a command with shell metacharacters or malicious intent.

#### T2: Environment variable exfiltration from shell subprocess

- **Flow**: DF4 (env dict -> execution policy -> subprocess environment)
- **Description**: The `env` dict passed to `ShellToolMiddleware.__init__` is forwarded to `BaseExecutionPolicy.spawn` without filtering. For `HostExecutionPolicy`, it becomes the subprocess environment via `subprocess.Popen(env=...)`. For `DockerExecutionPolicy`, each key-value pair becomes a `-e K=V` Docker flag. Commands executing in the shell can read all environment variables (e.g., `env`, `printenv`, `echo $SECRET_KEY`). If the operator passes API keys or secrets in the env dict, any LLM-generated or agent-executed command can access them.
- **Preconditions**: (1) User passes secrets in the `env` dict to `ShellToolMiddleware`; (2) an LLM-generated command reads environment variables.

#### T3: Symlink-following file read outside `root_path` in Python fallback

- **Flow**: DF7 (filesystem -> `_python_search` -> `rglob` -> `read_text`)
- **Description**: `_python_search` validates only the user-supplied base path via `_validate_and_resolve_path`. Individual files discovered by `Path.rglob("*")` are not re-validated. Python's `rglob` follows directory symlinks by default. `Path.read_text()` follows file symlinks. If a symlink inside `root_path` points to a file or directory outside `root_path`, the target's content is read and returned to the agent. The ripgrep backend is not affected (no `--follow` flag), so this only occurs when: (a) `use_ripgrep=False`, (b) ripgrep is not installed, or (c) ripgrep times out (triggering the Python fallback).
- **Preconditions**: (1) A symlink inside `root_path` points outside; (2) the Python fallback search is active (ripgrep unavailable, disabled, or timed out); (3) the agent issues a grep/glob pattern that matches the symlink.

#### T5: HITL edit decision allows arbitrary tool redirection

- **Flow**: DF9 (human reviewer -> `_process_decision` -> revised `ToolCall`)
- **Description**: When a human returns an `EditDecision`, the middleware constructs a new `ToolCall` from `edited_action["name"]` and `edited_action["args"]` with no validation. The `name` field is an unconstrained `str` -- it is not checked against `self.interrupt_on`, the agent's registered tool list, or any allowlist. The `args` field is `dict[str, Any]` with no schema validation. The `args_schema` field in `InterruptOnConfig` is declared in the type definition but never read or enforced in the implementation. The policy lookup at `after_model` uses the *original* tool name's config, not the edited name's config.
- **Preconditions**: (1) `HumanInTheLoopMiddleware` is configured with `"edit"` in `allowed_decisions` for at least one tool; (2) the human (or a compromised client submitting the resume payload) provides an `EditDecision` with a different tool name.

#### T8: Prompt injection escalation via shell tool

- **Flow**: DF3 -> DF4 (LLM processes untrusted content -> generates shell command)
- **Description**: In agentic workflows, the LLM may process untrusted external content (retrieved documents, tool outputs, web pages) that contains adversarial instructions. If the agent has `ShellToolMiddleware` enabled, a successful prompt injection can escalate to arbitrary shell command execution. This is the standard prompt injection escalation path for agents with shell access, amplified by the lack of command validation at TB3.
- **Preconditions**: (1) Agent processes untrusted external content; (2) the model follows adversarial instructions; (3) `ShellToolMiddleware` is enabled. All three conditions must be true.

### Chain Analysis

**T8 = T1 + prompt injection**: The combination of unrestricted shell access (T1) with prompt injection via untrusted content creates a critical escalation path. Individually, T1 is Medium-to-High (requires LLM to generate malicious commands) and prompt injection is an inherent LLM risk. Together, they form a path from untrusted document content to arbitrary code execution with full host access when `HostExecutionPolicy` is used.

**T3 + T6**: If an attacker can cause ripgrep to time out (e.g., via a very large directory tree or a slow filesystem), the Python fallback activates, enabling symlink-following file reads (T3). A separate ReDoS attack (T6) in the Python fallback could cause additional denial of service. However, these compose to DoS + information disclosure rather than escalating severity.

No other threat chains identified within langchain_v1 scope.

---

## Input Source Coverage

Maps each input source category to its data flows, threats, and validation. The "Responsibility" column reflects that users control many input paths in this open source library.

| Input Source | Data Flows | Threats | Validation Points | Responsibility | Gaps |
|-------------|-----------|---------|-------------------|----------------|------|
| LLM output (tool call arguments) | DF3, DF4, DF6 | T1, T2, T8 | `_ShellToolInput.validate_payload` (presence check only); `_validate_and_resolve_path` (file search paths); Pydantic schema on tool args (type only, no semantic validation) | User (chooses model, registers tools) / Project (provides shell tool with no command validation) | No command content validation in shell tool; no semantic validation of LLM-generated tool args |
| Filesystem content (symlink targets) | DF7 | T3, T4 | `_validate_and_resolve_path` (base path only); ripgrep no-follow default | Project (provides file search with containment check) | Python fallback `rglob` follows symlinks without per-file containment check |
| Human reviewer decisions (HITL) | DF9 | T5 | Decision count validation; decision type check (`allowed_decisions`) | Shared (project provides gating; human controls content) | No validation of edited tool name or args; `args_schema` declared but not enforced |
| User/LLM-supplied regex patterns | DF6 | T6 | `re.compile()` for syntax validation; ripgrep has built-in regex engine limits | User (supplies patterns) | No complexity/timeout limit on Python regex in fallback path; custom PII detector regex not validated for backtracking |
| Deployer configuration (env dict) | DF4 | T2 | `_normalize_env()` coerces values to str; no content filtering | User (controls env dict content) | No warning or filtering of secret-like env vars |
| Deployer configuration (model string) | DF2 | -- | `_BUILTIN_PROVIDERS` hardcoded registry allowlist in `init_chat_model` and `init_embeddings` | Project (controls provider registry) | None -- provider names are hardcoded; `importlib.import_module` only loads from known module paths |

---

## Out-of-Scope Threats

Threats that appear valid in isolation but fall outside project responsibility because they depend on conditions the project does not control.

| Pattern | Why Out of Scope | Project Responsibility Ends At |
|---------|-----------------|-------------------------------|
| Arbitrary code execution via LLM-directed shell commands when `ShellToolMiddleware` is explicitly enabled | `ShellToolMiddleware` is opt-in and designed to give the agent shell access. Users who enable it accept that the LLM can execute commands. The project's responsibility is providing execution policy options with documented isolation guarantees. | Providing `DockerExecutionPolicy` (container isolation) and `CodexSandboxExecutionPolicy` (syscall filtering) as alternatives to `HostExecutionPolicy`; documenting that `HostExecutionPolicy` provides no sandboxing |
| Prompt injection leading to tool misuse in agentic workflows | The project does not control model selection, prompt construction, or what tools users register. Prompt injection is an inherent LLM risk. | Providing `HumanInTheLoopMiddleware` for tool call approval; providing `ToolCallLimitMiddleware` and `ModelCallLimitMiddleware` for execution limits; Pydantic schema validation on tool arguments |
| Data exfiltration via user-registered tools | Users register custom tools with `create_agent`. A tool with network access can exfiltrate data if the LLM is manipulated. Tool capabilities are user-controlled. | Not shipping dangerous default tools; providing middleware hooks (`wrap_tool_call`) for custom tool call interception |
| PII leakage via user application logging of message content | The framework passes message content through middleware hooks. Users who log message content in callbacks or external systems control their own logging behavior. | Providing `PIIMiddleware` for optional PII detection and redaction; providing `SummarizationMiddleware` and `ContextEditingMiddleware` for reducing conversation history |
| LLM tool emulator generating incorrect/malicious content | `LLMToolEmulator` replaces real tool execution with LLM-generated fiction. It is explicitly designed for testing, not production. | Documenting that emulated responses are not real tool outputs; the middleware is opt-in |
| Supply chain attacks on LangGraph or partner SDKs | langchain_v1 depends on `langgraph` and dynamically loads partner packages via `importlib`. Compromise of these dependencies is outside the project's control. | Pinning dependency versions in `pyproject.toml` and `uv.lock`; loading only from hardcoded `_BUILTIN_PROVIDERS` registries |
| Docker container escape via `DockerExecutionPolicy` | Container security depends on host Docker daemon, kernel version, and container configuration. `DockerExecutionPolicy` is a best-effort isolation layer. | `DockerExecutionPolicy` defaults (`--network none`, `--rm`); documentation of host security requirements |

### Rationale

**Shell tool as opt-in accepted risk**: `ShellToolMiddleware` is explicitly designed to grant the LLM shell access. This is a deliberate, visible choice by the deployer -- analogous to giving a user SSH access. The project's responsibility is providing isolation options (`DockerExecutionPolicy`, `CodexSandboxExecutionPolicy`) and documenting the security properties of each policy. The `HostExecutionPolicy` docstring explicitly states: "best suited for trusted or single-tenant environments (CI jobs, developer workstations, pre-sandboxed containers)." In-scope threats (T1, T2) document the specific risks of the default policy; the out-of-scope pattern covers the broader "LLM runs commands" design decision.

**HITL as a shared-responsibility boundary**: `HumanInTheLoopMiddleware` is designed to add a human approval gate. The design assumes the human reviewer is trusted and the interrupt/resume infrastructure is secure. T5 documents the specific gap (no edit content validation), but the broader pattern of "malicious human reviewer" is out of scope because the middleware's purpose is to empower the human, not to constrain them.

---

## Investigated and Dismissed

Threats investigated during flaw validation that were found to be non-exploitable or already mitigated.

| ID | Original Threat | Investigation | Evidence | Conclusion |
|----|----------------|---------------|----------|------------|
| D1 | Shell injection via `subprocess.Popen` args list in `_launch_subprocess` | Traced `_execution.py:_launch_subprocess` -- uses `subprocess.Popen(list(command), ...)` with list arguments, not a string. The `# noqa: S603` suppression is appropriate. Shell injection via `Popen` args is not possible with list form. | `_execution.py:_launch_subprocess` -- `list(command)` passed to `Popen`; `shell=False` (default when list is provided) | Not exploitable. The injection risk is via stdin content (T1), not via the Popen args list. Bandit suppression is correct. |
| D2 | Flag injection in ripgrep subprocess via pattern argument | Traced `file_search.py:FilesystemFileSearchMiddleware._ripgrep_search` -- the `--` separator is placed before `pattern` in the command list. Ripgrep stops option parsing at `--`. | `file_search.py:FilesystemFileSearchMiddleware._ripgrep_search` -- `cmd.extend(["--", pattern, str(base_full)])` | Not exploitable. The `--` separator prevents the pattern from being interpreted as a ripgrep flag. |
| D3 | Provider registry injection via `init_chat_model` or `init_embeddings` | Traced `chat_models/base.py:init_chat_model` and `embeddings/base.py:init_embeddings` -- both use a hardcoded `_BUILTIN_PROVIDERS` dict. The `importlib.import_module` call only loads module paths from this registry. User-supplied `model_provider` is validated against the dict keys before any import. | `chat_models/base.py:_get_chat_model_creator` -- `if provider not in _BUILTIN_PROVIDERS: raise ValueError`; `embeddings/base.py:_get_embeddings_class_creator` -- same pattern | Not exploitable. Arbitrary module loading is prevented by the allowlist check before `importlib.import_module`. |
| D4 | Symlink file read via ripgrep backend in file search | Tested ripgrep symlink behavior -- `rg` does not follow symlinks by default (requires `--follow`/`-L` flag). The ripgrep command construction in `_ripgrep_search` does not include `--follow`. | `file_search.py:FilesystemFileSearchMiddleware._ripgrep_search` -- `cmd = ["rg", "--json"]` with no `--follow` flag | Not exploitable via ripgrep path. Symlink content read is limited to the Python fallback (`_python_search`), documented as T3. |

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-08 | langster-threat-model (deep mode, commit d3e60f5c03) | Initial langchain_v1 focused threat model -- 12 components, 7 data classifications (1 Critical, 3 High, 2 Medium, 1 Low), 5 trust boundaries, 11 data flows, 8 threats (2 High, 3 Medium, 3 Low; 6 Verified, 1 Likely, 1 Unverified), 7 out-of-scope patterns, 4 investigated and dismissed. Based on langchain-core THREAT_MODEL_CORE.md (2026-04-08). |
