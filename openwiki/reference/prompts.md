---
type: "Reference"
title: "Prompt Template Classes and Composition Methods"
description: "Technical reference for BasePromptTemplate hierarchy, string and chat prompt classes, composition operators, partial variables, variable validation, and factory methods. Covers PromptTemplate, ChatPromptTemplate, FewShot variants, and formatting pipelines."
tags: [prompt, template, variable-substitution, composition, factory, runnable, formatting]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-07T08:15:07.286Z
sources:
  - id: openwiki-source-5549894302ea4dfd5b8f4278
    resource: repo://libs/core/langchain_core/prompts/prompt.py
generated: { by: "openwiki/0.5.0", at: "2026-11-07T08:15:07.286Z" }
---

# Prompt Template Classes and Composition Methods

This reference documents the implementation and composition patterns of LangChain's prompt templating system, focusing on class hierarchies, variable management, composition operators, and format handling.

## Class Hierarchy

The prompt system is organized around a inheritance structure rooted in **`BasePromptTemplate`**:

```
BasePromptTemplate (RunnableSerializable[dict[str, Any], PromptValue])
├── StringPromptTemplate
│   ├── PromptTemplate
│   └── FewShotPromptTemplate
├── BaseChatPromptTemplate
│   ├── ChatPromptTemplate
│   └── FewShotChatMessagePromptTemplate
└── [Message-specific variants via BaseMessagePromptTemplate]
    ├── BaseStringMessagePromptTemplate
    │   ├── ChatMessagePromptTemplate
    │   ├── HumanMessagePromptTemplate
    │   ├── AIMessagePromptTemplate
    │   ├── SystemMessagePromptTemplate
    │   └── _StringImageMessagePromptTemplate (supports multi-modal)
    └── MessagesPlaceholder
```

All classes extend `RunnableSerializable[dict[str, Any], PromptValue]`, making them composable with chains and runnable operators.

## BasePromptTemplate Core Responsibilities

**`BasePromptTemplate`** (in `langchain_core.prompts.base`) is the abstract base for all prompts and provides:

### Input and Variable Management

- **`input_variables: list[str]`** — List of required variable names that must be supplied at format time. Extracted from templates by format-specific parsers.
- **`optional_variables: list[str]`** — Optional variable names (typically from `MessagesPlaceholder` with `optional=True`). These reduce the number of required inputs.
- **`input_types: dict[str, Any]`** — Optional type annotations for input variables (defaults to `str` for all variables if not provided).
- **`partial_variables: Mapping[str, Any]`** — Pre-filled variables reducing required inputs. Values can be strings or callables (evaluated at format time). Cannot overlap with `input_variables`.

### Variable Validation

The `validate_variable_names()` model validator enforces:
- Variable name `"stop"` is forbidden in both `input_variables` and `partial_variables` (reserved internally).
- No overlapping names between `input_variables` and `partial_variables`.

### Formatting Methods

- **`format_prompt(**kwargs) -> PromptValue`** — Abstract method; subclasses implement to return `StringPromptValue` or `ChatPromptValue`.
- **`format(**kwargs) -> FormatOutputType`** — Abstract method; returns string or message sequence depending on subclass.
- **`_format_prompt_with_error_handling(inner_input: dict) -> PromptValue`** — Validates inputs and calls `format_prompt()`.

Input validation via `_validate_input()` checks:
- Input is a dict (or single-variable templates accept non-dict values).
- All required `input_variables` are present.
- Raises `KeyError` with helpful escaping hints if variables are missing.

### Runnable Integration

- **`invoke(input: dict, config: RunnableConfig | None = None) -> PromptValue`** — Synchronous invocation with error handling and tracing metadata propagation.
- **`ainvoke(input: dict, config: RunnableConfig | None = None) -> PromptValue`** — Asynchronous counterpart.
- **`stream()` and `batch()`** — Inherited from `RunnableSerializable`.

Metadata (`metadata: dict | None`) and tags (`tags: list[str] | None`) are merged into the `RunnableConfig` at invocation time for observability.

### Immutable Composition

- **`partial(**kwargs) -> BasePromptTemplate`** — Returns a new instance with supplied kwargs moved from `input_variables` to `partial_variables`. Lazy callables are supported.
- **`__add__()` operator** — Subclass-specific; allows `prompt1 + prompt2` composition (details below).

## StringPromptTemplate and PromptTemplate

**`StringPromptTemplate`** (in `langchain_core.prompts.string`) is the abstract base for single-string templates, implementing:

- **`format(**kwargs) -> str`** — Returns formatted string.
- **`format_messages(**kwargs) -> list[BaseMessage]`** — Returns `[HumanMessage(content=<formatted_string>)]`.

**`PromptTemplate`** (in `langchain_core.prompts.prompt`) wraps a string template with variable substitution:

### Core Fields

- **`template: str`** — The template string.
- **`template_format: PromptTemplateFormat`** — One of `"f-string"` (default), `"mustache"`, or `"jinja2"`.
- **`validate_template: bool = False`** — If `True`, checks template syntax at instantiation (except for Mustache, which cannot be validated).

### Template Formatting Engines

**F-string** (default):
- Syntax: `{variable_name}` with `{{` and `}}` for literal braces.
- Variable extraction via Python's `string.Formatter`.
- Security: Safe; only variable substitution without execution.

**Mustache**:
- Syntax: `{{variable}}`, supports nested keys (`{{person.name}}`), sections (`{{#items}}...{{/items}}`), and conditionals.
- Variable extraction via `mustache_template_vars()` (top-level keys only; nested keys return only the top-level).
- Cannot be validated at instantiation.

**Jinja2**:
- Full Jinja2 syntax: conditionals, loops, filters, etc.
- Variable extraction via AST parsing (`jinja2.meta.find_undeclared_variables()`).
- **Security Warning**: Uses `SandboxedEnvironment` by default to block dunder attributes (`__class__`, `__globals__`, etc.), but sandboxing is best-effort. Do not use untrusted templates.

### Factory Methods

- **`from_template(template: str, template_format="f-string", partial_variables=None, **kwargs)`** — Recommended constructor; auto-detects `input_variables` from template.
- **`from_file(template_file: Path | str, encoding=None, **kwargs)`** — Loads template from file and calls `from_template()`.
- **`from_examples(examples: list[str], suffix: str, input_variables: list[str], example_separator="\n\n", prefix="", **kwargs)`** — Creates prompt by joining prefix + examples + suffix.

### Composition Operator

**`__add__()` for PromptTemplate**:
- Combines two `PromptTemplate` instances (or a `PromptTemplate` with a string).
- Requires same `template_format` for both operands.
- Merges `input_variables`, `partial_variables`, and template strings.
- Raises `ValueError` if same variable is partialed in both operands.
- Example: `prompt1 + prompt2` produces a new template concatenating both.

### Format Workflow

1. `_merge_partial_and_user_variables(**kwargs)` merges partial and user variables; evaluates callables in partial vars.
2. Calls format function from `DEFAULT_FORMATTER_MAPPING` (f-string, mustache, or jinja2 formatter).
3. Returns formatted string.

## ChatPromptTemplate and Message Composition

**`BaseChatPromptTemplate`** (in `langchain_core.prompts.chat`) is the abstract base for chat templates:

- **`format_messages(**kwargs) -> list[BaseMessage]`** — Abstract; returns list of `BaseMessage` objects.
- **`format_prompt(**kwargs) -> ChatPromptValue`** — Calls `format_messages()` and wraps result in `ChatPromptValue`.
- **`format(**kwargs) -> str`** — Calls `format_prompt()` and converts to string via `to_string()`.

### BaseMessagePromptTemplate

Abstract base for individual message templates (in `langchain_core.prompts.message`):

- **`format_messages(**kwargs) -> list[BaseMessage]`** — Abstract; formats into one or more messages.
- **`input_variables`** — Property extracted from underlying prompt.
- **`partial()`, `__add__()`** — Composition support.

### Message Prompt Template Variants

**`ChatMessagePromptTemplate`**:
- Wraps `PromptTemplate` with a custom `role: str`.
- Produces `ChatMessage(content=<formatted_text>, role=role, additional_kwargs=...)`.

**`HumanMessagePromptTemplate`, `AIMessagePromptTemplate`, `SystemMessagePromptTemplate`**:
- Specialized variants for user, AI, and system roles.
- Inherit from `_StringImageMessagePromptTemplate` (supports mixed text/image content).
- Factory: `HumanMessagePromptTemplate.from_template(template, template_format="f-string", partial_variables=None)`.

**`_StringImageMessagePromptTemplate`** (multi-modal support):
- **`prompt`** — Can be a single `StringPromptTemplate` or a list of `StringPromptTemplate | ImagePromptTemplate | DictPromptTemplate`.
- **`from_template(template, template_format="f-string", partial_variables=None)`** — Accepts string or list of mixed format templates.
- Handles image inputs via `ImagePromptTemplate` (URL-based image refs).
- Supports `DictPromptTemplate` for arbitrary structured content.

### MessagesPlaceholder

A special message template that injects a list of pre-formatted messages without templating:

- **`variable_name: str`** — Name of the variable in kwargs containing message list.
- **`optional: bool = False`** — If `True`, `format_messages()` returns `[]` when variable is missing; if `False`, raises `KeyError`.
- **`n_messages: PositiveInt | None = None`** — Maximum messages to include (slices last N).
- Supports tuples like `("human", "text")` which are converted via `convert_to_messages()`.

### ChatPromptTemplate

**`ChatPromptTemplate`** (in `langchain_core.prompts.chat`) composes message templates into a chat conversation:

- **`messages: list[MessageLike]`** — List of message templates or raw messages.
- **`template_format: PromptTemplateFormat = "f-string"`** — Default format for any string templates.
- **`validate_template: bool = False`** — Optional template validation.

#### Input Resolution

Constructor automatically infers:
- **`input_variables`** — Union of `input_variables` from all message templates.
- **`optional_variables`** — Variable names from optional `MessagesPlaceholder` instances.
- **`partial_variables`** — Pre-filled with `[]` for optional placeholders (auto-inferred).

#### Message Representation

The `messages` field accepts multiple formats:

1. **`BaseMessagePromptTemplate`** — Direct message template instance.
2. **`BaseMessage`** — Pre-formed message (e.g., `SystemMessage(content="...")`).
3. **Tuple `(message_type, template)`** — Shorthand; e.g., `("human", "{user_input}")` becomes `HumanMessagePromptTemplate`.
   - Message types: `"system"`, `"human"`, `"ai"`, `"assistant"`, or a message class.
4. **Tuple `(message_class, template)`** — Direct class; e.g., `(SystemMessage, "...")`.
5. **String** — Shorthand for `("human", string)`.

#### Composition Operator

**`__add__()` for ChatPromptTemplate**:
- Combines two `ChatPromptTemplate` instances: `prompt1 + prompt2` → new `ChatPromptTemplate(messages=prompt1.messages + prompt2.messages)`.
- Supports adding `BaseMessagePromptTemplate`, `BaseMessage`, or `BaseChatPromptTemplate` instances.
- Accepts lists/tuples and converts them via `ChatPromptTemplate.from_messages()`.
- Accepts strings and wraps them as `HumanMessagePromptTemplate.from_template(string)`.
- Merges and preserves partial variables from both operands.

#### Factory Methods

- **`from_messages(messages: Sequence[MessageLikeRepresentation], template_format="f-string")`** — Creates from mixed message formats.
- **`from_template(template: str, **kwargs)`** — Shorthand for single-message template; creates `HumanMessagePromptTemplate` + wraps in `ChatPromptTemplate`.

#### Format Workflow

1. Merge partial and user variables.
2. Iterate over `messages`:
   - If `BaseMessage`: append as-is.
   - If `BaseMessagePromptTemplate` or `BaseChatPromptTemplate`: call `format_messages(**kwargs)`, extend result.
3. Return list of formatted `BaseMessage` objects.

## FewShotPromptTemplate

**`FewShotPromptTemplate`** (in `langchain_core.prompts.few_shot`) extends `StringPromptTemplate` to include examples:

- **`examples: list[dict[str, Any]] | None`** — Fixed list of examples (dicts with example data).
- **`example_selector: BaseExampleSelector | None`** — Dynamic selector for examples (mutually exclusive with `examples`).
- **`example_prompt: PromptTemplate`** — Template for formatting each individual example.
- **`prefix: str = ""`** — Template string before examples.
- **`suffix: str`** — Template string after examples.
- **`example_separator: str = "\n\n"`** — Separator between prefix, examples, and suffix.
- **`template_format: Literal["f-string", "jinja2"] = "f-string"`** — Note: Mustache not supported for few-shot.

### Format Workflow

1. Merge partial and user variables.
2. Call `_get_examples(**kwargs)` or `_aget_examples(**kwargs)`:
   - If `examples` is provided, return as-is.
   - If `example_selector` is provided, call `example_selector.select_examples(kwargs)` (or async variant).
3. Filter each example to only include keys in `example_prompt.input_variables`.
4. Format each example using `example_prompt.format(**example)`.
5. Construct overall template: `[prefix, *formatted_examples, suffix]`.
6. Join with `example_separator`.
7. Format combined template with input variables using the format engine.

The `_FewShotPromptTemplateMixin` enforces mutual exclusivity: exactly one of `examples` or `example_selector` must be provided.

## FewShotChatMessagePromptTemplate

**`FewShotChatMessagePromptTemplate`** (in `langchain_core.prompts.few_shot`) is the chat variant of few-shot prompting:

- **`example_prompt: BaseMessagePromptTemplate | BaseChatPromptTemplate`** — Formats each example into one or more messages.
- **`examples: list[dict[str, Any]] | None`** — Fixed examples.
- **`example_selector: BaseExampleSelector | None`** — Dynamic selector (mutually exclusive with `examples`).
- **`input_variables: list[str]`** — Variables passed to example selector.

### Format Workflow

1. Get examples via `_get_examples(**kwargs)` or `_aget_examples(**kwargs)`.
2. For each example, format using `example_prompt.format_messages(**example)` to produce a list of messages.
3. Flatten results into a single message list.
4. Return prefix messages + flattened example messages + suffix messages.

Supports composition with `+` operator for message prompt templates (e.g., `SystemMessage + few_shot_prompt + HumanMessage`).

## Partial Variables and Lazy Evaluation

Partial variables support lazy evaluation via callables:

```python
prompt = PromptTemplate.from_template("Timestamp: {timestamp}")
prompt_with_partial = prompt.partial(timestamp=lambda: str(time.time()))
# At format time, timestamp is evaluated to current time
result = prompt_with_partial.format()  # "Timestamp: 1699123456.789"
```

The `_merge_partial_and_user_variables()` method:
- Iterates over `partial_variables`.
- If value is callable, invokes it; otherwise uses value as-is.
- Merges with user-supplied kwargs (user values override partials).

## Serialization and Metadata

- **`lc_attributes: dict[str, Any]`** — Attributes included in serialization (e.g., `template_format` for `PromptTemplate`).
- **`get_lc_namespace()`** — Returns serialization namespace (e.g., `["langchain", "prompts", "prompt"]` for `PromptTemplate`).
- **`is_lc_serializable()`** — Returns `True` if serializable; some templates (e.g., `FewShotPromptTemplate` with custom selectors) return `False`.
- **`_serialized`** — Cached property; serialized representation for tracing.

## PromptValue Routing

All prompts return `PromptValue` (abstract base) from `format_prompt()` or `invoke()`:

- **`StringPromptValue`** — Single string; has `to_string()` and `to_messages()` (wraps string as `HumanMessage`).
- **`ChatPromptValue`** — Message sequence; has `to_string()` (concatenates messages) and `to_messages()` (returns as-is).
- **`ChatPromptValueConcrete`** — Explicit variant for external schemas.

This enables downstream flexibility: models, chains, and output parsers can accept either representation.

## Input Schema Generation

- **`get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]`** — Returns Pydantic model with fields for all `input_variables` (required) and `optional_variables` (with defaults).
- **`mustache_schema(template: str) -> type[BaseModel]`** — Special handling for Mustache templates (nested variable schemas).

## Error Handling and Validation

### Pre-instantiation Validation

`PromptTemplate.pre_init_validation()` (model validator):
- Auto-detects `input_variables` from template if not provided.
- If `validate_template=True`, calls `check_valid_template()` to verify syntax.
- Raises `ValueError` if input variables mismatch (for f-string/jinja2) or if Mustache validation is requested.

### Format-Time Validation

`_validate_input()`:
- Checks input is dict (or single-variable templates coerce non-dict inputs).
- Ensures all required `input_variables` are present.
- Provides helpful error messages with escaping hints for missing variables.

### Variable Name Restrictions

`validate_variable_names()` (model validator):
- Forbids variable name `"stop"` (reserved internally).
- Enforces non-overlapping `input_variables` and `partial_variables`.

## Testing and Observability

Prompts integrate with LangSmith tracing:

- **`metadata: dict | None`** — Custom metadata (tags, version, context).
- **`tags: list[str] | None`** — Trace tags.
- At invocation, metadata and tags are merged into `RunnableConfig` for downstream observability.
- Serialized representation is cached for efficient tracing.

## Common Patterns

### Building a Multi-Message Prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{user_question}"),
])
# Composition:
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
]) + ("human", "{user_question}")
```

### Few-Shot with Dynamic Selection

```python
few_shot = FewShotChatMessagePromptTemplate(
    example_selector=semantic_selector,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ]),
)
final = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    few_shot,
    ("human", "{question}"),
])
```

### Partial Application

```python
base_prompt = PromptTemplate.from_template("Name: {name}\nAge: {age}")
name_filled = base_prompt.partial(name="Alice")
# Now only requires 'age'
result = name_filled.format(age=30)  # "Name: Alice\nAge: 30"
```

### Composition with Operators

```python
system = ChatPromptTemplate.from_template("You are a helpful assistant.")
user_input = ChatPromptTemplate.from_template("{input}")
combined = system + user_input
# Equivalent to: ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant."),
#     ("human", "{input}"),
# ])
```
