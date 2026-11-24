---
type: "Reference"
title: "Prompt Template APIs"
description: "API reference for LangChain prompt template classes, including BasePromptTemplate, PromptTemplate, ChatPromptTemplate, and FewShotPromptTemplate with their signatures and methods."
tags: [prompt, template, reference, api, base-class, string-template, chat-template, few-shot]
verified:
  - by: openwiki/0.5.0
    at: 2026-11-13T08:16:35.902Z
sources:
  - id: openwiki-source-269af2bb49a21bd6c548b3f0
    resource: repo://libs/core/langchain_core/prompt_values.py
  - id: openwiki-source-03d7415879ed05a392edd62d
    resource: repo://libs/core/langchain_core/prompts/base.py
  - id: openwiki-source-15fdd645c1ee76ae559799c1
    resource: repo://libs/core/langchain_core/prompts/chat.py
  - id: openwiki-source-bc32774051e0e8a931a6fecd
    resource: repo://libs/core/langchain_core/prompts/few_shot.py
  - id: openwiki-source-5549894302ea4dfd5b8f4278
    resource: repo://libs/core/langchain_core/prompts/prompt.py
  - id: openwiki-source-cf81d0ba0a387a7cd9b5dfb8
    resource: repo://libs/core/langchain_core/prompts/string.py
generated: { by: "openwiki/0.5.0", at: "2026-11-13T08:16:35.902Z" }
---

# Prompt Template APIs

This page documents the core prompt template class signatures and methods in LangChain. For conceptual overview and usage patterns, see [Prompts: Templates & Composition](/openwiki/prompts.md).

## BasePromptTemplate

**Module**: `langchain_core.prompts.base`

**Base class**: `RunnableSerializable[dict[str, Any], PromptValue]`

The abstract base class for all prompt templates. All prompt template implementations inherit from `BasePromptTemplate` and adhere to the runnable protocol.

### Fields

- **`input_variables: list[str]`** — Required variable names that must be provided at format time. Cannot include `"stop"` (reserved).
- **`optional_variables: list[str] = []`** — Optional variable names (e.g., from `MessagesPlaceholder` with `optional=True`). Auto-inferred from template structure.
- **`input_types: dict[str, Any] = {}`** — Type annotations for input variables. If not provided, all variables are assumed to be `str`.
- **`partial_variables: Mapping[str, Any] = {}`** — Pre-filled variables that reduce required inputs. Values can be strings or callables (invoked at format time).
- **`output_parser: BaseOutputParser | None = None`** — Optional parser to apply to the formatted prompt output.
- **`metadata: dict[str, Any] | None = None`** — Tracing metadata (tags, version, etc.) propagated to LangSmith and observability backends.
- **`tags: list[str] | None = None`** — Tags for tracing and filtering.

### Core Methods

#### `format(**kwargs: Any) -> FormatOutputType`
Format the template with provided inputs and return the formatted output (string for `PromptTemplate`, list of `BaseMessage` for `ChatPromptTemplate`).

```python
prompt = PromptTemplate.from_template("Say {text}")
result = prompt.format(text="hello")  # "Say hello"
```

#### `format_prompt(**kwargs: Any) -> PromptValue`
Format the template and return a `PromptValue` object (abstract representation of formatted output as either string or messages).

```python
result = prompt.format_prompt(text="hello")
assert isinstance(result, StringPromptValue)
assert result.to_string() == "Say hello"
```

#### `invoke(input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> PromptValue`
Runnable interface for synchronous invocation. Validates input, applies partial variables, calls `_format_prompt_with_error_handling`, and propagates metadata and tags to the runnable config.

```python
result = prompt.invoke({"text": "hello"})
assert isinstance(result, StringPromptValue)
```

#### `ainvoke(input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> Awaitable[PromptValue]`
Async version of `invoke()`. Default implementation delegates to `aformat_prompt()` with async error handling.

#### `partial(**kwargs: str | Callable[[], str]) -> BasePromptTemplate`
Return a new prompt template instance with specified variables pre-filled. Original template is not mutated. Callable values are invoked at format time.

```python
prompt = PromptTemplate.from_template("User: {name}, Topic: {topic}")
partial_prompt = prompt.partial(name="Alice")
result = partial_prompt.format(topic="AI")  # name is pre-filled
```

#### `aformat(**kwargs: Any) -> Awaitable[FormatOutputType]`
Async format method. Default implementation calls synchronous `format()`.

#### `asdict(**kwargs: Any) -> dict[str, Any]`
Return dictionary representation of the prompt template (Pydantic model dump).

### Validation

- **Variable Name Validation**: Constructor validates that `"stop"` is not used in `input_variables` or `partial_variables` (reserved for internal use).
- **Variable Overlap**: Input variables and partial variables must not overlap.
- **Input Validation**: `invoke()` and `ainvoke()` validate that all required `input_variables` are provided.

---

## PromptTemplate (StringPromptTemplate)

**Module**: `langchain_core.prompts.prompt`

**Base class**: `StringPromptTemplate` (extends `BasePromptTemplate`)

A single-string prompt template with variable substitution. Supports three template engines with different syntax and security profiles.

### Fields

- **`template: str`** — The template string (required).
- **`template_format: Literal["f-string", "mustache", "jinja2"] = "f-string"`** — Template engine to use.
- **`input_variables: list[str]`** — Auto-extracted variable names from template (required for construction).
- **`validate_template: bool = False`** — Whether to validate template syntax at initialization.
- **Inherited from `BasePromptTemplate`**: `optional_variables`, `partial_variables`, `output_parser`, `metadata`, `tags`, `input_types`.

### Template Formats

#### F-string (Default)

- **Syntax**: `{variable_name}` with `{{` and `}}` for literal braces.
- **Engine**: Python's `string.Formatter`.
- **Validation**: `get_template_variables()` extracts variable names.
- **Security**: Safe—only variable substitution, no expression evaluation.

```python
prompt = PromptTemplate.from_template("Say {text}")
prompt.format(text="hello")  # "Say hello"

# Escape literal braces
prompt = PromptTemplate.from_template("Literal {{braces}}")
prompt.format()  # "Literal {braces}"
```

#### Mustache

- **Syntax**: `{{variable}}`, `{{#section}}...{{/section}}`, `{{^inverted}}...{{/inverted}}`.
- **Validation**: Lightweight extraction (nested keys return only top-level key).
- **Security**: Safer for user-controlled templates.

```python
prompt = PromptTemplate.from_template(
    "Hello {{name}}, you have {{count}} items.",
    template_format="mustache"
)
prompt.format(name="Alice", count=5)
```

#### Jinja2

- **Syntax**: Full Jinja2 with conditionals, loops, filters, and expressions.
- **Validation**: AST parsing to extract undeclared variables.
- **Security**: Uses `SandboxedEnvironment` by default to block dunder attribute access. **Treat as best-effort; do not use with untrusted templates**.

```python
prompt = PromptTemplate(
    template="{% if detailed %}Full:{% endif %} {content}",
    template_format="jinja2",
    input_variables=["content"],
    partial_variables={"detailed": True}
)
prompt.format(content="summary")
```

### Construction

#### Direct Constructor

```python
prompt = PromptTemplate(
    template="User: {name}, Topic: {topic}",
    input_variables=["topic"],
    partial_variables={"name": "Bob"},
    template_format="f-string"
)
```

#### Factory: `from_template(template: str, *, template_format: PromptTemplateFormat = "f-string", partial_variables: dict[str, Any] | None = None, **kwargs) -> PromptTemplate`

Recommended approach. Auto-extracts `input_variables` from template.

```python
prompt = PromptTemplate.from_template("Say {text}")
```

#### Factory: `from_file(template_file: str | Path, encoding: str | None = None, **kwargs) -> PromptTemplate`

Load template from file.

```python
prompt = PromptTemplate.from_file("path/to/template.txt")
```

#### Factory: `from_examples(examples: list[str], suffix: str, input_variables: list[str], example_separator: str = "\n\n", prefix: str = "", **kwargs) -> PromptTemplate`

Build a template from a list of examples with prefix and suffix.

```python
prompt = PromptTemplate.from_examples(
    examples=["example1", "example2"],
    prefix="Here are examples:",
    suffix="Now solve: {problem}",
    input_variables=["problem"]
)
```

### Methods

#### `format(**kwargs: Any) -> str`
Format the template and return a string.

```python
output = prompt.format(topic="AI")
```

#### `aformat(**kwargs: Any) -> Awaitable[str]`
Async format.

#### `__add__(other: PromptTemplate | str) -> PromptTemplate`
Combine two templates (both must use same `template_format`). Merges input and partial variables; raises error on conflicts.

```python
prompt1 = PromptTemplate.from_template("Question: {q}")
prompt2 = PromptTemplate.from_template(" Answer: {a}")
combined = prompt1 + prompt2  # "Question: {q} Answer: {a}"
```

#### `get_input_schema(config: RunnableConfig | None = None) -> type[BaseModel]`
Get Pydantic model schema for input validation (respects `input_types`).

### Class Methods

#### `get_lc_namespace() -> list[str]`
Returns `["langchain", "prompts", "prompt"]`.

#### `is_lc_serializable() -> bool`
Returns `True`.

---

## ChatPromptTemplate

**Module**: `langchain_core.prompts.chat`

**Base class**: `BaseChatPromptTemplate` (extends `BasePromptTemplate`)

Sequences message prompt templates into a conversation structure. Each message has a role (system, human, ai, tool, function) and content. Returns `ChatPromptValue` with a list of `BaseMessage` objects.

### Fields

- **`messages: list[MessageLike]`** — List of message templates. Can be `BaseMessage`, `BaseMessagePromptTemplate`, `BaseChatPromptTemplate`, tuples, or strings.
- **`validate_template: bool = False`** — Whether to validate template structure.
- **Inherited from `BasePromptTemplate`**: `input_variables` (auto-inferred), `optional_variables`, `partial_variables`, `metadata`, `tags`, `input_types`.

### Message Types (Shorthand Tuple Syntax)

In `from_messages()` or constructor, messages can be specified as tuples:

- `("system", template)` → `SystemMessagePromptTemplate`
- `("human", template)` or `("user", template)` → `HumanMessagePromptTemplate`
- `("ai", template)` or `("assistant", template)` → `AIMessagePromptTemplate`
- `("tool", template)` → `ToolMessagePromptTemplate`
- `("function", template)` → `FunctionMessagePromptTemplate`
- `("placeholder", "{variable_name}")` → `MessagesPlaceholder(variable_name="variable_name", optional=True)`
- Plain string `"{text}"` → Shorthand for `("human", "{text}")`

### Construction

#### Direct Constructor

```python
template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
    ("ai", "Hi there!"),
    ("human", "{user_input}"),
])
```

#### Factory: `from_messages(messages: Sequence[MessageLikeRepresentation], template_format: PromptTemplateFormat = "f-string") -> ChatPromptTemplate`

Create from message representations. Auto-infers `input_variables`.

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Hello, {name}"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])
```

#### Factory: `from_template(template: str, **kwargs) -> ChatPromptTemplate`

Create from a single template string (wrapped as a human message).

```python
template = ChatPromptTemplate.from_template("Tell me about {topic}")
```

#### Factory: `from_template_file(template_file: str | Path, input_variables: list[str], **kwargs) -> ChatPromptTemplate`

Load from template file.

```python
template = ChatPromptTemplate.from_template_file("path/to/template.json", input_variables=["topic"])
```

### Methods

#### `format_messages(**kwargs: Any) -> list[BaseMessage]`
Format the chat template into a list of `BaseMessage` objects.

```python
messages = template.format_messages(name="Alice", user_input="How are you?")
```

#### `aformat_messages(**kwargs: Any) -> Awaitable[list[BaseMessage]]`
Async version of `format_messages()`.

#### `format(**kwargs: Any) -> str`
Format into a single string (concatenates all messages). Useful for debugging or string-based APIs.

```python
text = template.format(name="Alice", user_input="How are you?")
```

#### `aformat(**kwargs: Any) -> Awaitable[str]`
Async format to string.

#### `format_prompt(**kwargs: Any) -> ChatPromptValue`
Format and return `ChatPromptValue` containing the list of `BaseMessage` objects.

```python
prompt_value = template.format_prompt(name="Alice", user_input="How are you?")
assert isinstance(prompt_value, ChatPromptValue)
messages = prompt_value.to_messages()
```

#### `aformat_prompt(**kwargs: Any) -> Awaitable[ChatPromptValue]`
Async format to `ChatPromptValue`.

#### `invoke(input: dict[str, Any] | Any, config: RunnableConfig | None = None) -> ChatPromptValue`
Runnable interface. If template has exactly one input variable and input is not a dict, it is auto-wrapped.

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{text}"),
])
# Auto-wrapping: invoke(input) → invoke({"text": input})
result = template.invoke("hello")  # Same as invoke({"text": "hello"})
```

#### `partial(**kwargs: Any) -> ChatPromptTemplate`
Return new template with specified variables pre-filled.

```python
partial_template = template.partial(name="Alice")
result = partial_template.format_messages(user_input="Hello")
```

#### `append(message: MessageLikeRepresentation) -> None`
Append a single message to the template (mutates in-place).

```python
template.append(("human", "Goodbye!"))
```

#### `extend(messages: Sequence[MessageLikeRepresentation]) -> None`
Extend template with multiple messages (mutates in-place).

```python
template.extend([
    ("tool", "{tool_output}"),
    ("human", "Thank you!"),
])
```

#### `__add__(other: ChatPromptTemplate | BaseMessagePromptTemplate | BaseMessage | BaseChatPromptTemplate | list | tuple | str) -> ChatPromptTemplate`
Combine templates. Merges partial variables; auto-wraps strings as human messages.

```python
template1 = ChatPromptTemplate.from_messages([("system", "You are helpful.")])
template2 = ChatPromptTemplate.from_messages([("human", "{input}")])
combined = template1 + template2
```

#### `__getitem__(index: int | slice) -> MessageLike | ChatPromptTemplate`
Index into messages. Returns single `MessageLike` for int index; returns new `ChatPromptTemplate` for slice.

```python
msg = template[0]  # First message template
sub_template = template[1:3]  # Messages 1–2 as new template
```

### Class Methods

#### `get_lc_namespace() -> list[str]`
Returns `["langchain", "prompts", "chat"]`.

#### `is_lc_serializable() -> bool`
Returns `True`.

### Input Variable Auto-Inference

When constructed, `ChatPromptTemplate` automatically infers `input_variables` and `optional_variables`:

- Variables from templated message content are added to `input_variables`.
- Variables from optional `MessagesPlaceholder` instances are added to `optional_variables` and pre-filled with `[]` in `partial_variables`.
- If a variable appears in both input and partial, it is removed from input.

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history", optional=True),  # optional_variables
    ("human", "{question}"),  # input_variables
])
# input_variables == ["question"]
# optional_variables == ["history"]

template.format_messages(question="What?")  # history defaults to []
```

---

## MessagesPlaceholder

**Module**: `langchain_core.prompts.chat`

**Base class**: `BaseMessagePromptTemplate`

Placeholder for a dynamic list of messages passed at format time. Useful for inserting conversation history or multi-turn exchanges.

### Fields

- **`variable_name: str`** — Name of the variable containing the message list (required).
- **`optional: bool = False`** — If `True`, placeholder can be omitted at format time (returns empty list). If `False`, variable must be provided.
- **`n_messages: PositiveInt | None = None`** — Maximum number of messages to include (e.g., last 5 messages).

### Construction

```python
from langchain_core.prompts import MessagesPlaceholder

# Required variable
placeholder = MessagesPlaceholder("history")

# Optional variable (defaults to [])
placeholder = MessagesPlaceholder("history", optional=True)

# Limit to last 5 messages
placeholder = MessagesPlaceholder("history", n_messages=5)
```

### Methods

#### `format_messages(**kwargs: Any) -> list[BaseMessage]`
Return the message list from kwargs, optionally limiting to `n_messages` most recent.

```python
messages = placeholder.format_messages(
    history=[
        ("human", "Hello"),
        ("ai", "Hi!"),
        ("human", "How are you?"),
    ]
)
# Returns list of 3 BaseMessage objects
```

### Properties

#### `input_variables: list[str]`
Returns `[variable_name]` if required; empty list if optional.

---

## FewShotPromptTemplate

**Module**: `langchain_core.prompts.few_shot`

**Base class**: `StringPromptTemplate` (extends `BasePromptTemplate`)

String-based prompt template that includes fixed or dynamically selected examples before a suffix template.

### Fields

- **`examples: list[dict[str, Any]] | None`** — Static list of examples. Either `examples` or `example_selector` must be provided (mutually exclusive).
- **`example_selector: BaseExampleSelector | None`** — Selector that chooses examples dynamically based on input.
- **`example_prompt: PromptTemplate`** — Template used to format each individual example (required).
- **`suffix: str`** — Template string to append after examples (required).
- **`prefix: str = ""`** — Template string to prepend before examples.
- **`example_separator: str = "\n\n"`** — String separator between prefix, examples, and suffix.
- **`template_format: Literal["f-string", "jinja2"] = "f-string"`** — Format of prefix/suffix templates. **Note**: `"mustache"` is not supported.
- **`validate_template: bool = False`** — Whether to validate prefix and suffix templates.
- **Inherited from `BasePromptTemplate`**: `input_variables`, `optional_variables`, `partial_variables`, `metadata`, `tags`.

### Construction

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Q: {input}\nA: {output}"
)

few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Solve these math problems:\n",
    suffix="Q: {input}\nA:",
    input_variables=["input"],
)
```

### Methods

#### `format(**kwargs: Any) -> str`
Format the template:

1. Merge partial variables and user variables.
2. Get examples (static or selected).
3. Format each example using `example_prompt`.
4. Join prefix, example strings, and suffix with `example_separator`.
5. Format the combined template with input variables.

```python
output = few_shot.format(input="2+4")
# Prefix + formatted examples + formatted suffix
```

#### `aformat(**kwargs: Any) -> Awaitable[str]`
Async format. Awaits example selection if using `example_selector`.

### Class Methods

#### `is_lc_serializable() -> bool`
Returns `False` (few-shot templates with selectors are not serializable).

---

## FewShotChatMessagePromptTemplate

**Module**: `langchain_core.prompts.few_shot`

**Base class**: `BaseChatPromptTemplate` (extends `BasePromptTemplate`)

Chat-based few-shot template. Includes examples as message sequences, useful for multi-turn few-shot prompting.

### Fields

- **`examples: list[dict[str, Any]] | None`** — Static list of examples. Either `examples` or `example_selector` must be provided.
- **`example_selector: BaseExampleSelector | None`** — Selector for dynamic example selection.
- **`example_prompt: BaseMessagePromptTemplate | BaseChatPromptTemplate`** — Template to format each example as message sequence (required).
- **`input_variables: list[str] = []`** — Variables to pass to example selector. Auto-inferred if not provided.
- **Inherited from `BasePromptTemplate`**: `optional_variables`, `partial_variables`, `metadata`, `tags`.

### Construction

```python
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {input}?"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# Use in a larger template
final_template = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor."),
    few_shot,
    ("human", "{input}"),
])
```

### Methods

#### `format_messages(**kwargs: Any) -> list[BaseMessage]`
Format into a list of `BaseMessage` objects by interleaving examples and suffix messages.

```python
messages = few_shot.format_messages(input="2+4")
```

#### `aformat_messages(**kwargs: Any) -> Awaitable[list[BaseMessage]]`
Async format to messages.

### Class Methods

#### `is_lc_serializable() -> bool`
Returns `False`.

---

## PromptValue

**Module**: `langchain_core.prompt_values`

**Base class**: `Serializable` (abstract)

Abstract base type representing formatted prompt output in either string or message form. Enables flexible routing of prompts to different downstream consumers.

### Subclasses

#### `StringPromptValue`

Wraps a single formatted string.

**Fields**:
- `text: str` — The prompt text.
- `type: Literal["StringPromptValue"] = "StringPromptValue"`

**Methods**:
- `to_string() -> str` — Return the text.
- `to_messages() -> list[BaseMessage]` — Wrap text as a `HumanMessage`.

```python
spv = StringPromptValue(text="Say hello")
assert spv.to_string() == "Say hello"
assert len(spv.to_messages()) == 1
assert isinstance(spv.to_messages()[0], HumanMessage)
```

#### `ChatPromptValue`

Wraps a sequence of `BaseMessage` objects.

**Fields**:
- `messages: Sequence[BaseMessage]` — List of messages.

**Methods**:
- `to_string() -> str` — Concatenate messages as a buffer (newline-separated).
- `to_messages() -> list[BaseMessage]` — Return messages as-is.

```python
cpv = ChatPromptValue(messages=[
    SystemMessage(content="You are helpful."),
    HumanMessage(content="Hello!"),
])
assert len(cpv.to_messages()) == 2
text = cpv.to_string()  # Concatenated string
```

#### `ChatPromptValueConcrete`

Explicit variant of `ChatPromptValue` for external schemas and serialization.

---

## Message Prompt Template Classes

**Module**: `langchain_core.prompts.chat`

**Base class**: `BaseStringMessagePromptTemplate` (extends `BaseMessagePromptTemplate`)

Specialized message prompt templates for specific roles.

### Available Classes

- **`HumanMessagePromptTemplate`** — Produces `HumanMessage` (user input).
- **`AIMessagePromptTemplate`** — Produces `AIMessage` (assistant output).
- **`SystemMessagePromptTemplate`** — Produces `SystemMessage` (system instruction).
- **`ChatMessagePromptTemplate`** — Generic message with custom role.
- **`ToolMessagePromptTemplate`** — Produces `ToolMessage` (tool response).
- **`FunctionMessagePromptTemplate`** — Produces `FunctionMessage` (function call response).

### Fields

- **`prompt: StringPromptTemplate`** — Template for message content.
- **`additional_kwargs: dict[str, Any] = {}`** — Extra fields to attach to the message (e.g., `tool_call_id`).
- **Inherited from `BaseMessagePromptTemplate`**: `input_variables`, `optional_variables`, `partial_variables`.

### Construction

#### Direct Constructor

```python
from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate

template = PromptTemplate.from_template("Hello, {name}")
msg_template = HumanMessagePromptTemplate(prompt=template)
```

#### Factory: `from_template(template: str, template_format: PromptTemplateFormat = "f-string", partial_variables: dict[str, Any] | None = None, **kwargs) -> Self`

Recommended approach.

```python
msg_template = HumanMessagePromptTemplate.from_template("Hello, {name}")
message = msg_template.format(name="Alice")
assert isinstance(message, HumanMessage)
assert message.content == "Hello, Alice"
```

#### Factory: `from_template_file(template_file: str | Path, **kwargs) -> Self`

Load from file.

```python
msg_template = HumanMessagePromptTemplate.from_template_file("path/to/template.txt")
```

### Methods

#### `format(**kwargs: Any) -> BaseMessage`
Format the template and return a `BaseMessage` of the appropriate type.

#### `aformat(**kwargs: Any) -> Awaitable[BaseMessage]`
Async format.

#### `format_messages(**kwargs: Any) -> list[BaseMessage]`
Format and return list containing single message.

#### `aformat_messages(**kwargs: Any) -> Awaitable[list[BaseMessage]]`
Async format to message list.

### Properties

#### `input_variables: list[str]`
Returns `prompt.input_variables`.

---

## See Also

- [Prompts: Templates & Composition](/openwiki/prompts.md) — Conceptual overview and usage patterns
- [Runnable Protocol](/openwiki/concepts/runnables.md) — Invoke, stream, and batch semantics
<!-- openwiki: broken internal link [/openwiki/reference/example_selectors.md] file "/openwiki/reference/example_selectors.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Example Selectors](/openwiki/reference/example_selectors.md) — Dynamic example selection for few-shot learning
