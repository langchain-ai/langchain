---
type: "Reference"
title: "Prompts: Templating and Composability"
description: "Prompt templates define message sequences and variable substitution patterns for language models. Covers template syntax, PromptValue routing, runnable integration, and few-shot learning with dynamic example selection."
tags: [prompt, template, few-shot, variable-substitution, runnable, composition, chat-model]
verified:
  - by: openwiki/0.5.0
    at: 2026-10-25T08:23:03.589Z
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
generated: { by: "openwiki/0.5.0", at: "2026-10-25T08:23:03.589Z" }
---

# Prompts: Templating and Composability

Prompts are the **primary input interface** to language models in LangChain. They accept user-provided variables, format them into structured message sequences, and return `PromptValue` objects that can be routed to chat models, string-based APIs, or output parsers. The system provides three core abstractions: `PromptTemplate` for strings, `ChatPromptTemplate` for message sequences, and `FewShotPromptTemplate` for including examples that teach models by demonstration.

## Core Architecture: BasePromptTemplate

All prompts inherit from **`BasePromptTemplate`** (in `langchain_core.prompts.base`), which extends `RunnableSerializable[dict[str, Any], PromptValue]`. This base class manages variable validation, immutable composition, and runnable integration.

### Key Responsibilities

1. **Input Validation**: Tracks `input_variables` (required) and `optional_variables` (from `MessagesPlaceholder`), validates that provided inputs match expectations, and enforces that variable names do not conflict with reserved words like `"stop"`.

2. **Variable Management**: Maintains both `input_variables` (user-provided at format time) and `partial_variables` (pre-filled via `partial()` method). Partial variables support lazy evaluation—if callable, they are invoked at format time.

3. **Immutable Composition**: Methods like `partial()` and `+` return new instances without mutating the original, enabling safe reuse and caching in pipelines.

4. **PromptValue Output**: All prompts return `PromptValue` objects (via `format_prompt()` and `invoke()`) that abstract the formatted output as either string or message sequences, enabling flexible downstream consumption.

5. **Runnable Integration**: Implements `invoke()`, `ainvoke()`, `stream()`, and `batch()` via the runnable protocol, making prompts composable with models and output parsers in chains.

### Common Fields

- **`input_variables: list[str]`** — Required variable names that must be provided at format time (e.g., `["question", "context"]`).
- **`optional_variables: list[str]`** — Variable names that are optional (e.g., from `MessagesPlaceholder` with `optional=True`).
- **`partial_variables: dict[str, Any]`** — Pre-filled variables. Values can be strings or callables (evaluated at format time).
- **`metadata: dict | None`** — Tracing metadata (tags, version, etc.) propagated to LangSmith and observability backends.
- **`tags: list[str] | None`** — Tags for tracing and filtering.

## PromptTemplate: String Prompts

**`PromptTemplate`** (in `langchain_core.prompts.prompt`) wraps a single string template with variable placeholders. It supports three template engines:

### Template Formats

**F-string** (default):
```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Say {text}")
prompt.format(text="hello")  # "Say hello"

# Escape literal braces with double braces
prompt = PromptTemplate.from_template("Literal {{braces}}")  # template contains "{braces}"
```

- **Syntax**: `{variable_name}`, with `{{` and `}}` for literal braces.
- **Validation**: Python `string.Formatter` validates syntax and extracts variable names.
- **Security**: Safe for user-provided values (only variable substitution, no execution).

**Mustache**:
```python
prompt = PromptTemplate.from_template(
    "Hello {{name}}, you have {{count}} items.",
    template_format="mustache"
)

# Nested variables: {{person.name}}
# Sections: {{#items}}...{{/items}}
# Conditionals: {{#show_details}}...{{/show_details}}
```

- **Syntax**: Mustache syntax with `{{variable}}`, sections, and filters.
- **Validation**: Lightweight extraction of top-level variables (nested keys return only the top-level key).

**Jinja2**:
```python
prompt = PromptTemplate(
    template="{% if detailed %}Full:{% endif %} {content}",
    template_format="jinja2",
    input_variables=["content"],
    partial_variables={"detailed": True}
)
```

- **Syntax**: Full Jinja2 syntax (conditionals, loops, filters).
- **Validation**: Parses AST to extract undeclared variables.
- **Security**: Uses `SandboxedEnvironment` by default to block dunder attribute access. **Do not use with untrusted templates**; sandboxing is best-effort.

### Construction

```python
# Factory method (recommended)
prompt = PromptTemplate.from_template("Q: {question}\nA:")

# Explicit construction
prompt = PromptTemplate(
    template="User: {name}, Topic: {topic}",
    input_variables=["topic"],
    partial_variables={"name": "Bob"},
    template_format="f-string"
)
```

### Formatting and Invocation

```python
# format() returns a string
output = prompt.format(topic="AI")  # Only 'topic' required; 'name' is pre-filled

# invoke() returns a PromptValue (Runnable interface)
from langchain_core.prompt_values import StringPromptValue
result = prompt.invoke({"topic": "AI"})
assert isinstance(result, StringPromptValue)
assert result.to_string() == "User: Bob, Topic: AI"
```

## ChatPromptTemplate: Structured Message Sequences

**`ChatPromptTemplate`** (in `langchain_core.prompts.chat`) sequences message templates into a conversation structure. Each message has a **role** (system, human, ai, tool) and **template content**, enabling composition of multi-turn conversations aligned with chat model APIs.

### Construction

```python
from langchain_core.prompts import ChatPromptTemplate

# Tuple shorthand (role, template)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}."),
    ("human", "Question: {question}"),
    ("ai", "Answer: {answer}"),
    ("human", "Follow-up: {followup}"),
])

# Explicit message objects (less common)
from langchain_core.prompts import HumanMessagePromptTemplate
template = ChatPromptTemplate([
    ("system", "You are helpful."),
    HumanMessagePromptTemplate.from_template("{input}"),
])
```

### Supported Message Types

- **`"system"`** → `SystemMessagePromptTemplate`: Priming instructions (always first).
- **`"human"` / `"user"`** → `HumanMessagePromptTemplate`: User/human input.
- **`"ai"` / `"assistant"`** → `AIMessagePromptTemplate`: Model output (for few-shot examples).
- **`"tool"` / `"function"`** → `ToolMessagePromptTemplate`: Tool execution results.
- **`MessagesPlaceholder`**: Dynamic message lists (e.g., conversation history).

### Variable Inference and Auto-Wrapping

Variables are automatically inferred from all messages:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{input}"),
])

print(template.input_variables)  # ['input', 'role']

# If only one input variable, invoke() auto-wraps non-dict arguments
result = template.invoke("Hello")  # Auto-wrapped as {"input": "Hello"}
```

### MessagesPlaceholder for Conversation History

`MessagesPlaceholder` injects pre-formatted message lists at specific points:

```python
from langchain_core.prompts import MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{question}"),
])

result = template.invoke({
    "chat_history": [
        ("human", "What's 2+2?"),
        ("ai", "4"),
    ],
    "question": "And 3+3?",
})
# Messages: [System, Human(2+2?), AI(4), Human(3+3?)]
```

**Key parameters**:
- **`optional=True`**: Placeholder can be omitted from inputs; defaults to `[]`.
- **`optional=False`** (default): Placeholder must be provided, even if empty.
- **`n_messages`**: Limit to the last N messages (useful for token budgets).

```python
# Keep only last 2 messages
placeholder = MessagesPlaceholder("history", n_messages=2)
```

### Formatting and Output

```python
# format_messages() returns list[BaseMessage]
messages = template.format_messages(role="assistant", input="hello")

# invoke() returns ChatPromptValue
from langchain_core.prompt_values import ChatPromptValue
result = template.invoke({"role": "assistant", "input": "hello"})
assert isinstance(result, ChatPromptValue)

# Conversion to string (concatenates messages)
string_repr = result.to_string()
```

## PromptValue: Flexible Output Representation

All prompts return **`PromptValue`** objects, an abstract base that encodes formatted output in both string and message forms, enabling **routing** to different downstream consumers.

### PromptValue Types

**`StringPromptValue`**: Wraps a formatted string.
```python
from langchain_core.prompt_values import StringPromptValue

pv = StringPromptValue(text="Hello world")
pv.to_string()    # "Hello world"
pv.to_messages()  # [HumanMessage(content="Hello world")]
```

**`ChatPromptValue`**: Wraps a sequence of `BaseMessage` objects.
```python
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.messages import HumanMessage, AIMessage

pv = ChatPromptValue(messages=[
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!"),
])
pv.to_messages()  # [HumanMessage(...), AIMessage(...)]
pv.to_string()    # "Human: Hello\nAI: Hi there!"
```

### Routing Pattern

PromptValue enables **format-agnostic consumption**:
```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Both return PromptValue (different subclasses)
string_prompt = PromptTemplate.from_template("Say {text}")
chat_prompt = ChatPromptTemplate.from_messages([("human", "{text}")])

pv_string = string_prompt.invoke({"text": "hello"})
pv_chat = chat_prompt.invoke({"text": "hello"})

# Both can be converted to messages or strings
pv_string.to_messages()  # Works even though it started as string
pv_chat.to_string()      # Works even though it started as messages
```

## Variable Management and Partial Application

### Variable Validation

Required variables must be provided:
```python
prompt = PromptTemplate.from_template("User: {name}, Topic: {topic}")

try:
    prompt.format(topic="AI")  # Missing 'name'
except KeyError:
    print("Missing 'name' variable")
```

### Partial Variables and Lazy Evaluation

`partial()` method pre-fills variables, reducing required inputs. It returns a **new instance** without mutating the original:

```python
original = PromptTemplate.from_template("Say {text}")
partial = original.partial(text="hello")

print(original.input_variables)  # ['text']
print(partial.input_variables)   # []

# Original unchanged
output_original = original.format(text="world")  # "Say world"
output_partial = partial.format()                # "Say hello"
```

Partial values can be **callables** (evaluated at format time):
```python
import time

prompt = PromptTemplate(
    template="Time: {timestamp}",
    partial_variables={"timestamp": lambda: str(time.time())}
)

# timestamp is fresh each call
prompt.format()  # "Time: 1700000000.123"
prompt.format()  # "Time: 1700000001.456"
```

## Prompt Composition

Prompts compose via the `+` operator:

```python
system = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
])

conversation = ChatPromptTemplate.from_messages([
    ("human", "{user_input}"),
])

combined = system + conversation
result = combined.invoke({
    "role": "poet",
    "user_input": "Write a haiku",
})
```

**Composition rules**:
- **ChatPromptTemplate + ChatPromptTemplate**: Messages concatenate; variables merge.
- **PromptTemplate + PromptTemplate**: Templates concatenate; variables merge. Both must have the same `template_format`.
- **Partial variables merge**: Conflicting keys raise an error.
- **New instance created**: Original templates are unchanged.

## Few-Shot Learning: Examples in Prompts

Few-shot templates demonstrate desired behavior by including input-output examples. They support both **static example lists** and **dynamic example selection** via selectors.

### FewShotPromptTemplate (String-Based)

For string prompts with examples formatted as text:

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Provide antonyms:\n",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

output = prompt.format(input="big")
# Output:
# Provide antonyms:
#
# Input: happy
# Output: sad
#
# Input: tall
# Output: short
#
# Input: big
# Output:
```

**Key fields**:
- **`examples`** or **`example_selector`**: Source of examples (mutually exclusive).
- **`example_prompt`**: Template for formatting each example.
- **`prefix`**: Text before examples.
- **`suffix`**: Text after examples (contains the actual task).
- **`example_separator`**: Delimiter joining prefix, examples, and suffix (default: `"\n\n"`).

### FewShotChatMessagePromptTemplate (Chat-Based)

For chat prompts with example messages:

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
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

template = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor."),
    few_shot,
    ("human", "What is {input}?"),
])

result = template.invoke({"input": "4+4"})
# Messages: [System, Human(2+2?), AI(4), Human(2+3?), AI(5), Human(4+4?)]
```

## Example Selectors: Dynamic Example Selection

Instead of fixed example lists, **example selectors** dynamically choose relevant examples at format time, optimizing prompt length and relevance.

### BaseExampleSelector Interface

All selectors implement:

```python
class BaseExampleSelector:
    def add_example(self, example: dict) -> None:
        """Add an example to the selector."""
    
    def select_examples(self, input_variables: dict) -> list[dict]:
        """Select relevant examples based on inputs."""
    
    async def aadd_example(self, example: dict) -> None:
        """Async add."""
    
    async def aselect_examples(self, input_variables: dict) -> list[dict]:
        """Async select."""
```

### SemanticSimilarityExampleSelector

Selects examples by vector similarity (requires embeddings and a vector store):

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "sunny", "output": "gloomy"},
]

# Create vector store
to_vectorize = [" ".join(e.values()) for e in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Return 2 most similar
)

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# Selector retrieves 2 most similar examples to "bright"
output = prompt.format(input="bright")
```

**Parameters**:
- **`vectorstore`**: Vector store containing embedded examples.
- **`k`**: Number of examples to return (default: 4).
- **`input_keys`**: Keys to use for similarity search (e.g., only the "input" field).
- **`example_keys`**: Keys to include in returned examples.

### LengthBasedExampleSelector

Greedily selects examples up to a token/word limit:

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50,  # Approximate word limit
    get_text_length=lambda x: len(x.split()),  # Custom length function
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# Selector returns examples that fit within max_length
output = prompt.format(input="fast")
```

**Parameters**:
- **`examples`**: All available examples.
- **`max_length`**: Maximum prompt length (tokens or words).
- **`get_text_length`**: Function measuring length (default: word count).

**Behavior**: Iterates examples in order; stops adding when the next would exceed `max_length`. Fast and predictable, but greedy (not optimal).

## Runnable Interface and Chaining

All prompts are `Runnable` objects (from `langchain_core.runnables`), enabling composition with models and output parsers:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "You are a poet."),
    ("human", "Write a poem about {topic}"),
])

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = template | model | parser

result = chain.invoke({"topic": "the internet"})
print(result)  # Poem as string
```

### Key Runnable Methods

- **`invoke(dict) → PromptValue`**: Synchronous formatting.
- **`ainvoke(dict) → PromptValue`**: Asynchronous formatting.
- **`batch(list[dict]) → list[PromptValue]`**: Batch formatting.
- **`stream(dict)`**: Streaming mode (rarely used for prompts; more common downstream).

### Metadata and Configuration

Prompts support tracing via configuration:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Attach metadata
configured = template.with_config({
    "run_name": "my_prompt",
    "metadata": {"version": "1.0", "author": "alice"},
    "tags": ["production"],
})

result = configured.invoke({"input": "hello"})
# Metadata propagates to LangSmith and observability backends
```

## Serialization and Deserialization

Prompts support serialization via LangChain's `langchain_core.load` API:

```python
from langchain_core.load import dumps, dumpd, loads
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Serialize to dict
prompt_dict = dumpd(prompt)

# Serialize to JSON string
prompt_json = dumps(prompt)

# Deserialize from dict
restored_prompt = loads(prompt_dict)

# Deserialize from JSON
restored_prompt = loads(prompt_json)
```

**Important**: The older `save()` / `load_prompt_from_config()` API is **deprecated**. Use `dumpd()` / `dumps()` and `loads()` for new code.

Serialized prompts preserve:
- Template content and format.
- Input variables and partial variables.
- Message structure and roles.
- Example configurations and selectors (when serializable).

Selectors using external resources (e.g., `SemanticSimilarityExampleSelector` with a vector store) are **not serializable** via `dumpd()`.

## Security Considerations

### Template Injection Protection

Use variable substitution instead of string concatenation:

```python
# UNSAFE: Vulnerable to prompt injection
user_input = input("Enter text: ")
template = f"User said: {user_input}"

# SAFE: Use variable substitution
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("User said: {user_input}")
output = prompt.format(user_input=user_input)
```

Variables are substituted at format time, not at template definition time, preventing malicious users from injecting template directives (e.g., `{__import__('os')}`).

### Jinja2 Sandboxing

When using Jinja2 templates, LangChain applies `SandboxedEnvironment` by default:
- **Blocks dunder attributes**: `__class__`, `__globals__`, `__import__` are inaccessible.
- **Allows method calls**: Regular object methods (e.g., `obj.method()`) and attribute access still work.
- **Best-effort security**: Sandboxing is not foolproof; prefer f-string or mustache for untrusted inputs.

```python
# Safer: f-string template from untrusted source
prompt = PromptTemplate(
    template="Hello {name}",
    input_variables=["name"],
)
prompt.format(name=untrusted_input)

# Avoid: Jinja2 from untrusted source
# Even with sandboxing, there are potential bypasses
```

**Recommendation**: For untrusted templates, use `f-string` or `mustache` formats.

## Lifecycle and Immutability

Prompt templates are **functionally immutable**: calling `format()` or `invoke()` never mutates the template. Methods like `partial()` return new instances:

```python
original = PromptTemplate.from_template("Say {text}")
partial = original.partial(text="hello")

print(original.input_variables)  # ['text']
print(partial.input_variables)   # []

# Both are independent
output1 = original.format(text="world")  # "Say world"
output2 = partial.format()                # "Say hello"
```

This immutability enables:
- **Safe reuse**: Prompts can be reused in multiple chains without side effects.
- **Caching**: Immutable objects are safe to cache.
- **Composition**: Combining prompts via `+` does not modify originals.

## Integration with Chat Models and Agents

Prompts feed directly into chat models and agents:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

model = ChatOpenAI(model="gpt-4o")

# Chain invocation
response = (template | model).invoke({"input": "What is 2+2?"})
```

Prompts are also core to the **agent factory** pattern:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, calculator_tool],
    system_prompt="You are a helpful assistant with access to tools.",
)
```

The agent internally manages prompt composition with tool bindings and state machine logic.

## Validation and Error Handling

Prompts validate inputs at format/invoke time:

```python
prompt = PromptTemplate.from_template("Q: {question}\nA:")

# Missing variable
try:
    prompt.format()
except KeyError as e:
    print(f"Missing: {e}")

# Extra variable (ignored)
output = prompt.format(question="What?", extra="ignored")
print(output)  # "Q: What?\nA:"
```

Variable names are extracted automatically from templates:
- **F-string**: Python `string.Formatter` extracts from `{name}` syntax.
- **Mustache**: Lightweight parser extracts top-level keys.
- **Jinja2**: AST parser extracts undeclared variables.

## Common Patterns

### Multi-Turn Conversation with History

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# At runtime, inject conversation history
history = [
    ("human", "What's 2+2?"),
    ("ai", "4"),
]

result = template.invoke({
    "chat_history": history,
    "question": "And 3+3?",
})
```

### Few-Shot with Dynamic Selection

```python
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector

selector = LengthBasedExampleSelector(
    examples=[...],
    example_prompt=example_template,
    max_length=100,
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_template,
    suffix="...",
)

# Selector dynamically picks examples at format time
output = prompt.format(input=user_input)
```

### Parameterized System Prompts

```python
system = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. You speak in {style}."),
])

user_message = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
])

template = system + user_message

result = template.invoke({
    "role": "pirate",
    "style": "pirate speak",
    "input": "Hello!",
})
```
