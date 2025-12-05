# LangChain Agent tool_choice 深度解析

## 概述

`tool_choice` 是控制 LLM 模型如何选择调用工具的参数。它决定了模型是**必须**调用工具、**可以选择性**调用工具，还是**禁止**调用工具。

---

## 1. tool_choice 在 Agent 中的流转

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           tool_choice 在 Agent 中的完整流程                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         create_agent(...)           │
                    │                                     │
                    │  注意：create_agent 不接受          │
                    │  tool_choice 参数！                 │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 内部构建 ModelRequest
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         ModelRequest                │
                    │                                     │
                    │  tool_choice: Any | None = None     │
                    │  (初始值为 None)                    │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 中间件可以修改
                                      ▼
                    ┌─────────────────────────────────────┐
                    │    wrap_model_call 中间件           │
                    │                                     │
                    │  request.override(tool_choice=...)  │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 传递到 _get_bound_model
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      _get_bound_model(request)      │
                    │                                     │
                    │  根据 response_format 决定：        │
                    │                                     │
                    │  - ToolStrategy + 结构化输出工具:   │
                    │    tool_choice = "any"              │
                    │                                     │
                    │  - 普通情况:                        │
                    │    tool_choice = request.tool_choice│
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 调用 model.bind_tools
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     model.bind_tools(               │
                    │         tools,                      │
                    │         tool_choice=tool_choice,    │
                    │         **settings                  │
                    │     )                               │
                    │                                     │
                    │  不同模型对 tool_choice 的处理不同！│
                    │                                     │
                    └─────────────────────────────────────┘
```

---

## 2. tool_choice 的可选值

### 2.1 标准值（跨模型通用）

| 值 | 含义 | 行为 |
|---|------|------|
| `None` / `False` | 不指定 | 模型自行决定是否调用工具（通常等同于 `"auto"`） |
| `"auto"` | 自动选择 | 模型可以选择调用工具或不调用 |
| `"none"` | 禁止调用 | 模型不会调用任何工具，只返回文本 |
| `"any"` / `"required"` / `True` | 强制调用 | 模型必须调用至少一个工具 |
| `"tool_name"` | 指定工具 | 模型必须调用指定名称的工具 |
| `{"type": "function", "function": {"name": "xxx"}}` | 指定工具（dict格式） | 模型必须调用指定的工具 |

### 2.2 工作流程图

```text
                            ┌─────────────────┐
                            │   tool_choice   │
                            └────────┬────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │    "auto"     │        │    "none"     │        │ "any"/"required"|
    │    或 None    │        │               │        │    或 True     │
    └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
            │                        │                        │
            ▼                        ▼                        ▼
    ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
    │ 模型自动决定  │        │ 只返回文本    │        │ 强制调用工具  │
    │               │        │ 不调用工具    │        │               │
    │ 可能调用工具  │        │               │        │ 必须调用至少  │
    │ 也可能不调用  │        │               │        │ 一个工具      │
    └───────────────┘        └───────────────┘        └───────────────┘


            ┌─────────────────────────────────────────────────┐
            │                 指定特定工具                     │
            └─────────────────────────────────────────────────┘
                                     │
            ┌────────────────────────┴────────────────────────┐
            │                                                 │
            ▼                                                 ▼
    ┌───────────────────┐                        ┌───────────────────────────┐
    │  "tool_name"      │                        │  {"type": "function",     │
    │  (字符串)         │                        │   "function": {           │
    │                   │                        │     "name": "tool_name"   │
    │                   │                        │   }}                      │
    └─────────┬─────────┘                        └─────────────┬─────────────┘
              │                                                │
              └───────────────────┬────────────────────────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │ 模型必须调用      │
                        │ 指定的工具        │
                        │                   │
                        │ 只能调用这一个！  │
                        └───────────────────┘
```

---

## 3. 不同模型的 tool_choice 处理差异

### 3.1 OpenAI (ChatOpenAI)

**源码位置**: `libs/partners/openai/langchain_openai/chat_models/base.py`

```python
def bind_tools(
    self,
    tools: Sequence[...],
    *,
    tool_choice: dict | str | bool | None = None,
    strict: bool | None = None,
    parallel_tool_calls: bool | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**支持的 tool_choice 值**:

| 输入值 | 转换后的值 | 说明 |
|--------|-----------|------|
| `None` / `False` | 不传递 | 使用 OpenAI 默认行为（auto） |
| `"auto"` | `"auto"` | 自动选择 |
| `"none"` | `"none"` | 禁止调用工具 |
| `"any"` | `"required"` | **自动转换**！OpenAI 不支持 "any"，转为 "required" |
| `"required"` | `"required"` | 强制调用工具 |
| `True` | `"required"` | 转换为 "required" |
| `"tool_name"` | `{"type": "function", "function": {"name": "tool_name"}}` | 自动转换为 dict 格式 |
| `dict` | `dict` | 直接传递 |

**特殊功能**:
- 支持 `parallel_tool_calls` 参数控制并行工具调用
- 支持 `strict` 参数启用严格模式（JSON Schema 验证）

```python
# OpenAI bind_tools 核心转换逻辑
if tool_choice:
    if isinstance(tool_choice, str):
        if tool_choice in tool_names:
            # 工具名转为 dict 格式
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        elif tool_choice == "any":
            # 'any' 转为 'required'
            tool_choice = "required"
    elif isinstance(tool_choice, bool):
        tool_choice = "required"
```

---

### 3.2 Anthropic (ChatAnthropic)

**源码位置**: `libs/partners/anthropic/langchain_anthropic/chat_models.py`

```python
def bind_tools(
    self,
    tools: Sequence[...],
    *,
    tool_choice: dict[str, str] | str | None = None,
    parallel_tool_calls: bool | None = None,
    strict: bool | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**支持的 tool_choice 值**:

| 输入值 | 转换后的值 | 说明 |
|--------|-----------|------|
| `None` | 不传递 | 自动选择 |
| `"auto"` | `{"type": "auto"}` | 自动转换为 dict |
| `"any"` | `{"type": "any"}` | **原生支持 "any"！** |
| `"tool_name"` | `{"type": "tool", "name": "tool_name"}` | 转换格式与 OpenAI 不同！ |
| `dict` | `dict` | 直接传递 |

**特殊功能**:
- 原生支持 `"any"` 值
- 支持 `parallel_tool_calls=False` 禁用并行工具调用
- 支持 `strict=True` 启用严格工具使用

```python
# Anthropic bind_tools 核心转换逻辑
if not tool_choice:
    pass
elif isinstance(tool_choice, dict):
    kwargs["tool_choice"] = tool_choice
elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
    kwargs["tool_choice"] = {"type": tool_choice}
elif isinstance(tool_choice, str):
    # 工具名转为 Anthropic 格式
    kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
```

**Anthropic vs OpenAI 格式对比**:

| 场景 | OpenAI 格式 | Anthropic 格式 |
|------|------------|----------------|
| 强制调用 | `"required"` | `{"type": "any"}` |
| 指定工具 | `{"type": "function", "function": {"name": "xxx"}}` | `{"type": "tool", "name": "xxx"}` |
| 自动 | `"auto"` | `{"type": "auto"}` |

---

### 3.3 Groq (ChatGroq)

**源码位置**: `libs/partners/groq/langchain_groq/chat_models.py`

```python
def bind_tools(
    self,
    tools: Sequence[...],
    *,
    tool_choice: dict | str | bool | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**支持的 tool_choice 值**:

| 输入值 | 转换后的值 | 说明 |
|--------|-----------|------|
| `"any"` | `"required"` | **自动转换**！与 OpenAI 相同 |
| `"auto"` | `"auto"` | 自动选择 |
| `"none"` | `"none"` | 禁止调用 |
| `"required"` | `"required"` | 强制调用 |
| `True` | 指定第一个工具 | **要求只有一个工具！** |
| `"tool_name"` | `{"type": "function", "function": {"name": "tool_name"}}` | 转为 OpenAI 格式 |

```python
# Groq bind_tools 核心转换逻辑
if tool_choice == "any":
    tool_choice = "required"  # 转换为 OpenAI 兼容格式
if isinstance(tool_choice, bool):
    if len(tools) > 1:
        raise ValueError("tool_choice=True 只能在只有一个工具时使用")
    tool_choice = {"type": "function", "function": {"name": tool_name}}
```

---

### 3.4 MistralAI (ChatMistralAI)

**源码位置**: `libs/partners/mistralai/langchain_mistralai/chat_models.py`

```python
def bind_tools(
    self,
    tools: Sequence[...],
    tool_choice: dict | str | Literal["auto", "any"] | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**支持的 tool_choice 值**:

| 输入值 | 转换后的值 | 说明 |
|--------|-----------|------|
| `"auto"` | `"auto"` | 自动选择 |
| `"any"` | `"any"` | **原生支持！** |
| `"tool_name"` | `{"type": "function", "function": {"name": "tool_name"}}` | 转为 OpenAI 格式 |
| `dict` | `dict` | 直接传递 |

---

### 3.5 Ollama (ChatOllama)

**源码位置**: `libs/partners/ollama/langchain_ollama/chat_models.py`

```python
def bind_tools(
    self,
    tools: Sequence[...],
    *,
    tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,  # 被忽略！
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**⚠️ 重要**: **Ollama 不支持 tool_choice 参数！**

```python
# Ollama bind_tools 源码
def bind_tools(
    self,
    tools: Sequence[...],
    *,
    tool_choice: ... = None,  # ARG002 = 被忽略的参数
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
    """
    tool_choice: If provided, which tool for model to call.
    **This parameter is currently ignored as it is not supported by Ollama.**
    """
    formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
    return super().bind(tools=formatted_tools, **kwargs)  # 没有传递 tool_choice!
```

**影响**:
- 即使设置了 `tool_choice`，Ollama 也会忽略
- 模型总是以 "auto" 模式运行

---

### 3.6 Fireworks (ChatFireworks)

**源码位置**: `libs/partners/fireworks/langchain_fireworks/chat_models.py`

```python
def bind_tools(
    self,
    tools: Sequence[...],
    *,
    tool_choice: dict | str | bool | None = None,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, AIMessage]:
```

**支持的 tool_choice 值**:

| 输入值 | 转换后的值 | 说明 |
|--------|-----------|------|
| `"auto"` | `"auto"` | 自动选择 |
| `"any"` | `"any"` | **原生支持！** |
| `"none"` | `"none"` | 禁止调用 |
| `True` | 指定第一个工具 | **要求只有一个工具！** |
| `"tool_name"` | `{"type": "function", "function": {"name": "tool_name"}}` | 转为 OpenAI 格式 |

---

### 3.7 HuggingFace (ChatHuggingFace)

**源码位置**: `libs/partners/huggingface/langchain_huggingface/chat_models/huggingface.py`

**⚠️ 限制**: HuggingFace 要求指定 `tool_choice` 时**只能有一个工具**！

```python
if tool_choice is not None and tool_choice:
    if len(formatted_tools) != 1:
        raise ValueError(
            "When specifying `tool_choice`, you must provide exactly one tool."
        )
```

---

## 4. 模型支持对比表

| 模型 | `"auto"` | `"none"` | `"any"` | `"required"` | 指定工具名 | `parallel_tool_calls` | `strict` |
|------|----------|----------|---------|--------------|------------|---------------------|----------|
| **OpenAI** | ✅ | ✅ | ✅→required | ✅ | ✅ | ✅ | ✅ |
| **Anthropic** | ✅→dict | ❌ | ✅ 原生 | ❌ | ✅ | ✅ | ✅ |
| **Groq** | ✅ | ✅ | ✅→required | ✅ | ✅ | ❌ | ❌ |
| **MistralAI** | ✅ | ❌ | ✅ 原生 | ❌ | ✅ | ❌ | ❌ |
| **Fireworks** | ✅ | ✅ | ✅ 原生 | ❌ | ✅ | ❌ | ❌ |
| **Ollama** | ❌ 忽略 | ❌ 忽略 | ❌ 忽略 | ❌ 忽略 | ❌ 忽略 | ❌ | ❌ |
| **HuggingFace** | ✅ | ✅ | ❌ | ✅ | ✅(限1个) | ❌ | ❌ |

---

## 5. Agent 中 tool_choice 的自动处理

### 5.1 默认行为

在 `create_agent` 中，`tool_choice` 默认为 `None`：

```python
# factory.py 中的 model_node
request = ModelRequest(
    model=model,
    tools=default_tools,
    system_message=system_message,
    response_format=initial_response_format,
    messages=state["messages"],
    tool_choice=None,  # ← 默认为 None
    state=state,
    runtime=runtime,
)
```

### 5.2 结构化输出时的自动设置

当使用 `ToolStrategy` 结构化输出时，会**自动强制工具调用**：

```python
# factory.py 中的 _get_bound_model
if isinstance(effective_response_format, ToolStrategy):
    # Force tool use if we have structured output tools
    tool_choice = "any" if structured_output_tools else request.tool_choice
    return (
        request.model.bind_tools(
            final_tools, tool_choice=tool_choice, **request.model_settings
        ),
        effective_response_format,
    )
```

**流程图**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                      使用 response_format (ToolStrategy) 时的自动处理                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │  create_agent(                      │
                    │      model="openai:gpt-4o",         │
                    │      tools=[search, calculate],     │
                    │      response_format=MySchema,      │ ← 指定结构化输出
                    │  )                                  │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  检测 response_format 类型          │
                    │                                     │
                    │  - AutoStrategy → 自动检测         │
                    │  - ToolStrategy → 使用工具策略     │
                    │  - ProviderStrategy → 使用provider │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 如果是 ToolStrategy
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  自动设置 tool_choice = "any"       │
                    │                                     │
                    │  目的：强制模型调用结构化输出工具   │
                    │  确保获得预期的 schema 输出         │
                    │                                     │
                    └─────────────────────────────────────┘
```

---

## 6. 如何在中间件中修改 tool_choice

### 6.1 使用 wrap_model_call 修改

```python
from langchain.agents import AgentMiddleware, ModelRequest, ModelResponse

class ForceToolCallMiddleware(AgentMiddleware):
    """强制模型调用工具"""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        # 修改 tool_choice 为强制调用
        modified_request = request.override(tool_choice="any")
        return handler(modified_request)


class DisableToolCallMiddleware(AgentMiddleware):
    """在某些条件下禁用工具调用"""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        # 检查消息数量
        if len(request.messages) > 10:
            # 长对话时禁用工具调用，只回复文本
            modified_request = request.override(tool_choice="none")
            return handler(modified_request)
        return handler(request)


class SpecificToolMiddleware(AgentMiddleware):
    """强制使用特定工具"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        # 强制调用特定工具
        modified_request = request.override(tool_choice=self.tool_name)
        return handler(modified_request)
```

### 6.2 使用装饰器

```python
from langchain.agents import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def force_tool_call(request: ModelRequest, handler) -> ModelResponse:
    """强制工具调用的中间件"""
    modified = request.override(tool_choice="any")
    return handler(modified)


@wrap_model_call
def conditional_tool_choice(request: ModelRequest, handler) -> ModelResponse:
    """根据条件设置 tool_choice"""

    last_message = request.messages[-1] if request.messages else None

    if last_message and "必须搜索" in last_message.content:
        # 用户明确要求搜索时，强制使用搜索工具
        modified = request.override(tool_choice="search")
    elif last_message and "不要使用工具" in last_message.content:
        # 用户要求不使用工具
        modified = request.override(tool_choice="none")
    else:
        # 其他情况保持默认
        modified = request

    return handler(modified)


# 使用
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search, calculate],
    middleware=[force_tool_call, conditional_tool_choice]
)
```

---

## 7. 常见问题与解决方案

### 7.1 Ollama 不支持 tool_choice

**问题**: Ollama 会忽略 `tool_choice` 设置

**解决方案**: 使用 `system_prompt` 引导模型

```python
from langchain.agents import create_agent

agent = create_agent(
    model="ollama:llama3",
    tools=[search],
    system_prompt="""你是一个助手。

    重要：当用户询问需要查询的信息时，你必须使用 search 工具。
    不要猜测答案，一定要先搜索。
    """
)
```

### 7.2 "any" 在不同模型中的兼容性

**问题**: 不同模型对 "any" 的处理不同

| 模型 | "any" 的处理 |
|------|-------------|
| OpenAI | 转换为 "required" |
| Anthropic | 原生支持 |
| Groq | 转换为 "required" |
| MistralAI | 原生支持 |

**解决方案**: LangChain 已经在各模型的 `bind_tools` 中处理了兼容性，你可以放心使用 `"any"`。

### 7.3 HuggingFace 多工具限制

**问题**: HuggingFace 指定 `tool_choice` 时只能有一个工具

**解决方案**:

1. 不指定 `tool_choice`，让模型自动选择
2. 或使用其他模型

---

## 8. 完整流程图

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              tool_choice 完整处理流程                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     用户调用                                             │
│                                                                                         │
│   agent = create_agent(model="openai:gpt-4o", tools=[search, calc])                     │
│   agent.invoke({"messages": [HumanMessage("计算 1+1")]})                                │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 model_node 节点                                          │
│                                                                                         │
│   构建 ModelRequest:                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  ModelRequest(                                                                  │   │
│   │      model=ChatOpenAI("gpt-4o"),                                                │   │
│   │      tools=[search, calc],                                                      │   │
│   │      messages=[HumanMessage("计算 1+1")],                                        │   │
│   │      tool_choice=None,                  ← 默认值                                │   │
│   │      response_format=None,                                                      │   │
│   │  )                                                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              wrap_model_call 中间件链                                    │
│                                                                                         │
│   中间件可以修改 tool_choice:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  # 示例：强制工具调用中间件                                                      │   │
│   │  modified_request = request.override(tool_choice="any")                         │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              _get_bound_model(request)                                  │
│                                                                                         │
│   检查 response_format:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  if isinstance(effective_response_format, ToolStrategy):                        │   │
│   │      # 结构化输出时强制工具调用                                                  │   │
│   │      tool_choice = "any"                                                        │   │
│   │  else:                                                                          │   │
│   │      tool_choice = request.tool_choice  # 使用中间件设置的值                     │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              model.bind_tools(...)                                      │
│                                                                                         │
│   不同模型的处理:                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                 │   │
│   │  OpenAI:                                                                        │   │
│   │  ├── "any" → "required"                                                         │   │
│   │  ├── "tool_name" → {"type": "function", "function": {"name": "tool_name"}}      │   │
│   │  └── True → "required"                                                          │   │
│   │                                                                                 │   │
│   │  Anthropic:                                                                     │   │
│   │  ├── "any" → {"type": "any"}                                                    │   │
│   │  ├── "auto" → {"type": "auto"}                                                  │   │
│   │  └── "tool_name" → {"type": "tool", "name": "tool_name"}                        │   │
│   │                                                                                 │   │
│   │  Ollama:                                                                        │   │
│   │  └── 忽略 tool_choice 参数！                                                     │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                  LLM API 调用                                           │
│                                                                                         │
│   实际发送给 API 的请求:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  {                                                                              │   │
│   │      "model": "gpt-4o",                                                         │   │
│   │      "messages": [...],                                                         │   │
│   │      "tools": [{"type": "function", "function": {...}}, ...],                   │   │
│   │      "tool_choice": "required"   ← 最终值（如果设置了 "any"）                    │   │
│   │  }                                                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    LLM 响应                                             │
│                                                                                         │
│   tool_choice="required" 时:                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  AIMessage(                                                                     │   │
│   │      content="",                                                                │   │
│   │      tool_calls=[                                                               │   │
│   │          {"name": "calc", "args": {"expression": "1+1"}, "id": "call_xxx"}       │   │
│   │      ]                                                                          │   │
│   │  )                                                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│   tool_choice="none" 时:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  AIMessage(                                                                     │   │
│   │      content="1+1=2",                                                           │   │
│   │      tool_calls=[]                      ← 不会调用任何工具                       │   │
│   │  )                                                                              │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│   tool_choice="auto" 或 None 时:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  可能调用工具，也可能直接回复文本                                                 │   │
│   │  由模型自行判断                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 最佳实践

### 9.1 何时使用不同的 tool_choice

| 场景 | 推荐 tool_choice | 原因 |
|------|-----------------|------|
| 普通对话（可能需要工具） | `None` 或 `"auto"` | 让模型自行判断 |
| 必须使用工具的场景 | `"any"` 或 `"required"` | 强制调用工具 |
| 结构化输出 | 自动设置为 `"any"` | 框架自动处理 |
| 纯文本回复 | `"none"` | 禁止工具调用 |
| 特定任务（如搜索） | `"search"` (工具名) | 指定特定工具 |

### 9.2 跨模型兼容性建议

```python
# ✅ 推荐：使用 LangChain 兼容的值
tool_choice = "any"      # 所有模型都会正确处理
tool_choice = "auto"     # 所有模型都支持
tool_choice = None       # 最安全的默认值

# ⚠️ 谨慎：模型特定的格式
tool_choice = "required"  # OpenAI/Groq 原生支持，Anthropic 不支持
tool_choice = {"type": "tool", "name": "xxx"}  # Anthropic 格式，OpenAI 不支持

# ❌ 避免：Ollama 不支持
# 对于 Ollama，设置 tool_choice 没有效果
```
