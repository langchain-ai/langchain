
# LangChain Agent 中间件开发指南

## 目录

1. [概述](#概述)
2. [中间件生命周期](#中间件生命周期)
3. [可用的钩子方法](#可用的钩子方法)
4. [状态管理](#状态管理)
5. [工具注入](#工具注入)
6. [完整示例](#完整示例)
7. [最佳实践](#最佳实践)
8. [常见模式](#常见模式)
9. [调试和测试](#调试和测试)

---

## 概述

中间件（Middleware）是 LangChain Agent 系统的核心扩展机制，允许你在 Agent 执行的不同阶段插入自定义逻辑。中间件可以：

- 拦截和修改模型调用
- 拦截和修改工具调用
- 在 Agent 执行前后执行逻辑
- 扩展 Agent 状态
- 注入新的工具
- 控制执行流程

### 中间件执行顺序

多个中间件按照在 `middleware` 列表中的顺序执行，第一个中间件是最外层：

```python
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        MiddlewareA(),  # 最外层，最先执行
        MiddlewareB(),  # 中间层
        MiddlewareC(),  # 最内层，最后执行
    ]
)
```

---

## 中间件生命周期

Agent 执行流程中的中间件钩子调用顺序：

```text
Agent 开始
    ↓
before_agent (所有中间件)
    ↓
before_model (所有中间件)
    ↓
wrap_model_call (所有中间件，嵌套执行)
    ↓
模型调用
    ↓
after_model (所有中间件)
    ↓
工具调用（如果需要）
    ↓
wrap_tool_call (所有中间件，嵌套执行)
    ↓
after_agent (所有中间件)
    ↓
Agent 结束
```

### 各钩子特点总结

| 钩子 | 执行时机 | 执行方式 | 执行顺序 | 主要功能 | 返回值 | 特殊能力 |
|------|---------|---------|---------|---------|--------|---------|
| **before_agent** | Agent 执行开始前 | 所有中间件顺序执行 | **从前往后**（列表顺序） | 初始化状态、设置全局配置 | `dict[str, Any] \| None` | 状态初始化 |
| **before_model** | 每次模型调用前 | 所有中间件顺序执行 | **从前往后**（列表顺序） | 条件检查、限制控制、状态准备 | `dict[str, Any] \| None` | 可返回 `{"jump_to": "end"}` 跳过模型调用并结束 Agent |
| **wrap_model_call** | 模型调用时 | 所有中间件**嵌套执行**（洋葱模型） | **嵌套执行**（第一个是最外层） | 拦截、修改请求/响应、重试、缓存、短路执行 | `ModelResponse` | **最强大的钩子**，可完全控制模型调用流程 |
| **after_model** | 模型调用后 | 所有中间件顺序执行 | **从后往前**（列表逆序） | 处理响应、统计信息、更新状态 | `dict[str, Any] \| None` | 状态累积和统计 |
| **wrap_tool_call** | 工具调用时 | 所有中间件**嵌套执行**（洋葱模型） | **嵌套执行**（第一个是最外层） | 拦截、重试、监控、修改工具调用 | `ToolMessage \| Command` | 可完全控制工具调用流程 |
| **after_agent** | Agent 执行完成后 | 所有中间件顺序执行 | **从后往前**（列表逆序） | 清理资源、最终统计、状态持久化 | `dict[str, Any] \| None` | 资源清理和收尾工作 |

**关键区别：**

- **顺序执行（从前往后）**（before_agent, before_model）：所有中间件按列表顺序从前往后依次执行，互不嵌套
  - 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：A → B → C
- **顺序执行（从后往前）**（after_model, after_agent）：所有中间件按列表顺序从后往前依次执行，互不嵌套
  - 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：C → B → A
- **嵌套执行**（wrap_model_call, wrap_tool_call）：中间件形成洋葱模型，列表中的第一个是最外层，最后一个是最内层
  - 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：A进入 → B进入 → C进入 → 模型/工具调用 → C退出 → B退出 → A退出

---

## 可用的钩子方法

### 钩子总览

所有中间件都可以使用以下 **6 个钩子方法**（每个都有同步和异步版本）：

| 钩子方法 | 同步版本 | 异步版本 | 执行时机 | 执行方式 | 返回值 |
|---------|---------|---------|---------|---------|--------|
| **before_agent** | `before_agent` | `abefore_agent` | Agent 执行开始前 | 顺序执行（从前往后） | `dict[str, Any] \| None` |
| **before_model** | `before_model` | `abefore_model` | 每次模型调用前 | 顺序执行（从前往后） | `dict[str, Any] \| None` |
| **wrap_model_call** | `wrap_model_call` | `awrap_model_call` | 模型调用时 | 嵌套执行（洋葱模型） | `ModelResponse` |
| **after_model** | `after_model` | `aafter_model` | 模型调用后 | 顺序执行（从后往前） | `dict[str, Any] \| None` |
| **wrap_tool_call** | `wrap_tool_call` | `awrap_tool_call` | 工具调用时 | 嵌套执行（洋葱模型） | `ToolMessage \| Command` |
| **after_agent** | `after_agent` | `aafter_agent` | Agent 执行完成后 | 顺序执行（从后往前） | `dict[str, Any] \| None` |

**重要说明：**

1. **所有中间件都可以使用所有钩子**：每个中间件类都可以选择实现任意一个或多个钩子方法，不需要实现全部。
2. **同步和异步版本**：如果 Agent 以同步方式调用（`invoke()`），则使用同步版本；如果以异步方式调用（`ainvoke()`），则使用异步版本。建议同时实现两个版本以支持两种调用方式。
3. **钩子的可选性**：中间件只需要实现它需要的钩子，不需要的钩子可以不实现。

### 在 TodoListMiddleware 场景中的钩子使用

当使用 `TodoListMiddleware` 时，**所有其他中间件仍然可以使用所有 6 个钩子**，没有任何限制。

**TodoListMiddleware 本身的实现：**

- `TodoListMiddleware` 只实现了 `wrap_model_call` / `awrap_model_call` 钩子
- 它的作用是：在模型调用时注入 `write_todos` 工具的系统提示词
- 它不影响其他中间件使用任何钩子

**示例：多个中间件与 TodoListMiddleware 组合**

```python
from langchain.agents import create_agent
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware

# 创建包含多个中间件的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[
        # 中间件 A：使用 before_model 和 after_model
        ModelCallLimitMiddleware(max_calls=10),
        
        # 中间件 B：使用 wrap_model_call（注入 todo 工具提示）
        TodoListMiddleware(),
        
        # 中间件 C：使用 after_model（人工参与）
        HumanInTheLoopMiddleware(),
    ]
)
```

**执行流程：**

```text
Agent 开始
    ↓
before_agent (所有中间件，如果有实现)
    ↓
before_model (ModelCallLimitMiddleware) ← 检查调用限制
    ↓
wrap_model_call (嵌套执行)
    ├─ ModelCallLimitMiddleware.wrap_model_call (外层)
    ├─ TodoListMiddleware.wrap_model_call (中层) ← 注入 todo 提示
    └─ 实际模型调用 (内层)
    ↓
after_model (所有中间件，从后往前)
    ├─ HumanInTheLoopMiddleware.after_model ← 检查是否需要人工参与
    └─ ModelCallLimitMiddleware.after_model ← 更新调用计数
    ↓
工具调用（如果需要）
    ↓
wrap_tool_call (所有中间件，如果有实现)
    ↓
after_agent (所有中间件，如果有实现)
    ↓
Agent 结束
```

**关键点：**

- ✅ **所有中间件都可以使用所有钩子**，包括与 `TodoListMiddleware` 一起使用时
- ✅ `TodoListMiddleware` 只使用 `wrap_model_call`，不影响其他中间件
- ✅ 多个中间件可以同时使用同一个钩子，按照执行顺序/嵌套顺序执行
- ✅ 每个中间件可以选择性地实现它需要的钩子

---

### 1. before_agent / abefore_agent

在 Agent 执行开始前调用。

**执行顺序：**

- 多个中间件按列表顺序**从前往后**依次执行
- 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：A → B → C

**签名：**

```python
def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """在 Agent 执行前调用"""
    pass

async def abefore_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """异步版本"""
    pass
```

**返回值：**

- `dict[str, Any]`: 状态更新，会被合并到 AgentState
- `None`: 无状态更新

**示例：**

```python
def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # 初始化计数器
    return {"call_count": 0}
```

---

### 2. before_model / abefore_model

在每次模型调用前执行。

**执行顺序：**

- 多个中间件按列表顺序**从前往后**依次执行
- 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：A → B → C

**签名：**

```python
def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """在模型调用前执行"""
    pass
```

**返回值：**

- `dict[str, Any]`: 状态更新
- `None`: 无更新

**特殊功能：**

- 可以返回 `{"jump_to": "end"}` 来跳过模型调用并结束 Agent
- 需要 `@hook_config(can_jump_to=["end"])` 装饰器

**示例：**

```python
@hook_config(can_jump_to=["end"])
def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # 检查限制
    if state.get("call_count", 0) >= self.max_calls:
        return {"jump_to": "end", "messages": [AIMessage(content="Limit reached")]}
    return None
```

---

### 3. wrap_model_call / awrap_model_call

拦截和控制模型调用。这是最强大的钩子，可以：

- 修改请求
- 修改响应
- 实现重试逻辑
- 实现缓存
- 短路执行

**签名：**

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """拦截模型调用"""
    # 必须调用 handler 来执行实际的模型调用
    return handler(request)
```

**ModelRequest 包含：**

- `model`: BaseChatModel 实例
- `system_prompt`: 系统提示词
- `messages`: 消息列表
- `tools`: 工具列表
- `state`: 当前状态
- `runtime`: LangGraph 运行时

#### 嵌套执行机制（洋葱模型）

`wrap_model_call` 采用**嵌套执行**（洋葱模型）机制，多个中间件会形成嵌套结构：

**执行顺序：**

- 列表中的**第一个中间件是最外层**，最后一个是最内层
- 执行流程：外层 → 内层 → 模型调用 → 内层返回 → 外层返回
- 每个中间件接收的 `handler` 参数实际上是**内层中间件**（或最终的模型调用）

**示例：三个中间件的嵌套执行**

```python
# 定义三个中间件
class FirstMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        print("First: before")
        response = handler(request)  # handler 是 SecondMiddleware
        print("First: after")
        return response

class SecondMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        print("Second: before")
        response = handler(request)  # handler 是 ThirdMiddleware
        print("Second: after")
        return response

class ThirdMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        print("Third: before")
        response = handler(request)  # handler 是实际的模型调用
        print("Third: after")
        return response

# 创建 Agent（第一个是最外层）
agent = create_agent(
    model=model,
    middleware=[FirstMiddleware(), SecondMiddleware(), ThirdMiddleware()]
)
```

**执行流程：**

```text
First.wrap_model_call 开始
  ↓ print("First: before")
  ↓ 调用 handler (即 SecondMiddleware.wrap_model_call)
    ↓ Second.wrap_model_call 开始
      ↓ print("Second: before")
      ↓ 调用 handler (即 ThirdMiddleware.wrap_model_call)
        ↓ Third.wrap_model_call 开始
          ↓ print("Third: before")
          ↓ 调用 handler (即实际的模型调用)
            ↓ [模型执行]
          ↓ print("Third: after")
        ↓ Third.wrap_model_call 结束
      ↓ print("Second: after")
    ↓ Second.wrap_model_call 结束
  ↓ print("First: after")
↓ First.wrap_model_call 结束
```

**输出顺序：**

```text
First: before
Second: before
Third: before
[模型调用]
Third: after
Second: after
First: after
```

**关键理解：**

1. **handler 参数**：每个中间件接收的 `handler` 是内层中间件（或最终模型调用）的包装函数
2. **必须调用 handler**：只有调用 `handler(request)` 才能继续执行内层逻辑
3. **可以短路**：不调用 `handler` 可以直接返回，跳过内层和模型调用
4. **可以修改请求/响应**：在调用 `handler` 前后都可以修改 `request` 或 `response`

**实际应用：多个中间件协作**

嵌套执行使得多个中间件可以协作完成复杂功能。例如：日志中间件（外层）→ 重试中间件（中层）→ 缓存中间件（内层）

```python
# 1. 缓存中间件（最内层）- 先检查缓存
class CacheMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.cache = {}
    
    def wrap_model_call(self, request, handler):
        cache_key = str(request.messages)
        if cache_key in self.cache:
            print("Cache hit!")
            return self.cache[cache_key]
        response = handler(request)  # 调用重试中间件
        self.cache[cache_key] = response
        return response

# 2. 重试中间件（中层）- 处理错误重试
class RetryMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        for attempt in range(3):
            try:
                return handler(request)  # 调用缓存中间件或模型
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry attempt {attempt + 1}")

# 3. 日志中间件（最外层）- 记录所有调用
class LoggingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        print(f"[LOG] Model call started: {len(request.messages)} messages")
        start_time = time.time()
        response = handler(request)  # 调用重试中间件
        elapsed = time.time() - start_time
        print(f"[LOG] Model call completed in {elapsed:.2f}s")
        return response

# 使用：第一个是最外层
agent = create_agent(
    model=model,
    middleware=[
        LoggingMiddleware(),    # 外层：日志
        RetryMiddleware(),       # 中层：重试
        CacheMiddleware(),       # 内层：缓存
    ]
)
```

**执行顺序：**

1. LoggingMiddleware 开始记录
2. RetryMiddleware 尝试调用
3. CacheMiddleware 检查缓存（如果命中则直接返回，跳过模型）
4. 如果缓存未命中，执行模型调用
5. 响应依次返回：CacheMiddleware → RetryMiddleware → LoggingMiddleware

**示例：重试逻辑**

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(self.max_retries):
        try:
            return handler(request)
        except Exception as e:
            if attempt == self.max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指数退避
    raise Exception("Max retries exceeded")
```

**示例：修改请求**

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 修改系统提示词
    request.system_prompt = (
        (request.system_prompt or "") + "\n\nAdditional instructions..."
    )
    return handler(request)
```

**示例：修改响应**

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    response = handler(request)
    # 修改响应内容
    if response.result:
        ai_msg = response.result[0]
        modified_msg = AIMessage(
            content=f"[PREFIX]{ai_msg.content}[SUFFIX]",
            tool_calls=ai_msg.tool_calls,
        )
        return ModelResponse(
            result=[modified_msg],
            structured_response=response.structured_response,
        )
    return response
```

**示例：缓存**

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    cache_key = self._generate_cache_key(request)
    if cached := self.cache.get(cache_key):
        return cached  # 短路执行
    
    response = handler(request)
    self.cache.set(cache_key, response)
    return response
```

---

### 4. after_model / aafter_model

在模型调用后执行，用于处理响应和更新状态。

**执行顺序：**

- 多个中间件按列表顺序**从后往前**依次执行（与 before_model 相反）
- 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：C → B → A
- 这样设计是为了与 before_model 形成对称的执行顺序

**签名：**

```python
def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """在模型调用后执行"""
    pass
```

**返回值：**

- `dict[str, Any]`: 状态更新，会被合并到 AgentState
- `None`: 无更新

**示例：统计调用次数**

```python
def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    return {
        "call_count": state.get("call_count", 0) + 1
    }
```

**示例：提取和保存信息**

```python
def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None
    
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "usage_metadata"):
        usage = last_msg.usage_metadata
        if usage:
            return {
                "total_tokens": state.get("total_tokens", 0) + (usage.input_tokens or 0) + (usage.output_tokens or 0)
            }
    return None
```

---

### 5. wrap_tool_call / awrap_tool_call

拦截工具调用，用于重试、监控、修改等。

**签名：**

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """拦截工具调用"""
    return handler(request)
```

**ToolCallRequest 包含：**

- `tool_call`: 工具调用信息（dict）
- `tool`: BaseTool 实例
- `state`: 当前状态
- `runtime`: LangGraph 运行时

**示例：工具重试**

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    for attempt in range(self.max_retries):
        try:
            return handler(request)
        except Exception as e:
            if attempt == self.max_retries - 1:
                # 最后一次尝试失败，返回错误消息
                return ToolMessage(
                    content=f"Error after {self.max_retries} attempts: {str(e)}",
                    tool_call_id=request.tool_call["id"],
                )
            time.sleep(2 ** attempt)
    raise Exception("Should not reach here")
```

---

### 6. after_agent / aafter_agent

在 Agent 执行完成后调用。

**执行顺序：**

- 多个中间件按列表顺序**从后往前**依次执行（与 before_agent 相反）
- 示例：`[MiddlewareA, MiddlewareB, MiddlewareC]` → 执行顺序：C → B → A
- 这样设计是为了与 before_agent 形成对称的执行顺序

**签名：**

```python
def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
    """在 Agent 执行完成后调用"""
    pass
```

**示例：清理资源**

```python
def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # 清理临时文件、关闭连接等
    self.cleanup()
    return None
```

---

## 状态管理

### 扩展状态模式

中间件可以通过定义 `state_schema` 来扩展 AgentState：

```python
from typing_extensions import NotRequired, TypedDict, Annotated
from langgraph.channels.untracked_value import UntrackedValue
from langchain.agents.middleware.types import AgentState, PrivateStateAttr

class MyMiddlewareState(AgentState):
    """扩展的状态模式"""
    
    # 普通字段（会出现在输入/输出中）
    my_field: NotRequired[str]
    
    # 私有字段（不会出现在输入/输出中）
    internal_counter: NotRequired[Annotated[int, PrivateStateAttr]]
    
    # 未跟踪字段（不会持久化到 checkpointer）
    temp_data: NotRequired[Annotated[dict, UntrackedValue, PrivateStateAttr]]

class MyMiddleware(AgentMiddleware[MyMiddlewareState, Any]):
    state_schema = MyMiddlewareState
    
    def after_model(self, state: MyMiddlewareState, runtime: Runtime) -> dict[str, Any] | None:
        return {
            "my_field": "updated",
            "internal_counter": state.get("internal_counter", 0) + 1,
        }
```

### 状态更新规则

1. **返回字典格式**：`after_model` 等方法返回的字典会被合并到状态

2. **序列化兼容**：如果使用 dataclass，需要转换为字典：

   ```python
   from dataclasses import asdict
   
   return {
       "my_dataclass_field": asdict(my_dataclass_instance)
   }
   ```

3. **状态读取**：从 state 读取时，需要处理可能不存在的情况：

   ```python
   value = state.get("my_field", default_value)
   ```

---

## 工具注入

中间件可以通过 `self.tools` 属性注入新工具：

```python
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.tools import InjectedToolCallId
from typing import Annotated

class MyMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        
        @tool(description="我的自定义工具")
        def my_tool(
            param: str,
            tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command:
            """工具实现"""
            return Command(
                update={
                    "my_state_field": param,
                    "messages": [
                        ToolMessage(
                            content=f"Tool executed with {param}",
                            tool_call_id=tool_call_id
                        )
                    ]
                }
            )
        
        self.tools = [my_tool]
```

---

## 完整示例

### 示例 1: 简单的日志中间件

```python
"""简单的日志中间件示例"""

from typing import TYPE_CHECKING, Any
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class LoggingMiddleware(AgentMiddleware):
    """记录所有模型调用的中间件"""
    
    def __init__(self, log_level: str = "INFO"):
        super().__init__()
        self.log_level = log_level
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """记录模型调用"""
        print(f"[{self.log_level}] Model call started")
        print(f"  Messages: {len(request.messages)}")
        print(f"  Tools: {len(request.tools)}")
        
        response = handler(request)
        
        print(f"[{self.log_level}] Model call completed")
        if response.result:
            print(f"  Response length: {len(response.result[0].content)}")
        
        return response
    
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """记录调用后的状态"""
        messages = state.get("messages", [])
        print(f"[{self.log_level}] Total messages: {len(messages)}")
        return None
```

### 示例 2: 成本追踪中间件（简化版）

```python
"""成本追踪中间件示例"""

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Annotated
from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


@dataclass
class TokenUsage:
    """Token 使用统计"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class CostTrackingState(AgentState):
    """成本追踪状态"""
    run_token_usage: NotRequired[
        Annotated[TokenUsage, UntrackedValue, PrivateStateAttr]
    ]


class CostTrackingMiddleware(AgentMiddleware[CostTrackingState, Any]):
    """成本追踪中间件"""
    
    state_schema = CostTrackingState
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """拦截模型调用并追踪 token"""
        response = handler(request)
        
        # 提取 token 使用
        if response.result:
            last_msg = response.result[-1]
            if isinstance(last_msg, AIMessage) and hasattr(last_msg, "usage_metadata"):
                usage = last_msg.usage_metadata
                if usage:
                    print(f"Input tokens: {usage.input_tokens}")
                    print(f"Output tokens: {usage.output_tokens}")
        
        return response
    
    def after_model(
        self,
        state: CostTrackingState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """更新 token 统计"""
        messages = state.get("messages", [])
        if not messages:
            return None
        
        # 获取最后一个 AI 消息
        last_msg = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_msg = msg
                break
        
        if not last_msg or not hasattr(last_msg, "usage_metadata"):
            return None
        
        usage = last_msg.usage_metadata
        if not usage:
            return None
        
        # 提取本次 token 使用
        current_usage = TokenUsage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            total_tokens=(usage.input_tokens or 0) + (usage.output_tokens or 0),
        )
        
        # 累积统计
        run_usage = state.get("run_token_usage")
        if isinstance(run_usage, dict):
            run_usage = TokenUsage(**run_usage)
        elif run_usage is None:
            run_usage = TokenUsage()
        
        updated_usage = TokenUsage(
            input_tokens=run_usage.input_tokens + current_usage.input_tokens,
            output_tokens=run_usage.output_tokens + current_usage.output_tokens,
            total_tokens=run_usage.total_tokens + current_usage.total_tokens,
        )
        
        # 转换为字典格式
        return {
            "run_token_usage": asdict(updated_usage)
        }
```

---

## 最佳实践

### 1. 同步和异步支持

始终同时实现同步和异步版本：

```python
def wrap_model_call(self, request, handler):
    # 同步实现
    return handler(request)

async def awrap_model_call(self, request, handler):
    # 异步实现
    return await handler(request)
```

### 2. 错误处理

在中间件中妥善处理错误：

```python
def wrap_model_call(self, request, handler):
    try:
        return handler(request)
    except Exception as e:
        # 记录错误但不中断执行
        self.logger.error(f"Error in middleware: {e}")
        # 可以选择返回默认响应或重新抛出异常
        raise
```

### 3. 状态兼容性

处理状态可能不存在或格式不同的情况：

```python
def after_model(self, state, runtime):
    # 安全地读取状态
    value = state.get("my_field", default_value)
    
    # 处理可能是字典或对象的情况
    if isinstance(value, dict):
        value = MyClass(**value)
    elif value is None:
        value = MyClass()
```

### 4. 性能考虑

避免在中间件中执行耗时操作：

```python
def wrap_model_call(self, request, handler):
    # ❌ 不好：同步阻塞操作
    # time.sleep(1)
    
    # ✅ 好：异步操作或快速检查
    if self.should_skip(request):
        return handler(request)
    
    return handler(request)
```

### 5. 文档和类型提示

提供清晰的文档和类型提示：

```python
class MyMiddleware(AgentMiddleware[MyState, Any]):
    """中间件的详细描述。
    
    Example:
        ```python
        middleware = MyMiddleware(config="value")
        agent = create_agent(model, middleware=[middleware])
        ```
    """
    
    state_schema = MyState
    
    def __init__(self, *, config: str):
        """初始化中间件。
        
        Args:
            config: 配置参数说明
        """
        super().__init__()
        self.config = config
```

---

## 常见模式

### 模式 1: 条件执行

```python
def before_model(self, state, runtime):
    if self.should_skip(state):
        return {"jump_to": "end"}
    return None
```

### 模式 2: 状态累积

```python
def after_model(self, state, runtime):
    current = state.get("counter", 0)
    return {"counter": current + 1}
```

### 模式 3: 请求修改

```python
def wrap_model_call(self, request, handler):
    # 修改请求
    request.system_prompt = f"{request.system_prompt}\n\nAdditional context"
    return handler(request)
```

### 模式 4: 响应修改

```python
def wrap_model_call(self, request, handler):
    response = handler(request)
    # 修改响应
    if response.result:
        modified = AIMessage(content=f"[PREFIX]{response.result[0].content}")
        return ModelResponse(result=[modified])
    return response
```

### 模式 5: 重试逻辑

```python
def wrap_model_call(self, request, handler):
    for attempt in range(self.max_retries):
        try:
            return handler(request)
        except Exception as e:
            if attempt == self.max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

---

## 调试和测试

### 调试技巧

1. **添加日志**：

   ```python
   def wrap_model_call(self, request, handler):
       print(f"[DEBUG] Request: {request.messages}")
       response = handler(request)
       print(f"[DEBUG] Response: {response.result}")
       return response
   ```

2. **检查状态**：

   ```python
   def after_model(self, state, runtime):
       print(f"[DEBUG] State keys: {state.keys()}")
       print(f"[DEBUG] Messages count: {len(state.get('messages', []))}")
       return None
   ```

### 测试示例

```python
import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

def test_my_middleware():
    middleware = MyMiddleware()
    
    # 创建模拟请求
    request = ModelRequest(
        model=mock_model,
        messages=[],
        tools=[],
        # ... 其他参数
    )
    
    # 创建模拟处理器
    def mock_handler(req):
        return ModelResponse(result=[AIMessage(content="test")])
    
    # 测试中间件
    response = middleware.wrap_model_call(request, mock_handler)
    assert response.result[0].content == "test"
```

---

## 总结

开发中间件的关键步骤：

1. **确定需求**：明确中间件要实现的功能
2. **选择钩子**：选择合适的生命周期钩子
3. **定义状态**：如果需要，扩展状态模式
4. **实现逻辑**：实现同步和异步版本
5. **处理错误**：添加适当的错误处理
6. **编写文档**：提供清晰的文档和示例
7. **测试验证**：编写测试确保功能正常

遵循这些指南，你就能开发出高质量、可维护的中间件！
