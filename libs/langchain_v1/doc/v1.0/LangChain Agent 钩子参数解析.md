
# LangChain Agent 钩子参数解析

## 钩子总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Agent 中间件 7 个钩子                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   生命周期钩子 (4个)                        拦截器钩子 (2个)            便捷钩子 (1个)        │
│   ┌──────────────────────┐                ┌──────────────────────┐   ┌──────────────────┐   │
│   │ 1. before_agent      │                │ 5. wrap_model_call   │   │ 7. dynamic_prompt│   │
│   │ 2. before_model      │                │ 6. wrap_tool_call    │   │                  │   │
│   │ 3. after_model       │                │                      │   │                  │   │
│   │ 4. after_agent       │                │                      │   │                  │   │
│   └──────────────────────┘                └──────────────────────┘   └──────────────────┘   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 钩子 1: `before_agent` / `abefore_agent`

### 触发时机
**Agent 执行开始前，只执行一次**

### 函数签名

```python
def before_agent(
    self,
    state: StateT,           # AgentState 或自定义状态
    runtime: Runtime[ContextT]  # LangGraph 运行时
) -> dict[str, Any] | None
```

### 参数详解

#### `state: AgentState` (或自定义状态)

```python
class AgentState(TypedDict, Generic[ResponseT]):
    """Agent 状态 schema"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    # 必填字段，消息列表
    # add_messages 表示新消息会追加到列表中（而非替换）

    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    # 可选，控制流程跳转目标: "tools" | "model" | "end"
    # EphemeralValue 表示这个值不会持久化
    # PrivateStateAttr 表示不暴露给输入/输出 schema

    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
    # 可选，结构化输出结果
    # OmitFromInput 表示不在输入 schema 中显示
```

**state 数据示例：**

```python
{
    "messages": [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮助你的？"),
        HumanMessage(content="帮我搜索 AI 新闻"),
    ],
    "user_id": "user_123",           # 自定义字段（如果使用自定义 state_schema）
    "thread_model_call_count": 0,    # 中间件添加的私有字段
}
```

#### `runtime: Runtime[ContextT]`

```python
# Runtime 来自 langgraph.runtime，包含以下属性：
runtime.store      # BaseStore | None - 持久化存储（create_agent 传入的 store）
runtime.context    # ContextT | None - 用户上下文（create_agent 传入的 context_schema）
runtime.config     # RunnableConfig - 运行配置
runtime.previous   # 前一个节点的输出
```

**runtime 数据示例：**

```python
runtime.store        # <InMemoryStore> 或 None
runtime.context      # {"user_name": "张三", "session_id": "abc123"} 或 None
runtime.config       # {"configurable": {"thread_id": "thread_1"}, "callbacks": [...]}
```

### 返回值

| 返回值类型 | 含义 | 示例 |
|-----------|------|------|
| `None` | 无状态更新，继续执行 | `return None` |
| `dict[str, Any]` | 状态更新，合并到 state | `return {"call_count": 0}` |
| `{"jump_to": "end"}` | 跳转到结束（需配置 `can_jump_to`） | `return {"jump_to": "end", "messages": [AIMessage(...)]}` |

### 目的与用途

| 用途 | 示例 |
|------|------|
| **初始化状态** | 设置计数器、标志位 |
| **加载用户数据** | 从 store 读取用户偏好 |
| **条件检查** | 检查用户权限，不满足直接跳转 end |
| **日志记录** | 记录 Agent 开始时间 |

### 完整示例

```python
from langchain.agents import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from typing import Any

class InitializationMiddleware(AgentMiddleware):
    state_schema = AgentState  # 可以自定义

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 执行前初始化"""

        # 1. 访问当前状态
        message_count = len(state["messages"])
        print(f"[before_agent] 开始执行，消息数: {message_count}")

        # 2. 访问 runtime 属性
        if runtime.store:
            # 从 store 加载用户数据
            user_data = runtime.store.mget(["user:preferences"])[0]
            print(f"[before_agent] 用户偏好: {user_data}")

        if runtime.context:
            # 访问用户上下文
            user_name = runtime.context.get("user_name", "未知")
            print(f"[before_agent] 用户: {user_name}")

        # 3. 返回状态更新
        return {
            "start_time": time.time(),  # 添加新字段（需要在 state_schema 中定义）
        }


# 带条件跳转的版本
class AuthMiddleware(AgentMiddleware):

    @hook_config(can_jump_to=["end"])  # 声明可以跳转到 end
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """检查用户权限"""

        if not self._check_auth(runtime.context):
            # 权限不足，直接结束
            return {
                "jump_to": "end",
                "messages": [AIMessage(content="抱歉，您没有权限使用此功能。")]
            }
        return None
```

---

## 钩子 2: `before_model` / `abefore_model`

### 触发时机
**每次调用 LLM 模型前（可能执行多次，因为 agent 循环）**

### 函数签名

```python
def before_model(
    self,
    state: StateT,
    runtime: Runtime[ContextT]
) -> dict[str, Any] | None
```

### 参数（与 before_agent 相同）

- `state`: 当前 Agent 状态（包含最新消息）
- `runtime`: LangGraph 运行时

### 返回值

| 返回值类型 | 含义 |
|-----------|------|
| `None` | 继续调用模型 |
| `dict[str, Any]` | 状态更新后继续调用模型 |
| `{"jump_to": "end"}` | **跳过模型调用**，直接结束 |
| `{"jump_to": "tools"}` | **跳过模型调用**，直接去工具节点 |

### 目的与用途

| 用途 | 示例 |
|------|------|
| **限制检查** | 检查调用次数是否超限 |
| **条件跳过** | 某些情况下跳过模型调用 |
| **状态准备** | 在模型调用前准备数据 |
| **日志/监控** | 记录每次模型调用 |

### 完整示例

```python
class ModelCallLimitMiddleware(AgentMiddleware[ModelCallLimitState, Any]):
    """限制模型调用次数"""

    state_schema = ModelCallLimitState  # 包含 thread_model_call_count 字段

    def __init__(self, *, thread_limit: int = 10, run_limit: int = 5):
        self.thread_limit = thread_limit
        self.run_limit = run_limit

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
        """检查是否超过调用限制"""

        # 获取当前计数
        thread_count = state.get("thread_model_call_count", 0)
        run_count = state.get("run_model_call_count", 0)

        print(f"[before_model] 线程调用: {thread_count}/{self.thread_limit}, "
              f"本次调用: {run_count}/{self.run_limit}")

        # 检查是否超限
        if thread_count >= self.thread_limit:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=f"已达到调用上限 ({thread_count} 次)")]
            }

        if run_count >= self.run_limit:
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=f"本次对话已达上限 ({run_count} 次)")]
            }

        return None  # 继续调用模型
```

---

## 钩子 3: `after_model` / `aafter_model`

### 触发时机
**每次 LLM 模型调用完成后（可能执行多次）**

### 函数签名

```python
def after_model(
    self,
    state: StateT,
    runtime: Runtime[ContextT]
) -> dict[str, Any] | None
```

### 参数

- `state`: 当前 Agent 状态（**已包含模型刚返回的 AIMessage**）
- `runtime`: LangGraph 运行时

**此时 state["messages"][-1] 是模型刚返回的 AIMessage！**

### 返回值

| 返回值类型 | 含义 |
|-----------|------|
| `None` | 继续流程（去工具节点或结束） |
| `dict[str, Any]` | 状态更新 |
| `{"jump_to": "end"}` | 强制结束（跳过工具调用） |
| `{"jump_to": "model"}` | 重新调用模型 |

### 目的与用途

| 用途 | 示例 |
|------|------|
| **计数更新** | 增加模型调用计数 |
| **响应处理** | 分析/记录模型响应 |
| **条件控制** | 根据响应决定流程 |
| **缓存** | 缓存模型响应 |

### 完整示例

```python
class CallCounterMiddleware(AgentMiddleware[ModelCallLimitState, Any]):
    """计数模型调用次数"""

    state_schema = ModelCallLimitState

    def after_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
        """模型调用后增加计数"""

        # 获取模型刚返回的消息
        last_message = state["messages"][-1]
        print(f"[after_model] 模型响应: {last_message.content[:50]}...")

        # 检查是否有 tool_calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print(f"[after_model] 模型请求调用 {len(last_message.tool_calls)} 个工具")

        # 更新计数
        return {
            "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
            "run_model_call_count": state.get("run_model_call_count", 0) + 1,
        }


class ResponseAnalyzerMiddleware(AgentMiddleware):
    """分析模型响应"""

    @hook_config(can_jump_to=["end", "model"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]

        # 检查是否包含敏感内容
        if self._contains_sensitive_content(last_message.content):
            # 强制结束，返回安全消息
            return {
                "jump_to": "end",
                "messages": [AIMessage(content="抱歉，我无法回答这个问题。")]
            }

        # 检查是否需要重试
        if self._is_low_quality_response(last_message.content):
            return {"jump_to": "model"}  # 重新调用模型

        return None
```

---

## 钩子 4: `after_agent` / `aafter_agent`

### 触发时机
**Agent 执行完成后，只执行一次**

### 函数签名

```python
def after_agent(
    self,
    state: StateT,
    runtime: Runtime[ContextT]
) -> dict[str, Any] | None
```

### 参数

- `state`: **最终状态**（包含所有消息和工具调用结果）
- `runtime`: LangGraph 运行时

### 返回值

| 返回值类型 | 含义 |
|-----------|------|
| `None` | 结束 |
| `dict[str, Any]` | 最终状态更新 |
| `{"jump_to": "model"}` | 重新开始循环（谨慎使用！） |

### 目的与用途

| 用途 | 示例 |
|------|------|
| **清理资源** | 关闭连接、释放资源 |
| **保存数据** | 将结果写入 store |
| **统计汇总** | 计算总耗时、token 用量 |
| **日志记录** | 记录完成信息 |

### 完整示例

```python
class CleanupMiddleware(AgentMiddleware):

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 执行完成后清理"""

        # 1. 统计信息
        total_messages = len(state["messages"])
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

        print(f"[after_agent] 完成！总消息: {total_messages}, "
              f"AI 响应: {len(ai_messages)}, 工具调用: {len(tool_messages)}")

        # 2. 保存到 store
        if runtime.store:
            runtime.store.mset([
                ("stats:last_run", {
                    "total_messages": total_messages,
                    "ai_responses": len(ai_messages),
                    "tool_calls": len(tool_messages),
                })
            ])

        # 3. 计算耗时（假设 before_agent 设置了 start_time）
        if "start_time" in state:
            elapsed = time.time() - state["start_time"]
            print(f"[after_agent] 总耗时: {elapsed:.2f}s")

        return None
```

---

## 钩子 5: `wrap_model_call` / `awrap_model_call`

### 触发时机
**拦截每次模型调用（洋葱模型，最强大的钩子）**

### 函数签名

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelCallResult  # ModelResponse | AIMessage
```

### 参数详解

#### `request: ModelRequest`

```python
@dataclass
class ModelRequest:
    """模型请求信息"""

    model: BaseChatModel
    # 要使用的 LLM 模型实例

    messages: list[AnyMessage]
    # 消息列表（不包含 system message）

    system_message: SystemMessage | None
    # 系统提示词消息

    tool_choice: Any | None
    # 工具选择配置 (如 "auto", "none", {"type": "function", "function": {"name": "xxx"}})

    tools: list[BaseTool | dict]
    # 可用的工具列表

    response_format: ResponseFormat | None
    # 结构化输出配置

    state: AgentState
    # 当前 Agent 状态（只读引用）

    runtime: Runtime[ContextT]
    # LangGraph 运行时

    model_settings: dict[str, Any]
    # 额外的模型设置
```

**request 数据示例：**

```python
request.model          # ChatOpenAI(model="gpt-4o")
request.system_message # SystemMessage(content="你是一个助手...")
request.messages       # [HumanMessage("搜索 AI"), AIMessage("好的"), ToolMessage(...)]
request.tools          # [<BaseTool: search>, <BaseTool: calculator>]
request.tool_choice    # "auto"
request.state          # {"messages": [...], "user_id": "123"}
request.runtime.store  # <InMemoryStore>
request.runtime.context # {"user_name": "张三"}
```

**修改 request（使用 override 方法）：**

```python
# 修改系统提示
new_request = request.override(
    system_message=SystemMessage(content="新的系统提示")
)

# 修改模型
new_request = request.override(
    model=ChatOpenAI(model="gpt-4o-mini")
)

# 修改工具列表
new_request = request.override(
    tools=[tool1, tool2]
)

# 同时修改多个
new_request = request.override(
    system_message=SystemMessage(content="..."),
    model=fallback_model,
    tool_choice="none"
)
```

#### `handler: Callable[[ModelRequest], ModelResponse]`

```python
# handler 是下一层（内层中间件或实际模型调用）
# 调用 handler(request) 会执行模型调用并返回 ModelResponse

response = handler(request)  # 执行模型调用
response = handler(modified_request)  # 用修改后的 request 执行

# handler 可以多次调用（重试场景）
for attempt in range(3):
    try:
        return handler(request)
    except Exception:
        if attempt == 2:
            raise
```

### 返回值

#### `ModelResponse`

```python
@dataclass
class ModelResponse:
    """模型响应"""

    result: list[BaseMessage]
    # 通常是 [AIMessage(...)]
    # 如果使用工具做结构化输出，可能是 [AIMessage(...), ToolMessage(...)]

    structured_response: Any = None
    # 解析后的结构化输出（如果指定了 response_format）
```

#### `AIMessage`（简写形式）

```python
# 可以直接返回 AIMessage，框架会自动转换为 ModelResponse
return AIMessage(content="响应内容")
# 等价于
return ModelResponse(result=[AIMessage(content="响应内容")])
```

### 执行顺序（洋葱模型）

```
middleware=[A, B, C]  # A 最外层，C 最内层

执行顺序：
A.wrap_model_call 开始
  ├── A 的前置逻辑
  ├── 调用 handler (实际是 B)
  │   └── B.wrap_model_call 开始
  │       ├── B 的前置逻辑
  │       ├── 调用 handler (实际是 C)
  │       │   └── C.wrap_model_call 开始
  │       │       ├── C 的前置逻辑
  │       │       ├── 调用 handler (实际的模型调用)
  │       │       │   └── 模型执行，返回 ModelResponse
  │       │       └── C 的后置逻辑
  │       │       └── C 返回
  │       └── B 的后置逻辑
  │       └── B 返回
  └── A 的后置逻辑
  └── A 返回
```

### 目的与用途

| 用途 | 能力 |
|------|------|
| **修改请求** | 改 prompt、改 model、改 tools |
| **修改响应** | 重写、过滤、增强响应 |
| **重试逻辑** | 失败时多次调用 handler |
| **模型切换** | 主模型失败，切换备用模型 |
| **缓存** | 缓存命中时跳过 handler |
| **短路** | 直接返回，不调用 handler |
| **监控** | 记录耗时、token 用量 |

### 完整示例

```python
from langchain.agents import AgentMiddleware, ModelRequest, ModelResponse

class RetryMiddleware(AgentMiddleware):
    """重试逻辑"""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                print(f"[wrap_model_call] 尝试 {attempt+1} 失败: {e}")
                if attempt == 2:
                    # 最后一次失败，返回错误消息
                    return ModelResponse(
                        result=[AIMessage(content="服务暂时不可用，请稍后重试")]
                    )


class FallbackMiddleware(AgentMiddleware):
    """模型降级"""

    def __init__(self, fallback_model):
        self.fallback_model = fallback_model

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        try:
            return handler(request)
        except Exception as e:
            print(f"[wrap_model_call] 主模型失败，切换备用: {e}")
            # 使用备用模型
            fallback_request = request.override(model=self.fallback_model)
            return handler(fallback_request)


class CacheMiddleware(AgentMiddleware):
    """缓存响应"""

    def __init__(self):
        self.cache = {}

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        # 生成缓存键
        cache_key = self._make_key(request)

        # 缓存命中
        if cache_key in self.cache:
            print("[wrap_model_call] 缓存命中！")
            return self.cache[cache_key]  # 短路：不调用 handler

        # 缓存未命中，调用模型
        response = handler(request)
        self.cache[cache_key] = response
        return response


class ResponseRewriterMiddleware(AgentMiddleware):
    """重写响应"""

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        response = handler(request)

        # 获取原始响应
        ai_message = response.result[0]
        original_content = ai_message.content

        # 重写响应
        new_content = f"[经过处理] {original_content}"

        return ModelResponse(
            result=[AIMessage(content=new_content, tool_calls=ai_message.tool_calls)],
            structured_response=response.structured_response
        )
```

---

## 钩子 6: `wrap_tool_call` / `awrap_tool_call`

### 触发时机
**拦截每次工具调用（洋葱模型）**

### 函数签名

```python
def wrap_tool_call(
    self,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command
```

### 参数详解

#### `request: ToolCallRequest`

```python
# ToolCallRequest 来自 langgraph.prebuilt.tool_node
@dataclass
class ToolCallRequest:
    tool_call: dict
    # 工具调用信息
    # {"name": "search", "args": {"query": "AI"}, "id": "call_123", "type": "tool_call"}

    tool: BaseTool | None
    # 工具实例

    state: dict
    # 当前 Agent 状态

    runtime: ToolRuntime | None
    # 工具运行时（包含 store, context, config 等）
```

**request 数据示例：**

```python
request.tool_call      # {"name": "search", "args": {"query": "AI 新闻"}, "id": "call_abc123"}
request.tool           # <BaseTool: search>
request.tool.name      # "search"
request.tool.description  # "搜索互联网"
request.state          # {"messages": [...], "user_id": "123"}
request.runtime.store  # <InMemoryStore>
request.runtime.context # {"user_name": "张三"}
request.runtime.tool_call_id  # "call_abc123"
```

**修改 request：**

```python
# 修改工具调用参数
modified_call = {
    **request.tool_call,
    "args": {
        **request.tool_call["args"],
        "query": request.tool_call["args"]["query"] + " 2024"  # 添加年份
    }
}
new_request = request.override(tool_call=modified_call)
```

#### `handler`

```python
# 调用 handler 执行工具
result = handler(request)  # ToolMessage 或 Command
```

### 返回值

#### `ToolMessage`

```python
ToolMessage(
    content="搜索结果: ...",    # 工具执行结果
    tool_call_id="call_123",   # 必须与 request.tool_call["id"] 匹配
    name="search",             # 工具名
    status="success",          # "success" | "error"
)
```

#### `Command`

```python
from langgraph.types import Command

# 用于控制流程
Command(goto="model")  # 跳转到模型节点
Command(goto="end")    # 结束
```

### 目的与用途

| 用途 | 能力 |
|------|------|
| **重试逻辑** | 工具调用失败时重试 |
| **参数修改** | 修改工具调用参数 |
| **结果修改** | 修改/增强工具结果 |
| **缓存** | 缓存工具调用结果 |
| **监控** | 记录工具调用耗时、结果 |
| **模拟/Mock** | 不实际调用工具，返回模拟数据 |

### 完整示例

```python
from langchain.agents import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

class ToolRetryMiddleware(AgentMiddleware):
    """工具调用重试"""

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        tool_name = request.tool.name if request.tool else request.tool_call["name"]
        tool_call_id = request.tool_call["id"]

        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                print(f"[wrap_tool_call] {tool_name} 尝试 {attempt+1} 失败: {e}")
                if attempt == 2:
                    return ToolMessage(
                        content=f"工具 {tool_name} 执行失败: {e}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error"
                    )


class ToolCacheMiddleware(AgentMiddleware):
    """工具调用缓存"""

    def __init__(self):
        self.cache = {}

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        # 生成缓存键
        cache_key = f"{request.tool_call['name']}:{request.tool_call['args']}"

        if cache_key in self.cache:
            print("[wrap_tool_call] 缓存命中！")
            cached = self.cache[cache_key]
            return ToolMessage(
                content=cached,
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"]
            )

        result = handler(request)
        if isinstance(result, ToolMessage) and result.status != "error":
            self.cache[cache_key] = result.content
        return result


class ToolEmulatorMiddleware(AgentMiddleware):
    """模拟工具调用（测试用）"""

    def __init__(self, tools_to_emulate: list[str]):
        self.tools_to_emulate = tools_to_emulate

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]

        if tool_name in self.tools_to_emulate:
            # 不实际调用，返回模拟数据
            return ToolMessage(
                content=f"[模拟] {tool_name} 执行成功",
                tool_call_id=request.tool_call["id"],
                name=tool_name
            )

        return handler(request)


class ToolMonitorMiddleware(AgentMiddleware):
    """监控工具调用"""

    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call["args"]

        print(f"[wrap_tool_call] 开始调用: {tool_name}")
        print(f"[wrap_tool_call] 参数: {tool_args}")

        start_time = time.time()
        result = handler(request)
        elapsed = time.time() - start_time

        print(f"[wrap_tool_call] 完成: {tool_name}, 耗时: {elapsed:.2f}s")

        # 保存到 store
        if request.runtime and request.runtime.store:
            request.runtime.store.mset([
                (f"tool_stats:{tool_name}", {
                    "last_call": time.time(),
                    "duration": elapsed,
                    "success": isinstance(result, ToolMessage) and result.status != "error"
                })
            ])

        return result
```

---

## 钩子 7: `dynamic_prompt` (便捷装饰器)

### 触发时机
**每次模型调用前，动态生成系统提示**

### 函数签名

```python
@dynamic_prompt
def my_prompt(request: ModelRequest) -> str | SystemMessage:
    ...
```

### 参数

- `request: ModelRequest`：同 `wrap_model_call` 的 request

### 返回值

| 类型 | 说明 |
|------|------|
| `str` | 系统提示文本 |
| `SystemMessage` | 完整的系统消息（可包含 metadata） |

### 完整示例

```python
from langchain.agents import dynamic_prompt, ModelRequest
from langchain_core.messages import SystemMessage

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据上下文生成个性化提示"""

    # 访问用户上下文
    user_name = "用户"
    if request.runtime and request.runtime.context:
        user_name = request.runtime.context.get("user_name", "用户")

    # 根据消息数量调整
    msg_count = len(request.state["messages"])

    if msg_count > 10:
        return f"你正在与 {user_name} 进行长对话，请简洁回复。"
    else:
        return f"你是 {user_name} 的助手，请热情友好地回复。"


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> SystemMessage:
    """返回带 metadata 的 SystemMessage"""

    # 构建复杂的系统消息
    blocks = [
        {"type": "text", "text": "你是一个专业助手。"},
        {"type": "text", "text": "当前时间: " + datetime.now().isoformat()},
    ]

    # 添加缓存控制（如果模型支持）
    if len(request.messages) > 5:
        blocks[0]["cache_control"] = {"type": "ephemeral"}

    return SystemMessage(content=blocks)


# 使用
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[personalized_prompt, context_aware_prompt]
)
```

---

## 总览对比表

| 钩子 | 触发时机 | 参数 | 返回值 | 执行方式 | 主要用途 |
|------|---------|------|--------|---------|---------|
| `before_agent` | Agent 开始前 | state, runtime | dict/None | 顺序执行 | 初始化、权限检查 |
| `before_model` | 每次模型调用前 | state, runtime | dict/None (可跳转) | 顺序执行 | 限制检查、条件跳过 |
| `after_model` | 每次模型调用后 | state, runtime | dict/None (可跳转) | 顺序执行 | 计数、响应分析 |
| `after_agent` | Agent 结束后 | state, runtime | dict/None | 顺序执行 | 清理、保存、统计 |
| `wrap_model_call` | 拦截模型调用 | request, handler | ModelResponse | **洋葱模型** | 重试、缓存、改写 |
| `wrap_tool_call` | 拦截工具调用 | request, handler | ToolMessage | **洋葱模型** | 重试、缓存、监控 |
| `dynamic_prompt` | 模型调用前 | request | str/SystemMessage | 洋葱模型 | 动态提示词 |
