
# LangChain Agent Context解析

## 一、Context 数据结构

```python
# 来源：langgraph/libs/langgraph/langgraph/typing.py
ContextT = TypeVar("ContextT", bound=StateLike | None, default=None)
```

**Context 可以是以下类型：**
1. **`dataclass`** - 最推荐的方式
2. **`Pydantic BaseModel`**
3. **`TypedDict`**
4. **普通字典 `dict`**
5. **`None`** - 默认值

```23:27:langgraph/libs/langgraph/langgraph/typing.py
ContextT = TypeVar("ContextT", bound=StateLike | None, default=None)
"""Type variable used to represent graph run scoped context.

Defaults to `None`.
"""
```

**在 Runtime 中的定义：**

```86:89:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/runtime.py
    context: ContextT = field(default=None)  # type: ignore[assignment]
    """Static context for the graph run, like `user_id`, `db_conn`, etc.

    Can also be thought of as 'run dependencies'."""
```

---

## 二、数据流程图

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                          Context 完整数据流                                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║   用户代码层                                                                   ║
║   ═════════                                                                    ║
║   ┌────────────────────────────────────────────────────────────────────────┐   ║
║   │ # 1. 定义 Context Schema                                               │   ║
║   │ @dataclass                                                             │   ║
║   │ class UserContext:                                                     │   ║
║   │     user_id: str                                                       │   ║
║   │     db_conn: Any                                                       │   ║
║   │     api_key: str                                                       │   ║
║   │                                                                        │   ║
║   │ # 2. 创建 Agent 时指定 context_schema                                  │   ║
║   │ agent = create_agent(                                                  │   ║
║   │     model="openai:gpt-4o",                                            │   ║
║   │     tools=[...],                                                       │   ║
║   │     context_schema=UserContext,  ← context 类型约束                    │   ║
║   │ )                                                                      │   ║
║   └────────────────────────────────────────────────────────────────────────┘   ║
║                                      │                                         ║
║                                      ▼                                         ║
║   ┌────────────────────────────────────────────────────────────────────────┐   ║
║   │ # 3. 调用 invoke/stream 时传入 context                                 │   ║
║   │                                                                        │   ║
║   │ result = agent.invoke(                                                 │   ║
║   │     {"messages": [HumanMessage("Hi")]},                               │   ║
║   │     context=UserContext(       ← 传入点                                │   ║
║   │         user_id="user_123",                                           │   ║
║   │         db_conn=db,                                                   │   ║
║   │         api_key="sk-xxx"                                              │   ║
║   │     )                                                                  │   ║
║   │ )                                                                      │   ║
║   └────────────────────────────────────────────────────────────────────────┘   ║
║                                      │                                         ║
║                                      ▼                                         ║
║   LangGraph 内部层 (pregel/main.py)                                           ║
║   ════════════════════════════════                                            ║
║   ┌────────────────────────────────────────────────────────────────────────┐   ║
║   │ # 4. stream() 方法处理 context                                         │   ║
║   │                                                                        │   ║
║   │ # 4.1 类型转换 (_coerce_context)                                       │   ║
║   │ context = _coerce_context(self.context_schema, context)                │   ║
║   │                                                                        │   ║
║   │ # 4.2 创建 Runtime 对象                                                │   ║
║   │ runtime = Runtime(                                                     │   ║
║   │     context=context,            ← context 注入到 Runtime               │   ║
║   │     store=store,                                                       │   ║
║   │     stream_writer=stream_writer,                                       │   ║
║   │     previous=None,                                                     │   ║
║   │ )                                                                      │   ║
║   │                                                                        │   ║
║   │ # 4.3 合并父级 Runtime                                                 │   ║
║   │ parent_runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME) │   ║
║   │ runtime = parent_runtime.merge(runtime)                                │   ║
║   │                                                                        │   ║
║   │ # 4.4 存储到 config                                                    │   ║
║   │ config[CONF][CONFIG_KEY_RUNTIME] = runtime                             │   ║
║   └────────────────────────────────────────────────────────────────────────┘   ║
║                       │                                                        ║
║                       ▼                                                        ║
║   ┌────────────────────────────────────────────────────────────────────────┐   ║
║   │ # 5. 节点执行时从 config 获取 Runtime (pregel/_algo.py)                │   ║
║   │                                                                        │   ║
║   │ runtime = config[CONF].get(CONFIG_KEY_RUNTIME, DEFAULT_RUNTIME)        │   ║
║   │ runtime = runtime.override(store=store, previous=...)                  │   ║
║   └────────────────────────────────────────────────────────────────────────┘   ║
║                       │                                                        ║
║       ┌───────────────┼───────────────┬─────────────────────┐                 ║
║       ▼               ▼               ▼                     ▼                 ║
║  ┌────────────┐ ┌────────────┐ ┌────────────────┐ ┌──────────────────┐        ║
║  │ model_node │ │ Middleware │ │   ToolNode     │ │   User Tools     │        ║
║  │            │ │            │ │                │ │                  │        ║
║  │ runtime.   │ │ runtime.   │ │ 注入 runtime   │ │ runtime.context. │        ║
║  │ context    │ │ context.   │ │ 到工具         │ │ user_id          │        ║
║  │ 可访问     │ │ user_id    │ │                │ │                  │        ║
║  └────────────┘ └────────────┘ └────────────────┘ └──────────────────┘        ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## 三、传入时机

**Context 在 `invoke()` / `stream()` / `ainvoke()` / `astream()` 调用时传入：**

```3021:3041:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/pregel/main.py
    def invoke(
        self,
        input: InputT | Command | None,
        config: RunnableConfig | None = None,
        *,
        context: ContextT | None = None,  # ← 这里传入 context
        stream_mode: StreamMode = "values",
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Any:
        """Run the graph with a single input and config.

        Args:
            input: The input data for the graph. It can be a dictionary or any other type.
            config: The configuration for the graph run.
            context: The static context to use for the run.
                !!! version-added "Added in version 0.6.0"
```

**类型转换处理：**

```3292:3319:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/pregel/main.py
def _coerce_context(
    context_schema: type[ContextT] | None, context: Any
) -> ContextT | None:
    """Coerce context input to the appropriate schema type.

    If context is a dict and context_schema is a dataclass or pydantic model, we coerce.
    Else, we return the context as-is.

    Args:
        context_schema: The schema type to coerce to (BaseModel, dataclass, or TypedDict)
        context: The context value to coerce

    Returns:
        The coerced context value or None if context is None
    """
    if context is None:
        return None

    if context_schema is None:
        return context

    schema_is_class = issubclass(context_schema, BaseModel) or is_dataclass(
        context_schema
    )
    if isinstance(context, dict) and schema_is_class:
        return context_schema(**context)  # type: ignore[misc]

    return cast(ContextT, context)
```

---

## 四、使用方式

### 4.1 定义 Context Schema

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class UserContext:
    """运行时上下文 - 存储本次运行的静态依赖"""
    user_id: str
    db_conn: Any = None
    api_key: str = ""
    region: str = "CN"
```

### 4.2 在 create_agent 中指定

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tool],
    middleware=[MyMiddleware()],
    context_schema=UserContext,  # ← 指定 context 类型
)
```

### 4.3 调用时传入 context

```python
result = agent.invoke(
    {"messages": [HumanMessage("请帮我查询信息")]},
    context=UserContext(
        user_id="user_123",
        db_conn=database_connection,
        api_key="sk-xxx",
        region="US"
    )
)
```

### 4.4 在中间件中使用

```python
from langchain.agents import AgentMiddleware
from langgraph.runtime import Runtime

class RegionMiddleware(AgentMiddleware):
    """根据 context.region 调整行为的中间件"""

    def before_model(self, state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
        # 通过 runtime.context 访问上下文
        region = runtime.context.region
        user_id = runtime.context.user_id

        print(f"用户 {user_id} 来自 {region}")

        # 可以根据 region 动态调整行为
        if region == "EU":
            # GDPR 相关处理
            pass

        return None
```

### 4.5 在工具中使用

```python
from langchain.tools import tool, ToolRuntime

@tool
def query_user_data(query: str, runtime: ToolRuntime) -> str:
    """查询用户数据"""
    # 通过 runtime.context 访问上下文
    user_id = runtime.context.user_id
    db = runtime.context.db_conn

    # 使用 user_id 进行数据查询
    result = db.query(f"SELECT * FROM data WHERE user_id = '{user_id}' AND {query}")
    return str(result)
```

---

## 五、Context 的作用

| 作用 | 说明 |
|------|------|
| **运行时依赖注入** | 将数据库连接、API 客户端等依赖注入到节点和工具中 |
| **用户身份标识** | 传递 `user_id`、`session_id` 等用户标识信息 |
| **配置参数传递** | 传递 API Key、区域设置、功能开关等运行时配置 |
| **跨节点数据共享** | 在同一次运行中，所有节点共享相同的静态上下文 |
| **与 Store 区分** | Context 是**运行级别**的静态数据，Store 是**持久化**的数据存储 |

**Context vs Store 对比：**

```
┌──────────────────────┬───────────────────────────────────────────────────┐
│       Context        │                    Store                          │
├──────────────────────┼───────────────────────────────────────────────────┤
│ 运行级别 (per-run)   │ 持久化级别 (跨 thread/用户)                       │
│ 静态、只读           │ 可读写                                            │
│ 用于依赖注入         │ 用于记忆存储                                      │
│ 在 invoke 时传入     │ 在 create_agent 时配置                            │
│ 每次调用可以不同     │ 所有调用共享同一实例                              │
└──────────────────────┴───────────────────────────────────────────────────┘
```

---

## 六、完整使用示例

```python
from dataclasses import dataclass
from typing import Any, Annotated
from langchain.agents import create_agent, AgentMiddleware, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


# 1. 定义 Context 结构
@dataclass
class AppContext:
    user_id: str
    api_key: str
    region: str = "CN"


# 2. 中间件使用 context
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime[AppContext]) -> dict | None:
        # 访问 context
        print(f"[LOG] User: {runtime.context.user_id}, Region: {runtime.context.region}")
        return None


# 3. 工具使用 context
@tool
def call_external_api(query: str, runtime: ToolRuntime) -> str:
    """调用外部 API"""
    api_key = runtime.context.api_key
    user_id = runtime.context.user_id
    # 使用 api_key 调用外部服务...
    return f"API Response for {user_id}: {query}"


# 4. 创建 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[call_external_api],
    middleware=[LoggingMiddleware()],
    context_schema=AppContext,
    store=InMemoryStore(),
)


# 5. 调用时传入 context
result = agent.invoke(
    {"messages": [HumanMessage("帮我查询天气")]},
    context=AppContext(
        user_id="user_456",
        api_key="sk-secret-key",
        region="US"
    )
)
```

`context` 本质上是 LangGraph 提供的**运行级别依赖注入机制**，用于在单次 agent 执行过程中向所有节点传递静态的上下文信息。
