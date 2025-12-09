
# LangChain v1 Agent 长期记忆 (Store) 处理机制完全解析

## 一、什么是 Store（长期记忆）？

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                        Store vs Checkpointer 核心区别                                 ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   ┌─────────────────────────────────┬───────────────────────────────────────────────┐ ║
║   │     Checkpointer (短期记忆)      │           Store (长期记忆)                    │ ║
║   ├─────────────────────────────────┼───────────────────────────────────────────────┤ ║
║   │                                 │                                               │ ║
║   │  作用域：单个 thread_id         │  作用域：跨所有 thread_id                      │ ║
║   │  (单次对话)                     │  (多个对话/多个用户)                          │ ║
║   │                                 │                                               │ ║
║   │  存储内容：                      │  存储内容：                                   │ ║
║   │  - state["messages"]            │  - 用户配置/偏好                              │ ║
║   │  - 对话历史                      │  - 共享知识库                                 │ ║
║   │  - 中间状态                      │  - 缓存数据                                   │ ║
║   │                                 │  - 跨会话统计                                 │ ║
║   │                                 │                                               │ ║
║   │  访问方式：自动                  │  访问方式：手动                               │ ║
║   │  - invoke 时自动加载/保存        │  - 通过 runtime.store 访问                   │ ║
║   │                                 │  - 通过 InjectedStore 注入                    │ ║
║   │                                 │                                               │ ║
║   │  生命周期：                      │  生命周期：                                   │ ║
║   │  - 与 thread_id 绑定            │  - 独立于 thread_id                          │ ║
║   │  - 每个对话独立                  │  - 全局共享                                  │ ║
║   │                                 │                                               │ ║
║   └─────────────────────────────────┴───────────────────────────────────────────────┘ ║
║                                                                                      ║
║   示例场景：                                                                          ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │   用户 A 的对话 1 (thread_id="A-1")                                             │ ║
║   │   用户 A 的对话 2 (thread_id="A-2")   ──┬──▶  共享 Store（用户 A 的偏好）        │ ║
║   │   用户 B 的对话 1 (thread_id="B-1")   ──┘                                       │ ║
║   │                                                                                 │ ║
║   │   每个 thread 有独立的 checkpointer 状态                                         │ ║
║   │   但所有 thread 共享同一个 store                                                 │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 二、Store 存在哪里？

### 2.1 存储架构总览

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                              Store 存储架构                                          ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │                        用户代码层                                               │ ║
║   │   ┌───────────────────────────────────────────────────────────────────────────┐ │ ║
║   │   │  store = InMemoryStore()    # 用户创建 store 实例                         │ │ ║
║   │   │                             # Store 实际数据存储在这里                     │ │ ║
║   │   │  store.store = {            # InMemoryStore 内部字典                       │ │ ║
║   │   │      "user:123:name": "张三",                                             │ │ ║
║   │   │      "user:123:prefs": {"theme": "dark"},                                 │ │ ║
║   │   │      "cache:weather:beijing": "晴天",                                     │ │ ║
║   │   │  }                                                                        │ │ ║
║   │   └───────────────────────────────────────────────────────────────────────────┘ │ ║
║   │                                      │                                          │ ║
║   │                                      ▼                                          │ ║
║   │   ┌───────────────────────────────────────────────────────────────────────────┐ │ ║
║   │   │  agent = create_agent(                                                    │ │ ║
║   │   │      model="openai:gpt-4o",                                               │ │ ║
║   │   │      tools=[...],                                                         │ │ ║
║   │   │      store=store,           # ← store 引用传入                            │ │ ║
║   │   │  )                                                                        │ │ ║
║   │   └───────────────────────────────────────────────────────────────────────────┘ │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                              ║
║                                      ▼                                              ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │                     LangGraph 框架层                                            │ ║
║   │   ┌───────────────────────────────────────────────────────────────────────────┐ │ ║
║   │   │  graph.compile(                                                           │ │ ║
║   │   │      store=store,           # ← store 注册到编译后的图                     │ │ ║
║   │   │      checkpointer=...,                                                    │ │ ║
║   │   │  )                                                                        │ │ ║
║   │   │                                                                           │ │ ║
║   │   │  # 编译后的图持有 store 引用                                               │ │ ║
║   │   │  compiled_graph._store = store                                            │ │ ║
║   │   └───────────────────────────────────────────────────────────────────────────┘ │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                              ║
║                                      ▼                                              ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │                     运行时层 (Runtime)                                          │ ║
║   │   ┌───────────────────────────────────────────────────────────────────────────┐ │ ║
║   │   │  # 每次节点执行时，LangGraph 创建 Runtime 对象                             │ │ ║
║   │   │  runtime = Runtime(                                                       │ │ ║
║   │   │      store=compiled_graph._store,  # ← store 注入到 runtime               │ │ ║
║   │   │      context=...,                                                         │ │ ║
║   │   │      config=...,                                                          │ │ ║
║   │   │  )                                                                        │ │ ║
║   │   │                                                                           │ │ ║
║   │   │  # runtime 被传递给：                                                      │ │ ║
║   │   │  # 1. 中间件钩子 (before_model, after_model, etc.)                        │ │ ║
║   │   │  # 2. model_node / tool_node                                              │ │ ║
║   │   │  # 3. 工具函数 (通过 ToolRuntime)                                          │ │ ║
║   │   └───────────────────────────────────────────────────────────────────────────┘ │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 2.2 Store 的类型

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                        两种 Store 类型体系                                            ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   1. langchain_core.stores.BaseStore (简单键值存储)                                  ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │   from langchain_core.stores import InMemoryStore                               │ ║
║   │                                                                                 │ ║
║   │   store = InMemoryStore()                                                       │ ║
║   │   store.mset([("key1", "value1"), ("key2", {"data": 123})])                     │ ║
║   │   values = store.mget(["key1", "key2"])  # ["value1", {"data": 123}]            │ ║
║   │                                                                                 │ ║
║   │   特点：                                                                         │ ║
║   │   - 简单的 key-value 存储                                                       │ ║
║   │   - key 是字符串                                                                │ ║
║   │   - value 可以是任意类型                                                        │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                      ║
║   2. langgraph.store.base.BaseStore (命名空间存储)                                   ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │   from langgraph.store.memory import InMemoryStore                              │ ║
║   │                                                                                 │ ║
║   │   store = InMemoryStore()                                                       │ ║
║   │   store.put(("users", "123"), "profile", {"name": "张三"})                      │ ║
║   │   item = store.get(("users", "123"), "profile")                                 │ ║
║   │   # Item(value={"name": "张三"}, key="profile", namespace=("users", "123"))     │ ║
║   │                                                                                 │ ║
║   │   特点：                                                                         │ ║
║   │   - 支持命名空间 (namespace) 层级结构                                           │ ║
║   │   - 返回 Item 对象，包含元数据                                                  │ ║
║   │   - 更适合复杂的数据组织                                                        │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                      ║
║   在 create_agent 中使用的是 langgraph.store.base.BaseStore                          ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 2.3 源码引用：BaseStore 接口定义

来自 `libs/core/langchain_core/stores.py`:

```26:90:libs/core/langchain_core/stores.py
class BaseStore(ABC, Generic[K, V]):
    """Abstract interface for a key-value store."""

    @abstractmethod
    def mget(self, keys: Sequence[K]) -> list[V | None]:
        """Get the values associated with the given keys."""

    async def amget(self, keys: Sequence[K]) -> list[V | None]:
        """Async get the values associated with the given keys."""
        return await run_in_executor(None, self.mget, keys)

    @abstractmethod
    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """Set the values for the given keys."""

    async def amset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """Async set the values for the given keys."""
        return await run_in_executor(None, self.mset, key_value_pairs)

    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """Delete the given keys and their associated values."""

    async def amdelete(self, keys: Sequence[K]) -> None:
        """Async delete the given keys and their associated values."""
        return await run_in_executor(None, self.mdelete, keys)

    @abstractmethod
    def yield_keys(self, *, prefix: str | None = None) -> Iterator[K]:
        """Get an iterator over keys that match the given prefix."""
```

---

## 三、Store 什么时候传入？

### 3.1 完整的 Store 传入流程

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                         Store 传入的 4 个阶段                                         ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │  阶段 1：用户创建阶段                                                            │ ║
║   │  ═══════════════════════════════════════════════════════════════════════════    │ ║
║   │                                                                                 │ ║
║   │  from langgraph.store.memory import InMemoryStore                               │ ║
║   │                                                                                 │ ║
║   │  store = InMemoryStore()    # ← Store 实例在用户代码中创建                      │ ║
║   │                                                                                 │ ║
║   │  # 可选：预加载数据                                                              │ ║
║   │  store.put(("users",), "123", {"name": "张三", "level": "vip"})                 │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                              ║
║                                      ▼                                              ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │  阶段 2：Agent 创建阶段                                                          │ ║
║   │  ═══════════════════════════════════════════════════════════════════════════    │ ║
║   │                                                                                 │ ║
║   │  agent = create_agent(                                                          │ ║
║   │      model="openai:gpt-4o",                                                     │ ║
║   │      tools=[my_tool],                                                           │ ║
║   │      store=store,           # ← Store 传入 create_agent                         │ ║
║   │  )                                                                              │ ║
║   │                                                                                 │ ║
║   │  # 源码位置: factory.py 第 551 行                                                │ ║
║   │  def create_agent(                                                              │ ║
║   │      ...                                                                        │ ║
║   │      store: BaseStore | None = None,  # ← 参数定义                              │ ║
║   │      ...                                                                        │ ║
║   │  )                                                                              │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                              ║
║                                      ▼                                              ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │  阶段 3：图编译阶段                                                              │ ║
║   │  ═══════════════════════════════════════════════════════════════════════════    │ ║
║   │                                                                                 │ ║
║   │  # 源码位置: factory.py 第 1471-1479 行                                          │ ║
║   │  return graph.compile(                                                          │ ║
║   │      checkpointer=checkpointer,                                                 │ ║
║   │      store=store,           # ← Store 注册到编译后的图                           │ ║
║   │      interrupt_before=interrupt_before,                                         │ ║
║   │      interrupt_after=interrupt_after,                                           │ ║
║   │      debug=debug,                                                               │ ║
║   │      name=name,                                                                 │ ║
║   │      cache=cache,                                                               │ ║
║   │  )                                                                              │ ║
║   │                                                                                 │ ║
║   │  # LangGraph 内部：                                                              │ ║
║   │  # compiled_graph._store = store                                                │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                      │                                              ║
║                                      ▼                                              ║
║   ┌─────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                 │ ║
║   │  阶段 4：运行时注入阶段                                                          │ ║
║   │  ═══════════════════════════════════════════════════════════════════════════    │ ║
║   │                                                                                 │ ║
║   │  # 当调用 agent.invoke() 时，LangGraph 框架自动：                                │ ║
║   │                                                                                 │ ║
║   │  1. 创建 Runtime 对象                                                           │ ║
║   │     runtime = Runtime(                                                          │ ║
║   │         store=compiled_graph._store,    # ← store 注入                          │ ║
║   │         context=user_context,                                                   │ ║
║   │         config=runnable_config,                                                 │ ║
║   │     )                                                                           │ ║
║   │                                                                                 │ ║
║   │  2. 传递给每个节点                                                               │ ║
║   │     def model_node(state: AgentState, runtime: Runtime):                        │ ║
║   │         # runtime.store 可用                                                    │ ║
║   │                                                                                 │ ║
║   │  3. ToolNode 注入到工具                                                         │ ║
║   │     # 检测 InjectedStore 注解，将 runtime.store 传入工具参数                    │ ║
║   │                                                                                 │ ║
║   └─────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 3.2 源码引用：create_agent 中的 store 参数

```541:632:libs/langchain_v1/langchain/agents/factory.py
def create_agent(  # noqa: PLR0915
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    ...
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,      # ← Store 参数定义
    ...
):
    """Creates an agent graph that calls tools in a loop...

        store: An optional store object.

            Used for persisting data across multiple threads (e.g., multiple
            conversations / users).
    """
```

### 3.3 源码引用：graph.compile 传入 store

```1471:1479:libs/langchain_v1/langchain/agents/factory.py
    return graph.compile(
        checkpointer=checkpointer,
        store=store,                      # ← Store 传入编译
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 10_000})
```

---

## 四、用户可用的接口

### 4.1 Store 可以使用的 4 个位置

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          Store 使用位置总览                                           ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐              ║
║   │                   │   │                   │   │                   │              ║
║   │   1. Tools        │   │   2. Middleware   │   │   3. Agent 外部   │              ║
║   │      (工具)       │   │      (中间件)     │   │      (直接访问)   │              ║
║   │                   │   │                   │   │                   │              ║
║   │ InjectedStore()   │   │  runtime.store    │   │  store.mget()     │              ║
║   │ ToolRuntime.store │   │                   │   │  store.mset()     │              ║
║   │                   │   │                   │   │                   │              ║
║   └───────────────────┘   └───────────────────┘   └───────────────────┘              ║
║                                                                                      ║
║   ┌───────────────────┐                                                              ║
║   │                   │                                                              ║
║   │ 4. wrap_tool_call │                                                              ║
║   │    (工具拦截)     │                                                              ║
║   │                   │                                                              ║
║   │ request.runtime   │                                                              ║
║   │    .store         │                                                              ║
║   │                   │                                                              ║
║   └───────────────────┘                                                              ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 4.2 位置 1：在 Tools 中使用

```python
from typing import Annotated, Any
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.memory import InMemoryStore

# ========== 方式 A: 使用 InjectedStore 注解 ==========
@tool
def save_user_preference(
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore()]  # ← 注入 store
) -> str:
    """保存用户偏好设置"""
    store.put(("preferences",), key, {"value": value})
    return f"已保存 {key}={value}"


@tool
def get_user_preference(
    key: str,
    store: Annotated[Any, InjectedStore()]  # ← 注入 store
) -> str:
    """获取用户偏好设置"""
    item = store.get(("preferences",), key)
    if item:
        return f"{key}={item.value['value']}"
    return f"{key} 不存在"


# ========== 方式 B: 使用 ToolRuntime (包含 store) ==========
@tool
def tool_with_runtime(query: str, runtime: ToolRuntime) -> str:
    """通过 ToolRuntime 访问 store"""
    if runtime.store:
        # runtime.store 是同一个 store 实例
        runtime.store.put(("cache",), "last_query", {"query": query})
    return f"处理了查询: {query}"


# ========== 创建 Agent ==========
store = InMemoryStore()
agent = create_agent(
    model="openai:gpt-4o",
    tools=[save_user_preference, get_user_preference, tool_with_runtime],
    store=store,  # ← 传入 store
)
```

### 4.3 位置 2：在 Middleware 中使用

```python
from langchain.agents import AgentMiddleware, AgentState, before_model, after_agent
from langgraph.runtime import Runtime

# ========== 方式 A: 类方式中间件 ==========
class UserPreferenceMiddleware(AgentMiddleware):
    """在中间件中通过 runtime.store 访问 store"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 开始前加载用户数据"""
        if runtime.store:
            # 从 store 读取用户信息
            user_item = runtime.store.get(("users",), "current_user")
            if user_item:
                print(f"[Middleware] 欢迎 {user_item.value.get('name')}!")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """每次模型调用前"""
        if runtime.store:
            # 记录调用统计
            stats_item = runtime.store.get(("stats",), "model_calls")
            count = stats_item.value.get("count", 0) if stats_item else 0
            runtime.store.put(("stats",), "model_calls", {"count": count + 1})
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """每次模型调用后缓存响应"""
        if runtime.store:
            last_msg = state["messages"][-1] if state["messages"] else None
            if last_msg:
                runtime.store.put(("cache",), "last_response", {"content": str(last_msg)})
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 完成后保存会话摘要"""
        if runtime.store:
            runtime.store.put(("sessions",), "summary", {
                "message_count": len(state["messages"]),
                "completed": True
            })
        return None


# ========== 方式 B: 装饰器方式中间件 ==========
@before_model
def log_model_call(state: AgentState, runtime: Runtime) -> None:
    """装饰器方式访问 store"""
    if runtime.store:
        runtime.store.put(("logs",), "last_input", {
            "message": str(state["messages"][-1]) if state["messages"] else None
        })


@after_agent
def save_completion(state: AgentState, runtime: Runtime) -> None:
    """Agent 完成后保存"""
    if runtime.store:
        runtime.store.put(("completions",), "latest", {
            "total_messages": len(state["messages"])
        })
```

### 4.4 位置 3：在 Agent 外部直接使用

```python
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 创建 store
store = InMemoryStore()

# ========== Agent 执行前：预加载数据 ==========
store.put(("users",), "123", {"name": "张三", "level": "vip"})
store.put(("settings",), "global", {"max_tokens": 1000, "temperature": 0.7})

# 创建 agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    store=store,
)

# ========== Agent 执行 ==========
result = agent.invoke(
    {"messages": [HumanMessage("帮我查询天气")]},
    config={"configurable": {"thread_id": "thread-1"}}
)

# ========== Agent 执行后：读取数据 ==========
# 读取 agent 执行过程中保存的数据
stats = store.get(("stats",), "model_calls")
print(f"模型调用次数: {stats.value if stats else 'N/A'}")

# 遍历所有缓存数据
for item in store.search(("cache",)):
    print(f"缓存项: {item.key} = {item.value}")

# 导出所有数据（用于持久化到数据库）
all_items = list(store.search(()))  # 空 namespace 搜索所有
for item in all_items:
    print(f"{item.namespace}:{item.key} = {item.value}")
```

### 4.5 位置 4：在 wrap_tool_call 中使用

```python
from langchain.agents import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

class ToolCacheMiddleware(AgentMiddleware):
    """在 wrap_tool_call 中实现工具结果缓存"""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler
    ) -> ToolMessage:
        """拦截工具调用，实现缓存"""

        # 通过 request.runtime.store 访问 store
        store = request.runtime.store if request.runtime else None

        if store:
            # 构造缓存键
            tool_name = request.tool_call["name"]
            tool_args = str(request.tool_call["args"])
            cache_key = f"{tool_name}:{hash(tool_args)}"

            # 检查缓存
            cached = store.get(("tool_cache",), cache_key)
            if cached:
                print(f"[Cache Hit] {tool_name}")
                return ToolMessage(
                    content=cached.value["result"],
                    tool_call_id=request.tool_call["id"]
                )

        # 执行工具
        result = handler(request)

        # 保存缓存
        if store:
            store.put(("tool_cache",), cache_key, {"result": result.content})
            print(f"[Cache Saved] {tool_name}")

        return result
```

---

## 五、数据流动图

### 5.1 完整数据流动

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          Store 数据完整流动图                                         ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   用户代码                                                                           ║
║   ────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                      ║
║   store = InMemoryStore()                                                            ║
║   store.put(("users",), "u1", {"name": "张三"})  # 预加载数据                        ║
║        │                                                                             ║
║        │                                                                             ║
║        ▼                                                                             ║
║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║   │  agent = create_agent(model=..., tools=[...], store=store)                   │   ║
║   │                                                │                             │   ║
║   │                                                ▼                             │   ║
║   │  ┌────────────────────────────────────────────────────────────────────────┐  │   ║
║   │  │  graph.compile(store=store)                                            │  │   ║
║   │  │       │                                                                │  │   ║
║   │  │       │  LangGraph 保存 store 引用                                     │  │   ║
║   │  │       │  compiled_graph._store = store                                 │  │   ║
║   │  │       ▼                                                                │  │   ║
║   │  │  ┌──────────────────────────────────────────────────────────────────┐  │  │   ║
║   │  │  │  CompiledStateGraph                                              │  │  │   ║
║   │  │  │  ._store = store (引用)                                          │  │  │   ║
║   │  │  └──────────────────────────────────────────────────────────────────┘  │  │   ║
║   │  └────────────────────────────────────────────────────────────────────────┘  │   ║
║   └──────────────────────────────────────────────────────────────────────────────┘   ║
║        │                                                                             ║
║        │                                                                             ║
║        ▼                                                                             ║
║   agent.invoke({"messages": [HumanMessage("你好")]})                                 ║
║        │                                                                             ║
║        │                                                                             ║
║        ▼                                                                             ║
║   LangGraph 运行时                                                                   ║
║   ────────────────────────────────────────────────────────────────────────────────   ║
║                                                                                      ║
║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║   │                                                                              │   ║
║   │  每个节点执行前，LangGraph 创建 Runtime：                                     │   ║
║   │                                                                              │   ║
║   │  runtime = Runtime(                                                          │   ║
║   │      store=compiled_graph._store,  ← 同一个 store 实例                       │   ║
║   │      context=...,                                                            │   ║
║   │      config=...,                                                             │   ║
║   │  )                                                                           │   ║
║   │                                                                              │   ║
║   └──────────────────────────────────────────────────────────────────────────────┘   ║
║        │                                                                             ║
║        ├──────────────────────────────────────────────────────────────┐              ║
║        │                                                              │              ║
║        ▼                                                              ▼              ║
║   ┌─────────────────────────┐                              ┌─────────────────────┐   ║
║   │  中间件钩子             │                              │  model_node         │   ║
║   │                         │                              │                     │   ║
║   │  before_agent(          │                              │  def model_node(    │   ║
║   │    state,               │                              │    state,           │   ║
║   │    runtime  ────────────┼──▶ runtime.store 可用        │    runtime          │   ║
║   │  )                      │                              │  ):                 │   ║
║   │                         │                              │    # runtime.store  │   ║
║   │  before_model(...)      │                              │    # 可用           │   ║
║   │  after_model(...)       │                              │                     │   ║
║   │  after_agent(...)       │                              └─────────────────────┘   ║
║   │                         │                                        │              ║
║   └─────────────────────────┘                                        │              ║
║                                                                      │              ║
║                                                                      ▼              ║
║                                                        ┌─────────────────────────┐   ║
║                                                        │  ToolNode               │   ║
║                                                        │                         │   ║
║                                                        │  执行工具时：            │   ║
║                                                        │                         │   ║
║                                                        │  1. 检查参数类型注解    │   ║
║                                                        │  2. 识别 InjectedStore  │   ║
║                                                        │  3. 从 runtime 获取     │   ║
║                                                        │     store               │   ║
║                                                        │  4. 注入到工具参数      │   ║
║                                                        │                         │   ║
║                                                        └───────────┬─────────────┘   ║
║                                                                    │                 ║
║                                                                    ▼                 ║
║                                                        ┌─────────────────────────┐   ║
║                                                        │  工具函数               │   ║
║                                                        │                         │   ║
║                                                        │  @tool                  │   ║
║                                                        │  def my_tool(           │   ║
║                                                        │    query: str,          │   ║
║                                                        │    store: Annotated[    │   ║
║                                                        │      Any,               │   ║
║                                                        │      InjectedStore()    │   ║
║                                                        │    ]  ← 注入的 store    │   ║
║                                                        │  ):                     │   ║
║                                                        │    store.put(...)       │   ║
║                                                        │    store.get(...)       │   ║
║                                                        │                         │   ║
║                                                        └─────────────────────────┘   ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 5.2 Store 操作数据流

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          Store 操作详细数据流                                         ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   写入操作 (put/mset)                                                                ║
║   ─────────────────────────────────────────────────────────────────────────────────  ║
║                                                                                      ║
║   工具/中间件代码                         Store 内部                                  ║
║   ┌────────────────────────────┐         ┌────────────────────────────────────────┐  ║
║   │                            │         │                                        │  ║
║   │  store.put(                │   ───▶  │  InMemoryStore.store = {               │  ║
║   │    ("users",),             │         │      ("users",): {                     │  ║
║   │    "u123",                 │         │          "u123": Item(                 │  ║
║   │    {"name": "张三"}        │         │              value={"name": "张三"},   │  ║
║   │  )                         │         │              key="u123",               │  ║
║   │                            │         │              namespace=("users",)      │  ║
║   └────────────────────────────┘         │          )                             │  ║
║                                          │      }                                 │  ║
║                                          │  }                                     │  ║
║                                          └────────────────────────────────────────┘  ║
║                                                                                      ║
║   读取操作 (get/mget)                                                                ║
║   ─────────────────────────────────────────────────────────────────────────────────  ║
║                                                                                      ║
║   工具/中间件代码                         返回值                                      ║
║   ┌────────────────────────────┐         ┌────────────────────────────────────────┐  ║
║   │                            │         │                                        │  ║
║   │  item = store.get(         │   ◀───  │  Item(                                 │  ║
║   │    ("users",),             │         │      value={"name": "张三"},           │  ║
║   │    "u123"                  │         │      key="u123",                       │  ║
║   │  )                         │         │      namespace=("users",),             │  ║
║   │                            │         │      created_at=...,                   │  ║
║   │  # 访问数据                 │         │      updated_at=...                    │  ║
║   │  name = item.value["name"] │         │  )                                     │  ║
║   │  # "张三"                  │         │                                        │  ║
║   │                            │         │  或 None (如果不存在)                   │  ║
║   └────────────────────────────┘         └────────────────────────────────────────┘  ║
║                                                                                      ║
║   搜索操作 (search)                                                                  ║
║   ─────────────────────────────────────────────────────────────────────────────────  ║
║                                                                                      ║
║   工具/中间件代码                         返回值                                      ║
║   ┌────────────────────────────┐         ┌────────────────────────────────────────┐  ║
║   │                            │         │                                        │  ║
║   │  items = store.search(     │   ◀───  │  [                                     │  ║
║   │    ("users",),             │         │      Item(key="u123", value=...),      │  ║
║   │    limit=10                │         │      Item(key="u456", value=...),      │  ║
║   │  )                         │         │      ...                               │  ║
║   │                            │         │  ]                                     │  ║
║   │  for item in items:        │         │                                        │  ║
║   │    print(item.key)         │         │  迭代器，惰性加载                       │  ║
║   │                            │         │                                        │  ║
║   └────────────────────────────┘         └────────────────────────────────────────┘  ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 六、Store API 参考

### 6.1 langgraph.store.base.BaseStore 接口

```python
# langgraph.store.base.BaseStore 主要方法

class BaseStore(ABC):
    """跨线程持久化存储的基类"""

    # ========== 写入方法 ==========
    def put(
        self,
        namespace: tuple[str, ...],   # 命名空间，如 ("users", "preferences")
        key: str,                      # 键
        value: dict[str, Any],         # 值（字典）
    ) -> None:
        """存储一个值"""

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """异步存储一个值"""

    # ========== 读取方法 ==========
    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Item | None:
        """获取一个值，返回 Item 或 None"""

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> Item | None:
        """异步获取一个值"""

    # ========== 搜索方法 ==========
    def search(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> Iterable[Item]:
        """搜索命名空间下的所有项"""

    async def asearch(
        self,
        namespace: tuple[str, ...],
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> AsyncIterable[Item]:
        """异步搜索"""

    # ========== 删除方法 ==========
    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """删除一个值"""

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """异步删除一个值"""


# Item 数据结构
@dataclass
class Item:
    """Store 返回的数据项"""
    value: dict[str, Any]           # 存储的值
    key: str                         # 键
    namespace: tuple[str, ...]       # 命名空间
    created_at: datetime            # 创建时间
    updated_at: datetime            # 更新时间
```

### 6.2 langchain_core.stores.BaseStore 接口

```python
# langchain_core.stores.BaseStore 主要方法（简单键值存储）

class BaseStore(ABC, Generic[K, V]):
    """简单键值存储的基类"""

    # ========== 批量获取 ==========
    @abstractmethod
    def mget(self, keys: Sequence[K]) -> list[V | None]:
        """批量获取值"""

    async def amget(self, keys: Sequence[K]) -> list[V | None]:
        """异步批量获取"""

    # ========== 批量设置 ==========
    @abstractmethod
    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """批量设置值"""

    async def amset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """异步批量设置"""

    # ========== 批量删除 ==========
    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """批量删除"""

    async def amdelete(self, keys: Sequence[K]) -> None:
        """异步批量删除"""

    # ========== 遍历键 ==========
    @abstractmethod
    def yield_keys(self, *, prefix: str | None = None) -> Iterator[K]:
        """遍历键（可选前缀过滤）"""

    async def ayield_keys(self, *, prefix: str | None = None) -> AsyncIterator[K]:
        """异步遍历键"""
```

---

## 七、完整代码示例

### 7.1 综合示例：用户偏好管理

```python
"""
完整示例：使用 Store 实现跨会话用户偏好管理
"""
from typing import Annotated, Any
from langchain.agents import AgentMiddleware, AgentState, create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


# ========== 1. 定义工具 ==========
@tool
def save_preference(
    category: str,
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """保存用户偏好设置

    Args:
        category: 偏好类别（如 'theme', 'language'）
        key: 偏好键
        value: 偏好值
    """
    store.put(("preferences", category), key, {"value": value})
    return f"✅ 已保存偏好: {category}/{key} = {value}"


@tool
def get_preference(
    category: str,
    key: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """获取用户偏好设置

    Args:
        category: 偏好类别
        key: 偏好键
    """
    item = store.get(("preferences", category), key)
    if item:
        return f"偏好 {category}/{key} = {item.value['value']}"
    return f"❌ 偏好 {category}/{key} 不存在"


@tool
def list_preferences(
    category: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """列出某类别下的所有偏好

    Args:
        category: 偏好类别
    """
    items = list(store.search(("preferences", category), limit=100))
    if not items:
        return f"类别 {category} 下没有偏好设置"

    result = f"类别 {category} 的偏好设置:\n"
    for item in items:
        result += f"  - {item.key} = {item.value['value']}\n"
    return result


# ========== 2. 定义中间件 ==========
class UserContextMiddleware(AgentMiddleware):
    """加载用户上下文的中间件"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 开始前加载用户数据"""
        if runtime.store:
            # 检查用户是否存在
            user = runtime.store.get(("users",), "current")
            if user:
                print(f"🔄 [Middleware] 欢迎回来, {user.value.get('name', '用户')}!")
            else:
                print("🆕 [Middleware] 新用户访问")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 完成后保存会话统计"""
        if runtime.store:
            # 更新会话计数
            stats = runtime.store.get(("stats",), "sessions")
            count = stats.value.get("count", 0) if stats else 0
            runtime.store.put(("stats",), "sessions", {
                "count": count + 1,
                "last_message_count": len(state["messages"])
            })
            print(f"📊 [Middleware] 总会话数: {count + 1}")
        return None


# ========== 3. 创建和使用 Agent ==========
def main():
    # 创建 store
    store = InMemoryStore()

    # 预加载用户数据
    store.put(("users",), "current", {"name": "张三", "level": "vip"})
    store.put(("preferences", "ui"), "theme", {"value": "dark"})
    store.put(("preferences", "ui"), "language", {"value": "zh-CN"})

    # 创建 agent
    agent = create_agent(
        model="openai:gpt-4o",
        tools=[save_preference, get_preference, list_preferences],
        middleware=[UserContextMiddleware()],
        store=store,
        system_prompt="你是一个帮助用户管理偏好设置的助手。",
    )

    # ========== 第一次对话 ==========
    print("\n" + "=" * 60)
    print("第一次对话 (thread-1)")
    print("=" * 60)

    result1 = agent.invoke(
        {"messages": [HumanMessage("列出我的 UI 偏好设置")]},
        config={"configurable": {"thread_id": "thread-1"}}
    )
    print(f"回复: {result1['messages'][-1].content}")

    # ========== 第二次对话（不同 thread）==========
    print("\n" + "=" * 60)
    print("第二次对话 (thread-2) - 不同对话，但共享 store")
    print("=" * 60)

    result2 = agent.invoke(
        {"messages": [HumanMessage("把我的主题改成 light")]},
        config={"configurable": {"thread_id": "thread-2"}}
    )
    print(f"回复: {result2['messages'][-1].content}")

    # ========== 第三次对话（验证修改）==========
    print("\n" + "=" * 60)
    print("第三次对话 (thread-3) - 验证修改是否生效")
    print("=" * 60)

    result3 = agent.invoke(
        {"messages": [HumanMessage("查看我的主题设置")]},
        config={"configurable": {"thread_id": "thread-3"}}
    )
    print(f"回复: {result3['messages'][-1].content}")

    # ========== 直接访问 store 查看数据 ==========
    print("\n" + "=" * 60)
    print("直接访问 Store 数据")
    print("=" * 60)

    # 查看所有偏好
    print("\n所有偏好设置:")
    for item in store.search(("preferences", "ui"), limit=100):
        print(f"  {item.key}: {item.value}")

    # 查看统计
    stats = store.get(("stats",), "sessions")
    if stats:
        print(f"\n会话统计: {stats.value}")


if __name__ == "__main__":
    main()
```

---

## 八、总结

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                      LangChain v1 Store (长期记忆) 机制总结                          │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   1. Store 是什么？                                                                  │
│      - 跨 thread_id 的持久化存储                                                    │
│      - 用于存储用户数据、共享知识、缓存等                                            │
│      - 与 checkpointer（单线程状态）互补                                            │
│                                                                                     │
│   2. Store 存在哪里？                                                                │
│      - 用户创建的 Store 实例中（如 InMemoryStore）                                   │
│      - 通过 create_agent(store=store) 传入                                          │
│      - 通过 graph.compile(store=store) 注册到图中                                    │
│      - 运行时通过 runtime.store 访问                                                │
│                                                                                     │
│   3. Store 什么时候传入？                                                            │
│      - 阶段 1: 用户创建 store = InMemoryStore()                                     │
│      - 阶段 2: 传入 create_agent(store=store)                                       │
│      - 阶段 3: 内部 graph.compile(store=store)                                      │
│      - 阶段 4: 运行时自动注入到 runtime.store                                        │
│                                                                                     │
│   4. 用户可用的接口                                                                  │
│      - 工具中: InjectedStore(), ToolRuntime.store                                   │
│      - 中间件中: runtime.store                                                      │
│      - wrap_tool_call 中: request.runtime.store                                     │
│      - Agent 外部: 直接操作 store 实例                                               │
│                                                                                     │
│   5. 数据流动                                                                        │
│      用户创建 store → create_agent → graph.compile → Runtime → 工具/中间件          │
│      所有位置访问的是同一个 store 实例（引用传递）                                    │
│                                                                                     │
│   6. 与 Checkpointer 的区别                                                          │
│      - Checkpointer: 单线程，自动加载/保存，存储 messages                           │
│      - Store: 跨线程，手动操作，存储任意数据                                         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

