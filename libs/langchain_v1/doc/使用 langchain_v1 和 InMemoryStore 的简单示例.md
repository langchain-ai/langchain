# 使用 langchain_v1 用 Store 


## Store 的使用场景汇总

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Store 可以使用的 4 个地方                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│                   │   │                   │   │                   │   │                   │
│   1. Tools        │   │   2. Middleware   │   │   3. Agent 外部   │   │  4. wrap_tool_call│
│      (工具)       │   │      (中间件)     │   │      (直接访问)   │   │      (工具拦截)   │
│                   │   │                   │   │                   │   │                   │
│ InjectedStore()   │   │  runtime.store    │   │  store.mget()     │   │ request.runtime   │
│ ToolRuntime.store │   │                   │   │  store.mset()     │   │    .store         │
│                   │   │                   │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘   └───────────────────┘
```

---

### 1. 在 Tools 中使用（已讨论）

```python
from typing import Annotated, Any
from langchain.tools import InjectedStore, tool, ToolRuntime

# 方式 A: InjectedStore
@tool
def my_tool(query: str, store: Annotated[Any, InjectedStore()]) -> str:
    store.mset([("key", "value")])
    return store.mget(["key"])[0]

# 方式 B: ToolRuntime (包含 store)
@tool
def my_tool2(query: str, runtime: ToolRuntime) -> str:
    if runtime.store:
        runtime.store.put(("namespace",), "key", {"data": "value"})
    return "done"
```

---

### 2. 在 Middleware 中使用 ⭐

中间件的所有钩子方法都接收 `runtime` 参数，可以通过 `runtime.store` 访问 store：

```python
from langchain.agents import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

class MyMiddleware(AgentMiddleware):
    
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 执行前，通过 runtime.store 访问 store"""
        if runtime.store:
            # 读取用户偏好
            user_prefs = runtime.store.get(("users",), state.get("user_id"))
            print(f"用户偏好: {user_prefs}")
        return None
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """每次模型调用前"""
        if runtime.store:
            # 记录调用次数
            runtime.store.put(("stats",), "call_count", {"count": 1})
        return None
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """每次模型调用后"""
        if runtime.store:
            # 缓存结果
            runtime.store.put(("cache",), "last_response", state["messages"][-1])
        return None
    
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 执行完成后"""
        if runtime.store:
            # 保存会话摘要
            runtime.store.put(("sessions",), "summary", {"messages": len(state["messages"])})
        return None
```

也可以使用装饰器方式：

```python
from langchain.agents import before_model, after_model

@before_model
def log_and_cache(state: AgentState, runtime: Runtime) -> None:
    if runtime.store:
        runtime.store.put(("logs",), "last_input", state["messages"][-1])

@after_model  
def cache_response(state: AgentState, runtime: Runtime) -> None:
    if runtime.store:
        runtime.store.put(("cache",), "last_output", state["messages"][-1])
```

---

### 3. 在 Agent 外部直接使用

因为 `store` 是你自己创建的对象，可以在代码任何地方直接访问：

```python
from langchain_core.stores import InMemoryStore
from langchain.agents import create_agent

# 创建 store
store = InMemoryStore()

# 创建 agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    store=store,
)

# ========== Agent 执行前 ==========
# 预加载用户数据
store.mset([
    ("user:123:name", "张三"),
    ("user:123:preferences", {"theme": "dark", "language": "zh-CN"}),
])

# ========== Agent 执行 ==========
result = agent.invoke({"messages": [HumanMessage("...")]})

# ========== Agent 执行后 ==========
# 读取 agent 执行过程中保存的数据
all_keys = list(store.yield_keys())
print(f"Store 中的所有键: {all_keys}")

# 导出数据用于持久化
data = {key: store.mget([key])[0] for key in all_keys}
print(f"所有数据: {data}")
```

---

### 4. 在 wrap_tool_call 中使用

拦截工具调用时，也可以通过 `request.runtime.store` 访问：

```python
from langchain.agents import AgentMiddleware
from langchain.tools import ToolCallRequest
from langchain_core.messages import ToolMessage

class ToolCacheMiddleware(AgentMiddleware):
    
    def wrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage:
        """拦截工具调用，实现缓存"""
        store = request.runtime.store if request.runtime else None
        
        if store:
            # 检查缓存
            cache_key = f"tool_cache:{request.tool_call['name']}:{request.tool_call['args']}"
            cached = store.mget([cache_key])[0]
            if cached:
                return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
        
        # 执行工具
        result = handler(request)
        
        if store:
            # 保存缓存
            store.mset([(cache_key, result.content)])
        
        return result
```

---

## 完整示例：综合使用 Store

```python
from typing import Annotated, Any
from langchain.agents import AgentState, AgentMiddleware, create_agent, before_agent
from langchain.tools import InjectedStore, tool
from langchain_core.messages import HumanMessage
from langchain_core.stores import InMemoryStore
from langgraph.runtime import Runtime


# ========== 1. 工具中使用 ==========
@tool
def save_preference(key: str, value: str, store: Annotated[Any, InjectedStore()]) -> str:
    """保存偏好设置"""
    store.mset([(f"pref:{key}", value)])
    return f"Saved {key}={value}"


# ========== 2. 中间件中使用 ==========
class UserContextMiddleware(AgentMiddleware):
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        if runtime.store:
            # 加载用户上下文
            user_name = runtime.store.mget(["user:name"])[0]
            if user_name:
                print(f"[Middleware] 欢迎回来，{user_name}!")
        return None


@before_agent
def load_preferences(state: AgentState, runtime: Runtime) -> None:
    if runtime.store:
        prefs = list(runtime.store.yield_keys(prefix="pref:"))
        print(f"[Decorator Middleware] 已加载 {len(prefs)} 个偏好设置")


# ========== 3. 外部直接使用 ==========
store = InMemoryStore()

# 预设数据
store.mset([("user:name", "小明")])

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_preference],
    store=store,
    middleware=[UserContextMiddleware(), load_preferences],
)

# 执行
result = agent.invoke({"messages": [HumanMessage("保存我的主题为 dark")]})

# 读取结果
print(f"Store 数据: {list(store.yield_keys())}")
```

---

## 总结对比

| 使用场景 | 访问方式 | 适用情况 |
|---------|---------|---------|
| **Tools** | `Annotated[Any, InjectedStore()]` 或 `ToolRuntime.store` | 工具需要持久化数据 |
| **Middleware 钩子** | `runtime.store` | 在 agent 生命周期各阶段操作数据 |
| **wrap_tool_call** | `request.runtime.store` | 拦截工具调用时缓存/记录 |
| **Agent 外部** | 直接使用 `store` 对象 | 预加载数据、导出数据、调试 |

## 简单示例
```python
"""
使用 langchain_v1 和 InMemoryStore 的简单示例

这个例子展示了如何：
1. 创建 InMemoryStore
2. 在 create_agent 中传入 store
3. 在工具中使用 InjectedStore 来访问 store
4. 使用 store 的 mget/mset 方法读写数据
"""

from typing import Annotated, Any

from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedStore, tool
from langchain_core.messages import HumanMessage
from langchain_core.stores import InMemoryStore


# 定义自定义状态，可以包含用户信息等
class CustomState(AgentState):
    """自定义状态，包含用户ID"""
    user_id: str


# 创建一个使用 store 的工具
@tool
def save_user_preference(
    preference_key: str,
    preference_value: str,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """保存用户偏好设置到 store 中。

    Args:
        preference_key: 偏好设置的键名（如 'theme', 'language'）
        preference_value: 偏好设置的值
        store: 注入的 store 对象，用于持久化存储
    """
    if store is None:
        return "错误：store 未配置"

    # 使用 mset 方法保存数据
    # mset 接受一个键值对序列
    store.mset([(f"preference:{preference_key}", preference_value)])
    return f"已保存偏好设置：{preference_key} = {preference_value}"


@tool
def get_user_preference(
    preference_key: str,
    store: Annotated[Any, InjectedStore()],
) -> str:
    """从 store 中获取用户偏好设置。

    Args:
        preference_key: 偏好设置的键名
        store: 注入的 store 对象
    """
    if store is None:
        return "错误：store 未配置"

    # 使用 mget 方法读取数据
    # mget 接受一个键序列，返回对应的值列表
    values = store.mget([f"preference:{preference_key}"])
    value = values[0]

    if value is None:
        return f"未找到偏好设置：{preference_key}"
    return f"偏好设置 {preference_key} = {value}"


@tool
def list_all_preferences(
    store: Annotated[Any, InjectedStore()],
) -> str:
    """列出所有保存的偏好设置。

    Args:
        store: 注入的 store 对象
    """
    if store is None:
        return "错误：store 未配置"

    # 使用 yield_keys 方法遍历所有键
    # prefix 参数可以过滤特定前缀的键
    keys = list(store.yield_keys(prefix="preference:"))

    if not keys:
        return "没有找到任何偏好设置"

    # 批量读取所有偏好设置
    values = store.mget(keys)
    preferences = []
    for key, value in zip(keys, values):
        if value is not None:
            # 去掉前缀，只显示键名
            pref_key = key.replace("preference:", "")
            preferences.append(f"{pref_key}={value}")

    return f"所有偏好设置：{', '.join(preferences)}"


def main():
    """主函数：演示 store 的使用"""

    # 1. 创建 InMemoryStore 实例
    store = InMemoryStore()

    print("=" * 60)
    print("创建 InMemoryStore 并传入 create_agent")
    print("=" * 60)

    # 2. 创建 agent，传入 store
    agent = create_agent(
        model="openai:gpt-4o-mini",  # 或者使用其他模型
        tools=[save_user_preference, get_user_preference, list_all_preferences],
        state_schema=CustomState,
        store=store,  # 传入 store
        system_prompt="你是一个帮助用户管理偏好设置的助手。",
    )

    print("\n1. 保存用户偏好设置")
    print("-" * 60)

    # 3. 调用 agent，保存偏好设置
    result1 = agent.invoke({
        "messages": [HumanMessage("请保存我的主题偏好为 dark")],
        "user_id": "user_123",
    })

    print(f"Agent 响应: {result1['messages'][-1].content}")

    # 直接使用 store 验证数据已保存
    values = store.mget(["preference:theme"])
    print(f"Store 验证: preference:theme = {values[0]}")

    print("\n2. 获取用户偏好设置")
    print("-" * 60)

    # 4. 调用 agent，获取偏好设置
    result2 = agent.invoke({
        "messages": [HumanMessage("我的主题偏好是什么？")],
        "user_id": "user_123",
    })

    print(f"Agent 响应: {result2['messages'][-1].content}")

    print("\n3. 保存更多偏好设置")
    print("-" * 60)

    # 5. 保存更多设置
    result3 = agent.invoke({
        "messages": [HumanMessage("请保存我的语言偏好为 zh-CN")],
        "user_id": "user_123",
    })

    print(f"Agent 响应: {result3['messages'][-1].content}")

    print("\n4. 列出所有偏好设置")
    print("-" * 60)

    # 6. 列出所有偏好设置
    result4 = agent.invoke({
        "messages": [HumanMessage("列出我所有的偏好设置")],
        "user_id": "user_123",
    })

    print(f"Agent 响应: {result4['messages'][-1].content}")

    print("\n5. 直接使用 store API")
    print("-" * 60)

    # 7. 直接使用 store 的 API
    # 批量设置多个值
    store.mset([
        ("preference:timezone", "Asia/Shanghai"),
        ("preference:currency", "CNY"),
    ])

    # 批量获取多个值
    values = store.mget(["preference:timezone", "preference:currency"])
    print(f"时区: {values[0]}, 货币: {values[1]}")

    # 列出所有键
    all_keys = list(store.yield_keys())
    print(f"Store 中的所有键: {all_keys}")

    # 使用前缀过滤
    pref_keys = list(store.yield_keys(prefix="preference:"))
    print(f"所有偏好设置键: {pref_keys}")

    # 删除一个键
    store.mdelete(["preference:currency"])
    deleted_value = store.mget(["preference:currency"])
    print(f"删除后 currency 的值: {deleted_value[0]}")

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

要点说明：

1. 导入 `InMemoryStore`：
   ```python
   from langchain_core.stores import InMemoryStore
   ```

2. 创建 store 并传入 `create_agent`：
   ```python
   store = InMemoryStore()
   agent = create_agent(..., store=store)
   ```

3. 在工具中使用 `InjectedStore` 注入 store：
   ```python
   store: Annotated[Any, InjectedStore()]
   ```

4. Store API 使用：
   - `mset([(key, value), ...])`：批量设置
   - `mget([key, ...])`：批量获取
   - `mdelete([key, ...])`：批量删除
   - `yield_keys(prefix=...)`：遍历键（支持前缀过滤）

5. Store 的作用：
   - `checkpointer`：用于单线程（单次对话）的状态持久化
   - `store`：用于跨线程（多个对话/用户）的数据持久化

## 一个 `InjectedStore` 注入机制的流程图：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        InjectedStore 注入机制流程图                              │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              阶段 1: 定义时（代码编写阶段）
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│  用户定义工具函数                                                                │
│                                                                                 │
│  @tool                                                                          │
│  def my_tool(                                                                   │
│      query: str,                           ← LLM 控制的参数                      │
│      store: Annotated[Any, InjectedStore()] ← 标记为注入的参数                   │
│  ) -> str:                                                                      │
│      ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  @tool 装饰器分析函数签名                                                        │
│                                                                                 │
│  检测 Annotated[..., InjectedStore()] 类型注解                                   │
│                                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────┐                             │
│  │ LLM 可见参数        │    │ 注入参数（隐藏）     │                             │
│  │ - query: str        │    │ - store: InjectedStore │                          │
│  │                     │    │                     │                             │
│  │ 会包含在 tool schema│    │ 不会发送给 LLM      │                             │
│  │ 发送给 LLM          │    │ 运行时自动注入      │                             │
│  └─────────────────────┘    └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              阶段 2: 构建时（create_agent）
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  store = InMemoryStore()                                                        │
│                                                                                 │
│  agent = create_agent(                                                          │
│      model="openai:gpt-4o",                                                     │
│      tools=[my_tool],                                                           │
│      store=store,          ← 传入 store 实例                                    │
│  )                                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  create_agent 内部处理 (factory.py)                                             │
│                                                                                 │
│  1. 创建 ToolNode                                                               │
│     ┌──────────────────────────────────────────────────────────────────┐        │
│     │  tool_node = ToolNode(tools=[my_tool, ...])                      │        │
│     │                                                                  │        │
│     │  ToolNode 负责执行工具，并处理参数注入                            │        │
│     └──────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  2. 构建状态图（StateGraph）                                                     │
│     ┌──────────────────────────────────────────────────────────────────┐        │
│     │  graph.add_node("model", model_node)                             │        │
│     │  graph.add_node("tools", tool_node)  ← ToolNode 作为节点          │        │
│     └──────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
│  3. 编译图，传入 store                                                          │
│     ┌──────────────────────────────────────────────────────────────────┐        │
│     │  graph.compile(                                                  │        │
│     │      store=store,  ← store 被注册到编译后的图中                   │        │
│     │      ...                                                         │        │
│     │  )                                                               │        │
│     └──────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              阶段 3: 运行时（agent.invoke）
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│  用户调用 agent                                                                 │
│                                                                                 │
│  result = agent.invoke({"messages": [HumanMessage("搜索 AI")]})                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                   │
│  │              │      │              │      │              │                   │
│  │    START     │─────▶│    MODEL     │─────▶│    TOOLS     │──────┐            │
│  │              │      │              │      │  (ToolNode)  │      │            │
│  └──────────────┘      └──────────────┘      └──────────────┘      │            │
│                               ▲                                     │            │
│                               └─────────────────────────────────────┘            │
│                                        (循环直到无工具调用)                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ 当 MODEL 返回 tool_calls 时
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  MODEL 节点输出                                                                  │
│                                                                                 │
│  AIMessage(                                                                     │
│      content="",                                                                │
│      tool_calls=[                                                               │
│          {                                                                      │
│              "name": "my_tool",                                                 │
│              "args": {"query": "AI"},  ← 只有 LLM 控制的参数                     │
│              "id": "call_123"                                                   │
│          }                                                                      │
│      ]                                                                          │
│  )                                                                              │
│                                                                                 │
│  注意：args 中没有 store，因为 InjectedStore 不会发送给 LLM                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  TOOLS 节点（ToolNode）执行                                                      │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Step 1: ToolNode 接收 tool_call                                          │  │
│  │          {"name": "my_tool", "args": {"query": "AI"}, "id": "call_123"}   │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Step 2: 检查工具函数签名，发现需要注入的参数                              │  │
│  │                                                                           │  │
│  │  my_tool 签名:                                                            │  │
│  │    - query: str                    → 从 tool_call["args"] 获取            │  │
│  │    - store: Annotated[..., InjectedStore()]  → 需要注入                   │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Step 3: 从图的运行时环境获取 store                                       │  │
│  │                                                                           │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐      │  │
│  │  │  CompiledGraph                                                  │      │  │
│  │  │    ├── state (当前状态)                                         │      │  │
│  │  │    ├── config (运行配置)                                        │      │  │
│  │  │    └── store ─────────────────────────────────────────────────┐│      │  │
│  │  │              │                                                 ││      │  │
│  │  └──────────────│─────────────────────────────────────────────────┘│      │  │
│  │                 │                                                  │      │  │
│  │                 ▼                                                  │      │  │
│  │           InMemoryStore (之前传入的实例)                           │      │  │
│  │                                                                    │      │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Step 4: 构建完整的参数字典                                               │  │
│  │                                                                           │  │
│  │  {                                                                        │  │
│  │      "query": "AI",                ← 来自 LLM 的 tool_call                 │  │
│  │      "store": <InMemoryStore>      ← 注入的 store 实例                     │  │
│  │  }                                                                        │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                                                                           │  │
│  │  Step 5: 调用工具函数                                                     │  │
│  │                                                                           │  │
│  │  result = my_tool(query="AI", store=<InMemoryStore>)                      │  │
│  │                                                                           │  │
│  │  工具函数内部可以使用 store:                                               │  │
│  │    store.mset([("key", "value")])                                         │  │
│  │    store.mget(["key"])                                                    │  │
│  │                                                                           │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              关键概念总结
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  1. InjectedStore 是什么？                                                       │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │  一个"标记类"，用于告诉框架：                                         │     │
│     │  "这个参数不是 LLM 控制的，请在运行时自动注入 store"                   │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  2. 为什么 LLM 看不到 store 参数？                                               │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │  @tool 装饰器在生成 tool schema 时，                                  │     │
│     │  会过滤掉所有标记为 InjectedToolArg 的参数                            │     │
│     │                                                                     │     │
│     │  LLM 只能看到:                                                       │     │
│     │  {                                                                  │     │
│     │      "name": "my_tool",                                             │     │
│     │      "parameters": {                                                │     │
│     │          "query": {"type": "string"}  ← 只有 LLM 控制的参数          │     │
│     │      }                                                              │     │
│     │  }                                                                  │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  3. 注入发生在哪里？                                                             │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │  ToolNode (来自 langgraph) 在执行工具前：                            │     │
│     │                                                                     │     │
│     │  1. 检查工具函数的类型注解                                           │     │
│     │  2. 识别 Annotated[..., InjectedStore()] 标记的参数                  │     │
│     │  3. 从图的运行时环境获取 store                                       │     │
│     │  4. 将 store 作为参数传入工具函数                                    │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
│  4. 还有哪些可注入的类型？                                                       │
│     ┌─────────────────────────────────────────────────────────────────────┐     │
│     │  - InjectedState      → 注入当前图状态                               │     │
│     │  - InjectedStore      → 注入持久化存储                               │     │
│     │  - InjectedToolCallId → 注入工具调用 ID                              │     │
│     │  - ToolRuntime        → 注入运行时上下文（包含 state, store 等）      │     │
│     └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              数据流向图
═══════════════════════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────────────┐
                    │           用户代码                   │
                    │                                     │
                    │  store = InMemoryStore()           │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 传入
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         create_agent(store=store)   │
                    │                                     │
                    │  将 store 保存到图的配置中            │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 编译
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         CompiledStateGraph          │
                    │                                     │
                    │  store 作为图的属性被保存             │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 运行时
                                      ▼
                    ┌─────────────────────────────────────┐
                    │            ToolNode                 │
                    │                                     │
                    │  执行工具时从图获取 store            │
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                                      │ 注入
                                      ▼
                    ┌─────────────────────────────────────┐
                    │           工具函数                   │
                    │                                     │
                    │  def my_tool(query, store):        │
                    │      store.mset(...)               │
                    │      store.mget(...)               │
                    │                                     │
                    └─────────────────────────────────────┘
```

## 简单类比

可以把 `InjectedStore` 理解为一种**依赖注入**机制：

| 概念 | 类比 |
|------|------|
| `InjectedStore` | 像 Spring 的 `@Autowired` |
| `create_agent(store=...)` | 像 Spring 的 Bean 配置 |
| `ToolNode` | 像 Spring 的 IoC 容器 |
| 工具函数 | 像需要依赖注入的服务类 |

**核心思想**：
- **声明式**：你只需要在函数签名中声明"我需要 store"
- **自动注入**：框架会在合适的时机自动提供 store 实例
- **对 LLM 透明**：LLM 不知道 store 的存在，只负责提供它能控制的参数
