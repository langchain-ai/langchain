# LangChain Stores 实现分析与 Agent 中的使用

## 一、两套 Store 体系

LangChain 生态中存在 **两套完全不同的 Store 体系**，这是很多用户容易混淆的地方：

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                        两套 Store 体系对比                                            ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   ┌─────────────────────────────────────┬───────────────────────────────────────────┐║
║   │  langchain_core.stores.BaseStore    │  langgraph.store.base.BaseStore           │║
║   │  (简单键值存储)                      │  (命名空间存储)                            │║
║   ├─────────────────────────────────────┼───────────────────────────────────────────┤║
║   │                                     │                                           │║
║   │  用途：                              │  用途：                                   │║
║   │  - 缓存                              │  - Agent 长期记忆                         │║
║   │  - Embeddings 缓存                   │  - 跨线程数据持久化                       │║
║   │  - 检索器辅助存储                    │  - 工具/中间件数据共享                    │║
║   │                                     │                                           │║
║   │  API 风格：                          │  API 风格：                               │║
║   │  - mget([keys])                     │  - get(namespace, key)                   │║
║   │  - mset([(k, v), ...])              │  - put(namespace, key, value)            │║
║   │  - mdelete([keys])                  │  - delete(namespace, key)                │║
║   │  - yield_keys(prefix)               │  - search(namespace, limit)              │║
║   │                                     │                                           │║
║   │  Key 类型：字符串                    │  Key 类型：(namespace, key) 元组          │║
║   │  Value 类型：泛型 V                  │  Value 类型：dict + 元数据 Item           │║
║   │                                     │                                           │║
║   │  ⚠️ 不能用于 create_agent            │  ✅ 用于 create_agent(store=...)         │║
║   │                                     │                                           │║
║   └─────────────────────────────────────┴───────────────────────────────────────────┘║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 二、langchain_core.stores 详解（您打开的文件）

### 2.1 类继承结构

```
BaseStore[K, V]                 # 抽象基类 (stores.py:26)
    │
    ├── InMemoryBaseStore[V]    # 内存实现基类 (stores.py:174)
    │       │
    │       ├── InMemoryStore   # 任意类型存储 (stores.py:242)
    │       │
    │       └── InMemoryByteStore  # 字节存储 (stores.py:265)
    │
    └── ByteStore = BaseStore[str, bytes]  # 类型别名 (stores.py:171)

# langchain_classic.storage 扩展
ByteStore
    │
    └── LocalFileStore          # 文件系统存储 (file_system.py:12)

BaseStore
    │
    └── EncoderBackedStore      # 编码包装器 (encoder_backed.py:13)
```

### 2.2 源码分析：核心接口

```26:78:libs/core/langchain_core/stores.py
class BaseStore(ABC, Generic[K, V]):
    """Abstract interface for a key-value store.

    设计特点：
    1. 泛型：K 是键类型，V 是值类型
    2. 批量操作：所有方法都是批量的 (mget, mset, mdelete)
    3. 同步/异步：每个方法都有 async 版本 (amget, amset, amdelete)
    """

    @abstractmethod
    def mget(self, keys: Sequence[K]) -> list[V | None]:
        """批量获取，不存在返回 None"""

    @abstractmethod
    def mset(self, key_value_pairs: Sequence[tuple[K, V]]) -> None:
        """批量设置"""

    @abstractmethod
    def mdelete(self, keys: Sequence[K]) -> None:
        """批量删除"""

    @abstractmethod
    def yield_keys(self, *, prefix: str | None = None) -> Iterator[K]:
        """遍历键，支持前缀过滤"""
```

### 2.3 可用的 Store 实现

| Store 类 | 来源 | 存储位置 | 适用场景 |
|---------|------|---------|---------|
| `InMemoryStore` | `langchain_core.stores` | 内存 | 开发测试、临时缓存 |
| `InMemoryByteStore` | `langchain_core.stores` | 内存 | 字节数据缓存 |
| `LocalFileStore` | `langchain.storage` | 文件系统 | 本地持久化 |
| `EncoderBackedStore` | `langchain.storage` | 包装其他 Store | 自定义序列化 |
| `RedisStore` | `langchain_community.storage` | Redis | 生产环境缓存 |
| `UpstashRedisStore` | `langchain_community.storage` | Upstash Redis | Serverless 环境 |

### 2.4 使用示例

```python
# ========== 1. InMemoryStore 基本使用 ==========
from langchain_core.stores import InMemoryStore

store = InMemoryStore()

# 批量设置
store.mset([
    ("user:123", {"name": "张三", "age": 25}),
    ("user:456", {"name": "李四", "age": 30}),
])

# 批量获取
values = store.mget(["user:123", "user:456", "user:789"])
# [{"name": "张三", ...}, {"name": "李四", ...}, None]

# 遍历键
for key in store.yield_keys(prefix="user:"):
    print(key)  # "user:123", "user:456"

# 批量删除
store.mdelete(["user:123"])


# ========== 2. LocalFileStore 文件存储 ==========
from langchain.storage import LocalFileStore

file_store = LocalFileStore("/path/to/cache")

# 存储字节数据
file_store.mset([
    ("embeddings/doc1", b"...binary data..."),
    ("embeddings/doc2", b"...binary data..."),
])

# 读取
data = file_store.mget(["embeddings/doc1"])


# ========== 3. EncoderBackedStore 自定义序列化 ==========
import json
from langchain.storage import EncoderBackedStore, InMemoryByteStore

# 底层存储字节
byte_store = InMemoryByteStore()

# 包装为 JSON 序列化的 Store
json_store = EncoderBackedStore(
    store=byte_store,
    key_encoder=str,
    value_serializer=lambda v: json.dumps(v).encode(),
    value_deserializer=lambda b: json.loads(b.decode()),
)

json_store.mset([(1, {"data": "hello"})])
values = json_store.mget([1])  # [{"data": "hello"}]
```

---

## 三、在 Agent 中使用的 Store（langgraph.store）

### 3.1 关键区别

**`create_agent(store=...)` 只接受 `langgraph.store.base.BaseStore`，不是 `langchain_core.stores.BaseStore`！**

```python
# factory.py 第 59 行
from langgraph.store.base import BaseStore  # ← 注意是 langgraph 的

# factory.py 第 551 行
def create_agent(
    ...
    store: BaseStore | None = None,  # ← 类型是 langgraph.store.base.BaseStore
    ...
)
```

### 3.2 langgraph Store 的 API

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# ========== 写入：put(namespace, key, value) ==========
store.put(
    ("users", "preferences"),  # namespace 是元组
    "theme",                    # key
    {"value": "dark"}           # value 必须是 dict
)

# ========== 读取：get(namespace, key) -> Item | None ==========
item = store.get(("users", "preferences"), "theme")
if item:
    print(item.value)      # {"value": "dark"}
    print(item.key)        # "theme"
    print(item.namespace)  # ("users", "preferences")
    print(item.created_at) # datetime
    print(item.updated_at) # datetime

# ========== 搜索：search(namespace) -> Iterable[Item] ==========
items = store.search(("users",), limit=10)
for item in items:
    print(f"{item.namespace}/{item.key}: {item.value}")

# ========== 删除：delete(namespace, key) ==========
store.delete(("users", "preferences"), "theme")
```

### 3.3 在 Agent 中的 4 种使用位置

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                           Agent 中 Store 的 4 种访问方式                              │
├───────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│   1. 工具中 - InjectedStore                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │  @tool                                                                          │ │
│   │  def my_tool(query: str, store: Annotated[Any, InjectedStore()]) -> str:        │ │
│   │      store.put(("cache",), "key", {"data": "value"})                            │ │
│   │      item = store.get(("cache",), "key")                                        │ │
│   │      return item.value["data"]                                                  │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                       │
│   2. 工具中 - ToolRuntime.store                                                       │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │  @tool                                                                          │ │
│   │  def my_tool(query: str, runtime: ToolRuntime) -> str:                          │ │
│   │      if runtime.store:                                                          │ │
│   │          runtime.store.put(("logs",), "last", {"query": query})                 │ │
│   │      return "done"                                                              │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                       │
│   3. 中间件中 - runtime.store                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │  class MyMiddleware(AgentMiddleware):                                           │ │
│   │      def before_model(self, state, runtime):                                    │ │
│   │          if runtime.store:                                                      │ │
│   │              runtime.store.put(("stats",), "calls", {"count": 1})               │ │
│   │          return None                                                            │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                       │
│   4. Agent 外部 - 直接操作                                                            │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│   │  store = InMemoryStore()                                                        │ │
│   │  store.put(("users",), "u1", {"name": "张三"})  # 预加载                        │ │
│   │                                                                                 │ │
│   │  agent = create_agent(model=..., store=store)                                   │ │
│   │  agent.invoke(...)                                                              │ │
│   │                                                                                 │ │
│   │  # 执行后读取                                                                    │ │
│   │  for item in store.search(("cache",)):                                          │ │
│   │      print(item.value)                                                          │ │
│   └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、完整使用示例

```python
"""在 Agent 中使用 Store 的完整示例"""
from typing import Annotated, Any
from langchain.agents import create_agent, AgentMiddleware, AgentState
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


# ========== 1. 创建 Store ==========
store = InMemoryStore()

# 预加载数据
store.put(("users",), "current", {"name": "张三", "vip": True})


# ========== 2. 定义工具 ==========
@tool
def get_user_info(
    user_key: str,
    store: Annotated[Any, InjectedStore()]  # 注入 Store
) -> str:
    """获取用户信息"""
    item = store.get(("users",), user_key)
    if item:
        return f"用户: {item.value}"
    return f"用户 {user_key} 不存在"


@tool
def save_note(
    content: str,
    runtime: ToolRuntime  # 通过 ToolRuntime 访问 Store
) -> str:
    """保存笔记"""
    if runtime.store:
        import time
        key = f"note_{int(time.time())}"
        runtime.store.put(("notes",), key, {"content": content})
        return f"笔记已保存: {key}"
    return "Store 不可用"


# ========== 3. 定义中间件 ==========
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        if runtime.store:
            # 记录模型调用
            runtime.store.put(("logs",), "last_model_call", {
                "message_count": len(state["messages"])
            })
        return None


# ========== 4. 创建 Agent ==========
agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_user_info, save_note],
    middleware=[LoggingMiddleware()],
    store=store,  # ← 传入 langgraph 的 InMemoryStore
    system_prompt="你是一个助手",
)


# ========== 5. 使用 Agent ==========
result = agent.invoke(
    {"messages": [HumanMessage("查看当前用户信息")]},
    config={"configurable": {"thread_id": "thread-1"}}
)


# ========== 6. 查看 Store 数据 ==========
print("\n=== Store 中的数据 ===")

# 查看用户数据
for item in store.search(("users",)):
    print(f"用户: {item.key} = {item.value}")

# 查看笔记
for item in store.search(("notes",)):
    print(f"笔记: {item.key} = {item.value}")

# 查看日志
for item in store.search(("logs",)):
    print(f"日志: {item.key} = {item.value}")
```

---

## 五、总结对比表

| 特性 | langchain_core.stores | langgraph.store |
|-----|----------------------|-----------------|
| **用于 Agent** | ❌ 不支持 | ✅ 支持 |
| **Key 格式** | 字符串 | (namespace, key) 元组 |
| **Value 格式** | 任意类型 | 必须是 dict，返回 Item |
| **批量操作** | mget/mset/mdelete | 单个操作 get/put/delete |
| **搜索** | yield_keys(prefix) | search(namespace, limit) |
| **元数据** | 无 | created_at, updated_at |
| **典型用途** | 缓存、Embeddings | Agent 长期记忆 |
| **主要实现** | InMemoryStore, LocalFileStore, Redis | InMemoryStore |

**结论**：在 `langchain_v1` 的 `create_agent` 中，必须使用 `langgraph.store.memory.InMemoryStore`（或其他 langgraph Store 实现），而不是 `langchain_core.stores.InMemoryStore`。两者虽然名字相同，但 API 完全不同！

## 六、在[LangChain Stores 实现分析与 Agent 中的使用](./LangChain%20Stores%20实现分析与%20Agent%20中的使用.md)中是否正确

虽然langchain_core.stores也传递到了agent中，但是仅仅只能通过api进行存和取的操作。

因为在agent中stores是透传的。

在agent中还是推荐 langgraph.store 的使用，直接支持命名空间储存，且集成了search方法
