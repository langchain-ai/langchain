# LangChain v1 Agent `cache` 参数完全解析

## 一、源码定义位置

### 1.1 `create_agent` 中的参数定义

```541:557:/Users/cong/code/github/langchain/libs/langchain_v1/langchain/agents/factory.py
def create_agent(  # noqa: PLR0915
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,  # ← cache 参数
)
```

### 1.2 参数传递到 `graph.compile()`

```1471:1479:/Users/cong/code/github/langchain/libs/langchain_v1/langchain/agents/factory.py
    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,  # ← 传递给编译后的图
    ).with_config({"recursion_limit": 10_000})
```

---

## 二、BaseCache 数据结构

### 2.1 抽象基类定义

```15:49:/Users/cong/code/github/langgraph/libs/checkpoint/langgraph/cache/base/__init__.py
class BaseCache(ABC, Generic[ValueT]):
    """Base class for a cache."""

    serde: SerializerProtocol = JsonPlusSerializer(pickle_fallback=True)

    def __init__(self, *, serde: SerializerProtocol | None = None) -> None:
        """Initialize the cache with a serializer."""
        self.serde = serde or self.serde

    @abstractmethod
    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""

    @abstractmethod
    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get the cached values for the given keys."""

    @abstractmethod
    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set the cached values for the given keys and TTLs."""

    @abstractmethod
    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set the cached values for the given keys and TTLs."""

    @abstractmethod
    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""

    @abstractmethod
    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete the cached values for the given namespaces.
        If no namespaces are provided, clear all cached values."""
```

### 2.2 核心类型定义

```python
# 命名空间：元组形式，如 ("__cache_writes__", "func_hash", "node_name")
Namespace = tuple[str, ...]

# 完整的 Key：(命名空间, 键)
FullKey = tuple[Namespace, str]

# 值类型：泛型，通常是节点的输出 writes
ValueT = TypeVar("ValueT")
```

### 2.3 CacheKey 数据结构（任务级别）

```222:230:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/types.py
class CacheKey(NamedTuple):
    """Cache key for a task."""

    ns: tuple[str, ...]
    """Namespace for the cache entry."""
    key: str
    """Key for the cache entry."""
    ttl: int | None
    """Time to live for the cache entry in seconds."""
```

### 2.4 CachePolicy 配置

```125:134:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/types.py
@dataclass(**_DC_KWARGS)
class CachePolicy(Generic[KeyFuncT]):
    """Configuration for caching nodes."""

    key_func: KeyFuncT = default_cache_key  # type: ignore[assignment]
    """Function to generate a cache key from the node's input.
    Defaults to hashing the input with pickle."""

    ttl: int | None = None
    """Time to live for the cache entry in seconds. If `None`, the entry never expires."""
```

---

## 三、三种 Cache 实现

### 3.1 InMemoryCache（内存缓存）

```11:73:/Users/cong/code/github/langgraph/libs/checkpoint/langgraph/cache/memory/__init__.py
class InMemoryCache(BaseCache[ValueT]):
    def __init__(self, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys."""
        with self._lock:
            if not keys:
                return {}
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            values: dict[FullKey, ValueT] = {}
            for ns_tuple, key in keys:
                ns = Namespace(ns_tuple)
                if ns in self._cache and key in self._cache[ns]:
                    enc, val, expiry = self._cache[ns][key]
                    if expiry is None or now < expiry:
                        values[(ns, key)] = self.serde.loads_typed((enc, val))
                    else:
                        del self._cache[ns][key]  # 过期自动删除
            return values
    # ... set, clear 等方法
```

**特点：**
- 存储在进程内存中
- 进程重启后丢失
- 最快的缓存方式
- 适合开发测试

### 3.2 SqliteCache（文件缓存）

```13:120:/Users/cong/code/github/langgraph/libs/checkpoint-sqlite/langgraph/cache/sqlite/__init__.py
class SqliteCache(BaseCache[ValueT]):
    """File-based cache using SQLite."""

    def __init__(self, *, path: str, serde: SerializerProtocol | None = None) -> None:
        super().__init__(serde=serde)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                ns TEXT,
                key TEXT,
                expiry REAL,
                encoding TEXT NOT NULL,
                val BLOB NOT NULL,
                PRIMARY KEY (ns, key)
            )"""
        )
```

**特点：**
- 持久化到本地文件
- 进程重启后保留
- 适合单机生产环境

### 3.3 RedisCache（分布式缓存）

```10:144:/Users/cong/code/github/langgraph/libs/checkpoint/langgraph/cache/redis/__init__.py
class RedisCache(BaseCache[ValueT]):
    """Redis-based cache implementation with TTL support."""

    def __init__(
        self,
        redis: Any,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",
    ) -> None:
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix
```

**特点：**
- 分布式共享缓存
- 支持原生 TTL
- 适合多实例生产环境

---

## 四、Cache 工作流程图

```
╔════════════════════════════════════════════════════════════════════════════════════════╗
║                              Cache 完整工作流程                                         ║
╠════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                        ║
║  用户代码层                                                                            ║
║  ═════════                                                                             ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │ # 1. 创建 Cache 实例                                                           │    ║
║  │ from langgraph.cache.memory import InMemoryCache                               │    ║
║  │ # 或                                                                           │    ║
║  │ from langgraph.cache.sqlite import SqliteCache                                 │    ║
║  │                                                                                │    ║
║  │ cache = InMemoryCache()                                                        │    ║
║  │ # 或                                                                           │    ║
║  │ cache = SqliteCache(path="./cache.db")                                         │    ║
║  │                                                                                │    ║
║  │ # 2. 传入 create_agent                                                         │    ║
║  │ agent = create_agent(                                                          │    ║
║  │     model="openai:gpt-4o",                                                     │    ║
║  │     tools=[expensive_tool],                                                    │    ║
║  │     cache=cache,  ← 启用缓存                                                   │    ║
║  │ )                                                                              │    ║
║  └────────────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                              ║
║                                         ▼                                              ║
║  LangGraph 编译层 (graph.compile)                                                      ║
║  ════════════════════════════                                                         ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │ # 3. cache 存储到 Pregel 实例                                                  │    ║
║  │                                                                                │    ║
║  │ class Pregel:                                                                  │    ║
║  │     cache: BaseCache | None = None                                             │    ║
║  │     cache_policy: CachePolicy | None = None  # 全局默认策略                    │    ║
║  │                                                                                │    ║
║  └────────────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                              ║
║                                         ▼                                              ║
║  运行时执行层 (invoke/stream)                                                          ║
║  ════════════════════════════                                                         ║
║  ┌────────────────────────────────────────────────────────────────────────────────┐    ║
║  │ # 4. 创建 PregelLoop，传入 cache                                               │    ║
║  │                                                                                │    ║
║  │ with SyncPregelLoop(                                                           │    ║
║  │     input,                                                                     │    ║
║  │     cache=cache,          ← cache 传入 loop                                    │    ║
║  │     cache_policy=self.cache_policy,                                            │    ║
║  │     ...                                                                        │    ║
║  │ ) as loop:                                                                     │    ║
║  │     ...                                                                        │    ║
║  └────────────────────────────────────────────────────────────────────────────────┘    ║
║                                         │                                              ║
║                    ┌────────────────────┴────────────────────┐                         ║
║                    ▼                                         ▼                         ║
║  ┌─────────────────────────────────┐    ┌─────────────────────────────────────────┐   ║
║  │    任务准备阶段 (_algo.py)       │    │     任务执行阶段 (_loop.py)              │   ║
║  │                                 │    │                                         │   ║
║  │  # 5. 生成 cache_key            │    │  # 7. 执行前检查缓存                     │   ║
║  │  if cache_policy:               │    │  def match_cached_writes():             │   ║
║  │      args_key = cache_policy.   │    │      if self.cache is None:             │   ║
║  │          key_func(input)        │    │          return ()                      │   ║
║  │                                 │    │      cached = {                         │   ║
║  │      cache_key = CacheKey(      │    │          (t.cache_key.ns,               │   ║
║  │          ns=(CACHE_NS_WRITES,   │    │           t.cache_key.key): t           │   ║
║  │               func_hash,        │    │          for t in self.tasks.values()   │   ║
║  │               node_name),       │    │          if t.cache_key and not t.writes│   ║
║  │          key=xxh3_hash(args),   │    │      }                                  │   ║
║  │          ttl=cache_policy.ttl   │    │      for key, values in                 │   ║
║  │      )                          │    │          self.cache.get(cached):        │   ║
║  │                                 │    │          task.writes.extend(values)     │   ║
║  │  # 6. 存入 task                 │    │          matched.append(task) ← 命中    │   ║
║  │  task = PregelExecutableTask(   │    │                                         │   ║
║  │      ...,                       │    │  # 如果命中，跳过实际执行!               │   ║
║  │      cache_key=cache_key,       │    │                                         │   ║
║  │  )                              │    └─────────────────────────────────────────┘   ║
║  │                                 │                      │                           ║
║  └─────────────────────────────────┘                      ▼                           ║
║                                         ┌─────────────────────────────────────────┐   ║
║                                         │     任务完成后 (_loop.py)                │   ║
║                                         │                                         │   ║
║                                         │  # 8. 写入缓存                          │   ║
║                                         │  def put_writes(task_id, writes):       │   ║
║                                         │      ...                                │   ║
║                                         │      if task.cache_key:                 │   ║
║                                         │          self.cache.set({               │   ║
║                                         │              (task.cache_key.ns,        │   ║
║                                         │               task.cache_key.key): (    │   ║
║                                         │                  task.writes,           │   ║
║                                         │                  task.cache_key.ttl     │   ║
║                                         │              )                          │   ║
║                                         │          })                             │   ║
║                                         │  # 下次相同输入直接返回缓存!            │   ║
║                                         └─────────────────────────────────────────┘   ║
║                                                                                        ║
╚════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 五、Cache Key 生成机制

### 5.1 默认 key 函数

```26:31:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/_internal/_cache.py
def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments
    to generate a hashable key."""
    import pickle

    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
```

### 5.2 完整的 cache_key 生成（_algo.py）

```645:664:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/pregel/_algo.py
cache_policy = proc.cache_policy or cache_policy
if cache_policy:
    args_key = cache_policy.key_func(val)  # 将输入序列化为 key
    cache_key = CacheKey(
        (
            CACHE_NS_WRITES,           # 命名空间前缀: "__cache_writes__"
            (identifier(proc) or "__dynamic__"),  # 函数的唯一标识（源码 hash）
            name,                       # 节点名称
        ),
        xxh3_128_hexdigest(            # 对输入进行 xxhash 哈希
            (
                args_key.encode()
                if isinstance(args_key, str)
                else args_key
            ),
        ),
        cache_policy.ttl,              # TTL 过期时间
    )
else:
    cache_key = None
```

**关键点：**
- **命名空间 `ns`**: `("__cache_writes__", "函数hash", "节点名")` - 确保不同节点/函数的缓存隔离
- **key**: 对输入参数进行 xxhash 哈希 - 确保相同输入产生相同 key
- **ttl**: 缓存过期时间（秒），`None` 表示永不过期

---

## 六、缓存命中与写入逻辑

### 6.1 缓存命中检查（执行前）

```1040:1053:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/pregel/_loop.py
def match_cached_writes(self) -> Sequence[PregelExecutableTask]:
    if self.cache is None:
        return ()
    matched: list[PregelExecutableTask] = []
    if cached := {
        (t.cache_key.ns, t.cache_key.key): t
        for t in self.tasks.values()
        if t.cache_key and not t.writes  # 有 cache_key 且还没有 writes
    }:
        for key, values in self.cache.get(tuple(cached)).items():
            task = cached[key]
            task.writes.extend(values)  # 直接使用缓存的 writes！
            matched.append(task)
    return matched
```

### 6.2 缓存写入（执行后）

```1063:1079:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/pregel/_loop.py
def put_writes(self, task_id: str, writes: WritesT) -> None:
    """Put writes for a task, to be read by the next tick."""
    super().put_writes(task_id, writes)
    if not writes or self.cache is None or not hasattr(self, "tasks"):
        return
    task = self.tasks.get(task_id)
    if task is None or task.cache_key is None:
        return
    self.submit(
        self.cache.set,
        {
            (task.cache_key.ns, task.cache_key.key): (
                task.writes,          # 缓存任务的输出
                task.cache_key.ttl,   # 使用配置的 TTL
            )
        },
    )
```

---

## 七、使用场景

### 7.1 场景一：缓存昂贵的 API 调用

```python
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy
from langchain.agents import create_agent
from langchain.tools import tool

# 创建缓存
cache = InMemoryCache()

@tool
def expensive_api_call(query: str) -> str:
    """需要调用外部 API 的昂贵操作"""
    # 假设这是一个需要几秒钟的操作
    import time
    time.sleep(3)
    return f"API Result for: {query}"

# 创建启用缓存的 Agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[expensive_api_call],
    cache=cache,
)

# 第一次调用 - 实际执行，耗时 3 秒
result1 = agent.invoke({"messages": [HumanMessage("搜索 Python")]})

# 第二次相同调用 - 命中缓存，几乎瞬间返回！
result2 = agent.invoke({"messages": [HumanMessage("搜索 Python")]})
```

### 7.2 场景二：持久化缓存（跨进程重启）

```python
from langgraph.cache.sqlite import SqliteCache

# 使用 SQLite 持久化缓存
cache = SqliteCache(path="./agent_cache.db")

agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tools],
    cache=cache,
)

# 即使进程重启，之前的缓存仍然有效！
```

### 7.3 场景三：分布式缓存（多实例共享）

```python
import redis
from langgraph.cache.redis import RedisCache

# 连接 Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)
cache = RedisCache(redis=redis_client, prefix="myagent:cache:")

agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tools],
    cache=cache,
)

# 多个 Agent 实例可以共享缓存！
```

### 7.4 场景四：带 TTL 的节点级缓存策略

如果你需要更精细的控制，可以在 LangGraph 中直接使用 `CachePolicy`：

```python
from langgraph.graph import StateGraph
from langgraph.types import CachePolicy

# 定义自定义 key 函数
def my_cache_key(input_data):
    # 只根据特定字段生成 key
    return f"{input_data.get('user_id')}:{input_data.get('query')}"

# 创建图
graph = StateGraph(State)

# 添加节点时指定缓存策略
graph.add_node(
    "expensive_node",
    expensive_function,
    cache_policy=CachePolicy(
        key_func=my_cache_key,  # 自定义 key 函数
        ttl=3600,               # 1 小时后过期
    )
)

# 编译时传入 cache
compiled = graph.compile(cache=InMemoryCache())
```

---

## 八、Cache vs Checkpointer vs Store 对比

| 特性 | Cache | Checkpointer | Store |
|------|-------|--------------|-------|
| **作用** | 缓存节点输出，避免重复计算 | 保存图的完整状态，支持暂停/恢复 | 持久化存储跨线程数据 |
| **粒度** | 节点级别（基于输入 hash） | 步骤级别（基于 thread_id） | 自定义命名空间 |
| **生命周期** | 可配置 TTL | 永久（直到手动删除） | 永久 |
| **典型场景** | 缓存 LLM 调用、API 请求 | 对话历史、状态恢复 | 用户记忆、长期知识 |
| **key 生成** | 输入内容的 hash | thread_id + checkpoint_id | 用户自定义 namespace + key |
| **是否可跨线程** | ✅ 是（相同输入即可命中） | ❌ 否（绑定 thread_id） | ✅ 是 |

---

## 九、总结

**`cache: BaseCache | None = None`** 参数的核心作用是：

1. **缓存节点的输出（writes）**，避免相同输入重复执行
2. **基于输入内容的 hash 生成 key**，而不是基于 thread_id
3. **支持 TTL 过期机制**
4. **跨线程/跨调用共享**，只要输入相同就能命中缓存
5. **三种实现**：InMemoryCache（开发）、SqliteCache（单机生产）、RedisCache（分布式）

**最佳实践：**
- 对于 LLM 调用和昂贵 API，启用缓存可以显著提升性能和降低成本
- 生产环境推荐使用 SqliteCache 或 RedisCache 进行持久化
- 对于需要实时性的场景，合理设置 TTL 过期时间
