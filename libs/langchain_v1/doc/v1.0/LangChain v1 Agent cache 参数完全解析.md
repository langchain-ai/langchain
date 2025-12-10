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

## 三、三种 Cache 实现及构造参数详解

### 3.1 InMemoryCache（内存缓存）

**导入路径：**
```python
from langgraph.cache.memory import InMemoryCache
```

**构造函数参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `serde` | `SerializerProtocol \| None` | `None` | 序列化器，默认使用 `JsonPlusSerializer` |

**源码定义：**
```python
class InMemoryCache(BaseCache[ValueT]):
    def __init__(self, *, serde: SerializerProtocol | None = None):
        super().__init__(serde=serde)
        self._cache: dict[Namespace, dict[str, tuple[str, bytes, float | None]]] = {}
        self._lock = threading.RLock()
```

**内部数据结构：**
```python
# _cache 的结构：
# {
#     ("__cache_writes__", "func_hash", "node_name"): {  # Namespace
#         "input_hash_key": (encoding, value_bytes, expiry_timestamp),
#     }
# }
```

**使用示例：**
```python
from langgraph.cache.memory import InMemoryCache

# 最简单的用法 - 无参数
cache = InMemoryCache()

# 使用自定义序列化器（高级用法）
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
cache = InMemoryCache(serde=JsonPlusSerializer(pickle_fallback=True))
```

**特点：**
- ✅ 最快的缓存方式（纯内存）
- ✅ 无需外部依赖
- ✅ 线程安全（使用 `threading.RLock`）
- ❌ 进程重启后丢失
- ❌ 不支持多进程共享
- 📍 适合：开发测试、单进程应用

---

### 3.2 SqliteCache（文件缓存）

**导入路径：**
```python
from langgraph.cache.sqlite import SqliteCache
```

**构造函数参数：**

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `path` | `str` | - | ✅ **必填** | SQLite 数据库文件路径 |
| `serde` | `SerializerProtocol \| None` | `None` | ❌ | 序列化器 |

**源码定义：**
```python
class SqliteCache(BaseCache[ValueT]):
    def __init__(
        self,
        *,
        path: str,  # ← 必填参数
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        # 自动创建表...
```

**数据库表结构：**
```sql
CREATE TABLE IF NOT EXISTS cache (
    ns TEXT,           -- 命名空间 (如 "__cache_writes__,func_hash,node_name")
    key TEXT,          -- 缓存 key (输入的 xxhash)
    expiry REAL,       -- 过期时间戳 (NULL 表示永不过期)
    encoding TEXT NOT NULL,  -- 序列化编码类型
    val BLOB NOT NULL,       -- 序列化后的值
    PRIMARY KEY (ns, key)
);
```

**使用示例：**
```python
from langgraph.cache.sqlite import SqliteCache

# 指定数据库文件路径
cache = SqliteCache(path="./my_cache.db")

# 使用绝对路径
cache = SqliteCache(path="/var/data/agent_cache.db")

# 内存数据库（不持久化，但比 InMemoryCache 慢）
cache = SqliteCache(path=":memory:")
```

**特点：**
- ✅ 持久化存储（进程重启后保留）
- ✅ 无需外部服务
- ✅ 线程安全
- ✅ 使用 WAL 模式提高并发性能
- ❌ 单机部署，不支持分布式
- 📍 适合：单机生产环境、需要持久化的场景

---

### 3.3 RedisCache（分布式缓存）

**导入路径：**
```python
from langgraph.cache.redis import RedisCache
```

**构造函数参数：**

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `redis` | `Any` | - | ✅ **必填** | Redis 客户端实例（如 `redis.Redis`） |
| `serde` | `SerializerProtocol \| None` | `None` | ❌ | 序列化器 |
| `prefix` | `str` | `"langgraph:cache:"` | ❌ | Redis key 前缀，用于命名空间隔离 |

**源码定义：**
```python
class RedisCache(BaseCache[ValueT]):
    def __init__(
        self,
        redis: Any,  # ← 必填：Redis 客户端
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "langgraph:cache:",  # ← 默认前缀
    ) -> None:
        super().__init__(serde=serde)
        self.redis = redis
        self.prefix = prefix
```

**Redis Key 格式：**
```
{prefix}{namespace}:{key}

示例：
langgraph:cache:__cache_writes__:func_hash:model:abc123def456
```

**使用示例：**
```python
import redis
from langgraph.cache.redis import RedisCache

# 基本用法
redis_client = redis.Redis(host='localhost', port=6379, db=0)
cache = RedisCache(redis=redis_client)

# 指定自定义前缀（推荐，用于区分不同应用）
cache = RedisCache(
    redis=redis_client,
    prefix="myapp:agent:cache:"  # 自定义前缀
)

# 带密码的 Redis
redis_client = redis.Redis(
    host='redis.example.com',
    port=6379,
    password='your_password',
    db=0,
    decode_responses=False  # 重要：保持 False 以处理二进制数据
)
cache = RedisCache(redis=redis_client)

# 使用连接池（生产推荐）
pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
redis_client = redis.Redis(connection_pool=pool)
cache = RedisCache(redis=redis_client)
```

**特点：**
- ✅ 分布式共享（多实例/多进程共享）
- ✅ 原生 TTL 支持
- ✅ 高性能
- ❌ 需要 Redis 服务
- ❌ 需要额外依赖 `pip install redis`
- 📍 适合：分布式部署、多实例共享缓存

---

### 3.4 参数来源说明

**`serde` 参数从哪里来？**

```python
# 默认使用 JsonPlusSerializer，来自：
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# 默认配置
serde = JsonPlusSerializer(pickle_fallback=True)

# 序列化流程：
# 1. 对象 → JsonPlusSerializer.dumps_typed() → (encoding, bytes)
# 2. (encoding, bytes) → JsonPlusSerializer.loads_typed() → 对象

# 你可以自定义：
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# 加密存储（需要安装 pycryptodome）
encrypted_serde = EncryptedSerializer.from_pycryptodome_aes()
cache = SqliteCache(path="./cache.db", serde=encrypted_serde)
```

**Redis 客户端从哪里来？**

```python
# 需要安装：pip install redis
import redis

# 同步客户端
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# 或者使用 Redis URL
redis_client = redis.from_url("redis://localhost:6379/0")

# 如果使用 Redis Cluster
from redis.cluster import RedisCluster
redis_client = RedisCluster(host="localhost", port=6379)
```

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

## 五、Cache 在 Agent 主循环中的详细位置

### 5.1 Agent invoke() 执行主循环源码

以下是 `Pregel.invoke()` 方法的核心循环部分，展示 cache 的精确调用位置：

```2640:2648:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/pregel/main.py
# Similarly to Bulk Synchronous Parallel / Pregel model
# computation proceeds in steps, while there are channel updates.
# Channel updates from step N are only visible in step N+1
# channels are guaranteed to be immutable for the duration of the step,
# with channel updates applied only at the transition between steps.
while loop.tick():                                          # ← 主循环
    for task in loop.match_cached_writes():                 # ← 🔥 CACHE 命中检查
        loop.output_writes(task.id, task.writes, cached=True)
    for _ in runner.tick(
        [t for t in loop.tasks.values() if not t.writes],   # ← 只执行未命中缓存的任务
        timeout=self.step_timeout,
        get_waiter=get_waiter,
        schedule_task=loop.accept_push,
    ):
        # emit output
```

### 5.2 详细执行流程图（带源码位置）

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                        Agent 执行主循环中 Cache 的精确位置                                     ║
╠══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                              ║
║   agent.invoke({"messages": [HumanMessage("hello")]})                                        ║
║                            │                                                                 ║
║                            ▼                                                                 ║
║   ┌──────────────────────────────────────────────────────────────────────────────────────┐  ║
║   │  Pregel.invoke() / Pregel.stream()                                                   │  ║
║   │  📍 源码位置: langgraph/pregel/main.py:2579-2599                                     │  ║
║   │                                                                                      │  ║
║   │  with SyncPregelLoop(                                                                │  ║
║   │      input,                                                                          │  ║
║   │      cache=cache,              # ← cache 传入                                        │  ║
║   │      cache_policy=self.cache_policy,  # ← 全局缓存策略                               │  ║
║   │      ...                                                                             │  ║
║   │  ) as loop:                                                                          │  ║
║   └──────────────────────────────────────────────────────────────────────────────────────┘  ║
║                            │                                                                 ║
║                            ▼                                                                 ║
║   ╔══════════════════════════════════════════════════════════════════════════════════════╗  ║
║   ║                         主循环: while loop.tick()                                    ║  ║
║   ║                         📍 源码位置: langgraph/pregel/main.py:2640                   ║  ║
║   ╠══════════════════════════════════════════════════════════════════════════════════════╣  ║
║   ║                                                                                      ║  ║
║   ║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║  ║
║   ║  │  STEP 1: loop.tick() - 准备任务                                              │   ║  ║
║   ║  │  📍 源码位置: langgraph/pregel/_loop.py:459-536                              │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  self.tasks = prepare_next_tasks(                                            │   ║  ║
║   ║  │      ...,                                                                    │   ║  ║
║   ║  │      cache_policy=self.cache_policy,  # ← 传入 cache_policy                  │   ║  ║
║   ║  │  )                                                                           │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  内部调用 _algo.py 生成每个任务的 cache_key:                                 │   ║  ║
║   ║  │  📍 源码位置: langgraph/pregel/_algo.py:645-664                              │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  if cache_policy:                                                            │   ║  ║
║   ║  │      args_key = cache_policy.key_func(input)  # 序列化输入                   │   ║  ║
║   ║  │      cache_key = CacheKey(                                                   │   ║  ║
║   ║  │          ns=("__cache_writes__", func_hash, node_name),                      │   ║  ║
║   ║  │          key=xxh3_128_hexdigest(args_key),    # 输入的 hash                  │   ║  ║
║   ║  │          ttl=cache_policy.ttl                                                │   ║  ║
║   ║  │      )                                                                       │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  task = PregelExecutableTask(..., cache_key=cache_key)                       │   ║  ║
║   ║  └──────────────────────────────────────────────────────────────────────────────┘   ║  ║
║   ║                            │                                                         ║  ║
║   ║                            ▼                                                         ║  ║
║   ║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║  ║
║   ║  │  STEP 2: loop.match_cached_writes() - 🔥 缓存命中检查                        │   ║  ║
║   ║  │  📍 源码位置: langgraph/pregel/_loop.py:1040-1053                            │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  def match_cached_writes(self) -> Sequence[PregelExecutableTask]:            │   ║  ║
║   ║  │      if self.cache is None:                                                  │   ║  ║
║   ║  │          return ()  # ← 没有 cache，直接返回                                 │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │      # 收集所有有 cache_key 且还没有 writes 的任务                           │   ║  ║
║   ║  │      cached = {                                                              │   ║  ║
║   ║  │          (t.cache_key.ns, t.cache_key.key): t                                │   ║  ║
║   ║  │          for t in self.tasks.values()                                        │   ║  ║
║   ║  │          if t.cache_key and not t.writes                                     │   ║  ║
║   ║  │      }                                                                       │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │      # 批量查询缓存                                                          │   ║  ║
║   ║  │      for key, values in self.cache.get(tuple(cached)).items():               │   ║  ║
║   ║  │          task = cached[key]                                                  │   ║  ║
║   ║  │          task.writes.extend(values)  # ← 🎯 命中！直接使用缓存的 writes      │   ║  ║
║   ║  │          matched.append(task)                                                │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │      return matched                                                          │   ║  ║
║   ║  └──────────────────────────────────────────────────────────────────────────────┘   ║  ║
║   ║                            │                                                         ║  ║
║   ║              ┌─────────────┴─────────────┐                                          ║  ║
║   ║              │                           │                                          ║  ║
║   ║              ▼                           ▼                                          ║  ║
║   ║  ┌─────────────────────────┐  ┌─────────────────────────────────────────────────┐   ║  ║
║   ║  │   缓存命中 (CACHE HIT)  │  │  缓存未命中 (CACHE MISS)                        │   ║  ║
║   ║  │                         │  │                                                 │   ║  ║
║   ║  │  # 输出缓存的结果       │  │  STEP 3: runner.tick() - 实际执行任务           │   ║  ║
║   ║  │  loop.output_writes(    │  │  📍 源码位置: langgraph/pregel/main.py:2643     │   ║  ║
║   ║  │      task.id,           │  │                                                 │   ║  ║
║   ║  │      task.writes,       │  │  for _ in runner.tick(                          │   ║  ║
║   ║  │      cached=True  # ←   │  │      [t for t in loop.tasks.values()            │   ║  ║
║   ║  │  )                      │  │       if not t.writes],  # ← 只执行未命中的     │   ║  ║
║   ║  │                         │  │      ...                                        │   ║  ║
║   ║  │  ⚡ 跳过实际执行！      │  │  ):                                             │   ║  ║
║   ║  │                         │  │      ...                                        │   ║  ║
║   ║  └─────────────────────────┘  │                                                 │   ║  ║
║   ║                               │  # 任务执行（调用 LLM、工具等）                 │   ║  ║
║   ║                               │  # 执行完成后调用 put_writes                    │   ║  ║
║   ║                               └─────────────────────────────────────────────────┘   ║  ║
║   ║                                             │                                        ║  ║
║   ║                                             ▼                                        ║  ║
║   ║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║  ║
║   ║  │  STEP 4: loop.put_writes() - 🔥 写入缓存                                     │   ║  ║
║   ║  │  📍 源码位置: langgraph/pregel/_loop.py:1063-1079                            │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  def put_writes(self, task_id: str, writes: WritesT) -> None:                │   ║  ║
║   ║  │      super().put_writes(task_id, writes)  # 先调用父类方法                   │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │      # 检查是否需要写入缓存                                                  │   ║  ║
║   ║  │      if not writes or self.cache is None:                                    │   ║  ║
║   ║  │          return                                                              │   ║  ║
║   ║  │      task = self.tasks.get(task_id)                                          │   ║  ║
║   ║  │      if task is None or task.cache_key is None:                              │   ║  ║
║   ║  │          return  # ← 没有 cache_key，不写入缓存                              │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │      # 异步写入缓存（不阻塞主流程）                                          │   ║  ║
║   ║  │      self.submit(                                                            │   ║  ║
║   ║  │          self.cache.set,                                                     │   ║  ║
║   ║  │          {                                                                   │   ║  ║
║   ║  │              (task.cache_key.ns, task.cache_key.key): (                      │   ║  ║
║   ║  │                  task.writes,           # ← 缓存任务的输出                   │   ║  ║
║   ║  │                  task.cache_key.ttl     # ← 使用配置的 TTL                   │   ║  ║
║   ║  │              )                                                               │   ║  ║
║   ║  │          },                                                                  │   ║  ║
║   ║  │      )                                                                       │   ║  ║
║   ║  │      # 🎯 下次相同输入的任务将直接命中缓存！                                 │   ║  ║
║   ║  └──────────────────────────────────────────────────────────────────────────────┘   ║  ║
║   ║                            │                                                         ║  ║
║   ║                            ▼                                                         ║  ║
║   ║  ┌──────────────────────────────────────────────────────────────────────────────┐   ║  ║
║   ║  │  STEP 5: loop.after_tick() - 完成当前步骤                                    │   ║  ║
║   ║  │  📍 源码位置: langgraph/pregel/_loop.py:538-571                              │   ║  ║
║   ║  │                                                                              │   ║  ║
║   ║  │  # 应用 writes 到 channels                                                   │   ║  ║
║   ║  │  # 保存 checkpoint                                                           │   ║  ║
║   ║  │  # 检查是否需要中断                                                          │   ║  ║
║   ║  └──────────────────────────────────────────────────────────────────────────────┘   ║  ║
║   ║                            │                                                         ║  ║
║   ║                            ▼                                                         ║  ║
║   ║                     继续下一轮 tick() 或退出循环                                     ║  ║
║   ╚══════════════════════════════════════════════════════════════════════════════════════╝  ║
║                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

### 5.3 关键时序图

```
时间轴 →

第一次调用 (CACHE MISS):
┌───────────────────────────────────────────────────────────────────────────────────┐
│ tick()           match_cached_writes()     runner.tick()       put_writes()       │
│   │                      │                      │                   │             │
│   ▼                      ▼                      ▼                   ▼             │
│ 生成             查询缓存                 实际执行任务        写入缓存            │
│ cache_key        (未命中)                 (调用 LLM)         (异步)              │
│   │                      │                      │                   │             │
│   │ cache_key ──────────►│ get() ──► {}        │                   │             │
│   │ = CacheKey(          │                      │                   │             │
│   │   ns, hash, ttl      │                      │ LLM API call      │             │
│   │ )                    │                      │ ............      │             │
│   │                      │                      │ → writes          │             │
│   │                      │                      │                   │             │
│   │                      │                      └───────────────────► set({       │
│   │                      │                                           │   key:     │
│   │                      │                                           │   (writes, │
│   │                      │                                           │    ttl)    │
│   │                      │                                           │ })         │
└───────────────────────────────────────────────────────────────────────────────────┘

第二次相同调用 (CACHE HIT):
┌───────────────────────────────────────────────────────────────────────────────────┐
│ tick()           match_cached_writes()     runner.tick()       put_writes()       │
│   │                      │                      │                   │             │
│   ▼                      ▼                                                        │
│ 生成             查询缓存                 ⚡ 跳过执行！                           │
│ cache_key        (命中!)                                                          │
│   │                      │                                                        │
│   │ cache_key ──────────►│ get() ──► {key: writes}                               │
│   │ (相同的 hash)        │                                                        │
│   │                      │                                                        │
│   │                      │ task.writes = cached_writes                            │
│   │                      │ output_writes(cached=True)                             │
│   │                      │                                                        │
│   │                      │ ⚡ 直接返回缓存结果，无需调用 LLM！                    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Cache 相关的源码文件索引

| 文件路径 | 关键函数/类 | 作用 |
|----------|-------------|------|
| `langgraph/pregel/main.py` | `Pregel.invoke()` | 主入口，创建 loop 并传入 cache |
| `langgraph/pregel/_loop.py` | `SyncPregelLoop` | 主循环，包含 match_cached_writes, put_writes |
| `langgraph/pregel/_loop.py:459` | `tick()` | 单次迭代，准备任务 |
| `langgraph/pregel/_loop.py:1040` | `match_cached_writes()` | 批量查询缓存命中 |
| `langgraph/pregel/_loop.py:1063` | `put_writes()` | 任务完成后写入缓存 |
| `langgraph/pregel/_algo.py:645` | cache_key 生成逻辑 | 生成 CacheKey(ns, key, ttl) |
| `langgraph/cache/base/__init__.py` | `BaseCache` | 缓存抽象基类 |
| `langgraph/cache/memory/__init__.py` | `InMemoryCache` | 内存缓存实现 |
| `langgraph/cache/sqlite/__init__.py` | `SqliteCache` | SQLite 缓存实现 |
| `langgraph/cache/redis/__init__.py` | `RedisCache` | Redis 缓存实现 |

---

## 六、Cache Key 生成机制

### 6.1 默认 key 函数

```26:31:/Users/cong/code/github/langgraph/libs/langgraph/langgraph/_internal/_cache.py
def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments
    to generate a hashable key."""
    import pickle

    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
```

### 6.2 完整的 cache_key 生成（_algo.py）

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

## 七、缓存命中与写入逻辑

### 7.1 缓存命中检查（执行前）

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

### 7.2 缓存写入（执行后）

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

## 八、使用场景

### 8.1 场景一：缓存昂贵的 API 调用

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

### 8.2 场景二：持久化缓存（跨进程重启）

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

### 8.3 场景三：分布式缓存（多实例共享）

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

### 8.4 场景四：带 TTL 的节点级缓存策略

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

## 九、查询和调试缓存数据

### 9.1 InMemoryCache - 直接访问内部数据

```python
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()

# ... 使用 agent 一段时间后 ...

# 直接查看内部缓存结构
print("缓存的命名空间：", list(cache._cache.keys()))

# 查看某个命名空间下的所有 key
for ns, entries in cache._cache.items():
    print(f"\n命名空间: {ns}")
    for key, (encoding, value_bytes, expiry) in entries.items():
        print(f"  Key: {key[:20]}...")  # key 通常很长，只显示前 20 位
        print(f"  过期时间: {expiry}")
        # 解码查看值
        value = cache.serde.loads_typed((encoding, value_bytes))
        print(f"  值: {value}")

# 手动查询特定 key
from langgraph.cache.base import FullKey
keys_to_query: list[FullKey] = [
    (("__cache_writes__", "some_hash", "model"), "input_hash_key")
]
results = cache.get(keys_to_query)
print("查询结果：", results)

# 手动清除所有缓存
cache.clear()
```

### 9.2 SqliteCache - 使用 SQL 查询

```python
from langgraph.cache.sqlite import SqliteCache
import sqlite3

cache = SqliteCache(path="./cache.db")

# ... 使用 agent 一段时间后 ...

# 方法 1：通过 cache 对象的内部连接查询
with cache._lock:
    cursor = cache._conn.execute("SELECT ns, key, expiry FROM cache")
    for row in cursor.fetchall():
        ns, key, expiry = row
        print(f"命名空间: {ns}")
        print(f"Key: {key[:20]}...")
        print(f"过期时间: {expiry}")
        print("---")

# 方法 2：直接打开数据库文件查询
conn = sqlite3.connect("./cache.db")

# 查看所有缓存条目数量
cursor = conn.execute("SELECT COUNT(*) FROM cache")
print(f"缓存条目总数: {cursor.fetchone()[0]}")

# 查看所有命名空间
cursor = conn.execute("SELECT DISTINCT ns FROM cache")
for (ns,) in cursor.fetchall():
    print(f"命名空间: {ns}")

# 查看特定节点的缓存
cursor = conn.execute("""
    SELECT key, expiry, encoding
    FROM cache
    WHERE ns LIKE '%model%'
""")
for key, expiry, encoding in cursor.fetchall():
    print(f"Key: {key[:20]}..., 过期: {expiry}, 编码: {encoding}")

# 删除过期条目
import time
now = time.time()
conn.execute("DELETE FROM cache WHERE expiry IS NOT NULL AND expiry < ?", (now,))
conn.commit()

# 删除特定节点的缓存
conn.execute("DELETE FROM cache WHERE ns LIKE '%model%'")
conn.commit()

conn.close()
```

### 9.3 RedisCache - 使用 Redis 命令查询

```python
import redis
from langgraph.cache.redis import RedisCache

redis_client = redis.Redis(host='localhost', port=6379, db=0)
cache = RedisCache(redis=redis_client, prefix="myapp:cache:")

# ... 使用 agent 一段时间后 ...

# 查看所有缓存 key
keys = redis_client.keys("myapp:cache:*")
print(f"缓存 key 数量: {len(keys)}")

for key in keys[:10]:  # 只显示前 10 个
    key_str = key.decode() if isinstance(key, bytes) else key
    print(f"Key: {key_str}")

    # 查看 TTL
    ttl = redis_client.ttl(key)
    print(f"  TTL: {ttl} 秒 (-1 表示永不过期, -2 表示不存在)")

    # 查看值大小
    value = redis_client.get(key)
    if value:
        print(f"  值大小: {len(value)} bytes")

# 使用 Redis CLI 查看（命令行）
# redis-cli KEYS "myapp:cache:*"
# redis-cli TTL "myapp:cache:__cache_writes__:xxx:model:abc123"
# redis-cli GET "myapp:cache:__cache_writes__:xxx:model:abc123"

# 删除特定前缀的所有缓存
keys_to_delete = redis_client.keys("myapp:cache:*model*")
if keys_to_delete:
    redis_client.delete(*keys_to_delete)
    print(f"删除了 {len(keys_to_delete)} 个缓存条目")

# 清除所有缓存
cache.clear()
```

---

## 十、清除缓存的方法

### 10.1 通过 Agent 的 `clear_cache()` 方法（推荐）

```python
from langchain.agents import create_agent
from langgraph.cache.memory import InMemoryCache

cache = InMemoryCache()
agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tool],
    cache=cache,
)

# 使用 agent...

# 清除所有节点的缓存
agent.clear_cache()

# 清除特定节点的缓存
agent.clear_cache(nodes=["model"])  # 只清除 model 节点
agent.clear_cache(nodes=["tools"])  # 只清除 tools 节点

# 异步版本
await agent.aclear_cache()
await agent.aclear_cache(nodes=["model"])
```

**源码实现：**
```python
# langgraph/pregel/main.py
def clear_cache(self, nodes: Sequence[str] | None = None) -> None:
    """Clear the cache for the given nodes."""
    if not self.cache:
        raise ValueError("No cache is set for this graph. Cannot clear cache.")
    nodes = nodes or self.nodes.keys()
    # 收集要清除的命名空间
    namespaces: list[tuple[str, ...]] = []
    for node in nodes:
        if node in self.nodes:
            namespaces.append(
                (
                    CACHE_NS_WRITES,  # "__cache_writes__"
                    (identifier(self.nodes[node]) or "__dynamic__"),
                    node,
                ),
            )
    # 清除缓存
    self.cache.clear(namespaces)
```

### 10.2 直接调用 Cache 的 `clear()` 方法

```python
# 清除所有缓存
cache.clear()

# 清除特定命名空间
cache.clear(namespaces=[
    ("__cache_writes__", "func_hash_abc", "model"),
    ("__cache_writes__", "func_hash_def", "tools"),
])

# 异步版本
await cache.aclear()
```

### 10.3 针对不同 Cache 实现的清除方式

**InMemoryCache：**
```python
# 清除所有
cache._cache.clear()

# 清除特定命名空间
ns_to_delete = ("__cache_writes__", "xxx", "model")
if ns_to_delete in cache._cache:
    del cache._cache[ns_to_delete]
```

**SqliteCache：**
```python
# 清除所有
cache._conn.execute("DELETE FROM cache")
cache._conn.commit()

# 清除特定节点
cache._conn.execute("DELETE FROM cache WHERE ns LIKE '%model%'")
cache._conn.commit()

# 清除过期条目
import time
cache._conn.execute("DELETE FROM cache WHERE expiry < ?", (time.time(),))
cache._conn.commit()
```

**RedisCache：**
```python
# 清除所有（带前缀）
keys = cache.redis.keys(f"{cache.prefix}*")
if keys:
    cache.redis.delete(*keys)

# 清除特定模式
keys = cache.redis.keys(f"{cache.prefix}*model*")
if keys:
    cache.redis.delete(*keys)

# 使用 Redis CLI
# redis-cli DEL myapp:cache:__cache_writes__:xxx:model:abc123
# redis-cli --scan --pattern "myapp:cache:*" | xargs redis-cli DEL
```

---

## 十一、Cache vs Checkpointer vs Store 对比

| 特性 | Cache | Checkpointer | Store |
|------|-------|--------------|-------|
| **作用** | 缓存节点输出，避免重复计算 | 保存图的完整状态，支持暂停/恢复 | 持久化存储跨线程数据 |
| **粒度** | 节点级别（基于输入 hash） | 步骤级别（基于 thread_id） | 自定义命名空间 |
| **生命周期** | 可配置 TTL | 永久（直到手动删除） | 永久 |
| **典型场景** | 缓存 LLM 调用、API 请求 | 对话历史、状态恢复 | 用户记忆、长期知识 |
| **key 生成** | 输入内容的 hash | thread_id + checkpoint_id | 用户自定义 namespace + key |
| **是否可跨线程** | ✅ 是（相同输入即可命中） | ❌ 否（绑定 thread_id） | ✅ 是 |

---

## 十二、总结

### 核心作用

**`cache: BaseCache | None = None`** 参数的核心作用是：

1. **缓存节点的输出（writes）**，避免相同输入重复执行
2. **基于输入内容的 hash 生成 key**，而不是基于 thread_id
3. **支持 TTL 过期机制**
4. **跨线程/跨调用共享**，只要输入相同就能命中缓存

### 三种实现选择指南

| 实现 | 构造参数 | 适用场景 |
|------|----------|----------|
| `InMemoryCache()` | `serde` (可选) | 开发测试、单进程应用 |
| `SqliteCache(path="./cache.db")` | `path` (必填), `serde` (可选) | 单机生产、需要持久化 |
| `RedisCache(redis=client)` | `redis` (必填), `prefix` (可选), `serde` (可选) | 分布式部署、多实例共享 |

### 最佳实践

```python
# 开发环境
from langgraph.cache.memory import InMemoryCache
cache = InMemoryCache()

# 单机生产
from langgraph.cache.sqlite import SqliteCache
cache = SqliteCache(path="/var/data/agent_cache.db")

# 分布式生产
import redis
from langgraph.cache.redis import RedisCache
redis_client = redis.Redis(host='redis-host', port=6379)
cache = RedisCache(redis=redis_client, prefix="myapp:agent:")

# 使用缓存
agent = create_agent(
    model="openai:gpt-4o",
    tools=[my_tools],
    cache=cache,
)

# 查看缓存（调试）
# InMemoryCache: cache._cache
# SqliteCache: SELECT * FROM cache
# RedisCache: redis-cli KEYS "myapp:agent:*"

# 清除缓存
agent.clear_cache()  # 清除所有
agent.clear_cache(nodes=["model"])  # 清除特定节点
```

### 注意事项

- ⚠️ 缓存基于**输入内容**的 hash，输入相同才会命中
- ⚠️ 缓存的是节点的**完整输出**（writes），包括所有返回值
- ⚠️ 默认使用 `pickle` 序列化，确保你的数据类型是可序列化的
- ⚠️ RedisCache 需要安装 `pip install redis`
- ⚠️ SqliteCache 的数据库文件路径需要有写入权限
