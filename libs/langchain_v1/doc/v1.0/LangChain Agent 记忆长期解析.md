
# LangGraph Store 深度解析：Agent 长期记忆解析

## 一、Store 核心架构

### 1.1 整体架构图

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                  ║
║                              LangGraph Store 架构总览                                             ║
║                                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                  ║
║   ┌────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                   BaseStore (抽象基类)                                     │ ║
║   │                        langgraph/store/base/__init__.py                                   │ ║
║   │                                                                                            │ ║
║   │   核心方法：                                                                                │ ║
║   │   ├─ put(namespace, key, value)       # 存储数据                                          │ ║
║   │   ├─ get(namespace, key)              # 获取单个数据                                      │ ║
║   │   ├─ search(namespace_prefix, ...)    # 搜索数据（支持语义搜索）                          │ ║
║   │   ├─ delete(namespace, key)           # 删除数据                                          │ ║
║   │   ├─ list_namespaces(...)             # 列出命名空间                                      │ ║
║   │   └─ batch(ops)                       # 批量操作                                          │ ║
║   │                                                                                            │ ║
║   │   异步版本：aput, aget, asearch, adelete, alist_namespaces, abatch                        │ ║
║   │                                                                                            │ ║
║   └────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                            │                                                     ║
║                                            ▼                                                     ║
║   ┌────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                    实现类层级                                               │ ║
║   │                                                                                            │ ║
║   │   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐             │ ║
║   │   │                     │   │                     │   │                     │             │ ║
║   │   │   InMemoryStore     │   │   PostgresStore     │   │   SqliteStore       │             │ ║
║   │   │   (内存存储)        │   │   (PostgreSQL)      │   │   (SQLite)          │             │ ║
║   │   │                     │   │                     │   │                     │             │ ║
║   │   │ langgraph/store/    │   │ langgraph/store/    │   │ langgraph/store/    │             │ ║
║   │   │ memory/__init__.py  │   │ postgres/base.py    │   │ sqlite/base.py      │             │ ║
║   │   │                     │   │                     │   │                     │             │ ║
║   │   │ ⚠️ 进程退出数据丢失  │   │ ✅ 持久化存储       │   │ ✅ 持久化存储       │             │ ║
║   │   │ ✅ 开发测试使用     │   │ ✅ 生产环境使用     │   │ ✅ 单机部署使用     │             │ ║
║   │   │                     │   │ ✅ 支持向量搜索     │   │ ✅ 支持向量搜索     │             │ ║
║   │   │                     │   │ ✅ 支持 TTL         │   │ ✅ 支持 TTL         │             │ ║
║   │   │                     │   │                     │   │                     │             │ ║
║   │   └─────────────────────┘   └─────────────────────┘   └─────────────────────┘             │ ║
║   │                                                                                            │ ║
║   │   ┌─────────────────────┐   ┌─────────────────────┐                                       │ ║
║   │   │                     │   │                     │                                       │ ║
║   │   │ AsyncPostgresStore  │   │  AsyncSqliteStore   │                                       │ ║
║   │   │ (异步 PostgreSQL)   │   │  (异步 SQLite)      │                                       │ ║
║   │   │                     │   │                     │                                       │ ║
║   │   └─────────────────────┘   └─────────────────────┘                                       │ ║
║   │                                                                                            │ ║
║   └────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 二、核心数据结构

### 2.1 Item 类（存储的数据项）

源码位置：`langgraph/store/base/__init__.py` 第 51-116 行

```python
class Item:
    """存储的数据项，包含值和元数据"""

    __slots__ = ("value", "key", "namespace", "created_at", "updated_at")

    def __init__(
        self,
        *,
        value: dict[str, Any],           # 存储的数据（必须是字典）
        key: str,                          # 唯一标识符
        namespace: tuple[str, ...],        # 命名空间路径
        created_at: datetime,              # 创建时间
        updated_at: datetime,              # 更新时间
    ):
        self.value = value
        self.key = key
        self.namespace = tuple(namespace)
        self.created_at = created_at
        self.updated_at = updated_at
```

**Item 结构示意图：**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                    Item                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   namespace: ("users", "preferences")    ← 命名空间（类似文件夹路径）           │
│                     ▲                                                           │
│                     │                                                           │
│   key: "theme"      │                    ← 键（类似文件名）                      │
│           ▲         │                                                           │
│           │         │                                                           │
│           └─────────┴───▶  完整路径: ("users", "preferences") / "theme"         │
│                                                                                 │
│   value: {"mode": "dark", "color": "blue"}  ← 值（必须是 dict）                 │
│                                                                                 │
│   created_at: 2024-01-15 10:30:00+00:00     ← 创建时间                          │
│   updated_at: 2024-01-15 14:20:00+00:00     ← 最后更新时间                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 SearchItem 类（搜索结果项）

源码位置：`langgraph/store/base/__init__.py` 第 118-154 行

```python
class SearchItem(Item):
    """搜索返回的数据项，额外包含相似度分数"""

    __slots__ = ("score",)

    def __init__(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        created_at: datetime,
        updated_at: datetime,
        score: float | None = None,  # ← 相似度分数（语义搜索时有值）
    ):
        super().__init__(...)
        self.score = score
```

### 2.3 操作类型（Op 类型）

源码位置：`langgraph/store/base/__init__.py` 第 157-536 行

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              四种操作类型                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────┐   ┌──────────────────┐                                   │
│   │     GetOp        │   │     PutOp        │                                   │
│   │                  │   │                  │                                   │
│   │ namespace: tuple │   │ namespace: tuple │                                   │
│   │ key: str         │   │ key: str         │                                   │
│   │ refresh_ttl:bool │   │ value: dict|None │  ← None 表示删除                  │
│   │                  │   │ index: list|None │  ← 指定索引字段                   │
│   │                  │   │ ttl: float|None  │  ← 过期时间（分钟）               │
│   └──────────────────┘   └──────────────────┘                                   │
│                                                                                 │
│   ┌──────────────────┐   ┌──────────────────┐                                   │
│   │    SearchOp      │   │ ListNamespacesOp │                                   │
│   │                  │   │                  │                                   │
│   │ namespace_prefix │   │ match_conditions │                                   │
│   │ filter: dict     │   │ max_depth: int   │                                   │
│   │ limit: int       │   │ limit: int       │                                   │
│   │ offset: int      │   │ offset: int      │                                   │
│   │ query: str       │   │                  │                                   │
│   │ refresh_ttl:bool │   │                  │                                   │
│   └──────────────────┘   └──────────────────┘                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、BaseStore API 完整解析

### 3.1 put / aput - 存储数据

源码位置：`langgraph/store/base/__init__.py` 第 848-927 行

```python
def put(
    self,
    namespace: tuple[str, ...],           # 命名空间路径
    key: str,                              # 键
    value: dict[str, Any],                 # 值（必须是字典）
    index: Literal[False] | list[str] | None = None,  # 索引配置
    *,
    ttl: float | None | NotProvided = NOT_PROVIDED,   # TTL（分钟）
) -> None:
    """存储或更新数据

    参数说明：
    - namespace: 命名空间路径，如 ("users", "123", "preferences")
    - key: 唯一标识符，如 "theme"
    - value: 存储的数据，必须是 dict
    - index:
        - None: 使用 store 默认的索引配置
        - False: 不索引此项（不支持语义搜索）
        - ["field1", "field2"]: 索引指定字段
    - ttl: 过期时间（分钟），None 表示不过期
    """
```

**用法示例：**

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 基本存储
store.put(
    ("users", "u123"),      # namespace
    "profile",               # key
    {"name": "张三", "age": 25}  # value
)

# 带索引的存储（支持语义搜索）
store.put(
    ("documents",),
    "doc1",
    {"title": "Python 教程", "content": "Python 是一门优秀的编程语言"},
    index=["title", "content"]  # 索引这两个字段
)

# 带 TTL 的存储
store.put(
    ("cache",),
    "weather",
    {"city": "北京", "temp": 25},
    ttl=60  # 60 分钟后过期
)
```

### 3.2 get / aget - 获取数据

源码位置：`langgraph/store/base/__init__.py` 第 748-769 行

```python
def get(
    self,
    namespace: tuple[str, ...],
    key: str,
    *,
    refresh_ttl: bool | None = None,  # 是否刷新 TTL
) -> Item | None:
    """获取单个数据项

    返回：Item 对象或 None（不存在时）
    """
```

**用法示例：**

```python
# 获取数据
item = store.get(("users", "u123"), "profile")

if item:
    print(f"用户名: {item.value['name']}")
    print(f"创建时间: {item.created_at}")
    print(f"更新时间: {item.updated_at}")
else:
    print("用户不存在")
```

### 3.3 search / asearch - 搜索数据

源码位置：`langgraph/store/base/__init__.py` 第 771-846 行

```python
def search(
    self,
    namespace_prefix: tuple[str, ...],  # 命名空间前缀
    /,
    *,
    query: str | None = None,           # 语义搜索查询
    filter: dict[str, Any] | None = None,  # 过滤条件
    limit: int = 10,                    # 返回数量限制
    offset: int = 0,                    # 偏移量（分页）
    refresh_ttl: bool | None = None,    # 是否刷新 TTL
) -> list[SearchItem]:
    """搜索数据

    支持两种搜索模式：
    1. 过滤搜索：使用 filter 参数
    2. 语义搜索：使用 query 参数（需要配置 index）
    """
```

**用法示例：**

```python
# 1. 简单过滤搜索
results = store.search(
    ("users",),  # 在 users 命名空间下搜索
    filter={"status": "active"},
    limit=10
)

# 2. 带比较运算符的过滤
results = store.search(
    ("orders",),
    filter={
        "amount": {"$gt": 100},     # 金额大于 100
        "status": {"$ne": "cancelled"}  # 状态不等于 cancelled
    }
)

# 3. 语义搜索（需要配置 index）
results = store.search(
    ("documents",),
    query="Python 编程入门",  # 自然语言查询
    limit=5
)

# 遍历结果
for item in results:
    print(f"Key: {item.key}")
    print(f"Value: {item.value}")
    print(f"Score: {item.score}")  # 语义搜索时有相似度分数
```

**支持的过滤运算符：**

| 运算符 | 含义 | 示例 |
|-------|------|------|
| `$eq` | 等于 | `{"status": {"$eq": "active"}}` |
| `$ne` | 不等于 | `{"status": {"$ne": "deleted"}}` |
| `$gt` | 大于 | `{"score": {"$gt": 80}}` |
| `$gte` | 大于等于 | `{"score": {"$gte": 60}}` |
| `$lt` | 小于 | `{"price": {"$lt": 100}}` |
| `$lte` | 小于等于 | `{"price": {"$lte": 50}}` |

### 3.4 delete / adelete - 删除数据

源码位置：`langgraph/store/base/__init__.py` 第 929-936 行

```python
def delete(self, namespace: tuple[str, ...], key: str) -> None:
    """删除数据项"""
    self.batch([PutOp(namespace, str(key), None, ttl=None)])
```

**用法示例：**

```python
# 删除单个数据
store.delete(("users", "u123"), "profile")
```

### 3.5 list_namespaces / alist_namespaces - 列出命名空间

源码位置：`langgraph/store/base/__init__.py` 第 938-991 行

```python
def list_namespaces(
    self,
    *,
    prefix: NamespacePath | None = None,   # 前缀匹配
    suffix: NamespacePath | None = None,   # 后缀匹配
    max_depth: int | None = None,          # 最大深度
    limit: int = 100,                      # 数量限制
    offset: int = 0,                       # 偏移量
) -> list[tuple[str, ...]]:
    """列出命名空间"""
```

**用法示例：**

```python
# 列出所有命名空间
all_namespaces = store.list_namespaces()
# [("users",), ("users", "u123"), ("documents",), ...]

# 列出以 users 开头的命名空间
user_namespaces = store.list_namespaces(prefix=("users",))
# [("users",), ("users", "u123"), ("users", "u456"), ...]

# 限制深度
namespaces = store.list_namespaces(prefix=("users",), max_depth=2)
# [("users",), ("users", "u123"), ("users", "u456")]  # 深度限制为 2
```

---

## 四、Store 实现详解

### 4.1 InMemoryStore - 内存存储

源码位置：`langgraph/store/memory/__init__.py`

```python
class InMemoryStore(BaseStore):
    """内存字典存储，支持可选的向量搜索

    特点：
    - 数据存储在内存中
    - 进程退出后数据丢失
    - 支持语义搜索（需配置 index）
    - 适合开发测试
    """

    __slots__ = ("_data", "_vectors", "index_config", "embeddings")

    def __init__(self, *, index: IndexConfig | None = None) -> None:
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)
        self._vectors: dict[tuple[str, ...], dict[str, dict[str, list[float]]]] = ...
        self.index_config = index
        if index:
            self.embeddings = ensure_embeddings(index.get("embed"))
```

**内部数据结构：**

```
InMemoryStore 内部结构
═══════════════════════════════════════════════════════════════════════════════

_data: dict[namespace, dict[key, Item]]
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  _data = {                                                                    │
│      ("users", "u123"): {                                                     │
│          "profile": Item(value={"name": "张三"}, key="profile", ...),         │
│          "settings": Item(value={"theme": "dark"}, key="settings", ...),      │
│      },                                                                       │
│      ("documents",): {                                                        │
│          "doc1": Item(value={"title": "Python"}, key="doc1", ...),            │
│      },                                                                       │
│  }                                                                            │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

_vectors: dict[namespace, dict[key, dict[field_path, embedding]]]
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  _vectors = {                                                                 │
│      ("documents",): {                                                        │
│          "doc1": {                                                            │
│              "title": [0.1, 0.2, ...],    # 1536 维向量                       │
│              "content": [0.3, 0.4, ...],                                      │
│          },                                                                   │
│      },                                                                       │
│  }                                                                            │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

**使用示例：**

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# 1. 基本使用（无向量搜索）
store = InMemoryStore()

# 2. 启用向量搜索
store = InMemoryStore(
    index={
        "dims": 1536,  # 向量维度
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text", "title"],  # 要索引的字段
    }
)
```

### 4.2 PostgresStore - PostgreSQL 存储

源码位置：`langgraph/store/postgres/base.py`

```python
class PostgresStore(BaseStore, BasePostgresStore):
    """PostgreSQL 存储，支持向量搜索（pgvector）

    特点：
    - 持久化存储
    - 支持 pgvector 向量搜索
    - 支持 TTL（自动过期）
    - 支持 HNSW/IVFFlat 索引
    - 适合生产环境
    """

    supports_ttl: bool = True  # ← 支持 TTL
```

**数据库表结构：**

```sql
-- 主数据表
CREATE TABLE store (
    prefix text NOT NULL,           -- 命名空间（.分隔）
    key text NOT NULL,              -- 键
    value jsonb NOT NULL,           -- 值（JSONB）
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,  -- 过期时间
    ttl_minutes INT,                -- TTL 分钟数
    PRIMARY KEY (prefix, key)
);

-- 向量表（可选）
CREATE TABLE store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,       -- 字段路径
    embedding vector(1536),         -- 向量（pgvector 类型）
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prefix, key, field_name),
    FOREIGN KEY (prefix, key) REFERENCES store(prefix, key) ON DELETE CASCADE
);
```

**使用示例：**

```python
from langgraph.store.postgres import PostgresStore
from langchain.embeddings import init_embeddings

conn_string = "postgresql://user:pass@localhost:5432/dbname"

# 1. 基本使用
with PostgresStore.from_conn_string(conn_string) as store:
    store.setup()  # 首次使用需要运行迁移

    store.put(("users",), "u1", {"name": "张三"})
    item = store.get(("users",), "u1")

# 2. 启用向量搜索
with PostgresStore.from_conn_string(
    conn_string,
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text"],
        "ann_index_config": {"kind": "hnsw"},  # 使用 HNSW 索引
        "distance_type": "cosine",             # 余弦相似度
    }
) as store:
    store.setup()

    # 存储并索引
    store.put(("docs",), "d1", {"text": "Python 教程"})

    # 语义搜索
    results = store.search(("docs",), query="编程入门")

# 3. 启用 TTL
with PostgresStore.from_conn_string(
    conn_string,
    ttl={
        "default_ttl": 60,           # 默认 60 分钟过期
        "refresh_on_read": True,     # 读取时刷新 TTL
        "sweep_interval_minutes": 5  # 每 5 分钟清理过期数据
    }
) as store:
    store.setup()
    store.start_ttl_sweeper()  # 启动清理线程

    store.put(("cache",), "key", {"data": "value"})  # 60 分钟后过期
```

### 4.3 SqliteStore - SQLite 存储

源码位置：`langgraph/store/sqlite/base.py`

```python
class SqliteStore(BaseStore, BaseSqliteStore):
    """SQLite 存储，支持向量搜索（sqlite-vec）

    特点：
    - 持久化到文件
    - 支持向量搜索（sqlite-vec）
    - 支持 TTL
    - 适合单机部署
    """

    supports_ttl: bool = True
```

**数据库表结构：**

```sql
CREATE TABLE store (
    prefix text NOT NULL,
    key text NOT NULL,
    value text NOT NULL,           -- JSON 字符串
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    ttl_minutes REAL,
    PRIMARY KEY (prefix, key)
);

CREATE TABLE store_vectors (
    prefix text NOT NULL,
    key text NOT NULL,
    field_name text NOT NULL,
    embedding BLOB,                -- 向量二进制存储
    PRIMARY KEY (prefix, key, field_name)
);
```

**使用示例：**

```python
from langgraph.store.sqlite import SqliteStore

# 基本使用
with SqliteStore.from_conn_string("my_store.db") as store:
    store.setup()

    store.put(("users",), "u1", {"name": "张三"})
    item = store.get(("users",), "u1")
```

---

## 五、在 Agent 中使用 Store

### 5.1 Store 在 Agent 中的数据流

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                  ║
║                          Store 在 Agent 中的完整数据流                                            ║
║                                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                  ║
║   用户代码                                                                                       ║
║   ═══════════════════════════════════════════════════════════════════════════════════════════    ║
║                                                                                                  ║
║   ┌────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                            │ ║
║   │  # 1. 创建 Store 实例                                                                      │ ║
║   │  store = InMemoryStore()                                                                   │ ║
║   │                                                                                            │ ║
║   │  # 2. 可选：预加载数据                                                                     │ ║
║   │  store.put(("users",), "current", {"name": "张三", "level": "vip"})                        │ ║
║   │                                                                                            │ ║
║   └────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                              │                                                   ║
║                                              ▼                                                   ║
║   ┌────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║   │                                                                                            │ ║
║   │  # 3. 传入 create_agent                                                                    │ ║
║   │  agent = create_agent(                                                                     │ ║
║   │      model="openai:gpt-4o",                                                                │ ║
║   │      tools=[my_tool],                                                                      │ ║
║   │      middleware=[MyMiddleware()],                                                          │ ║
║   │      store=store,  ← ─────────────────────────────────────────────────────────────────┐    │ ║
║   │  )                                                                                    │    │ ║
║   │                                                                                       │    │ ║
║   └───────────────────────────────────────────────────────────────────────────────────────│────┘ ║
║                                              │                                            │      ║
║                                              ▼                                            │      ║
║   LangChain Agent 内部                                                                    │      ║
║   ═══════════════════════════════════════════════════════════════════════════════════════│═     ║
║                                                                                          │      ║
║   ┌──────────────────────────────────────────────────────────────────────────────────────│────┐ ║
║   │  # 4. factory.py 中传递给 graph.compile                                              │    │ ║
║   │                                                                                      │    │ ║
║   │  return graph.compile(                                                               │    │ ║
║   │      checkpointer=checkpointer,                                                      │    │ ║
║   │      store=store,  ← ────────────────────────────────────────────────────────────────┘    │ ║
║   │      ...                                                                                  │ ║
║   │  )                                                                                        │ ║
║   │                                                                                           │ ║
║   └───────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                              │                                                   ║
║                                              ▼                                                   ║
║   LangGraph 运行时                                                                              ║
║   ════════════════════════════════════════════════════════════════════════════════════════      ║
║                                                                                                  ║
║   ┌────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║   │  # 5. 每个节点执行时，LangGraph 自动创建 Runtime 对象                                      │ ║
║   │                                                                                            │ ║
║   │  runtime = Runtime(                                                                        │ ║
║   │      store=compiled_graph._store,  ← store 引用注入到 runtime                              │ ║
║   │      context=user_context,                                                                 │ ║
║   │      config=runnable_config,                                                               │ ║
║   │  )                                                                                         │ ║
║   │                                                                                            │ ║
║   └────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                       │                             │                             │              ║
║                       ▼                             ▼                             ▼              ║
║   ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────┐ ║
║   │                             │  │                             │  │                         │ ║
║   │      中间件钩子              │  │       model_node           │  │      ToolNode           │ ║
║   │                             │  │                             │  │                         │ ║
║   │  def before_model(         │  │  def model_node(            │  │  执行工具时：            │ ║
║   │    state,                  │  │    state,                   │  │                         │ ║
║   │    runtime  ───────────────│──│──▶ runtime.store 可用        │  │  1. 检查参数注解        │ ║
║   │  ):                        │  │  ):                         │  │  2. InjectedStore       │ ║
║   │    runtime.store.put(...)  │  │                             │  │  3. 注入 store          │ ║
║   │                             │  │                             │  │                         │ ║
║   └─────────────────────────────┘  └─────────────────────────────┘  └───────────┬─────────────┘ ║
║                                                                                  │              ║
║                                                                                  ▼              ║
║                                                                      ┌─────────────────────────┐ ║
║                                                                      │                         │ ║
║                                                                      │   工具函数               │ ║
║                                                                      │                         │ ║
║                                                                      │  @tool                  │ ║
║                                                                      │  def my_tool(           │ ║
║                                                                      │    query: str,          │ ║
║                                                                      │    store: Annotated[    │ ║
║                                                                      │      Any,               │ ║
║                                                                      │      InjectedStore()    │ ║
║                                                                      │    ]                    │ ║
║                                                                      │  ):                     │ ║
║                                                                      │    store.put(...)       │ ║
║                                                                      │    store.get(...)       │ ║
║                                                                      │                         │ ║
║                                                                      └─────────────────────────┘ ║
║                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### 5.2 四种使用方式

```python
"""Agent 中使用 Store 的四种方式"""
from typing import Annotated, Any
from langchain.agents import create_agent, AgentMiddleware, AgentState
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime


# ═══════════════════════════════════════════════════════════════════════════════
# 方式 1：在工具中使用 InjectedStore
# ═══════════════════════════════════════════════════════════════════════════════
@tool
def save_user_data(
    key: str,
    value: str,
    store: Annotated[Any, InjectedStore()]  # ← 自动注入 store
) -> str:
    """保存用户数据"""
    store.put(("user_data",), key, {"value": value})
    return f"已保存 {key}"


@tool
def get_user_data(
    key: str,
    store: Annotated[Any, InjectedStore()]  # ← 自动注入 store
) -> str:
    """获取用户数据"""
    item = store.get(("user_data",), key)
    return item.value["value"] if item else f"{key} 不存在"


# ═══════════════════════════════════════════════════════════════════════════════
# 方式 2：在工具中使用 ToolRuntime.store
# ═══════════════════════════════════════════════════════════════════════════════
@tool
def tool_with_runtime(query: str, runtime: ToolRuntime) -> str:
    """通过 ToolRuntime 访问 store"""
    if runtime.store:
        # 记录查询
        import time
        runtime.store.put(("queries",), str(int(time.time())), {"query": query})
    return f"处理: {query}"


# ═══════════════════════════════════════════════════════════════════════════════
# 方式 3：在中间件中使用 runtime.store
# ═══════════════════════════════════════════════════════════════════════════════
class UserContextMiddleware(AgentMiddleware):
    """用户上下文中间件"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 开始前加载用户数据"""
        if runtime.store:
            user = runtime.store.get(("users",), "current")
            if user:
                print(f"欢迎 {user.value.get('name')}!")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """每次模型调用前记录"""
        if runtime.store:
            runtime.store.put(("stats",), "model_calls", {
                "count": len(state["messages"])
            })
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Agent 完成后保存会话"""
        if runtime.store:
            runtime.store.put(("sessions",), "latest", {
                "message_count": len(state["messages"]),
                "completed": True
            })
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 方式 4：在 Agent 外部直接使用
# ═══════════════════════════════════════════════════════════════════════════════

# 创建 store
store = InMemoryStore()

# 预加载数据
store.put(("users",), "current", {"name": "张三", "level": "vip"})
store.put(("settings",), "global", {"max_tokens": 1000})

# 创建 agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[save_user_data, get_user_data, tool_with_runtime],
    middleware=[UserContextMiddleware()],
    store=store,  # ← 传入 store
)

# 执行
result = agent.invoke(
    {"messages": [HumanMessage("保存我的偏好：主题=深色")]},
    config={"configurable": {"thread_id": "thread-1"}}
)

# 执行后读取数据
print("\n=== Store 中的数据 ===")
for item in store.search(("user_data",)):
    print(f"user_data: {item.key} = {item.value}")

for item in store.search(("queries",)):
    print(f"queries: {item.key} = {item.value}")
```

---

## 六、高级功能

### 6.1 语义搜索配置

**IndexConfig 结构：**

```python
class IndexConfig(TypedDict, total=False):
    """向量索引配置"""

    dims: int
    """向量维度，取决于嵌入模型
    - openai:text-embedding-3-large: 3072
    - openai:text-embedding-3-small: 1536
    - openai:text-embedding-ada-002: 1536
    - cohere:embed-english-v3.0: 1024
    """

    embed: Embeddings | EmbeddingsFunc | AEmbeddingsFunc | str
    """嵌入函数，支持：
    1. LangChain Embeddings 实例
    2. 同步函数 (texts: list[str]) -> list[list[float]]
    3. 异步函数
    4. 提供商字符串 "openai:text-embedding-3-small"
    """

    fields: list[str] | None
    """要索引的字段路径
    - ["$"]: 索引整个文档（默认）
    - ["text", "title"]: 索引特定字段
    - ["messages[*].content"]: 索引数组中的字段
    """
```

**配置示例：**

```python
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# 使用 LangChain 嵌入
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["text", "metadata.description"],
    }
)

# 使用自定义嵌入函数
from openai import OpenAI
client = OpenAI()

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embed_texts,
        "fields": ["content"],
    }
)
```

### 6.2 TTL（过期时间）配置

**TTLConfig 结构：**

```python
class TTLConfig(TypedDict, total=False):
    """TTL 配置"""

    refresh_on_read: bool
    """读取时是否刷新 TTL（默认 True）"""

    default_ttl: float | None
    """默认 TTL（分钟），None 表示不过期"""

    sweep_interval_minutes: int | None
    """清理过期数据的间隔（分钟）"""
```

**使用示例：**

```python
from langgraph.store.postgres import PostgresStore

with PostgresStore.from_conn_string(
    conn_string,
    ttl={
        "default_ttl": 60,            # 默认 60 分钟过期
        "refresh_on_read": True,      # 读取时刷新
        "sweep_interval_minutes": 5,  # 每 5 分钟清理
    }
) as store:
    store.setup()
    store.start_ttl_sweeper()  # 启动清理线程

    # 使用默认 TTL
    store.put(("cache",), "key1", {"data": "value"})  # 60 分钟后过期

    # 自定义 TTL
    store.put(("cache",), "key2", {"data": "value"}, ttl=5)  # 5 分钟后过期

    # 不过期
    store.put(("persistent",), "key3", {"data": "value"}, ttl=None)

    # 停止清理线程
    store.stop_ttl_sweeper()
```

---

## 七、三种 Store 对比

| 特性 | InMemoryStore | PostgresStore | SqliteStore |
|-----|---------------|---------------|-------------|
| **持久化** | ❌ 进程退出丢失 | ✅ 持久化 | ✅ 持久化 |
| **适用场景** | 开发测试 | 生产环境 | 单机部署 |
| **向量搜索** | ✅ 支持 | ✅ 支持（pgvector） | ✅ 支持（sqlite-vec） |
| **TTL 支持** | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| **并发性能** | 一般 | 高 | 低 |
| **扩展性** | 单进程 | 分布式 | 单机 |
| **依赖** | 无 | PostgreSQL + pgvector | SQLite + sqlite-vec |
| **首次使用** | 无需配置 | 需要 `setup()` | 需要 `setup()` |

---

## 八、完整示例：跨会话用户偏好系统

```python
"""
完整示例：使用 Store 实现跨会话用户偏好系统
支持语义搜索用户的历史偏好
"""
from typing import Annotated, Any
from langchain.agents import create_agent, AgentMiddleware, AgentState
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain.embeddings import init_embeddings
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
import time


# ═══════════════════════════════════════════════════════════════════════════════
# 创建带语义搜索的 Store
# ═══════════════════════════════════════════════════════════════════════════════
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "fields": ["description", "value"],
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# 定义工具
# ═══════════════════════════════════════════════════════════════════════════════
@tool
def save_preference(
    category: str,
    key: str,
    value: str,
    description: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """保存用户偏好，支持描述以便后续语义搜索

    Args:
        category: 偏好类别（如 'theme', 'language', 'notification'）
        key: 偏好键
        value: 偏好值
        description: 偏好的自然语言描述
    """
    store.put(
        ("preferences", category),
        key,
        {
            "value": value,
            "description": description,
            "updated_at": time.time(),
        }
    )
    return f"✅ 已保存偏好: {category}/{key} = {value}"


@tool
def search_preferences(
    query: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """语义搜索用户偏好

    Args:
        query: 自然语言查询，如 "深色主题相关的设置"
    """
    results = store.search(
        ("preferences",),
        query=query,
        limit=5
    )

    if not results:
        return "没有找到相关偏好"

    output = "找到以下相关偏好:\n"
    for item in results:
        output += f"- [{item.namespace[-1]}/{item.key}] {item.value['description']}\n"
        output += f"  值: {item.value['value']}\n"
        if item.score:
            output += f"  相似度: {item.score:.2f}\n"

    return output


@tool
def list_preferences(
    category: str,
    store: Annotated[Any, InjectedStore()]
) -> str:
    """列出某类别下的所有偏好

    Args:
        category: 偏好类别
    """
    results = store.search(
        ("preferences", category),
        limit=100
    )

    if not results:
        return f"类别 '{category}' 下没有偏好"

    output = f"类别 '{category}' 的偏好:\n"
    for item in results:
        output += f"- {item.key}: {item.value['value']}\n"

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# 定义中间件
# ═══════════════════════════════════════════════════════════════════════════════
class SessionMiddleware(AgentMiddleware):
    """会话管理中间件"""

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        if runtime.store:
            # 更新会话计数
            session = runtime.store.get(("sessions",), "current")
            count = session.value.get("count", 0) if session else 0
            runtime.store.put(
                ("sessions",),
                "current",
                {"count": count + 1, "started_at": time.time()}
            )
            print(f"📊 这是第 {count + 1} 次会话")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        if runtime.store:
            runtime.store.put(
                ("sessions",),
                "last_conversation",
                {
                    "message_count": len(state["messages"]),
                    "completed_at": time.time()
                }
            )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 创建 Agent
# ═══════════════════════════════════════════════════════════════════════════════
agent = create_agent(
    model="openai:gpt-4o",
    tools=[save_preference, search_preferences, list_preferences],
    middleware=[SessionMiddleware()],
    store=store,
    system_prompt="""你是一个偏好管理助手。

    你可以帮助用户：
    1. 保存偏好设置（使用 save_preference）
    2. 搜索相关偏好（使用 search_preferences）
    3. 列出某类别的偏好（使用 list_preferences）

    保存偏好时，请提供清晰的描述以便后续搜索。
    """,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user-123"}}

    # 第一轮：保存偏好
    print("\n" + "=" * 60)
    print("第一轮：保存偏好")
    print("=" * 60)

    result = agent.invoke(
        {"messages": [HumanMessage("帮我保存以下偏好：1. 界面主题用深色 2. 语言用中文 3. 通知只接收重要的")]},
        config=config
    )
    print(result["messages"][-1].content)

    # 第二轮：语义搜索
    print("\n" + "=" * 60)
    print("第二轮：语义搜索")
    print("=" * 60)

    result = agent.invoke(
        {"messages": [HumanMessage("搜索和外观显示相关的设置")]},
        config=config
    )
    print(result["messages"][-1].content)

    # 第三轮：列出偏好
    print("\n" + "=" * 60)
    print("第三轮：列出所有偏好")
    print("=" * 60)

    result = agent.invoke(
        {"messages": [HumanMessage("列出所有主题相关的偏好")]},
        config=config
    )
    print(result["messages"][-1].content)

    # 查看 Store 数据
    print("\n" + "=" * 60)
    print("Store 数据概览")
    print("=" * 60)

    namespaces = store.list_namespaces()
    for ns in namespaces:
        print(f"命名空间: {ns}")
        for item in store.search(ns, limit=100):
            print(f"  - {item.key}: {item.value}")
```

---

## 九、总结

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                 │
│                           LangGraph Store 核心要点总结                                          │
│                                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   1. Store 是什么？                                                                              │
│      - 跨 thread_id 的持久化存储（长期记忆）                                                    │
│      - 支持命名空间层级组织数据                                                                 │
│      - 支持语义搜索（需配置 index）                                                             │
│      - 支持 TTL 自动过期（PostgresStore/SqliteStore）                                          │
│                                                                                                 │
│   2. 核心数据结构                                                                                │
│      - Item: 存储的数据项（value, key, namespace, created_at, updated_at）                     │
│      - SearchItem: 搜索结果（额外包含 score）                                                   │
│      - 四种操作：GetOp, PutOp, SearchOp, ListNamespacesOp                                      │
│                                                                                                 │
│   3. 三种实现                                                                                    │
│      - InMemoryStore: 内存存储，开发测试用                                                      │
│      - PostgresStore: PostgreSQL，生产环境推荐                                                  │
│      - SqliteStore: SQLite，单机部署用                                                          │
│                                                                                                 │
│   4. 在 Agent 中的使用方式                                                                       │
│      - 工具中：InjectedStore() 或 ToolRuntime.store                                            │
│      - 中间件中：runtime.store                                                                  │
│      - wrap_tool_call 中：request.runtime.store                                                │
│      - 外部直接操作：store.put/get/search                                                       │
│                                                                                                 │
│   5. 数据流动路径                                                                                │
│      store = InMemoryStore() → create_agent(store=...) → graph.compile(store=...)              │
│      → Runtime(store=...) → 工具/中间件                                                         │
│                                                                                                 │
│   6. 与 Checkpointer 的区别                                                                      │
│      - Checkpointer: 单线程状态，自动管理 messages，短期记忆                                    │
│      - Store: 跨线程共享，手动管理，长期记忆                                                    │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```


