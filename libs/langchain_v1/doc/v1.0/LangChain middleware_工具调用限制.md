
# LangChain middleware_工具调用限制

通过限制工具调用的数量，可以更好地控制智能体（Agent）的执行行为。限制可以是全局性的（针对所有工具），也可以针对指定的工具单独设置。

常见的应用场景包括：

- 防止对昂贵的外部 API 进行过多调用
- 限制网络搜索或数据库查询的频率
- 对特定工具实施速率限制
- 防止智能体陷入无限循环

## 配置选项

| 配置项         | 类型     | 说明                                                                                      |
|---------------|--------|----------------------------------------------------------------------------------------|
| `tool_name`   | 字符串   | 限制指定工具的名称。如果不填写，则全局应用于所有工具。                                        |
| `thread_limit`| 数字     | 单个线程（对话）中的工具调用最大次数。会跨具有相同线程 ID 的会话累积，需要 checkpointer 持久化。`None` 表示不限制。        |
| `run_limit`   | 数字     | 每次调用（一次用户消息至响应的完整流程）内允许的最大工具调用次数。每逢新一轮用户消息自动重置。`None` 表示不限制。<br/>**注意：`thread_limit` 和 `run_limit` 至少需指定一个。** |
| `exit_behavior` | 字符串（默认："继续"） | 达到限制后的行为：<br/>- `"继续"`（默认）- 用错误消息阻止本次超限工具调用，其他工具及模型继续，模型可决定何时终止。<br/>- `"error"` - 抛出 `ToolCallLimitExceededError` 异常，立即中断执行。<br/>- `"end"` - 使用 ToolMessage 和 AI 消息立即终止（仅限限制单一工具时有效，若其他工具有待处理调用会抛出 `NotImplementedError`）。 |

## ToolCallLimitMiddleware 流程解析

## 一、中间件概述

`ToolCallLimitMiddleware` 是一个**工具调用次数限制中间件**，通过 `after_model` 钩子实现。与 `ModelCallLimitMiddleware` 不同，它追踪的是**工具调用**次数，而非模型调用次数。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   核心区别：                                                             │
│                                                                         │
│   ModelCallLimitMiddleware                                              │
│   - 追踪：模型被调用的次数                                               │
│   - 钩子：before_model（调用前检查）+ after_model（调用后计数）           │
│   - 计数：int（单个数字）                                                │
│                                                                         │
│   ToolCallLimitMiddleware                                               │
│   - 追踪：工具被调用的次数                                               │
│   - 钩子：只有 after_model（模型返回后检查 tool_calls）                  │
│   - 计数：dict[str, int]（按工具名分别计数）                             │
│   - 特性：可针对特定工具限制，支持并行工具调用处理                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、输入输出定义

### State Schema 扩展

```python
class ToolCallLimitState(AgentState):
    """扩展状态，添加工具调用计数字段"""
    
    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    # 字典：{"search": 3, "weather": 2, "__all__": 5}
    # 线程级计数，会持久化，跨 run 累积
    
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]
    # 字典：{"search": 1, "__all__": 2}
    # 运行级计数，不持久化，每次 invoke 重置为 {}
```

**关键设计**：
- `"__all__"` 键用于追踪所有工具的总调用次数（当 `tool_name=None` 时）
- 每个工具可以有独立的计数键，支持多个中间件实例各自追踪不同工具

### after_model 输入输出

| 输入 | 类型 | 说明 |
|------|------|------|
| `state["messages"]` | `list[AnyMessage]` | 消息列表，用于获取最后的 AIMessage |
| `state["thread_tool_call_count"]` | `dict[str, int]` | 线程级工具调用计数（默认 {}） |
| `state["run_tool_call_count"]` | `dict[str, int]` | 运行级工具调用计数（默认 {}） |
| `runtime` | `Runtime` | LangGraph 运行时（未使用） |

| 输出 | 条件 | 类型 | 说明 |
|------|------|------|------|
| `None` | 无 tool_calls | `None` | 无工具调用，无需处理 |
| `{"thread_tool_call_count": {...}, "run_tool_call_count": {...}}` | 未超限 | `dict` | 更新计数 |
| `{..., "messages": [ToolMessage(error)]}` | 超限 + continue | `dict` | 注入错误 ToolMessage，阻止被限工具 |
| `{..., "jump_to": "end", "messages": [...]}` | 超限 + end | `dict` | 跳转到 END |
| `raise ToolCallLimitExceededError` | 超限 + error | Exception | 抛出异常 |

---

## 三、配置参数

```python
ToolCallLimitMiddleware(
    tool_name: str | None = None,       # 限制特定工具（None = 所有工具）
    thread_limit: int | None = None,    # 线程级限制（跨 run）
    run_limit: int | None = None,       # 运行级限制（单次 invoke）
    exit_behavior: "continue" | "error" | "end" = "continue",  # 超限行为
)
```

| 参数 | 说明 |
|------|------|
| `tool_name=None` | 限制所有工具的总调用次数，使用 `"__all__"` 键计数 |
| `tool_name="search"` | 只限制 `search` 工具，使用 `"search"` 键计数 |
| `thread_limit` | 整个对话（thread）最多调用该工具多少次 |
| `run_limit` | 单次 invoke 最多调用该工具多少次 |
| `exit_behavior="continue"` | **默认**，注入错误 ToolMessage 阻止超限工具，其他工具继续执行 |
| `exit_behavior="error"` | 抛出 `ToolCallLimitExceededError` 异常 |
| `exit_behavior="end"` | 跳转到 END（仅支持单个工具调用，有并行调用时抛 NotImplementedError） |

**验证规则**：
- `thread_limit` 和 `run_limit` 至少设置一个
- 如果两者都设置，`run_limit` 不能大于 `thread_limit`

---

## 四、完整流程图

```
                              ┌─────────────────────────────┐
                              │        model 节点           │
                              │      （调用 LLM）           │
                              └─────────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │   AIMessage 返回            │
                              │   可能包含 tool_calls       │
                              └─────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│                       after_model(state, runtime)                                    │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: 获取最后一个 AIMessage                                                 │  │
│  │                                                                                │  │
│  │   messages = state.get("messages", [])                                         │  │
│  │   last_ai_message = None                                                       │  │
│  │   for message in reversed(messages):                                           │  │
│  │       if isinstance(message, AIMessage):                                       │  │
│  │           last_ai_message = message                                            │  │
│  │           break                                                                │  │
│  │                                                                                │  │
│  │   if not last_ai_message or not last_ai_message.tool_calls:                    │  │
│  │       return None  # 无工具调用，跳过                                           │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: 获取当前计数                                                           │  │
│  │                                                                                │  │
│  │   count_key = self.tool_name if self.tool_name else "__all__"                  │  │
│  │                                                                                │  │
│  │   thread_counts = state.get("thread_tool_call_count", {}).copy()               │  │
│  │   run_counts = state.get("run_tool_call_count", {}).copy()                     │  │
│  │                                                                                │  │
│  │   current_thread_count = thread_counts.get(count_key, 0)  # e.g., 3            │  │
│  │   current_run_count = run_counts.get(count_key, 0)        # e.g., 1            │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: 分离工具调用 → allowed vs blocked                                      │  │
│  │                                                                                │  │
│  │   _separate_tool_calls(tool_calls, thread_count, run_count)                    │  │
│  │                                                                                │  │
│  │   遍历每个 tool_call:                                                          │  │
│  │     1. 检查是否匹配当前中间件的 tool_name 过滤器                                │  │
│  │     2. 检查是否会超限：_would_exceed_limit(temp_thread, temp_run)              │  │
│  │        - thread_count + 1 > thread_limit?                                      │  │
│  │        - run_count + 1 > run_limit?                                            │  │
│  │     3. 如果超限 → blocked_calls，否则 → allowed_calls                          │  │
│  │                                                                                │  │
│  │   返回: (allowed_calls, blocked_calls, new_thread_count, new_run_count)        │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                         ┌─────────────────┴─────────────────┐                        │
│                         │                                   │                        │
│                  blocked_calls 为空               blocked_calls 不为空               │
│                         │                                   │                        │
│                         ▼                                   ▼                        │
│                  ┌──────────────┐           ┌────────────────────────────────────┐   │
│                  │ 只更新计数   │           │ Step 4: 根据 exit_behavior 处理    │   │
│                  │              │           │                                    │   │
│                  │ return {     │           │  ┌─────────────────────────────┐   │   │
│                  │   "thread_   │           │  │ exit_behavior == "error"    │   │   │
│                  │   tool_call_ │           │  │                             │   │   │
│                  │   count": {..│           │  │ raise ToolCallLimitExceeded │   │   │
│                  │   },         │           │  │       Error(...)            │   │   │
│                  │   "run_tool_ │           │  │                             │   │   │
│                  │   call_count"│           │  └─────────────────────────────┘   │   │
│                  │   : {...}    │           │                                    │   │
│                  │ }            │           │  ┌─────────────────────────────┐   │   │
│                  │              │           │  │ exit_behavior == "end"      │   │   │
│                  └──────────────┘           │  │                             │   │   │
│                                             │  │ 检查是否有其他工具的并行调用│   │   │
│                                             │  │ 如果有 → NotImplementedError│   │   │
│                                             │  │                             │   │   │
│                                             │  │ return {                    │   │   │
│                                             │  │   ...,                      │   │   │
│                                             │  │   "jump_to": "end",         │   │   │
│                                             │  │   "messages": [             │   │   │
│                                             │  │     ToolMessage(error),     │   │   │
│                                             │  │     AIMessage(final)        │   │   │
│                                             │  │   ]                         │   │   │
│                                             │  │ }                           │   │   │
│                                             │  └─────────────────────────────┘   │   │
│                                             │                                    │   │
│                                             │  ┌─────────────────────────────┐   │   │
│                                             │  │ exit_behavior == "continue" │   │   │
│                                             │  │ (默认)                      │   │   │
│                                             │  │                             │   │   │
│                                             │  │ return {                    │   │   │
│                                             │  │   ...,                      │   │   │
│                                             │  │   "messages": [             │   │   │
│                                             │  │     ToolMessage(error)      │   │   │
│                                             │  │     for each blocked_call   │   │   │
│                                             │  │   ]                         │   │   │
│                                             │  │ }                           │   │   │
│                                             │  │                             │   │   │
│                                             │  │ → 阻止被限工具，其他继续    │   │   │
│                                             │  └─────────────────────────────┘   │   │
│                                             │                                    │   │
│                                             └────────────────────────────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 五、_separate_tool_calls 详细流程

这是核心分离逻辑，决定哪些工具调用被允许，哪些被阻止：

```
输入: tool_calls = [
    {"id": "call_1", "name": "search", "args": {...}},
    {"id": "call_2", "name": "weather", "args": {...}},
    {"id": "call_3", "name": "search", "args": {...}},
]

配置: tool_name = "search", thread_limit = 3, run_limit = 2
当前计数: thread_count = 2, run_count = 1

┌──────────────────────────────────────────────────────────────────────────┐
│ 遍历 tool_calls                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ call_1: {"name": "search"}                                               │
│   ├─ _matches_tool_filter? "search" == "search" → ✓ 匹配                │
│   ├─ _would_exceed_limit(2, 1)?                                          │
│   │   └─ 2 + 1 = 3 > 3? No, 1 + 1 = 2 > 2? No → ✓ 允许                  │
│   ├─ allowed_calls.append(call_1)                                        │
│   └─ temp_thread = 3, temp_run = 2                                       │
│                                                                          │
│ call_2: {"name": "weather"}                                              │
│   └─ _matches_tool_filter? "weather" == "search" → ✗ 不匹配，跳过       │
│      (这个调用不受当前中间件限制)                                        │
│                                                                          │
│ call_3: {"name": "search"}                                               │
│   ├─ _matches_tool_filter? "search" == "search" → ✓ 匹配                │
│   ├─ _would_exceed_limit(3, 2)?                                          │
│   │   └─ 3 + 1 = 4 > 3? Yes → ✗ 超限！                                  │
│   └─ blocked_calls.append(call_3)                                        │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ 返回:                                                                    │
│   allowed_calls = [call_1]                                               │
│   blocked_calls = [call_3]                                               │
│   new_thread_count = 3                                                   │
│   new_run_count = 2                                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 六、三种 exit_behavior 对比

### exit_behavior = "continue"（默认）

```
场景: 模型返回 tool_calls = [search, weather, search]
      search 被限制，只允许 1 次

结果:
┌────────────────────────────────────────────────────────────────┐
│ 返回:                                                          │
│ {                                                              │
│   "thread_tool_call_count": {"search": 3},                     │
│   "run_tool_call_count": {"search": 3},                        │
│   "messages": [                                                │
│     ToolMessage(                                               │
│       content="Tool call limit exceeded. Do not call           │
│                'search' again.",                               │
│       tool_call_id="call_3",                                   │
│       status="error"                                           │
│     )                                                          │
│   ]                                                            │
│ }                                                              │
├────────────────────────────────────────────────────────────────┤
│ 效果:                                                          │
│ - call_1 (search): 正常执行 ✓                                  │
│ - call_2 (weather): 正常执行 ✓（不受限制）                     │
│ - call_3 (search): 被阻止 ✗，注入错误 ToolMessage              │
│ - Agent 继续运行，模型会看到错误消息                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### exit_behavior = "error"

```
场景: 同上

结果:
┌────────────────────────────────────────────────────────────────┐
│ raise ToolCallLimitExceededError(                              │
│   thread_count=4,   # 假设执行后的计数                         │
│   run_count=3,                                                 │
│   thread_limit=3,                                              │
│   run_limit=2,                                                 │
│   tool_name="search"                                           │
│ )                                                              │
├────────────────────────────────────────────────────────────────┤
│ 效果:                                                          │
│ - Agent 立即终止                                               │
│ - 所有工具调用都不会执行                                       │
│ - 调用方需要 try/except 捕获异常                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### exit_behavior = "end"

```
场景 A: 模型返回 tool_calls = [search]（单个工具调用超限）

结果:
┌────────────────────────────────────────────────────────────────┐
│ 返回:                                                          │
│ {                                                              │
│   "thread_tool_call_count": {"search": 3},                     │
│   "run_tool_call_count": {"search": 3},                        │
│   "jump_to": "end",                                            │
│   "messages": [                                                │
│     ToolMessage(                                               │
│       content="Tool call limit exceeded...",                   │
│       tool_call_id="call_1",                                   │
│       status="error"                                           │
│     ),                                                         │
│     AIMessage(                                                 │
│       content="'search' tool call limit reached:               │
│                thread limit exceeded (4/3 calls)."             │
│     )                                                          │
│   ]                                                            │
│ }                                                              │
├────────────────────────────────────────────────────────────────┤
│ 效果:                                                          │
│ - 跳转到 END，Agent 结束                                       │
│ - 注入 ToolMessage + AIMessage 告知用户                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

场景 B: 模型返回 tool_calls = [search, weather]（有其他工具的并行调用）

结果:
┌────────────────────────────────────────────────────────────────┐
│ raise NotImplementedError(                                     │
│   "Cannot end execution with other tool calls pending. "       │
│   "Found calls to: weather. Use 'continue' or 'error' "        │
│   "behavior instead."                                          │
│ )                                                              │
├────────────────────────────────────────────────────────────────┤
│ 原因:                                                          │
│ - weather 调用没有被限制，应该继续执行                          │
│ - 但 "end" 行为要求立即结束                                    │
│ - 两者矛盾，所以抛出 NotImplementedError                       │
│ - 建议使用 "continue" 或 "error" 代替                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 七、状态变化示例

### 场景：限制所有工具（tool_name=None），run_limit=3

```
初始状态:
  thread_tool_call_count = {}
  run_tool_call_count = {}

第一次 model 调用返回: tool_calls = [search, weather]
  ├─ count_key = "__all__"
  ├─ 检查: 0 + 1 <= 3 ✓, 1 + 1 <= 3 ✓
  ├─ 两个都允许
  └─ after_model 返回: {"__all__": 2}

第二次 model 调用返回: tool_calls = [search, db_query]
  ├─ 检查: 2 + 1 <= 3 ✓, 3 + 1 <= 3 ✗
  ├─ search 允许，db_query 被阻止
  └─ after_model 返回:
     {
       "thread_tool_call_count": {"__all__": 3},
       "run_tool_call_count": {"__all__": 4},  # 包含被阻止的
       "messages": [ToolMessage(error for db_query)]
     }

实际执行:
  - search: 正常执行
  - db_query: 被注入错误 ToolMessage，不会执行

模型收到消息:
  - ToolMessage(search result)
  - ToolMessage("Tool call limit exceeded. Do not make additional tool calls.")
```

### 场景：限制特定工具（tool_name="search"），thread_limit=2

```
第一次 invoke (run #1):
  model 返回: [search, weather, search]
  ├─ weather 不匹配，跳过
  ├─ search #1: 0 + 1 <= 2 ✓ 允许
  ├─ search #2: 1 + 1 <= 2 ✓ 允许
  └─ 计数: {"search": 2}

第二次 invoke (run #2，同一 thread):
  初始: thread_tool_call_count = {"search": 2}  # 从 checkpointer 加载
        run_tool_call_count = {}                  # 重置

  model 返回: [search, weather]
  ├─ search: 2 + 1 > 2 ✗ 被阻止
  ├─ weather: 不匹配，正常执行
  └─ 返回:
     {
       "thread_tool_call_count": {"search": 2},
       "run_tool_call_count": {"search": 1},
       "messages": [ToolMessage(error for search)]
     }
```

---

## 八、完整数据流总结

```
用户发送消息
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ state 初始化                                                      │
│ - thread_tool_call_count: 从 checkpointer 加载（或 {}）          │
│ - run_tool_call_count: {}（每次 invoke 重置）                    │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Agent 循环开始                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ model 节点 (调用 LLM)                                   │     │
│  │                                                         │     │
│  │ 返回 AIMessage，可能包含 tool_calls:                    │     │
│  │ [                                                       │     │
│  │   {id: "call_1", name: "search", args: {...}},          │     │
│  │   {id: "call_2", name: "weather", args: {...}},         │     │
│  │ ]                                                       │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ after_model (ToolCallLimitMiddleware)                   │     │
│  │                                                         │     │
│  │   1. 获取最后一个 AIMessage 的 tool_calls               │     │
│  │   2. 分离: allowed_calls vs blocked_calls               │     │
│  │   3. 更新计数 dict                                      │     │
│  │   4. 根据 exit_behavior 处理 blocked_calls:             │     │
│  │      - continue: 注入错误 ToolMessage                   │     │
│  │      - error: 抛出异常                                  │     │
│  │      - end: 跳转到 END                                  │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ tools 节点                                              │     │
│  │                                                         │     │
│  │ - 只执行 allowed_calls                                  │     │
│  │ - blocked_calls 已有错误 ToolMessage，不会执行          │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│                   回到 model 节点                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 九、使用示例

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langgraph.checkpoint.memory import MemorySaver

# 场景 1: 限制所有工具的总调用次数（防止无限循环）
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, weather_tool, db_tool],
    middleware=[
        ToolCallLimitMiddleware(
            run_limit=20,              # 单次 invoke 最多 20 次工具调用
            exit_behavior="continue",  # 超限后阻止工具，模型自行决定是否结束
        ),
    ],
)

# 场景 2: 限制特定昂贵工具的调用次数
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, expensive_api_tool, weather_tool],
    checkpointer=MemorySaver(),
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="expensive_api",
            thread_limit=5,            # 整个对话最多调用 5 次
            run_limit=2,               # 单次任务最多调用 2 次
            exit_behavior="error",     # 超限抛异常，严格限制
        ),
    ],
)

# 场景 3: 多个中间件实例，分别限制不同工具
agent = create_agent(
    model="openai:gpt-4o",
    tools=[search_tool, db_tool, email_tool],
    checkpointer=MemorySaver(),
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=10,
            exit_behavior="continue",
        ),
        ToolCallLimitMiddleware(
            tool_name="email",
            run_limit=3,              # 单次任务最多发 3 封邮件
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            run_limit=50,             # 所有工具总计限制
            exit_behavior="continue",
        ),
    ],
)

# 使用
try:
    result = agent.invoke(
        {"messages": [HumanMessage("帮我搜索并发送报告邮件")]},
        config={"configurable": {"thread_id": "user-123"}}
    )
except ToolCallLimitExceededError as e:
    print(f"工具调用超限: {e.tool_name}, 次数: {e.thread_count}")
```

---

## 十、与 ModelCallLimitMiddleware 对比

| 维度 | ModelCallLimitMiddleware | ToolCallLimitMiddleware |
|------|--------------------------|-------------------------|
| **追踪对象** | 模型调用次数 | 工具调用次数 |
| **钩子** | `before_model` + `after_model` | 只有 `after_model` |
| **计数类型** | `int` | `dict[str, int]` |
| **是否支持特定工具** | ❌ | ✅ `tool_name` 参数 |
| **检查时机** | 模型调用前 | 模型调用后（检查 tool_calls） |
| **exit_behavior** | `end`, `error` | `continue`, `end`, `error` |
| **默认行为** | `end` | `continue` |
| **并行处理** | N/A | 支持（allowed vs blocked） |

---

## 十一、关键设计总结

| 组件 | 作用 |
|------|------|
| `thread_tool_call_count: dict` | 跨 run 持久化的计数，按工具名分别记录 |
| `run_tool_call_count: dict` | 单次 invoke 内的计数，UntrackedValue 不持久化 |
| `"__all__"` 键 | 当 `tool_name=None` 时，追踪所有工具的总调用次数 |
| `_separate_tool_calls` | 核心逻辑，将 tool_calls 分为 allowed 和 blocked |
| `exit_behavior="continue"` | 默认，最灵活，只阻止超限工具 |
| `ToolMessage(status="error")` | 注入错误消息，告诉模型不要再调用该工具 |
| `name` 属性 | 返回 `ToolCallLimitMiddleware[search]`，支持多实例 |

