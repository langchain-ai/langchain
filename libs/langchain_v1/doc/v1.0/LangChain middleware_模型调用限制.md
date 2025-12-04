# LangChain 中间件：模型调用限制

通过限制模型调用次数，可以防止出现无限循环或产生过高的 API 成本。模型调用限制常见的应用场景包括：

- 防止智能体（Agent）失控，频繁进行不必要的 API 调用。
- 对生产环境中的模型调用次数进行成本控制。
- 在固定调用预算下对智能体行为进行测试和评估。

## 一、中间件概述

`ModelCallLimitMiddleware` 是一个**模型调用次数限制中间件**，通过 `before_model` 和 `after_model` 两个钩子实现。它支持两种级别的计数：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Thread 级别（跨 run 持久化）                                           │
│   - thread_model_call_count                                             │
│   - 会被 checkpointer 持久化                                            │
│   - 适用于：限制整个对话的总调用次数                                     │
│                                                                         │
│   Run 级别（单次 invoke 内）                                             │
│   - run_model_call_count                                                │
│   - UntrackedValue，不持久化，每次 invoke 重置为 0                       │
│   - 适用于：限制单次任务的调用次数                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、输入输出定义

### State Schema 扩展

```python
class ModelCallLimitState(AgentState):
    """扩展状态，添加调用计数字段"""
    
    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    # 线程级计数，会持久化，跨 run 累积
    
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    # 运行级计数，不持久化，每次 invoke 从 0 开始
```

### before_model 输入输出

| 输入 | 类型 | 说明 |
|------|------|------|
| `state["thread_model_call_count"]` | `int` | 当前线程调用次数（默认 0） |
| `state["run_model_call_count"]` | `int` | 当前运行调用次数（默认 0） |
| `runtime` | `Runtime` | LangGraph 运行时（未使用） |

| 输出 | 条件 | 类型 | 说明 |
|------|------|------|------|
| `None` | 未超限 | `None` | 继续执行，进入 model 节点 |
| `{"jump_to": "end", "messages": [AIMessage]}` | 超限 + exit_behavior="end" | `dict` | 跳转到 END，注入限制消息 |
| `raise ModelCallLimitExceededError` | 超限 + exit_behavior="error" | Exception | 抛出异常，终止执行 |

### after_model 输入输出

| 输入 | 类型 | 说明 |
|------|------|------|
| `state` | `ModelCallLimitState` | 当前状态 |
| `runtime` | `Runtime` | LangGraph 运行时（未使用） |

| 输出 | 类型 | 说明 |
|------|------|------|
| `{"thread_model_call_count": N+1, "run_model_call_count": M+1}` | `dict` | 两个计数都 +1 |

---

## 三、配置参数

```python
ModelCallLimitMiddleware(
    thread_limit: int | None = None,    # 线程级限制（跨 run）
    run_limit: int | None = None,       # 运行级限制（单次 invoke）
    exit_behavior: "end" | "error" = "end",  # 超限时的行为
)
```

| 参数 | 说明 |
|------|------|
| `thread_limit` | 整个对话（thread）最多调用模型多少次，None 表示不限制 |
| `run_limit` | 单次 invoke 最多调用模型多少次，None 表示不限制 |
| `exit_behavior="end"` | 超限时注入 AIMessage 并跳转到 END |
| `exit_behavior="error"` | 超限时抛出 `ModelCallLimitExceededError` |

**注意**：`thread_limit` 和 `run_limit` 至少要设置一个，否则初始化报错。

---

## 四、完整流程图

```
                              ┌─────────────────────────────┐
                              │      Agent 执行循环          │
                              └─────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│                           before_model(state, runtime)                               │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: 读取当前计数                                                           │  │
│  │                                                                                │  │
│  │   thread_count = state.get("thread_model_call_count", 0)                       │  │
│  │   run_count = state.get("run_model_call_count", 0)                             │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: 检查是否超限                                                           │  │
│  │                                                                                │  │
│  │   thread_limit_exceeded = (thread_limit != None) and (thread_count >= thread_limit)│
│  │   run_limit_exceeded = (run_limit != None) and (run_count >= run_limit)        │  │
│  │                                                                                │  │
│  │   ⚠️ 注意：是 >= 而不是 >，因为检查发生在调用前                                 │  │
│  │      如果 thread_limit=5，当 thread_count=5 时就会阻止第 6 次调用               │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                         ┌─────────────────┴─────────────────┐                        │
│                         │                                   │                        │
│                   False (未超限)                       True (已超限)                 │
│                         │                                   │                        │
│                         ▼                                   ▼                        │
│                  ┌──────────────┐           ┌────────────────────────────────────┐   │
│                  │ return None  │           │ Step 3: 根据 exit_behavior 处理    │   │
│                  │              │           │                                    │   │
│                  │ → 进入 model │           │  if exit_behavior == "error":      │   │
│                  │    节点      │           │      raise ModelCallLimitExceeded  │   │
│                  │              │           │           Error(...)               │   │
│                  └──────────────┘           │                                    │   │
│                                             │  if exit_behavior == "end":        │   │
│                                             │      msg = "Model call limits      │   │
│                                             │             exceeded: ..."         │   │
│                                             │      return {                      │   │
│                                             │          "jump_to": "end",         │   │
│                                             │          "messages": [AIMessage(   │   │
│                                             │              content=msg           │   │
│                                             │          )]                        │   │
│                                             │      }                             │   │
│                                             │                                    │   │
│                                             └────────────────────────────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           │ (如果未超限)
                                           ▼
                              ┌─────────────────────────────┐
                              │        model 节点           │
                              │      （调用 LLM）           │
                              └─────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│                           after_model(state, runtime)                                │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 4: 增加计数                                                               │  │
│  │                                                                                │  │
│  │   return {                                                                     │  │
│  │       "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,  │  │
│  │       "run_model_call_count": state.get("run_model_call_count", 0) + 1,        │  │
│  │   }                                                                            │  │
│  │                                                                                │  │
│  │   # 两个计数都 +1                                                              │  │
│  │   # thread_model_call_count 会被 checkpointer 持久化                           │  │
│  │   # run_model_call_count 不会持久化（UntrackedValue）                          │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │  继续执行（tools 或 END）   │
                              └─────────────────────────────┘
```

---

## 五、状态变化示例

### 场景：thread_limit=5, run_limit=3

#### 第一次 invoke（run #1）

```
初始状态:
  thread_model_call_count = 0  (从 checkpointer 加载，或默认 0)
  run_model_call_count = 0     (每次 invoke 重置)

循环 1:
  before_model: 0 < 5 ✓, 0 < 3 ✓ → 继续
  model: 调用 LLM
  after_model: thread=1, run=1

循环 2:
  before_model: 1 < 5 ✓, 1 < 3 ✓ → 继续
  model: 调用 LLM
  after_model: thread=2, run=2

循环 3:
  before_model: 2 < 5 ✓, 2 < 3 ✓ → 继续
  model: 调用 LLM
  after_model: thread=3, run=3

循环 4:
  before_model: 3 < 5 ✓, 3 >= 3 ❌ → run_limit 超限！
  return {"jump_to": "end", "messages": [AIMessage("Model call limits exceeded: run limit (3/3)")]}

invoke 结束:
  thread_model_call_count = 3  (持久化到 checkpointer)
  run_model_call_count = 3     (不持久化)
```

#### 第二次 invoke（run #2，同一 thread）

```
初始状态:
  thread_model_call_count = 3  (从 checkpointer 加载)
  run_model_call_count = 0     (重置！)

循环 1:
  before_model: 3 < 5 ✓, 0 < 3 ✓ → 继续
  model: 调用 LLM
  after_model: thread=4, run=1

循环 2:
  before_model: 4 < 5 ✓, 1 < 3 ✓ → 继续
  model: 调用 LLM
  after_model: thread=5, run=2

循环 3:
  before_model: 5 >= 5 ❌ → thread_limit 超限！
  return {"jump_to": "end", "messages": [AIMessage("Model call limits exceeded: thread limit (5/5)")]}

invoke 结束:
  thread_model_call_count = 5  (持久化)
```

---

## 六、跳转机制详解

```python
@hook_config(can_jump_to=["end"])  # ← 声明此钩子可以跳转到 "end"
def before_model(self, state, runtime):
    ...
    if limit_exceeded and self.exit_behavior == "end":
        return {
            "jump_to": "end",       # ← 触发跳转到 END 节点
            "messages": [AIMessage(...)]  # ← 注入限制消息
        }
```

### 跳转流程

```
before_model 返回 {"jump_to": "end", ...}
       │
       ▼
LangGraph 检测到 jump_to = "end"
       │
       ▼
跳过 model 节点，直接进入 after_agent（如果有）或 END
       │
       ▼
state["messages"] 追加 AIMessage("Model call limits exceeded: ...")
       │
       ▼
Agent 执行结束
```

---

## 七、异常模式详解

```python
class ModelCallLimitExceededError(Exception):
    """超限异常"""
    
    def __init__(self, thread_count, run_count, thread_limit, run_limit):
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        
        msg = f"Model call limits exceeded: thread limit ({thread_count}/{thread_limit}), ..."
        super().__init__(msg)
```

当 `exit_behavior="error"` 时：

```
before_model 检测到超限
       │
       ▼
raise ModelCallLimitExceededError(...)
       │
       ▼
异常向上传播，Agent 执行终止
       │
       ▼
调用方可以 try/except 捕获并处理
```

---

## 八、完整数据流总结

```
用户发送消息
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ state 初始化                                                      │
│ - thread_model_call_count: 从 checkpointer 加载（或 0）          │
│ - run_model_call_count: 0（每次 invoke 重置）                    │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Agent 循环开始                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ before_model (ModelCallLimitMiddleware)                 │     │
│  │                                                         │     │
│  │   检查: thread_count >= thread_limit?                   │     │
│  │         run_count >= run_limit?                         │     │
│  │                                                         │     │
│  │   超限 + end → return {"jump_to": "end", "messages"}    │     │
│  │   超限 + error → raise ModelCallLimitExceededError      │     │
│  │   未超限 → return None (继续)                           │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ model 节点 (调用 LLM)                                   │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ after_model (ModelCallLimitMiddleware)                  │     │
│  │                                                         │     │
│  │   return {                                              │     │
│  │       thread_model_call_count: N + 1,                   │     │
│  │       run_model_call_count: M + 1                       │     │
│  │   }                                                     │     │
│  │                                                         │     │
│  └─────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│                   tools 节点或 END                               │
│                          │                                       │
│                          └──────────► 回到 before_model          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│ invoke 结束                                                       │
│ - thread_model_call_count: 持久化到 checkpointer                 │
│ - run_model_call_count: 丢弃（UntrackedValue）                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 九、使用示例

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import MemorySaver

# 场景 1: 限制单次任务复杂度（防止无限循环）
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        ModelCallLimitMiddleware(
            run_limit=10,           # 单次 invoke 最多 10 次模型调用
            exit_behavior="end",    # 超限时优雅结束
        ),
    ],
)

# 场景 2: 限制整个对话的 token 消耗（按次数估算）
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    checkpointer=MemorySaver(),  # 需要 checkpointer 来持久化 thread 计数
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=50,        # 整个对话最多 50 次模型调用
            exit_behavior="error",  # 超限时抛出异常
        ),
    ],
)

# 场景 3: 双重限制
agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    checkpointer=MemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=100,       # 对话总限制
            run_limit=20,           # 单次任务限制
            exit_behavior="end",
        ),
    ],
)

# 使用
result = agent.invoke(
    {"messages": [HumanMessage("帮我完成一个复杂任务")]},
    config={"configurable": {"thread_id": "user-123"}}
)

# 检查是否因超限结束
last_message = result["messages"][-1]
if "Model call limits exceeded" in last_message.content:
    print("任务因调用次数限制而提前结束")
```

---

## 十、关键设计总结

| 组件 | 作用 |
|------|------|
| `thread_model_call_count` | 跨 run 持久化的计数，由 checkpointer 保存 |
| `run_model_call_count` | 单次 invoke 内的计数，UntrackedValue 不持久化 |
| `before_model` | 检查是否超限，决定是否允许调用模型 |
| `after_model` | 模型调用成功后，两个计数都 +1 |
| `@hook_config(can_jump_to=["end"])` | 声明钩子可以跳转到 END，触发 LangGraph 的条件边 |
| `exit_behavior="end"` | 优雅结束，注入限制消息 |
| `exit_behavior="error"` | 抛出异常，由调用方处理 |

