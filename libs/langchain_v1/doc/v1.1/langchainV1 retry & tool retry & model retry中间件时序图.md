# langchainV1 retry & tool retry & model retry中间件时序图.md

## 场景设定

- 任务：`HumanMessage("帮我创建 login.tsx 和 register.tsx 两个文件")`
- 中间件：
  - `ModelRetryMiddleware(max_retries=2, retry_on=(TimeoutError,), on_failure="continue")`
  - `ToolRetryMiddleware(max_retries=1, retry_on=(IOError,), on_failure="continue")`
- 工具：`create_file`, `write_todos`
- 初始状态（LangGraph 会根据 AgentState schema 自动初始化）：

```
state_0 = {
    "messages": [HumanMessage("帮我创建 login.tsx 和 register.tsx 两个文件")],
    "structured_response": None,    # 默认
    "jump_to": None,                # 默认
    # todo middleware 还未运行，state 中无 todos
}
```

LangGraph 拓扑（简化）：

```
START
  │
  ▼
(before_model nodes?) → model node → (after_model nodes?)
                                     │
                                     ▼
                                   tools
                                     │
                                     └──（根据 tool_calls 回到 model 或结束）
```

ModelRetry 包裹的是 `model node`，ToolRetry 包裹的是 `tools node`。

---

## 详细时序 + 状态流动

### Step 1: `before_model`（如果有）

- 无变化，状态仍为 `state_0`

### Step 2: `model node`（含 ModelRetry）

#### 2.1 `wrap_model_call` attempt=0，第一次调用模型

- 输入消息：`[Human("帮我创建...")]` + system_prompt
- LLM 输出：**TimeoutError**（模拟超时）
- 此时 LangGraph 状态 **不变**：`state_0`
- ModelRetry 检测到异常：
  - `should_retry_exception(exc, retry_on=(TimeoutError,)) → True`
  - `attempt < max_retries (0 < 2) → True`
  - `delay = calculate_delay(retry_number=0, ...) ~ 1s`
  - `sleep(delay)`，并准备 attempt=1

#### 2.2 `wrap_model_call` attempt=1，第二次调用模型

- 输入消息仍然是 `[Human(...)]`
- LLM 输出成功：`AIMessage(tool_calls=[write_todos([...])])`
- ModelRetry 返回 `ModelResponse(result=[AIMessage(...)])`

LangGraph 更新状态：

```
state_1 = {
    "messages": [
        Human("帮我创建..."),
        AI(tool_calls=[write_todos(tasks=[login, register])])
    ],
    # todos 仍不存在（写入靠工具执行）
}
```

LangGraph 根据有 `tool_calls`，转向 `tools` 节点执行 `write_todos`。

---

### Step 3: `tools` 节点（执行 write_todos）含 ToolRetry

#### 3.1 `wrap_tool_call` attempt=0

- `request.tool_call = {"name": "write_todos", "args": {...}}`
- 假设工具执行成功，返回 `ToolMessage("Updated todo list...")`

LangGraph 状态更新：

```
state_2 = {
    "messages": [
        Human(...),
        AI(tool_calls=[write_todos(...)]),
        ToolMessage("Updated todo list ...")
    ],
    "todos": [
        {"content": "创建 login.tsx", "status": "in_progress"},
        {"content": "创建 register.tsx", "status": "pending"}
    ]
}
```

LangGraph 看到工具执行完毕 → 回到 `model node`（这是 ReAct 的“观察后继续思考”）。

---

### Step 4: `model node`（计划执行 login.tsx）

#### 4.1 `wrap_model_call` attempt=0

- 输入消息包括 `state_2["messages"]`（Human + AI(write_todos) + Tool(write_todos 结果)）
- LLM 输出成功：`AIMessage(tool_calls=[create_file(login.tsx)])`

LangGraph 状态：

```
state_3 = {
    "messages": [
        Human(...),
        AI(write_todos),
        ToolMessage(Update todo list),
        AI(tool_calls=[create_file(login.tsx)])
    ],
    "todos": 同 Step 3
}
```

无异常，ModelRetry 不介入。

LangGraph 进入 `tools` 节点执行 `create_file(login.tsx)`。

---

### Step 5: `tools` 节点执行 `create_file(login.tsx)`（ToolRetry 生效）

#### 5.1 `wrap_tool_call` attempt=0

- ToolRetry 判断该工具在 retry 范围内
- 第一次执行工具抛出 `IOError("Disk full")`
- ToolRetry 判定：
  - `should_retry_exception(IOError, retry_on=(IOError,)) → True`
  - `attempt < max_retries (0 < 1) → True`
  - `delay = calculate_delay(retry_number=0, ...) ≈ 1s`
  - `sleep(delay)`，继续 attempt=1

状态仍为 `state_3`，因为工具失败没有产生消息。

#### 5.2 `wrap_tool_call` attempt=1

- 再次执行 `create_file(login.tsx)`，这次成功
- 返回 `ToolMessage("文件 login.tsx 已创建")`

LangGraph 更新：

```
state_4 = {
    "messages": [
        Human(...),
        AI(write_todos),
        ToolMessage(Update todo list),
        AI(tool_calls=[create_file(login.tsx)]),
        ToolMessage("文件 login.tsx 已创建")
    ],
    "todos": [
        {"content": "创建 login.tsx", "status": "completed"},
        {"content": "创建 register.tsx", "status": "pending"}   ← 模型稍后更新
    ]
}
```

---

### Step 6: `model node`（继续执行 register.tsx）

- 输入 `state_4["messages"]`
- LLM 输出：`AIMessage(tool_calls=[create_file(register.tsx)])`

LangGraph：

```
state_5 = {
    "messages": [
        ...,                      # 之前所有消息
        AI(tool_calls=[create_file(register.tsx)])
    ],
    "todos": same as state_4
}
```

### Step 7: `tools` 节点执行 `create_file(register.tsx)`

- `wrap_tool_call` attempt=0 → 假设成功
- 返回 `ToolMessage("文件 register.tsx 已创建")`

LangGraph 更新：

```
state_6 = {
    "messages": [
        ...,
        AI(tool_calls=[create_file(register.tsx)]),
        ToolMessage("文件 register.tsx 已创建")
    ],
    "todos": [
        {"content": "创建 login.tsx", "status": "completed"},
        {"content": "创建 register.tsx", "status": "completed"}
    ]
}
```

---

### Step 8: `model node`（总结回答）

- 输入 `state_6["messages"]`（包含两次工具成功记录 + todos 状态）
- LLM 输出：`AIMessage("已完成 login.tsx 和 register.tsx")`

LangGraph 最终状态：

```
state_final = {
    "messages": [
        Human("帮我创建..."),
        AI(write_todos),
        Tool("Updated todo list"),
        AI(create_file login.tsx),
        Tool("文件 login.tsx 已创建"),
        AI(create_file register.tsx),
        Tool("文件 register.tsx 已创建"),
        AI("我已经完成两个文件")
    ],
    "todos": [
        {"content": "创建 login.tsx", "status": "completed"},
        {"content": "创建 register.tsx", "status": "completed"}
    ]
}
```

Graph 根据最后一条 AIMessage 无 `tool_calls`，因此走 `END`。

---

## 重试失败时对状态的影响

- **模型重试失败**（如连续 3 次超时）：
  - 如果 `on_failure="continue"`，`ModelRetry._handle_failure` 会返回 `ModelResponse(result=[AIMessage("Model call failed...")])`
  - LangGraph 在 `state["messages"]` 中追加这条 AIMessage，状态 **变为** “模型返回错误描述”
  - Graph 会继续根据错误消息决定是否结束或再次经过 tools

- **工具重试失败**：
  - `ToolRetry._handle_failure` 返回 `ToolMessage(status="error", content="Tool 'xxx' failed...")`
  - 这条 ToolMessage 会被添加到 `state["messages"]`
  - LLM 在下一次模型调用时能看到“某工具失败”并决定下一步

- **on_failure="error"** 时：中间件会 re-raise 原异常，LangGraph 捕获→整个 Agent 执行终止，状态保持在最后一次成功的样子。

---

## LangGraph 状态流动总结表

| 步骤 | 节点 | `state["messages"]` | `state["todos"]` | Retry 介入点 |
|------|------|----------------------|------------------|--------------|
| state_0 | START | [Human] | 未定义 | - |
| state_1 | model（重试后成功） | [Human, AI(write_todos)] | 未定义 | ModelRetry |
| state_2 | tools（write_todos） | +ToolMessage | 建立 todos 列表 | ToolRetry（未触发） |
| state_3 | model | +AI(create_file login) | 同上 | - |
| state_3（重试中） | tools（create_file login）第一次失败 | 状态不变 | 不变 | ToolRetry（attempt=0→1） |
| state_4 | tools 成功 | +ToolMessage("login 完成") | login 完成 | ToolRetry |
| state_5 | model | +AI(create_file register) | login 完成 | - |
| state_6 | tools | +ToolMessage("register 完成") | login/register 完成 | - |
| state_final | model（总结） | +AI("已完成") | login/register 完成 | - |

---

## 本质理解

1. **LangGraph 状态是单一的**：所有节点共享 `state`；重试发生在某个节点内部（wrap_model_call / wrap_tool_call），在重试成功之前 `state` 不会改变。
2. **ModelRetry 和 ToolRetry 是“透明”的**：它们只包裹 handler 调用；只有当最终选择 `on_failure="continue"` 时才主动向 `state["messages"]` 添加错误消息。
3. **todos 状态只在 write_todos 工具执行成功后更新**，重试期间不会有影响。
4. **每次 LLM 调用的输入消息= `state["messages"]`**（加上 system prompt）。重试次数不改变消息列表；只有成功或 on_failure 才会新增消息。

