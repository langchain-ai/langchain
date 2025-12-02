# agent执行流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent 执行流程（LangGraph 图）                        │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段1：图构建（create_agent 时，只执行一次）                             │
└─────────────────────────────────────────────────────────────────────────┘

1. 创建图节点
   ┌─────────────────────┐
   │ graph.add_node(      │
   │   "model",           │ ← 模型节点：执行 model_node()
   │   model_node         │
   │ )                    │
   └─────────────────────┘

   ┌─────────────────────┐
   │ graph.add_node(     │
   │   "tools",          │ ← 工具节点：执行 ToolNode
   │   tool_node         │   包含 write_todos 工具（来自 todo.py）
   │ )                   │
   └─────────────────────┘

2. 注册条件边（路由规则）
   ┌─────────────────────────────────────────────────────────────┐
   │ graph.add_conditional_edges(                               │
   │   "model",                    ← 从模型节点出发              │
   │   _make_model_to_tools_edge(...),  ← 路由函数              │
   │   ["tools", "end"]            ← 可能的目标节点             │
   │ )                                                            │
   └─────────────────────────────────────────────────────────────┘
   说明：当模型节点执行完成后，LangGraph 会调用这个函数决定下一步

   ┌─────────────────────────────────────────────────────────────┐
   │ graph.add_conditional_edges(                               │
   │   "tools",                    ← 从工具节点出发              │
   │   _make_tools_to_model_edge(...),  ← 路由函数              │
   │   ["model"]                   ← 可能的目标节点             │
   │ )                                                            │
   └─────────────────────────────────────────────────────────────┘
   说明：当工具节点执行完成后，LangGraph 会调用这个函数决定下一步

┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段2：运行时执行（Agent 循环）                                          │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
迭代1：创建 Todo List
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ 步骤1：模型节点执行                                                   │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    model_node(state) 执行
    ├─ state = {messages: [...], todos: None}
    ├─ TodoListMiddleware.wrap_model_call() 注入系统提示词
    │   └─ request.system_prompt += WRITE_TODOS_SYSTEM_PROMPT
    ├─ 调用 LLM（GPT-4）
    │   ├─ LLM 看到：系统提示词（包含 todo 使用指导）
    │   ├─ LLM 看到：工具列表（包含 write_todos）
    │   └─ LLM 看到：state["todos"] = None
    └─ LLM 返回：AIMessage(tool_calls=[write_todos(...)])
    ↓
    state 更新：messages 添加 AIMessage
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤2：模型节点完成 → LangGraph 调用路由函数                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 检测到 "model" 节点完成
    ↓
    LangGraph 查找条件边：从 "model" 出发的条件边
    ↓
    LangGraph 调用：_make_model_to_tools_edge(state)
    ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ _make_model_to_tools_edge() 执行逻辑：                      │
    │                                                             │
    │ 1. 检查 state["jump_to"] → 无                             │
    │                                                             │
    │ 2. 获取最后一条 AIMessage                                  │
    │    last_ai_message = AIMessage(tool_calls=[write_todos])  │
    │                                                             │
    │ 3. 检查 tool_calls 数量                                    │
    │    len(tool_calls) = 1 > 0  ✓                             │
    │    → 不退出，继续                                         │
    │                                                             │
    │ 4. 查找待执行的工具调用                                    │
    │    pending_tool_calls = [write_todos(...)]                │
    │                                                             │
    │ 5. 有待执行的工具调用                                      │
    │    → 返回 [Send("tools", write_todos(...))]               │
    └─────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 路由到 "tools" 节点
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤3：工具节点执行（todo.py 的 write_todos）                        │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    ToolNode 执行 write_todos 工具
    ├─ 调用 todo.py 中的 write_todos 函数
    │   └─ 参数：todos = [
    │         {content: "任务1", status: "in_progress"},
    │         {content: "任务2", status: "pending"}
    │       ]
    └─ 返回：Command(
        update={
          "todos": todos,  ← 更新 state["todos"]
          "messages": [ToolMessage(...)]
        }
      )
    ↓
    ToolNode 处理 Command
    ├─ 应用 Command.update 到 state
    └─ state["todos"] = todos  ← 状态已更新！
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤4：工具节点完成 → LangGraph 调用路由函数                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 检测到 "tools" 节点完成
    ↓
    LangGraph 查找条件边：从 "tools" 出发的条件边
    ↓
    LangGraph 调用：_make_tools_to_model_edge(state)
    ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ _make_tools_to_model_edge() 执行逻辑：                     │
    │                                                             │
    │ 1. 检查工具是否有 return_direct=True                        │
    │    write_todos 没有 → 继续                                │
    │                                                             │
    │ 2. 检查是否是 structured_output_tool                        │
    │    write_todos 不是 → 继续                                │
    │                                                             │
    │ 3. 默认：继续循环                                           │
    │    → 返回 "model"  ← 回到模型节点！                       │
    └─────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 路由到 "model" 节点（循环回到模型）

═══════════════════════════════════════════════════════════════════════════
迭代2：执行第一个任务
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ 步骤5：模型节点再次执行（回到模型）                                    │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    model_node(state) 再次执行
    ├─ state = {
    │     messages: [...ToolMessage("Updated todo list...")],
    │     todos: [  ← 模型能看到更新后的 todos！
    │       {content: "任务1", status: "in_progress"},
    │       {content: "任务2", status: "pending"}
    │     ]
    │   }
    ├─ TodoListMiddleware.wrap_model_call() 再次注入系统提示词
    ├─ 调用 LLM
    │   ├─ LLM 看到：state["todos"]（包含 in_progress 任务）
    │   ├─ LLM 看到：系统提示词（"mark todos as completed..."）
    │   └─ LLM 看到：ToolMessage（"Updated todo list..."）
    └─ LLM 决策：
        ├─ 看到 "任务1" status="in_progress"
        ├─ 执行任务1（调用其他工具，如代码分析工具）
        └─ 返回：AIMessage(tool_calls=[
              other_tool(...),      ← 执行任务
              write_todos([...])    ← 更新状态
            ])
    ↓
    state 更新：messages 添加 AIMessage
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤6：模型节点完成 → LangGraph 调用路由函数                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 调用：_make_model_to_tools_edge(state)
    ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ _make_model_to_tools_edge() 执行逻辑：                      │
    │                                                             │
    │ 1. 获取最后一条 AIMessage                                  │
    │    last_ai_message.tool_calls = [                          │
    │      other_tool(...),                                      │
    │      write_todos([...])                                    │
    │    ]                                                        │
    │                                                             │
    │ 2. 查找待执行的工具调用                                    │
    │    pending_tool_calls = [                                  │
    │      other_tool(...),      ← 待执行                        │
    │      write_todos([...])    ← 待执行                        │
    │    ]                                                        │
    │                                                             │
    │ 3. 有待执行的工具调用                                      │
    │    → 返回 [Send("tools", ...), Send("tools", ...)]        │
    └─────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 路由到 "tools" 节点（并行执行多个工具）
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤7：工具节点执行（执行任务 + 更新 todos）                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    ToolNode 执行工具
    ├─ 执行 other_tool（完成任务1的实际工作）
    └─ 执行 write_todos（来自 todo.py）
        └─ 返回：Command(update={
              "todos": [
                {content: "任务1", status: "completed"},  ← 完成
                {content: "任务2", status: "in_progress"}  ← 开始下一个
              ]
            })
    ↓
    state 更新：todos 和 messages 都更新了
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 步骤8：工具节点完成 → LangGraph 调用路由函数                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 调用：_make_tools_to_model_edge(state)
    ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ _make_tools_to_model_edge() 执行逻辑：                     │
    │                                                             │
    │ 1. 检查退出条件 → 都不满足                                  │
    │                                                             │
    │ 2. 默认：继续循环                                           │
    │    → 返回 "model"  ← 再次回到模型节点！                    │
    └─────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 路由到 "model" 节点（继续循环）

═══════════════════════════════════════════════════════════════════════════
迭代3、4、5...：继续执行任务
═══════════════════════════════════════════════════════════════════════════

重复步骤 5-8，直到所有任务完成...

═══════════════════════════════════════════════════════════════════════════
最后迭代：所有任务完成
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ 模型节点执行                                                         │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    model_node(state) 执行
    ├─ state["todos"] = [
    │     {content: "任务1", status: "completed"},
    │     {content: "任务2", status: "completed"},
    │     {content: "任务3", status: "completed"}
    │   ]
    └─ LLM 返回：AIMessage(tool_calls=[])  ← 没有工具调用！
    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 模型节点完成 → LangGraph 调用路由函数                                 │
└─────────────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 调用：_make_model_to_tools_edge(state)
    ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ _make_model_to_tools_edge() 执行逻辑：                      │
    │                                                             │
    │ 1. 获取最后一条 AIMessage                                  │
    │    last_ai_message.tool_calls = []  ← 空！                │
    │                                                             │
    │ 2. 检查 tool_calls 数量                                    │
    │    len(tool_calls) = 0                                     │
    │    → 返回 "end"  ← 退出循环！                             │
    └─────────────────────────────────────────────────────────────┘
    ↓
    LangGraph 路由到 "end" 节点
    ↓
    Agent 执行结束
```

## 关键关联点说明

### 1. `_make_model_to_tools_edge` 的作用

```python:1440:1493:langchain/agents/factory.py
def _make_model_to_tools_edge(...):
    def model_to_tools(state):
        # 检查模型是否调用了工具
        if len(last_ai_message.tool_calls) == 0:
            return "end"  # 没有工具调用 → 结束
        
        # 有待执行的工具调用
        if pending_tool_calls:
            return ["tools"]  # 有工具调用 → 路由到工具节点
        
        return "model"  # 其他情况 → 回到模型
```

- 作用：决定模型节点完成后是去工具节点还是结束
- 与 todo.py 的关联：模型调用 `write_todos` 时，该函数会路由到 `"tools"` 节点

### 2. `_make_tools_to_model_edge` 的作用

```python:1523:1552:langchain/agents/factory.py
def _make_tools_to_model_edge(...):
    def tools_to_model(state):
        # 检查退出条件...
        
        # 默认：继续循环
        return "model"  # 工具执行完成 → 回到模型节点
```

- 作用：决定工具节点完成后是回到模型节点还是结束
- 与 todo.py 的关联：`write_todos` 执行完成后，该函数会路由回 `"model"` 节点

### 3. todo.py 的驱动作用

```python:118:126:langchain/agents/middleware/todo.py
def write_todos(...) -> Command:
    return Command(
        update={
            "todos": todos,  # ← 更新状态
            "messages": [ToolMessage(...)]
        }
    )
```

- 更新 `state["todos"]`
- 状态更新后，下一轮模型调用能看到新的 `todos`
- 模型根据新的 `todos` 继续执行任务

## 驱动机制总结

```
┌─────────────────────────────────────────────────────────────┐
│ 驱动循环的核心机制                                            │
└─────────────────────────────────────────────────────────────┘

1. 图构建时注册路由函数
   ├─ 模型 → 工具：_make_model_to_tools_edge
   └─ 工具 → 模型：_make_tools_to_model_edge

2. 运行时 LangGraph 自动调用路由函数
   ├─ 模型节点完成 → 调用 _make_model_to_tools_edge
   └─ 工具节点完成 → 调用 _make_tools_to_model_edge

3. todo.py 通过更新状态驱动循环
   ├─ write_todos 更新 state["todos"]
   └─ 下一轮模型看到新状态，继续执行

4. 循环直到模型不再调用工具
   └─ _make_model_to_tools_edge 检测到 tool_calls=[] → 结束
```

## 关键理解

1. `_make_model_to_tools_edge` 不直接调用 todo.py
   - 它只是路由函数，决定下一步去哪里
   - todo.py 的 `write_todos` 在工具节点中执行

2. todo.py 通过状态更新驱动循环
   - `Command.update` 更新 `state["todos"]`
   - 下一轮模型调用时能看到更新后的状态
   - 模型根据状态决定下一步行动

3. LangGraph 的条件边机制是核心
   - 节点完成后自动调用路由函数
   - 路由函数返回下一个节点名称
   - LangGraph 自动路由到下一个节点

这就是整个驱动机制：LangGraph 的条件边 + todo.py 的状态更新 = 自驱动的任务执行循环。