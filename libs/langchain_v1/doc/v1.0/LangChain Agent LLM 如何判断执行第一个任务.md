# LLM 如何判断执行第一个任务

## 关键机制：通过 ToolMessage 传递状态信息

当 `write_todos` 执行后，会创建一个 `ToolMessage` 并添加到 `messages` 中：

```python:118:126:langchain/agents/middleware/todo.py
@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )
```

要点：
- `Command.update["todos"]` 更新 `state["todos"]`
- `Command.update["messages"]` 添加 `ToolMessage`，内容包含 `todos` 的完整信息

### 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 迭代1：创建 Todo List                                            │
└─────────────────────────────────────────────────────────────────┘

1. 模型节点执行
   ├─ LLM 调用 write_todos([...])
   └─ 返回：AIMessage(tool_calls=[write_todos(...)])

2. 工具节点执行
   ├─ write_todos 执行
   └─ 返回：Command(
       update={
         "todos": [
           {content: "任务1", status: "in_progress"},
           {content: "任务2", status: "pending"}
         ],
         "messages": [
           ToolMessage("Updated todo list to [...]")  ← 关键！
         ]
       }
     )

3. 状态更新
   ├─ state["todos"] = [...]
   └─ state["messages"].append(ToolMessage(...))  ← LLM 能看到这个！

┌─────────────────────────────────────────────────────────────────┐
│ 迭代2：回到模型节点（关键步骤）                                  │
└─────────────────────────────────────────────────────────────────┘

1. model_node 执行
   ├─ 创建 ModelRequest
   │   ├─ messages = state["messages"]  ← 包含 ToolMessage！
   │   ├─ state = state  ← 包含 todos
   │   └─ system_prompt = ... + WRITE_TODOS_SYSTEM_PROMPT
   │
   └─ 调用 LLM
       ├─ LLM 看到的消息列表：
       │   [
       │     HumanMessage("用户请求"),
       │     AIMessage(tool_calls=[write_todos(...)]),
       │     ToolMessage("Updated todo list to [  ← LLM 看到这个！
       │                  {content: '任务1', status: 'in_progress'},
       │                  {content: '任务2', status: 'pending'}
       │                ]")
       │   ]
       │
       ├─ LLM 看到的系统提示词：
       │   "... 标记为 in_progress 的任务应该立即执行 ..."
       │
       └─ LLM 看到的工具描述（write_todos）：
          "... 当你开始工作时，在开始之前标记为 in_progress ..."
```

### 系统提示词的指导作用

```python:103:115:langchain/agents/middleware/todo.py
WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""
```

### 工具描述中的关键规则

```python:56:82:langchain/agents/middleware/todo.py
## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.
```

关键规则：
- 第 81 行：创建 todo list 时，立即将第一个任务标记为 `in_progress`
- 第 82 行：除非所有任务完成，否则应至少有一个 `in_progress` 任务
- 第 79 行：完成当前任务后再开始新任务

### LLM 的判断逻辑

当 LLM 回到模型节点时，它会：

1. 读取 ToolMessage 中的 todos 信息
   ```
   ToolMessage: "Updated todo list to [
     {content: '任务1', status: 'in_progress'},
     {content: '任务2', status: 'pending'}
   ]"
   ```

2. 结合系统提示词和工具描述
   - 看到 `status: "in_progress"` → 应该执行这个任务
   - 看到 `status: "pending"` → 等待执行
   - 规则：完成 `in_progress` 任务后再开始 `pending` 任务

3. 决定下一步行动
   ```
   LLM 推理：
   - 当前有 "任务1" status="in_progress"
   - 根据规则，应该执行 "任务1"
   - 执行任务1（调用相关工具）
   - 完成后，标记 "任务1" 为 completed
   - 标记 "任务2" 为 in_progress
   ```

### 完整判断流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ LLM 判断执行第一个任务的完整流程                                │
└─────────────────────────────────────────────────────────────────┘

回到模型节点
    ↓
model_node(state) 执行
    ├─ 创建 ModelRequest
    │   └─ messages = state["messages"]
    │       └─ 包含：ToolMessage("Updated todo list to [...]")
    │
    └─ TodoListMiddleware.wrap_model_call() 注入系统提示词
        └─ request.system_prompt += WRITE_TODOS_SYSTEM_PROMPT
    ↓
LLM 接收到的信息：
    ├─ 消息历史：
    │   [
    │     HumanMessage("用户请求"),
    │     AIMessage(tool_calls=[write_todos(...)]),
    │     ToolMessage("Updated todo list to [
    │                  {content: '任务1', status: 'in_progress'},  ← 看到这个！
    │                  {content: '任务2', status: 'pending'}
    │                ]")
    │   ]
    │
    ├─ 系统提示词：
    │   "... 标记为 in_progress 的任务应该立即执行 ..."
    │   "... 完成当前任务后再开始新任务 ..."
    │
    └─ 工具列表：
        └─ write_todos (描述中包含任务管理规则)
    ↓
LLM 推理过程：
    1. 解析 ToolMessage，提取 todos 列表
    2. 发现 "任务1" status="in_progress"
    3. 根据系统提示词和工具描述：
       - "in_progress" 表示正在执行
       - 应该完成这个任务
       - 完成后标记为 "completed"
       - 然后开始下一个 "pending" 任务
    4. 决定：执行 "任务1"
    ↓
LLM 返回：
    AIMessage(
        tool_calls=[
            other_tool(...),           ← 执行任务1的实际工具
            write_todos([              ← 更新状态
                {content: "任务1", status: "completed"},
                {content: "任务2", status: "in_progress"}
            ])
        ]
    )
```

### 总结

LLM 判断执行第一个任务的依据：

1. 通过 ToolMessage 看到 todos 状态
   - `write_todos` 执行后，`ToolMessage` 包含完整的 todos 信息
   - 该消息在 `state["messages"]` 中，LLM 能看到

2. 系统提示词和工具描述提供规则
   - `in_progress` 表示正在执行，应完成它
   - `pending` 表示待执行，完成后开始
   - 完成当前任务后再开始新任务

3. LLM 的推理能力
   - 解析 ToolMessage 中的 todos
   - 识别 `in_progress` 状态
   - 根据规则决定执行该任务

核心机制：状态通过 ToolMessage 传递给 LLM，而不是直接访问 `state["todos"]`。LLM 通过消息历史中的 ToolMessage 了解当前任务状态，并结合系统提示词做出决策。