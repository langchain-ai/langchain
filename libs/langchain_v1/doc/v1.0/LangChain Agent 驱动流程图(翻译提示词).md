# 完整 Agent 执行流程图（带提示词翻译 todo中间件主要靠）

```
═══════════════════════════════════════════════════════════════════════════
                    Agent 完整执行流程（Todo Middleware）
═══════════════════════════════════════════════════════════════════════════

假设用户请求："帮我重构代码库，包括：1. 代码审查 2. 重构 3. 测试"

┌─────────────────────────────────────────────────────────────────────────┐
│ 阶段 0：图构建（create_agent 时，只执行一次）                            │
└─────────────────────────────────────────────────────────────────────────┘

注册的路由函数：
├─ 从 "model" 节点出发：_make_model_to_tools_edge()
│   └─ 可能的目标：["tools", "end", "model"]
│
└─ 从 "tools" 节点出发：_make_tools_to_model_edge()
    └─ 可能的目标：["model", "end"]

═══════════════════════════════════════════════════════════════════════════
迭代 1：创建 Todo List
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：START → model                                                      │
└─────────────────────────────────────────────────────────────────────────┘

【状态 state】
{
  "messages": [
    HumanMessage("帮我重构代码库，包括：1. 代码审查 2. 重构 3. 测试")
  ],
  "todos": None
}

【model_node 执行】
├─ 创建 ModelRequest
│   ├─ messages = state["messages"]
│   ├─ state = state
│   ├─ tools = [write_todos, ...]
│   └─ system_prompt = None
│
└─ TodoListMiddleware.wrap_model_call() 注入系统提示词
    └─ request.system_prompt = WRITE_TODOS_SYSTEM_PROMPT

【注入的系统提示词（中文翻译）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## `write_todos`

你可以使用 `write_todos` 工具来帮助管理和规划复杂的目标。
对于复杂的目标，使用此工具可以确保你跟踪每个必要的步骤，并向用户展示你的进度。
这个工具对于规划复杂目标非常有用，可以将这些较大的复杂目标分解为更小的步骤。

关键是要在完成一个步骤后立即标记 todos 为已完成。不要在标记多个步骤为已完成之前批量处理它们。
对于只需要几个步骤的简单目标，最好直接完成目标而不要使用此工具。
编写 todos 需要时间和 token，在管理复杂的多步骤问题时使用它很有帮助！但对于简单的几步请求则不需要。

## 重要的待办事项列表使用注意事项
- `write_todos` 工具永远不应该并行调用多次。
- 不要害怕在过程中修改待办事项列表。新信息可能会揭示需要完成的新任务，或者不再相关的旧任务。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【LLM 看到的工具描述（write_todos，中文翻译）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用此工具为当前工作会话创建和管理结构化任务列表。这有助于你跟踪进度、
组织复杂任务，并向用户展示你的彻底性。

只有在认为有助于保持组织性时才使用此工具。如果用户的请求是琐碎的且少于 3 步，
最好不要使用此工具，而是直接执行任务。

## 何时使用此工具
在以下场景中使用此工具：

1. 复杂的多步骤任务 - 当任务需要 3 个或更多不同的步骤或操作时
2. 非平凡且复杂的任务 - 需要仔细规划或多个操作的任务
3. 用户明确要求待办事项列表 - 当用户直接要求你使用待办事项列表时
4. 用户提供多个任务 - 当用户提供要完成的事情列表（编号或逗号分隔）时
5. 计划可能需要根据前几个步骤的结果进行未来的修订或更新

## 如何使用此工具
1. 当你开始处理任务时 - 在开始工作之前将其标记为 in_progress。
2. 完成任务后 - 将其标记为已完成，并添加在实施过程中发现的任何新的后续任务。
3. 你还可以更新未来的任务，例如如果不再需要则删除它们，或添加必要的新任务。
   不要更改以前已完成的任务。
4. 你可以一次对待办事项列表进行多次更新。例如，当你完成任务时，
   可以将需要开始的下一个任务标记为 in_progress。

## 何时不使用此工具
在以下情况下跳过使用此工具很重要：
1. 只有一个简单的任务
2. 任务是琐碎的，跟踪它没有好处
3. 任务可以在少于 3 个琐碎步骤中完成
4. 任务纯粹是对话性或信息性的

## 任务状态和管理

1. **任务状态**：使用这些状态来跟踪进度：
   - pending: 尚未开始的任务
   - in_progress: 当前正在处理（如果它们彼此不相关且可以并行运行，
     你可以同时有多个 in_progress 的任务）
   - completed: 成功完成的任务

2. **任务管理**：
   - 在你工作时实时更新任务状态
   - 完成后立即标记任务完成（不要批量完成）
   - 在开始新任务之前完成当前任务
   - 从列表中完全删除不再相关的任务
   - 重要：当你编写此待办事项列表时，你应该立即将第一个任务（或任务）标记为 in_progress！
   - 重要：除非所有任务都已完成，否则你应该始终至少有一个 in_progress 的任务，
     以向用户显示你正在处理某些事情。

3. **任务完成要求**：
   - 只有在完全完成时才将任务标记为已完成
   - 如果遇到错误、阻塞或无法完成，请将任务保持为 in_progress
   - 当被阻塞时，创建一个新任务描述需要解决的内容
   - 在以下情况下永远不要将任务标记为已完成：
     - 存在未解决的问题或错误
     - 工作是部分或不完整的
     - 遇到阻止完成的阻塞
     - 找不到必要的资源或依赖项
     - 未达到质量标准

4. **任务分解**：
   - 创建具体的、可操作的项目
   - 将复杂任务分解为更小的、可管理的步骤
   - 使用清晰、描述性的任务名称

积极主动的任务管理展示了你的专注度，并确保你成功完成所有要求。
记住：如果你只需要进行几次工具调用来完成任务，并且清楚你需要做什么，
最好直接执行任务，完全不要调用此工具。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【LLM 返回的 AIMessage】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "write_todos",
            "args": {
                "todos": [
                    {"content": "代码审查", "status": "in_progress"},
                    {"content": "重构代码", "status": "pending"},
                    {"content": "编写测试", "status": "pending"}
                ]
            },
            "id": "call_abc123"
        }
    ]
)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【状态更新】
state["messages"].append(AIMessage(...))

┌─────────────────────────────────────────────────────────────────────────┐
│ 路由判断：_make_model_to_tools_edge(state)                              │
└─────────────────────────────────────────────────────────────────────────┘

【判断逻辑（factory.py:1446-1493）】
1. 检查 jump_to → 无
2. 获取最后一条 AIMessage
   └─ last_ai_message.tool_calls = [write_todos(...)]
3. 检查 tool_calls 数量
   └─ len(tool_calls) = 1 > 0 ✓
4. 查找待执行的工具调用
   └─ pending_tool_calls = [write_todos(...)]
5. 有待执行的工具调用
   └─ 返回：["tools"] ← 路由到工具节点

【当前状态】
{
  "messages": [
    HumanMessage("..."),
    AIMessage(tool_calls=[write_todos(...)])
  ],
  "todos": None
}

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：tools                                                              │
└─────────────────────────────────────────────────────────────────────────┘

【ToolNode 执行 write_todos】
├─ 调用 write_todos(
│     todos=[
│       {"content": "代码审查", "status": "in_progress"},
│       {"content": "重构代码", "status": "pending"},
│       {"content": "编写测试", "status": "pending"}
│     ],
│     tool_call_id="call_abc123"
│   )
│
└─ 返回：Command(
    update={
        "todos": [
            {"content": "代码审查", "status": "in_progress"},
            {"content": "重构代码", "status": "pending"},
            {"content": "编写测试", "status": "pending"}
        ],
        "messages": [
            ToolMessage(
                content="Updated todo list to [{'content': '代码审查', 'status': 'in_progress'}, {'content': '重构代码', 'status': 'pending'}, {'content': '编写测试', 'status': 'pending'}]",
                tool_call_id="call_abc123"
            )
        ]
    }
)

【状态更新】
state["todos"] = [
    {"content": "代码审查", "status": "in_progress"},
    {"content": "重构代码", "status": "pending"},
    {"content": "编写测试", "status": "pending"}
]
state["messages"].append(ToolMessage(...))

┌─────────────────────────────────────────────────────────────────────────┐
│ 路由判断：_make_tools_to_model_edge(state)                              │
└─────────────────────────────────────────────────────────────────────────┘

【判断逻辑（factory.py:1530-1550）】
1. 获取最后一条 AIMessage 和 ToolMessages
   └─ last_ai_message.tool_calls = [write_todos(...)]
   └─ tool_messages = [ToolMessage(...)]
2. 检查客户端工具调用
   └─ client_side_tool_calls = [write_todos(...)]
3. 检查 return_direct
   └─ write_todos 没有 return_direct=True ✓
4. 检查结构化输出工具
   └─ write_todos 不是 structured_output_tool ✓
5. 默认：继续循环
   └─ 返回："model" ← 回到模型节点！

【当前状态】
{
  "messages": [
    HumanMessage("..."),
    AIMessage(tool_calls=[write_todos(...)]),
    ToolMessage("Updated todo list to [...]")
  ],
  "todos": [
    {"content": "代码审查", "status": "in_progress"},
    {"content": "重构代码", "status": "pending"},
    {"content": "编写测试", "status": "pending"}
  ]
}

═══════════════════════════════════════════════════════════════════════════
迭代 2：执行第一个任务（代码审查）
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：model（回到模型节点）                                              │
└─────────────────────────────────────────────────────────────────────────┘

【model_node 执行】
├─ 创建 ModelRequest
│   ├─ messages = state["messages"]  ← 包含 ToolMessage！
│   ├─ state = state  ← 包含 todos
│   ├─ tools = [write_todos, code_review_tool, ...]
│   └─ system_prompt = WRITE_TODOS_SYSTEM_PROMPT
│
└─ TodoListMiddleware.wrap_model_call() 再次注入系统提示词

【LLM 看到的消息历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  HumanMessage("帮我重构代码库，包括：1. 代码审查 2. 重构 3. 测试"),
  
  AIMessage(
    tool_calls=[{
      "name": "write_todos",
      "args": {
        "todos": [
          {"content": "代码审查", "status": "in_progress"},
          {"content": "重构代码", "status": "pending"},
          {"content": "编写测试", "status": "pending"}
        ]
      },
      "id": "call_abc123"
    }]
  ),
  
  ToolMessage(
    content="Updated todo list to [
      {'content': '代码审查', 'status': 'in_progress'},
      {'content': '重构代码', 'status': 'pending'},
      {'content': '编写测试', 'status': 'pending'}
    ]",
    tool_call_id="call_abc123"
  )
]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【LLM 推理过程】
1. 看到 ToolMessage 中的 todos 列表
2. 发现 "代码审查" status="in_progress"
3. 根据系统提示词和工具描述：
   - "in_progress" 表示正在执行
   - 应该完成这个任务
   - 完成后标记为 "completed"
   - 然后开始下一个 "pending" 任务
4. 决定：执行 "代码审查" 任务

【LLM 返回的 AIMessage】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "code_review_tool",
            "args": {"codebase_path": "/path/to/code"},
            "id": "call_def456"
        },
        {
            "name": "write_todos",
            "args": {
                "todos": [
                    {"content": "代码审查", "status": "completed"},
                    {"content": "重构代码", "status": "in_progress"},
                    {"content": "编写测试", "status": "pending"}
                ]
            },
            "id": "call_ghi789"
        }
    ]
)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【状态更新】
state["messages"].append(AIMessage(...))

┌─────────────────────────────────────────────────────────────────────────┐
│ 路由判断：_make_model_to_tools_edge(state)                              │
└─────────────────────────────────────────────────────────────────────────┘

【判断逻辑】
1. 检查 jump_to → 无
2. 获取最后一条 AIMessage
   └─ last_ai_message.tool_calls = [
         code_review_tool(...),
         write_todos(...)
       ]
3. 检查 tool_calls 数量
   └─ len(tool_calls) = 2 > 0 ✓
4. 查找待执行的工具调用
   └─ pending_tool_calls = [
         code_review_tool(...),
         write_todos(...)
       ]
5. 有待执行的工具调用
   └─ 返回：[
         Send("tools", code_review_tool(...)),
         Send("tools", write_todos(...))
       ] ← 路由到工具节点（并行执行）

【当前状态】
{
  "messages": [
    HumanMessage("..."),
    AIMessage(tool_calls=[write_todos(...)]),
    ToolMessage("Updated todo list to [...]"),
    AIMessage(tool_calls=[code_review_tool(...), write_todos(...)])
  ],
  "todos": [
    {"content": "代码审查", "status": "in_progress"},
    {"content": "重构代码", "status": "pending"},
    {"content": "编写测试", "status": "pending"}
  ]
}

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：tools（并行执行两个工具）                                         │
└─────────────────────────────────────────────────────────────────────────┘

【ToolNode 执行】
├─ 执行 code_review_tool(...)
│   └─ 返回：ToolMessage(
│       content="代码审查完成：发现 5 个问题需要修复",
│       tool_call_id="call_def456"
│     )
│
└─ 执行 write_todos(...)
    └─ 返回：Command(
        update={
            "todos": [
                {"content": "代码审查", "status": "completed"},
                {"content": "重构代码", "status": "in_progress"},
                {"content": "编写测试", "status": "pending"}
            ],
            "messages": [
                ToolMessage(
                    content="Updated todo list to [...]",
                    tool_call_id="call_ghi789"
                )
            ]
        }
    )

【状态更新】
state["todos"] = [
    {"content": "代码审查", "status": "completed"},
    {"content": "重构代码", "status": "in_progress"},
    {"content": "编写测试", "status": "pending"}
]
state["messages"].extend([
    ToolMessage("代码审查完成：发现 5 个问题需要修复"),
    ToolMessage("Updated todo list to [...]")
])

┌─────────────────────────────────────────────────────────────────────────┐
│ 路由判断：_make_tools_to_model_edge(state)                              │
└─────────────────────────────────────────────────────────────────────────┘

【判断逻辑】
1. 获取最后一条 AIMessage 和 ToolMessages
   └─ last_ai_message.tool_calls = [
         code_review_tool(...),
         write_todos(...)
       ]
   └─ tool_messages = [
         ToolMessage(code_review_tool 结果),
         ToolMessage(write_todos 结果)
       ]
2. 检查客户端工具调用
   └─ client_side_tool_calls = [
         code_review_tool(...),
         write_todos(...)
       ]
3. 检查 return_direct
   └─ 两个工具都没有 return_direct=True ✓
4. 检查结构化输出工具
   └─ 都不是 structured_output_tool ✓
5. 默认：继续循环
   └─ 返回："model" ← 再次回到模型节点！

【当前状态】
{
  "messages": [
    HumanMessage("..."),
    AIMessage(tool_calls=[write_todos(...)]),
    ToolMessage("Updated todo list to [...]"),
    AIMessage(tool_calls=[code_review_tool(...), write_todos(...)]),
    ToolMessage("代码审查完成：发现 5 个问题需要修复"),
    ToolMessage("Updated todo list to [...]")
  ],
  "todos": [
    {"content": "代码审查", "status": "completed"},
    {"content": "重构代码", "status": "in_progress"},
    {"content": "编写测试", "status": "pending"}
  ]
}

═══════════════════════════════════════════════════════════════════════════
迭代 3：执行第二个任务（重构代码）
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：model（回到模型节点）                                              │
└─────────────────────────────────────────────────────────────────────────┘

【LLM 看到的消息历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  HumanMessage("帮我重构代码库，包括：1. 代码审查 2. 重构 3. 测试"),
  
  AIMessage(tool_calls=[write_todos(...)]),
  ToolMessage("Updated todo list to [...]"),
  
  AIMessage(tool_calls=[code_review_tool(...), write_todos(...)]),
  ToolMessage("代码审查完成：发现 5 个问题需要修复"),
  ToolMessage("Updated todo list to [
    {'content': '代码审查', 'status': 'completed'},
    {'content': '重构代码', 'status': 'in_progress'},  ← 看到这个！
    {'content': '编写测试', "status": 'pending'}
  ]")
]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【LLM 推理过程】
1. 看到 ToolMessage 中的 todos 列表
2. 发现 "重构代码" status="in_progress"
3. 看到 "代码审查" 已完成
4. 决定：执行 "重构代码" 任务

【LLM 返回的 AIMessage】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "refactor_tool",
            "args": {"refactor_type": "cleanup"},
            "id": "call_jkl012"
        },
        {
            "name": "write_todos",
            "args": {
                "todos": [
                    {"content": "代码审查", "status": "completed"},
                    {"content": "重构代码", "status": "completed"},
                    {"content": "编写测试", "status": "in_progress"}
                ]
            },
            "id": "call_mno345"
        }
    ]
)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【路由判断】→ 路由到 "tools" 节点

【工具执行】
├─ refactor_tool(...) → ToolMessage("重构完成")
└─ write_todos(...) → 更新 todos

【状态更新】
state["todos"] = [
    {"content": "代码审查", "status": "completed"},
    {"content": "重构代码", "status": "completed"},
    {"content": "编写测试", "status": "in_progress"}
]

【路由判断】→ 返回 "model"

═══════════════════════════════════════════════════════════════════════════
迭代 4：执行第三个任务（编写测试）
═══════════════════════════════════════════════════════════════════════════

【LLM 返回的 AIMessage】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIMessage(
    content="",
    tool_calls=[
        {
            "name": "write_test_tool",
            "args": {"test_type": "unit"},
            "id": "call_pqr678"
        },
        {
            "name": "write_todos",
            "args": {
                "todos": [
                    {"content": "代码审查", "status": "completed"},
                    {"content": "重构代码", "status": "completed"},
                    {"content": "编写测试", "status": "completed"}
                ]
            },
            "id": "call_stu901"
        }
    ]
)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【工具执行】→ 更新 todos，所有任务完成

═══════════════════════════════════════════════════════════════════════════
迭代 5：所有任务完成，准备退出
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：model                                                              │
└─────────────────────────────────────────────────────────────────────────┘

【LLM 看到的消息历史】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[
  ... (之前的消息) ...
  ToolMessage("Updated todo list to [
    {'content': '代码审查', 'status': 'completed'},
    {'content': '重构代码', 'status': 'completed'},
    {'content': '编写测试', 'status': 'completed'}  ← 所有任务完成！
  ]")
]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【LLM 推理过程】
1. 看到所有任务都是 "completed"
2. 没有 "in_progress" 或 "pending" 的任务
3. 决定：任务完成，返回最终结果

【LLM 返回的 AIMessage】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIMessage(
    content="所有任务已完成！我已经完成了代码审查、重构和测试编写。",
    tool_calls=[]  ← 没有工具调用！
)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【状态更新】
state["messages"].append(AIMessage(content="...", tool_calls=[]))

┌─────────────────────────────────────────────────────────────────────────┐
│ 路由判断：_make_model_to_tools_edge(state)                              │
└─────────────────────────────────────────────────────────────────────────┘

【判断逻辑】
1. 检查 jump_to → 无
2. 获取最后一条 AIMessage
   └─ last_ai_message.tool_calls = []  ← 空！
3. 检查 tool_calls 数量
   └─ len(tool_calls) = 0
4. 退出条件满足
   └─ 返回："end" ← 退出循环！

【最终状态】
{
  "messages": [
    HumanMessage("..."),
    ... (所有中间消息) ...,
    AIMessage(content="所有任务已完成！...", tool_calls=[])
  ],
  "todos": [
    {"content": "代码审查", "status": "completed"},
    {"content": "重构代码", "status": "completed"},
    {"content": "编写测试", "status": "completed"}
  ]
}

┌─────────────────────────────────────────────────────────────────────────┐
│ 节点：END                                                                │
└─────────────────────────────────────────────────────────────────────────┘

Agent 执行结束，返回最终结果。
```

## 路由函数判断条件总结

### `_make_model_to_tools_edge`（模型 → 工具）

```python
判断优先级：
1. 检查 state["jump_to"] → 有则按 jump_to 路由
2. 检查 tool_calls 数量 → 0 则返回 "end"
3. 检查 pending_tool_calls → 有则返回 ["tools"]
4. 检查 structured_response → 有则返回 "end"
5. 默认 → 返回 "model"
```

### `_make_tools_to_model_edge`（工具 → 模型）

```python
判断优先级：
1. 检查所有工具是否 return_direct=True → 是则返回 "end"
2. 检查是否执行了 structured_output_tool → 是则返回 "end"
3. 默认 → 返回 "model"
```

## 关键机制总结

1. 状态传递：通过 `ToolMessage` 将 `todos` 信息传递给 LLM
2. 系统提示词：指导 LLM 如何使用 `write_todos` 工具
3. 工具描述：详细说明任务状态管理和执行规则
4. 路由函数：根据状态动态决定下一步
5. 循环机制：通过条件边实现 "模型 → 工具 → 模型" 的循环

这就是完整的 Agent 执行流程。