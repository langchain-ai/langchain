
# factory 的实现思路 与 todo中间件的关联

结合 `factory.py` 的实现，解释图构建时如何注册路由函数，以及整体实现思路：

## factory.py 的实现思路

### 整体架构：基于 LangGraph 的状态图（StateGraph）

`factory.py` 的核心思路是构建一个有向状态图，节点表示执行步骤，边表示路由逻辑。

### 实现流程

#### 阶段 1：准备阶段（工具和中间件收集）

```python:700:728:langchain/agents/factory.py
# Setup tools
tool_node: ToolNode | None = None
# Extract built-in provider tools (dict format) and regular tools (BaseTool/callables)
built_in_tools = [t for t in tools if isinstance(t, dict)]
regular_tools = [t for t in tools if not isinstance(t, dict)]

# Tools that require client-side execution (must be in ToolNode)
available_tools = middleware_tools + regular_tools

# Only create ToolNode if we have client-side tools
tool_node = (
    ToolNode(
        tools=available_tools,
        wrap_tool_call=wrap_tool_call_wrapper,
        awrap_tool_call=awrap_tool_call_wrapper,
    )
    if available_tools
    else None
)

# Default tools for ModelRequest initialization
# Use converted BaseTool instances from ToolNode (not raw callables)
# Include built-ins and converted tools (can be changed dynamically by middleware)
# Structured tools are NOT included - they're added dynamically based on response_format
if tool_node:
    default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
else:
    default_tools = list(built_in_tools)
```

要点：
- 收集所有工具（用户工具 + 中间件工具）
- 创建 `ToolNode`（统一执行工具）
- 准备默认工具列表供模型使用

#### 阶段 2：创建状态图

```python:797:805:langchain/agents/factory.py
# create graph, add nodes
graph: StateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
] = StateGraph(
    state_schema=resolved_state_schema,
    input_schema=input_schema,
    output_schema=output_schema,
    context_schema=context_schema,
)
```

#### 阶段 3：添加核心节点

```python:1130:1135:langchain/agents/factory.py
# Use sync or async based on model capabilities
graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

# Only add tools node if we have tools
if tool_node is not None:
    graph.add_node("tools", tool_node)
```

- `"model"` 节点：调用 LLM
- `"tools"` 节点：执行工具（包含 `write_todos` 等）

#### 阶段 4：添加中间件节点

```python:1137:1219:langchain/agents/factory.py
# Add middleware nodes
for m in middleware:
    if (
        m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ):
        # Use RunnableCallable to support both sync and async
        # Pass None for sync if not overridden to avoid signature conflicts
        sync_before_agent = (
            m.before_agent
            if m.__class__.before_agent is not AgentMiddleware.before_agent
            else None
        )
        async_before_agent = (
            m.abefore_agent
            if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
            else None
        )
        before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
        graph.add_node(
            f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema
        )
    # ... 类似的逻辑处理 before_model, after_model, after_agent
```

为每个中间件的钩子创建独立节点（如 `"todo.before_model"`）。

#### 阶段 5：确定关键节点位置

```python:1221:1247:langchain/agents/factory.py
# Determine the entry node (runs once at start): before_agent -> before_model -> model
if middleware_w_before_agent:
    entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
elif middleware_w_before_model:
    entry_node = f"{middleware_w_before_model[0].name}.before_model"
else:
    entry_node = "model"

# Determine the loop entry node (beginning of agent loop, excludes before_agent)
# This is where tools will loop back to for the next iteration
if middleware_w_before_model:
    loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
else:
    loop_entry_node = "model"

# Determine the loop exit node (end of each iteration, can run multiple times)
# This is after_model or model, but NOT after_agent
if middleware_w_after_model:
    loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
else:
    loop_exit_node = "model"

# Determine the exit node (runs once at end): after_agent or END
if middleware_w_after_agent:
    exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
else:
    exit_node = END
```

关键概念：
- `entry_node`：图的入口（只执行一次）
- `loop_entry_node`：循环入口（工具执行后回到这里）
- `loop_exit_node`：循环出口（模型执行后从这里判断路由）
- `exit_node`：图的出口（只执行一次）

#### 阶段 6：注册路由函数（核心）

```python:1261:1290:langchain/agents/factory.py
graph.add_conditional_edges(
    "tools",
    _make_tools_to_model_edge(
        tool_node=tool_node,
        model_destination=loop_entry_node,
        structured_output_tools=structured_output_tools,
        end_destination=exit_node,
    ),
    tools_to_model_destinations,
)

# base destinations are tools and exit_node
# we add the loop_entry node to edge destinations if:
# - there is an after model hook(s) -- allows jump_to to model
#   potentially artificially injected tool messages, ex HITL
# - there is a response format -- to allow for jumping to model to handle
#   regenerating structured output tool calls
model_to_tools_destinations = ["tools", exit_node]
if response_format or loop_exit_node != "model":
    model_to_tools_destinations.append(loop_entry_node)

graph.add_conditional_edges(
    loop_exit_node,
    _make_model_to_tools_edge(
        model_destination=loop_entry_node,
        structured_output_tools=structured_output_tools,
        end_destination=exit_node,
    ),
    model_to_tools_destinations,
)
```

### 路由函数的工作原理

#### 1. `_make_model_to_tools_edge`：模型 → 工具

```python:1440:1493:langchain/agents/factory.py
def _make_model_to_tools_edge(
    *,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. if there's an explicit jump_to in the state, use it
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])
        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 2. if the model hasn't called any tools, exit the loop
        # this is the classic exit condition for an agent loop
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        # 3. if there are pending tool calls, jump to the tool node
        if pending_tool_calls:
            return [
                Send(
                    "tools",
                    ToolCallWithContext(
                        __type="tool_call_with_context",
                        tool_call=tool_call,
                        state=state,
                    ),
                )
                for tool_call in pending_tool_calls
            ]

        # 4. if there is a structured response, exit the loop
        if "structured_response" in state:
            return end_destination

        # 5. AIMessage has tool calls, but there are no pending tool calls
        # which suggests the injection of artificial tool messages. jump to the model node
        return model_destination

    return model_to_tools
```

执行逻辑：
1. 检查 `jump_to`（中间件可强制跳转）
2. 无工具调用 → 返回 `end_destination`（结束）
3. 有待执行工具调用 → 返回 `["tools"]`（路由到工具节点）
4. 有结构化响应 → 返回 `end_destination`（结束）
5. 其他情况 → 返回 `model_destination`（回到模型）

#### 2. `_make_tools_to_model_edge`：工具 → 模型

```python:1523:1552:langchain/agents/factory.py
def _make_tools_to_model_edge(
    *,
    tool_node: ToolNode,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 1. Exit condition: All executed tools have return_direct=True
        # Filter to only client-side tools (provider tools are not in tool_node)
        client_side_tool_calls = [
            c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name
        ]
        if client_side_tool_calls and all(
            tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
        ):
            return end_destination

        # 2. Exit condition: A structured output tool was executed
        if any(t.name in structured_output_tools for t in tool_messages):
            return end_destination

        # 3. Default: Continue the loop
        #    Tool execution completed successfully, route back to the model
        #    so it can process the tool results and decide the next action.
        return model_destination

    return tools_to_model
```

执行逻辑：
1. 所有工具 `return_direct=True` → 返回 `end_destination`（结束）
2. 执行了结构化输出工具 → 返回 `end_destination`（结束）
3. 默认 → 返回 `model_destination`（回到模型，继续循环）

### 注册机制图解

```
┌─────────────────────────────────────────────────────────────────┐
│ 图构建时的路由函数注册过程                                       │
└─────────────────────────────────────────────────────────────────┘

步骤1：创建路由函数（闭包）
   ↓
   _make_model_to_tools_edge(
       model_destination="model",      ← 循环入口节点
       end_destination="end",          ← 退出节点
       structured_output_tools={...}    ← 结构化输出工具
   )
   ↓
   返回：model_to_tools(state) 函数
   └─ 这个函数会"记住" model_destination 和 end_destination

步骤2：注册条件边
   ↓
   graph.add_conditional_edges(
       "model",                        ← 从哪个节点出发
       model_to_tools,                 ← 路由函数
       ["tools", "end", "model"]       ← 可能的目标节点
   )
   ↓
   LangGraph 内部记录：
   {
       "from": "model",
       "routing_function": model_to_tools,
       "possible_destinations": ["tools", "end", "model"]
   }

步骤3：运行时自动调用
   ↓
   当 "model" 节点执行完成后：
   1. LangGraph 检测到节点完成
   2. 查找从 "model" 出发的条件边
   3. 调用 model_to_tools(state)
   4. 根据返回值路由到下一个节点
```

### 与 todo.py 的关联

```
┌─────────────────────────────────────────────────────────────────┐
│ todo.py 如何通过路由函数驱动循环                                 │
└─────────────────────────────────────────────────────────────────┘

迭代1：创建 Todo List
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 模型节点执行                                               │
  │    model_node(state)                                         │
  │    ├─ TodoListMiddleware.wrap_model_call() 注入提示词       │
  │    └─ LLM 返回：AIMessage(tool_calls=[write_todos(...)])   │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. LangGraph 调用路由函数                                    │
  │    _make_model_to_tools_edge()(state)                       │
  │    ├─ 检测到 tool_calls = [write_todos(...)]               │
  │    ├─ pending_tool_calls = [write_todos(...)]             │
  │    └─ 返回：["tools"]  ← 路由到工具节点                     │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. 工具节点执行                                               │
  │    ToolNode 执行 write_todos                                 │
  │    └─ 返回：Command(update={"todos": [...]})               │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. LangGraph 调用路由函数                                    │
  │    _make_tools_to_model_edge()(state)                      │
  │    ├─ write_todos 没有 return_direct=True                  │
  │    ├─ write_todos 不是 structured_output_tool               │
  │    └─ 返回："model"  ← 回到模型节点！                       │
  └─────────────────────────────────────────────────────────────┘
    ↓
  循环继续...

迭代2：执行第一个任务
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 模型节点再次执行                                           │
  │    model_node(state)                                         │
  │    ├─ state["todos"] = [                                    │
  │    │     {content: "任务1", status: "in_progress"},        │
  │    │     {content: "任务2", status: "pending"}             │
  │    │   ]                                                    │
  │    └─ LLM 看到 todos，决定执行任务1                         │
  └─────────────────────────────────────────────────────────────┘
    ↓
  ...（继续循环）
```

### 核心设计思想

1. 声明式路由：路由函数返回目标节点名称，LangGraph 负责执行路由
2. 状态驱动：路由决策基于 `state`（如 `messages`、`todos`）
3. 闭包捕获：路由函数通过闭包捕获 `model_destination`、`end_destination` 等配置
4. 条件边：使用 `add_conditional_edges` 注册动态路由，而非固定边

### 总结

`factory.py` 的实现思路：
1. 收集工具和中间件
2. 创建状态图并添加节点
3. 注册条件边和路由函数
4. 运行时由 LangGraph 调用路由函数，根据状态动态路由

路由函数的作用：
- `_make_model_to_tools_edge`：决定模型完成后是去工具节点还是结束
- `_make_tools_to_model_edge`：决定工具完成后是回到模型节点还是结束

与 todo.py 的关联：
- todo.py 通过 `write_todos` 更新 `state["todos"]`
- 路由函数根据状态决定下一步
- 形成“模型 → 工具 → 模型”的循环，直到所有任务完成

这就是整个驱动机制的核心：LangGraph 的条件边 + 路由函数 + 状态更新 = 自驱动的 Agent 循环。