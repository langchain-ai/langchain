
# LangChain Agent 完整架构深度解析

## 目录

1. [概述与核心概念](#一概述与核心概念)
2. [create_agent 函数详解](#二create_agent-函数详解)
3. [Agent 状态管理](#三agent-状态管理)
4. [Graph 节点与边](#四graph-节点与边)
5. [完整执行流程](#五完整执行流程)
6. [中间件系统](#七中间件系统)
7. [结构化输出](#八结构化输出)
8. [特殊工具配置](#九特殊工具配置)
9. [最终答案返回机制](#十一最终答案返回机制)
10. [LLM 数据交互详解](#十二llm-数据交互详解)
11. [实战示例](#十一实战示例)

---

## 一、概述与核心概念

### 1.1 什么是 LangChain Agent

LangChain Agent 是一个**可编程的 AI 代理系统**，它将 LLM（大语言模型）与工具调用能力结合，形成一个可以自主决策、执行任务并返回结果的智能循环系统。

### 1.2 核心设计理念

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Agent 的核心循环                                   │
│                                                                             │
│   用户输入 → Model 思考 → 需要工具？ ─→ 是 → 调用工具 → 获取结果 → 回到思考  │
│                              │                                              │
│                              └─→ 否 → 直接回答 → 结束                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 核心组件

| 组件 | 作用 | 源码位置 |
|------|------|----------|
| **Model 节点** | 调用 LLM 生成响应 | `factory.py:1114-1139` |
| **Tools 节点** | 执行工具调用 | `ToolNode` 类 |
| **Edge 函数** | 决定下一步走向 | `factory.py:1513-1625` |
| **Middleware** | 拦截和修改行为 | `types.py:330-500` |
| **State** | 管理对话状态 | `types.py:304-327` |

---

## 二、create_agent 函数详解

### 2.1 函数签名

```python
# factory.py:541-559
def create_agent(
    model: str | BaseChatModel,                                    # LLM 模型
    tools: Sequence[BaseTool | Callable | dict] | None = None,     # 工具列表
    *,
    system_prompt: str | SystemMessage | None = None,              # 系统提示词
    middleware: Sequence[AgentMiddleware] = (),                    # 中间件
    response_format: ResponseFormat | type | None = None,          # 结构化输出格式
    state_schema: type[AgentState] | None = None,                  # 状态模式
    context_schema: type | None = None,                            # 上下文模式
    checkpointer: Checkpointer | None = None,                      # 状态持久化
    store: BaseStore | None = None,                                # 数据存储
    interrupt_before: list[str] | None = None,                     # 中断点（前）
    interrupt_after: list[str] | None = None,                      # 中断点（后）
    debug: bool = False,                                           # 调试模式
    name: str | None = None,                                       # 图名称
    cache: BaseCache | None = None,                                # 缓存
) -> CompiledStateGraph
```

### 2.2 参数详解

#### model 参数
```python
# 方式1：字符串标识符
agent = create_agent("openai:gpt-4")
agent = create_agent("anthropic:claude-sonnet-4-5-20250929")

# 方式2：直接传入模型实例
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")
agent = create_agent(model)
```

#### tools 参数
```python
# 方式1：函数（自动转换为工具）
def check_weather(location: str) -> str:
    """查询天气"""
    return f"{location} 的天气是晴天"

# 方式2：BaseTool 实例
from langchain_core.tools import Tool
weather_tool = Tool(
    name="weather",
    description="查询天气",
    func=check_weather,
    return_direct=False  # 是否直接返回结果
)

# 方式3：字典格式（内置工具）
built_in_tool = {"type": "web_search"}

agent = create_agent("gpt-4", tools=[check_weather, weather_tool, built_in_tool])
```

#### system_prompt 参数
```python
# 字符串形式
agent = create_agent("gpt-4", system_prompt="你是一个有帮助的助手")

# SystemMessage 形式
from langchain_core.messages import SystemMessage
system_msg = SystemMessage(content="你是一个有帮助的助手")
agent = create_agent("gpt-4", system_prompt=system_msg)
```

---

## 三、Agent 状态管理

### 3.1 状态模式定义

```python
# types.py:304-323
class AgentState(TypedDict, Generic[ResponseT]):
    """Agent 的状态模式"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    # 消息历史列表，使用 add_messages 进行增量更新

    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    # 跳转指令：可选值为 "tools", "model", "end"

    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
    # 结构化响应（可选）

class _InputAgentState(TypedDict):
    """输入状态"""
    messages: Required[Annotated[list[AnyMessage | dict], add_messages]]

class _OutputAgentState(TypedDict, Generic[ResponseT]):
    """输出状态"""
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]
```

### 3.2 状态流转

```
初始状态                    执行中状态                    最终状态
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ messages: [     │        │ messages: [     │        │ messages: [     │
│   HumanMsg      │   →    │   HumanMsg,     │   →    │   HumanMsg,     │
│ ]               │        │   AIMsg,        │        │   AIMsg,        │
│                 │        │   ToolMsg       │        │   ToolMsg,      │
│                 │        │ ]               │        │   AIMsg(答案)   │
│                 │        │ jump_to: null   │        │ ]               │
│                 │        │                 │        │ structured_     │
│                 │        │                 │        │ response: {...} │
└─────────────────┘        └─────────────────┘        └─────────────────┘
```

---

## 四、Graph 节点与边

### 4.1 节点类型

#### Model 节点
```python
# factory.py:1114-1139
def model_node(state: AgentState, runtime: Runtime) -> dict:
    """模型节点：调用 LLM"""
    request = ModelRequest(
        model=model,
        tools=default_tools,
        system_message=system_message,
        messages=state["messages"],
    )

    response = _execute_model_sync(request)
    # 或通过中间件: wrap_model_call_handler(request, _execute_model_sync)

    return {"messages": response.result}
```

#### Tools 节点
```python
# 由 ToolNode 类处理
tool_node = ToolNode(
    tools=available_tools,
    wrap_tool_call=wrap_tool_call_wrapper,
    awrap_tool_call=awrap_tool_call_wrapper,
)
```

### 4.2 边函数

#### model_to_tools_edge（模型到工具的边）
```python
# factory.py:1513-1566
def model_to_tools(state: dict) -> str | list[Send] | None:
    """决定模型调用后的下一步"""

    # 1. 检查是否有显式跳转指令
    if jump_to := state.get("jump_to"):
        return _resolve_jump(jump_to, ...)

    last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

    # 2. 🔑 经典退出条件：模型没有调用任何工具
    if len(last_ai_message.tool_calls) == 0:
        return end_destination  # 跳转到 END

    # 3. 检查是否有待执行的工具调用
    pending_tool_calls = [
        c for c in last_ai_message.tool_calls
        if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
    ]

    if pending_tool_calls:
        return [Send("tools", ToolCallWithContext(...)) for tc in pending_tool_calls]

    # 4. 检查是否有结构化响应
    if "structured_response" in state:
        return end_destination

    # 5. 默认：回到模型节点
    return model_destination
```

#### tools_to_model_edge（工具到模型的边）
```python
# factory.py:1596-1625
def tools_to_model(state: dict) -> str | None:
    """决定工具执行后的下一步"""

    last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

    # 1. 🔴 特殊条件：所有工具都设置了 return_direct=True
    client_side_tool_calls = [
        c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name
    ]
    if client_side_tool_calls and all(
        tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
    ):
        return end_destination  # 直接结束

    # 2. 🔴 特殊条件：执行了结构化输出工具
    if any(t.name in structured_output_tools for t in tool_messages):
        return end_destination  # 直接结束

    # 3. 默认：返回模型节点，让 LLM 处理工具结果
    return model_destination
```

### 4.3 完整 Graph 结构

```
                                ┌─────────────────┐
                                │      START      │
                                └────────┬────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            ┌───────────────┐                        ┌─────────────────┐
            │ before_agent  │                        │ (无中间件时跳过) │
            │   中间件      │                        └────────┬────────┘
            └───────┬───────┘                                 │
                    │                                         │
                    ▼                                         │
            ┌───────────────┐                                 │
            │ before_model  │◀────────────────────────────────┘
            │   中间件      │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │     MODEL     │ ←──────────────────────┐
            │   (LLM调用)   │                        │
            └───────┬───────┘                        │
                    │                                │
                    ▼                                │
            ┌───────────────┐                        │
            │ after_model   │                        │
            │   中间件      │                        │
            └───────┬───────┘                        │
                    │                                │
                    ▼                                │
        ┌───────────────────────┐                    │
        │ model_to_tools_edge   │                    │
        │                       │                    │
        │ tool_calls == 0?      │                    │
        │  └── YES → END        │                    │
        │  └── NO  → TOOLS      │                    │
        └───────────┬───────────┘                    │
                    │                                │
                    ▼                                │
            ┌───────────────┐                        │
            │     TOOLS     │                        │
            │  (工具执行)   │                        │
            └───────┬───────┘                        │
                    │                                │
                    ▼                                │
        ┌───────────────────────┐                    │
        │ tools_to_model_edge   │                    │
        │                       │                    │
        │ return_direct=True?   │                    │
        │  └── YES → END        │                    │
        │  └── NO  → MODEL ─────┼────────────────────┘
        └───────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ after_agent   │
            │   中间件      │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │      END      │
            └───────────────┘
```

---

## 五、完整执行流程

### 5.1 标准流程示例

#### 场景：用户询问天气

```
用户输入: "北京今天天气怎么样？"
```

#### 执行步骤：

```
步骤 1: START → Model
─────────────────────────────────────────────
输入 messages:
  [HumanMessage("北京今天天气怎么样？")]

LLM 收到:
  System: "你是一个有帮助的助手"
  Human: "北京今天天气怎么样？"
  Tools: [weather_api]

LLM 输出:
  AIMessage(
    content="我来帮你查询天气",
    tool_calls=[{"name": "weather_api", "args": {"city": "北京"}}]
  )

状态更新:
  messages += [AIMessage(...)]
```

```
步骤 2: Model → model_to_tools_edge
─────────────────────────────────────────────
检查: last_ai_message.tool_calls
结果: len(tool_calls) > 0 → 有工具调用

决策: 跳转到 TOOLS
```

```
步骤 3: TOOLS 执行
─────────────────────────────────────────────
执行: weather_api(city="北京")
返回: "北京: 多云, 22°C, 无雨"

状态更新:
  messages += [ToolMessage("北京: 多云, 22°C, 无雨")]
```

```
步骤 4: TOOLS → tools_to_model_edge
─────────────────────────────────────────────
检查:
  - weather_api.return_direct = False → 不直接返回
  - 不是结构化输出工具

决策: 返回 MODEL
```

```
步骤 5: Model（第二次）
─────────────────────────────────────────────
输入 messages:
  [HumanMessage("北京今天天气怎么样？"),
   AIMessage(..., tool_calls=[...]),
   ToolMessage("北京: 多云, 22°C, 无雨")]

LLM 输出:
  AIMessage(
    content="北京今天多云，温度22°C，不需要带雨伞",
    tool_calls=[]  # 🔑 没有工具调用了
  )

状态更新:
  messages += [AIMessage("北京今天多云...")]
```

```
步骤 6: Model → model_to_tools_edge
─────────────────────────────────────────────
检查: len(last_ai_message.tool_calls) == 0

决策: 跳转到 END
```

```
步骤 7: END → 返回结果
─────────────────────────────────────────────
最终返回:
{
  "messages": [
    HumanMessage("北京今天天气怎么样？"),
    AIMessage("我来帮你查询天气", tool_calls=[...]),
    ToolMessage("北京: 多云, 22°C, 无雨"),
    AIMessage("北京今天多云，温度22°C，不需要带雨伞")  ← 最终答案
  ]
}
```

### 5.2 LLM 数据交互详解

#### 核心发现：LLM 不会收到 AgentState 的全部状态！

基于源码分析，每次 Model 节点调用时，**LLM 只接收精心筛选的数据**，而不是完整的 AgentState。

#### LLM 实际接收的数据结构

```python
# 传递给 LLM 的核心数据（来自 factory.py:1149-1155）
{
  "messages": [  // 🔑 只有消息历史传递给 LLM
    SystemMessage("你是一个有帮助的助手"),
    HumanMessage("用户问题"),
    AIMessage("我之前的响应", tool_calls=[...]),
    ToolMessage("工具执行结果"),
    // ... 完整对话历史
  ],

  "tools": [  // 🔑 可用工具列表
    {
      "name": "weather_api",
      "description": "查询天气信息",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}}
      }
    }
  ],

  "tool_choice": "auto",  // 工具选择策略
  "response_format": None  // 响应格式要求（可选）
}
```

#### 状态字段处理规则

| 状态字段 | 传递给LLM？ | 注解说明 | 源码位置 |
|---------|------------|----------|----------|
| **`messages`** | ✅ **是** | `add_messages` | `factory.py:1121` |
| **`todos`** | ❌ **否** | `OmitFromInput` | `types.py:40` |
| **`jump_to`** | ❌ **否** | `EphemeralValue + PrivateStateAttr` | `types.py:308` |
| **`structured_response`** | ❌ **否** | `OmitFromInput` | `types.py:309` |

#### 源码验证

**消息构造逻辑**：
```python
# factory.py:1114-1139
def model_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
    request = ModelRequest(
        model=model,
        tools=default_tools,
        system_message=system_message,
        response_format=initial_response_format,
        messages=state["messages"],  # 🔑 只传递 messages
        tool_choice=None,
        state=state,  # 完整state给中间件，但不传递给LLM
        runtime=runtime,
    )

    # ... 中间件处理 ...

    # factory.py:1149-1155
    def _execute_model_async(request: ModelRequest):
        messages = request.messages  # 只使用 messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = await model_.ainvoke(messages)  # 🔑 只发送 messages 给 LLM
```

#### 实际示例：Todo 中间件场景

**AgentState 包含的数据**：
```python
state = {
  "messages": [
    HumanMessage("重构代码库"),
    AIMessage("我来规划任务", tool_calls=[write_todos_call]),
    ToolMessage("Updated todo list..."),
    AIMessage("开始执行第一个任务")
  ],

  "todos": [  // 🔴 LLM 完全看不到！
    {"content": "分析代码", "status": "completed"},
    {"content": "重构函数", "status": "in_progress"},
    {"content": "测试修改", "status": "pending"}
  ],

  "jump_to": None,  // 🔴 LLM 看不到！
  "structured_response": None  // 🔴 LLM 看不到！
}
```

**LLM 实际接收的数据**：
```python
{
  "messages": [
    SystemMessage("你是一个有帮助的助手\n## write_todos\n你有权访问write_todos工具..."),
    HumanMessage("重构代码库"),
    AIMessage("我来规划任务", tool_calls=[write_todos_call]),
    ToolMessage("Updated todo list..."),
    AIMessage("开始执行第一个任务")
  ],

  "tools": [
    {
      "name": "write_todos",
      "description": "创建任务列表...",
      "parameters": {"type": "object", "properties": {...}}
    }
  ]
}
```

#### 设计哲学

1. **信息隔离**：LLM 只负责对话和工具决策，状态管理由框架处理
2. **Token 效率**：避免发送不必要的状态数据
3. **关注分离**：LLM 专注于对话上下文，不需要了解内部状态
4. **扩展性**：中间件可以添加任意状态字段，而不影响 LLM

**结论：即使 AgentState 有很多状态字段，LLM 每次只看到对话历史 + 工具列表！**

### 5.4 关键问题解答

#### Q1: LLM 如何知道需要基于工具结果回答？

**答案：LLM 不需要特殊提示词！**

```python
# factory.py:1151-1155
messages = request.messages  # 包含完整历史
if request.system_message:
    messages = [request.system_message, *messages]
output = await model_.ainvoke(messages)
```

LLM 收到的是**完整的消息历史**：
```
System: "你是一个助手"
Human: "北京天气？"
Assistant: "我来查询" [tool_call]
Tool: "多云, 22°C"
```

LLM 被训练成理解这种对话模式，会自然地基于工具结果生成最终答案。

#### Q2: 为什么最后一个 AIMessage 是最终答案？

**答案：因为退出条件是 `tool_calls == 0`**

```python
# factory.py:1533-1536
# 经典退出条件：模型没有调用任何工具
if len(last_ai_message.tool_calls) == 0:
    return end_destination
```

只有当 LLM 决定**不再调用任何工具**时，才会跳转到 END，此时的 AIMessage 就是最终答案。

---

## 六、中间件系统

### 6.1 中间件生命周期钩子

```python
# types.py:330-450
class AgentMiddleware:
    """中间件基类"""

    state_schema = AgentState       # 状态模式扩展
    tools: list[BaseTool]           # 注册的工具

    # 生命周期钩子
    def before_agent(self, state, runtime) -> dict | None:
        """Agent 开始前（只执行一次）"""

    def before_model(self, state, runtime) -> dict | None:
        """模型调用前（每次循环都执行）"""

    def after_model(self, state, runtime) -> dict | None:
        """模型调用后（每次循环都执行）"""

    def after_agent(self, state, runtime) -> dict | None:
        """Agent 结束后（只执行一次）"""

    # 包装器钩子
    def wrap_model_call(self, request, handler) -> ModelResponse:
        """包装模型调用，可以修改请求/响应"""

    def wrap_tool_call(self, request, handler) -> ToolMessage:
        """包装工具调用，可以修改请求/响应"""
```

### 6.2 中间件执行顺序

```
before_agent → before_model → MODEL → after_model → TOOLS → before_model → ...
     ↑                                                              │
     └──────────────────────────────────────────────────────────────┘
                              循环执行

最终: ... → after_model → after_agent → END
```

### 6.3 TodoListMiddleware 示例

```python
# todo.py:130-225
class TodoListMiddleware(AgentMiddleware):
    """任务规划中间件"""

    state_schema = PlanningState  # 扩展状态，添加 todos 字段

    def __init__(self):
        # 注册 write_todos 工具
        @tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
        def write_todos(todos: list[Todo], tool_call_id) -> Command:
            return Command(update={
                "todos": todos,
                "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id)]
            })

        self.tools = [write_todos]

    def wrap_model_call(self, request, handler):
        """修改系统提示词，指导 LLM 使用 todo 工具"""
        new_system_message = SystemMessage(content=[
            *request.system_message.content_blocks,
            {"type": "text", "text": self.system_prompt}
        ])
        return handler(request.override(system_message=new_system_message))
```

### 6.4 使用中间件

```python
from langchain.agents import create_agent
from langchain.agents.middleware.todo import TodoListMiddleware

agent = create_agent(
    "gpt-4",
    tools=[my_tools],
    middleware=[TodoListMiddleware()]
)

result = await agent.invoke({"messages": [HumanMessage("重构代码库")]})

# 结果包含任务进度
print(result["todos"])
# [{"content": "分析代码", "status": "completed"}, ...]
```

---

## 七、结构化输出

### 7.1 输出策略

```python
# structured_output.py

# 策略1: ToolStrategy（工具调用方式）
from langchain.agents.structured_output import ToolStrategy
response_format = ToolStrategy(schema=MySchema)

# 策略2: ProviderStrategy（提供商原生方式）
from langchain.agents.structured_output import ProviderStrategy
response_format = ProviderStrategy(schema=MySchema)

# 策略3: AutoStrategy（自动选择）
from langchain.agents.structured_output import AutoStrategy
response_format = AutoStrategy(schema=MySchema)

# 策略4: 直接传入 Pydantic 模型
from pydantic import BaseModel
class WeatherResponse(BaseModel):
    temperature: int
    condition: str
```

### 7.2 结构化输出流程

```
Model 调用
    │
    ▼
生成 tool_calls（结构化输出工具）
    │
    ▼
tools_to_model_edge 检查
    │
    └── t.name in structured_output_tools? → YES → END
    │
    ▼
返回 structured_response
```

### 7.3 使用示例

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

agent = create_agent(
    "gpt-4",
    tools=[],
    response_format=PersonInfo
)

result = await agent.invoke({
    "messages": [HumanMessage("张三是35岁的工程师")]
})

print(result["structured_response"])
# PersonInfo(name="张三", age=35, occupation="工程师")
```

---

## 八、特殊工具配置

### 8.1 return_direct=True

```python
# 工具执行后直接返回结果，不经过 LLM 处理
calculator = Tool(
    name="calculate",
    description="计算数学表达式",
    func=eval_expression,
    return_direct=True  # 🔑 直接返回
)
```

**执行流程**：
```
Model → tool_calls → TOOLS → tools_to_model_edge
                              │
                              └── return_direct=True → END（直接返回工具结果）
```

**适用场景**：
- 计算器工具：2+2=4 无需 LLM 重新表述
- 精确查询：数据库查询结果直接返回
- API 调用：结果已经是用户需要的格式

### 8.2 return_direct=False（默认）

```python
# 工具执行后返回 Model，LLM 处理结果
search_tool = Tool(
    name="search",
    description="搜索信息",
    func=search_web,
    return_direct=False  # 默认值
)
```

**执行流程**：
```
Model → tool_calls → TOOLS → tools_to_model_edge
                              │
                              └── return_direct=False → MODEL（LLM 处理结果）
```

**适用场景**：
- 搜索工具：需要 LLM 总结多个结果
- 数据分析：需要 LLM 解释分析结果
- 复杂任务：需要 LLM 决定下一步

---

## 九、最终答案返回机制

### 9.1 答案提取函数

```python
# factory.py:1497-1510
def _fetch_last_ai_and_tool_messages(messages: list[AnyMessage]):
    """获取最后一个 AIMessage 和其后的 ToolMessages"""

    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_index = i
            last_ai_message = cast(AIMessage, messages[i])
            break

    tool_messages = [m for m in messages[last_ai_index + 1:] if isinstance(m, ToolMessage)]
    return last_ai_message, tool_messages
```

### 9.2 最终答案的位置

```python
# 返回的状态结构
result = {
    "messages": [
        HumanMessage("用户问题"),
        AIMessage("调用工具", tool_calls=[...]),
        ToolMessage("工具结果"),
        AIMessage("最终答案")  # ← 这是最终答案
    ],
    "structured_response": {...}  # 可选的结构化响应
}
```

### 9.3 提取最终答案的方法

```python
def get_final_answer(result):
    """从 Agent 结果中提取最终答案"""

    # 方法1: 优先使用结构化响应
    if "structured_response" in result and result["structured_response"]:
        return result["structured_response"]

    # 方法2: 从消息历史中获取最后一个 AIMessage
    from langchain_core.messages import AIMessage
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content

    return None
```

### 9.4 边界情况分析

| 情况 | 最终答案来源 | 说明 |
|------|------------|------|
| **正常流程** | 最后一个 `AIMessage.content` | LLM 基于工具结果生成的答案 |
| **结构化输出** | `structured_response` | 符合预定义 schema 的数据 |
| **return_direct** | 最后一个 `ToolMessage.content` | 工具直接返回的结果 |
| **无工具调用** | 第一个 `AIMessage.content` | LLM 直接回答，未调用工具 |

---

## 十一、实战示例

### 11.1 基础 Agent

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 定义工具
def search_web(query: str) -> str:
    """搜索网页"""
    return f"关于 {query} 的搜索结果..."

def calculate(expression: str) -> str:
    """计算表达式"""
    return str(eval(expression))

# 创建 Agent
agent = create_agent(
    model="openai:gpt-4",
    tools=[search_web, calculate],
    system_prompt="你是一个有帮助的助手"
)

# 调用 Agent
result = await agent.invoke({
    "messages": [HumanMessage("帮我计算 123 * 456")]
})

print(result["messages"][-1].content)
# 输出: "123 * 456 = 56088"
```

### 11.2 带中间件的 Agent

```python
from langchain.agents import create_agent
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4",
    tools=[code_analysis_tool, refactoring_tool],
    middleware=[
        TodoListMiddleware(),           # 任务规划
        ToolCallLimitMiddleware(max=10) # 限制工具调用次数
    ],
    system_prompt="你是一个代码重构助手"
)

result = await agent.invoke({
    "messages": [HumanMessage("帮我重构这个代码库")]
})

# 查看任务进度
for todo in result["todos"]:
    print(f"[{todo['status']}] {todo['content']}")
```

### 11.3 结构化输出 Agent

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class AnalysisResult(BaseModel):
    summary: str
    key_points: list[str]
    sentiment: str

agent = create_agent(
    model="openai:gpt-4",
    tools=[],
    response_format=AnalysisResult,
    system_prompt="分析给定的文本"
)

result = await agent.invoke({
    "messages": [HumanMessage("分析这篇文章：...")]
})

analysis = result["structured_response"]
print(f"摘要: {analysis.summary}")
print(f"关键点: {analysis.key_points}")
print(f"情感: {analysis.sentiment}")
```

### 11.4 流式输出 Agent

```python
agent = create_agent(
    model="openai:gpt-4",
    tools=[search_tool],
    system_prompt="你是一个助手"
)

inputs = {"messages": [HumanMessage("搜索最新的 Python 新闻")]}

# 流式输出
for chunk in agent.stream(inputs, stream_mode="updates"):
    print(chunk)
```

### 11.5 带状态持久化的 Agent

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点保存器
checkpointer = MemorySaver()

agent = create_agent(
    model="openai:gpt-4",
    tools=[my_tools],
    checkpointer=checkpointer,  # 启用状态持久化
    system_prompt="你是一个助手"
)

# 第一次对话
config = {"configurable": {"thread_id": "user_123"}}
result1 = await agent.invoke(
    {"messages": [HumanMessage("我叫张三")]},
    config=config
)

# 第二次对话（记住上下文）
result2 = await agent.invoke(
    {"messages": [HumanMessage("我叫什么名字？")]},
    config=config
)
# 输出: "你叫张三"
```

---

## 附录：源码文件索引

| 文件 | 作用 |
|------|------|
| `factory.py` | Agent 创建和 Graph 构建 |
| `middleware/types.py` | 中间件类型定义和基类 |
| `middleware/todo.py` | TodoListMiddleware 实现 |
| `middleware/tool_call_limit.py` | 工具调用限制中间件 |
| `structured_output.py` | 结构化输出策略 |

---

