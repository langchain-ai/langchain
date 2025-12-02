

# LangChain v1 记忆处理机制完全解析

## 一、什么是"记忆"？

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          LangChain v1 中的"记忆"定义                                  ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   在 LangChain v1 中，"记忆"不是一个独立的模块，而是【状态持久化】的自然结果。          ║
║                                                                                      ║
║   记忆 = 对话历史 = state["messages"] = 被持久化的消息列表                            ║
║                                                                                      ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │                                                                            │     ║
║   │   state = {                                                                │     ║
║   │       "messages": [                        ← 这就是"记忆"！                │     ║
║   │           HumanMessage("你好，我是张三"),                                   │     ║
║   │           AIMessage("你好张三！很高兴认识你"),                               │     ║
║   │           HumanMessage("我的名字是什么？"),                                 │     ║
║   │           AIMessage("你的名字是张三"),                                      │     ║
║   │       ],                                                                   │     ║
║   │       "todos": [...],                      ← 中间件扩展的状态也会被记住     │     ║
║   │       "structured_response": ...                                          │     ║
║   │   }                                                                        │     ║
║   │                                                                            │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
║   核心理解：                                                                          ║
║   1. 记忆 = 状态中的 messages 字段                                                   ║
║   2. 记忆通过 checkpointer 持久化                                                    ║
║   3. 记忆通过 thread_id 区分不同对话                                                 ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 二、记忆在哪里？

### 2.1 记忆的存储位置

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                              记忆存储架构                                             ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   ┌──────────────────────────────────────────────────────────────────────────────┐   ║
║   │                           LangGraph 状态图                                   │   ║
║   │                                                                              │   ║
║   │   state["messages"]  ◄────────┐                                              │   ║
║   │         ▲                     │                                              │   ║
║   │         │                     │                                              │   ║
║   │         │  每次节点执行后      │  每次 invoke 开始时                           │   ║
║   │         │  自动保存            │  自动加载                                     │   ║
║   │         │                     │                                              │   ║
║   │         ▼                     │                                              │   ║
║   │   ┌─────────────────────────────────────────────────────────────────────┐    │   ║
║   │   │                      Checkpointer                                   │    │   ║
║   │   │                                                                     │    │   ║
║   │   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │    │   ║
║   │   │   │ MemorySaver │  │ SqliteSaver │  │PostgresSaver│  ...           │    │   ║
║   │   │   │   (内存)    │  │  (SQLite)   │  │(PostgreSQL) │                │    │   ║
║   │   │   └─────────────┘  └─────────────┘  └─────────────┘                │    │   ║
║   │   │                                                                     │    │   ║
║   │   │   存储结构（按 thread_id 分隔）：                                    │    │   ║
║   │   │   {                                                                 │    │   ║
║   │   │       "conversation-1": {checkpoint_data...},                       │    │   ║
║   │   │       "conversation-2": {checkpoint_data...},                       │    │   ║
║   │   │       "weather-thread": {checkpoint_data...},                       │    │   ║
║   │   │   }                                                                 │    │   ║
║   │   │                                                                     │    │   ║
║   │   └─────────────────────────────────────────────────────────────────────┘    │   ║
║   │                                                                              │   ║
║   └──────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 2.2 AgentState 的定义

从 `types.py` 可以看到记忆存储的核心数据结构：

```168:172:langchain/agents/middleware/types.py
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
```

**关键点：`add_messages` 是一个 reducer 函数**

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│   add_messages 的作用：                                                              │
│                                                                                     │
│   当节点返回 {"messages": [新消息]} 时：                                              │
│   - 不是【替换】原有消息                                                             │
│   - 而是【追加】到现有消息列表                                                        │
│                                                                                     │
│   示例：                                                                             │
│   原始状态: messages = [HumanMessage("A"), AIMessage("B")]                           │
│   节点返回: {"messages": [AIMessage("C")]}                                           │
│   新状态:   messages = [HumanMessage("A"), AIMessage("B"), AIMessage("C")]           │
│                                    └─────────────────────────────────────┘           │
│                                               被追加                                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、记忆什么时候传入？

### 3.1 完整的记忆加载和保存流程

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          记忆的加载和保存时机                                         ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   用户调用:                                                                          ║
║   agent.invoke(                                                                      ║
║       {"messages": [HumanMessage("新消息")]},                                        ║
║       config={"configurable": {"thread_id": "my-thread"}}                            ║
║   )                                                                                  ║
║                                                                                      ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║   ║ 阶段 1: 记忆加载 (自动)                                                     ║   ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║   │                                                                                  ║
║   │  1. LangGraph 检测到 thread_id = "my-thread"                                     ║
║   │                                                                                  ║
║   │  2. 从 checkpointer 加载该 thread 的最新状态                                     ║
║   │     checkpoint = checkpointer.get({"thread_id": "my-thread"})                    ║
║   │                                                                                  ║
║   │  3. 恢复状态                                                                     ║
║   │     state = {                                                                    ║
║   │         "messages": [                                                            ║
║   │             HumanMessage("之前的消息1"),   ← 从 checkpoint 恢复                   ║
║   │             AIMessage("之前的回复1"),      ← 从 checkpoint 恢复                   ║
║   │         ]                                                                        ║
║   │     }                                                                            ║
║   │                                                                                  ║
║   │  4. 合并用户输入（使用 add_messages reducer）                                    ║
║   │     state["messages"] = [                                                        ║
║   │         HumanMessage("之前的消息1"),                                             ║
║   │         AIMessage("之前的回复1"),                                                ║
║   │         HumanMessage("新消息"),           ← 新追加的消息                          ║
║   │     ]                                                                            ║
║   │                                                                                  ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║                                                                                      ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║   ║ 阶段 2: 图执行                                                              ║   ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║   │                                                                                  ║
║   │  model 节点:                                                                     ║
║   │  - 读取 state["messages"]（包含完整历史）                                        ║
║   │  - 发送给 AI API                                                                 ║
║   │  - 返回 {"messages": [AIMessage(...)]}                                           ║
║   │                                                                                  ║
║   │  add_messages 自动追加:                                                          ║
║   │  state["messages"] = [...历史..., AIMessage("新回复")]                           ║
║   │                                                                                  ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║                                                                                      ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║   ║ 阶段 3: 记忆保存 (自动)                                                     ║   ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║   │                                                                                  ║
║   │  每个节点执行完毕后，checkpointer 自动保存：                                      ║
║   │  checkpointer.put(                                                               ║
║   │      {"thread_id": "my-thread"},                                                 ║
║   │      {                                                                           ║
║   │          "messages": [...完整消息历史...],                                       ║
║   │          "todos": [...],  # 如果有                                               ║
║   │          ...其他状态...                                                          ║
║   │      }                                                                           ║
║   │  )                                                                               ║
║   │                                                                                  ║
║   ════════════════════════════════════════════════════════════════════════════════   ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

### 3.2 代码层面的记忆传递

从 `factory.py` 中可以看到：

```1050:1061:langchain/agents/factory.py
    def model_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Sync model request handler with sequential middleware processing."""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_prompt=system_prompt,
            response_format=initial_response_format,
            messages=state["messages"],  # ← 记忆从这里传入！
            tool_choice=None,
            state=state,
            runtime=runtime,
        )
```

**state["messages"] 包含了所有历史消息，这就是"记忆"传入模型的方式！**

---

## 四、用户可用的接口

### 4.1 create_agent 的记忆相关参数

```516:576:langchain/agents/factory.py
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    ...
        checkpointer: An optional checkpoint saver object.

            Used for persisting the state of the graph (e.g., as chat memory) for a
            single thread (e.g., a single conversation).
        store: An optional store object.

            Used for persisting data across multiple threads (e.g., multiple
            conversations / users).
```

### 4.2 完整的用户接口总结

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          用户可用的记忆相关接口                                       ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   1. create_agent 参数                                                               ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │                                                                            │     ║
║   │   from langchain.agents import create_agent                                │     ║
║   │   from langgraph.checkpoint.memory import MemorySaver                      │     ║
║   │                                                                            │     ║
║   │   agent = create_agent(                                                    │     ║
║   │       model="openai:gpt-4o",                                               │     ║
║   │       tools=[...],                                                         │     ║
║   │       checkpointer=MemorySaver(),  # ← 启用记忆！                          │     ║
║   │       store=...,                   # ← 跨线程共享存储（可选）              │     ║
║   │   )                                                                        │     ║
║   │                                                                            │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
║   2. invoke/stream 的 config 参数                                                    ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │                                                                            │     ║
║   │   # 指定对话线程 ID                                                         │     ║
║   │   config = {"configurable": {"thread_id": "unique-conversation-id"}}       │     ║
║   │                                                                            │     ║
║   │   result = agent.invoke(                                                   │     ║
║   │       {"messages": [HumanMessage("你好")]},                                │     ║
║   │       config=config,  # ← 传入线程配置                                     │     ║
║   │   )                                                                        │     ║
║   │                                                                            │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
║   3. 可用的 Checkpointer 实现                                                        ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │                                                                            │     ║
║   │   # 内存存储（开发/测试用，进程结束后丢失）                                 │     ║
║   │   from langgraph.checkpoint.memory import MemorySaver                      │     ║
║   │   checkpointer = MemorySaver()                                             │     ║
║   │                                                                            │     ║
║   │   # SQLite 存储（持久化到文件）                                             │     ║
║   │   from langgraph.checkpoint.sqlite import SqliteSaver                      │     ║
║   │   checkpointer = SqliteSaver.from_conn_string("chat.db")                   │     ║
║   │                                                                            │     ║
║   │   # PostgreSQL 存储（生产环境）                                             │     ║
║   │   from langgraph.checkpoint.postgres import PostgresSaver                  │     ║
║   │   checkpointer = PostgresSaver.from_conn_string(DATABASE_URL)              │     ║
║   │                                                                            │     ║
║   │   # Redis 存储                                                             │     ║
║   │   from langgraph.checkpoint.redis import RedisSaver                        │     ║
║   │   checkpointer = RedisSaver.from_conn_string(REDIS_URL)                    │     ║
║   │                                                                            │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
║   4. Checkpointer 的方法                                                             ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │                                                                            │     ║
║   │   # 获取检查点                                                              │     ║
║   │   checkpoint = checkpointer.get(config)                                    │     ║
║   │                                                                            │     ║
║   │   # 获取检查点中的消息                                                      │     ║
║   │   messages = checkpoint["channel_values"]["messages"]                      │     ║
║   │                                                                            │     ║
║   │   # 列出所有检查点                                                          │     ║
║   │   checkpoints = list(checkpointer.list(config))                            │     ║
║   │                                                                            │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 五、记忆工作流程图

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║                          完整记忆工作流程                                             ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║   用户第一次对话                                                                      ║
║   ═════════════════════════════════════════════════════════════════════════════      ║
║                                                                                      ║
║   agent.invoke({"messages": [HumanMessage("你好，我是张三")]},                        ║
║                config={"configurable": {"thread_id": "conv-1"}})                     ║
║                                                                                      ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │ 1. 检查 checkpointer，thread_id="conv-1" 不存在                            │     ║
║   │ 2. 初始化空状态: state = {"messages": []}                                  │     ║
║   │ 3. 合并输入: state["messages"] = [HumanMessage("你好，我是张三")]           │     ║
║   │ 4. 执行 model 节点 → AI 返回 "你好张三！"                                   │     ║
║   │ 5. 状态更新: state["messages"] = [Human, AI]                               │     ║
║   │ 6. checkpointer 保存 → {"conv-1": {messages: [Human, AI]}}                 │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
║   用户第二次对话（相同 thread_id）                                                    ║
║   ═════════════════════════════════════════════════════════════════════════════      ║
║                                                                                      ║
║   agent.invoke({"messages": [HumanMessage("我的名字是什么？")]},                      ║
║                config={"configurable": {"thread_id": "conv-1"}})                     ║
║                                                                                      ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │ 1. 检查 checkpointer，thread_id="conv-1" 存在！                            │     ║
║   │ 2. 加载状态: state = {"messages": [Human("你好..."), AI("你好张三...")]}   │     ║
║   │ 3. 合并输入: state["messages"] = [Human, AI, HumanMessage("我的名字...")]  │     ║
║   │ 4. 执行 model 节点 → AI 看到完整历史，回答 "你的名字是张三"                 │     ║
║   │ 5. 状态更新: state["messages"] = [Human, AI, Human, AI]                    │     ║
║   │ 6. checkpointer 保存新状态                                                 │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
║   用户使用新的 thread_id                                                             ║
║   ═════════════════════════════════════════════════════════════════════════════      ║
║                                                                                      ║
║   agent.invoke({"messages": [HumanMessage("你好")]},                                 ║
║                config={"configurable": {"thread_id": "conv-2"}})  ← 新线程           ║
║                                                                                      ║
║   ┌────────────────────────────────────────────────────────────────────────────┐     ║
║   │ 1. 检查 checkpointer，thread_id="conv-2" 不存在                            │     ║
║   │ 2. 初始化空状态 → 这是一个全新的对话，没有记忆！                            │     ║
║   │ 3. conv-1 和 conv-2 完全独立                                               │     ║
║   └────────────────────────────────────────────────────────────────────────────┘     ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 六、checkpointer vs store 的区别

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                        checkpointer vs store                                         │
│                                                                                     │
├─────────────────────────────────┬───────────────────────────────────────────────────┤
│          checkpointer           │                    store                          │
├─────────────────────────────────┼───────────────────────────────────────────────────┤
│                                 │                                                   │
│  用途：单线程内的状态持久化       │  用途：跨线程共享数据                             │
│                                 │                                                   │
│  存储内容：                       │  存储内容：                                       │
│  - messages（对话历史）          │  - 用户信息                                       │
│  - todos（任务列表）             │  - 共享知识                                       │
│  - 其他状态                      │  - 长期记忆                                       │
│                                 │                                                   │
│  生命周期：                       │  生命周期：                                       │
│  - 绑定到 thread_id             │  - 可跨 thread 访问                               │
│  - 每个线程独立                  │  - 全局共享                                       │
│                                 │                                                   │
│  访问方式：                       │  访问方式：                                       │
│  - 自动加载/保存                 │  - runtime.store.get()                           │
│  - 无需手动操作                  │  - runtime.store.put()                           │
│                                 │                                                   │
│  示例：                          │  示例：                                           │
│  - 对话 A 的历史不会影响对话 B   │  - 用户 ID 为 123 的偏好设置                      │
│                                 │  - 所有对话都能访问                               │
│                                 │                                                   │
└─────────────────────────────────┴───────────────────────────────────────────────────┘
```

---

## 七、完整代码示例

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# 1. 创建 checkpointer
checkpointer = MemorySaver()

# 2. 创建带记忆的 agent
agent = create_agent(
    model="openai:gpt-4o",
    tools=[],
    system_prompt="你是一个友好的助手。",
    checkpointer=checkpointer,  # ← 启用记忆
)

# 3. 定义线程配置
config = {"configurable": {"thread_id": "user-123-conversation"}}

# 4. 第一轮对话
result1 = agent.invoke(
    {"messages": [HumanMessage(content="我叫张三")]},
    config=config,
)
print(result1["messages"][-1].content)
# 输出: "你好张三！很高兴认识你！"

# 5. 第二轮对话 - agent 会记住之前的对话
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config,  # 使用相同的 thread_id
)
print(result2["messages"][-1].content)
# 输出: "你叫张三！"  ← 记住了！

# 6. 查看完整对话历史
for msg in result2["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

---

## 八、总结

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                     │
│                              LangChain v1 记忆机制总结                               │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   1. 什么是记忆？                                                                    │
│      记忆 = state["messages"]，即对话消息历史                                        │
│                                                                                     │
│   2. 记忆在哪里？                                                                    │
│      存储在 Checkpointer 中，按 thread_id 分隔                                       │
│                                                                                     │
│   3. 记忆什么时候传入？                                                              │
│      - invoke 开始时自动从 checkpointer 加载                                         │
│      - 每个节点执行后自动保存到 checkpointer                                          │
│      - 用户无需手动操作                                                              │
│                                                                                     │
│   4. 用户接口：                                                                      │
│      - create_agent(checkpointer=...) 启用记忆                                       │
│      - invoke(config={"configurable": {"thread_id": ...}}) 指定对话                  │
│      - 支持 MemorySaver, SqliteSaver, PostgresSaver 等多种存储                       │
│                                                                                     │
│   5. 核心设计理念：                                                                  │
│      - 记忆是状态持久化的自然结果，不是独立模块                                       │
│      - 通过 add_messages reducer 自动追加消息                                        │
│      - thread_id 隔离不同对话的记忆                                                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

