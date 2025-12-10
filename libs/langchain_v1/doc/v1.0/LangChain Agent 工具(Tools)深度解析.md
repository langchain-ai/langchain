# LangChain 工具(Tools)深度解析

## 1. 工具参数可以是什么

### 1.1 基本参数类型

根据 `libs/core/langchain_core/tools/convert.py` 和 `base.py` 源码：

```python
# 支持的参数类型（Python 类型提示）

# ✅ 基本类型
str, int, float, bool

# ✅ 容器类型
list[str], dict[str, Any], tuple[int, int], set[str]

# ✅ 可选类型
Optional[str], str | None

# ✅ Pydantic 模型
class MyArgs(BaseModel):
    query: str
    limit: int = 10

# ✅ 字面量类型
Literal["asc", "desc"]

# ✅ 联合类型
int | str
```

### 1.2 工具参数定义方式

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              工具参数定义的三种方式                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘

方式 1: 函数类型提示（自动推断）
────────────────────────────────
@tool
def search(query: str, limit: int = 10) -> str:
    """搜索内容"""
    return "results"

                    │
                    ▼  自动生成 args_schema
           ┌───────────────────┐
           │ {                 │
           │   "query": {      │
           │     "type": "string"
           │   },              │
           │   "limit": {      │
           │     "type": "integer",
           │     "default": 10 │
           │   }               │
           │ }                 │
           └───────────────────┘

方式 2: Pydantic 模型（显式定义）
────────────────────────────────
class SearchInput(BaseModel):
    """搜索参数"""
    query: str = Field(description="搜索关键词")
    limit: int = Field(default=10, description="返回数量")

@tool(args_schema=SearchInput)
def search(query: str, limit: int) -> str:
    return "results"

方式 3: Google 风格 docstring（解析描述）
────────────────────────────────────────
@tool(parse_docstring=True)
def search(query: str, limit: int = 10) -> str:
    """搜索内容。

    Args:
        query: 搜索关键词
        limit: 返回的结果数量
    """
    return "results"
```

### 1.3 特殊注入参数类型

根据 `base.py` 第 1297-1397 行的源码：

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           工具可注入的特殊参数                                            │
│                    （这些参数不会传给 LLM，运行时自动注入）                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│          类型                │                     说明                               │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ InjectedToolArg              │ 基类：标记参数为运行时注入                              │
│                              │                                                        │
│ InjectedToolCallId           │ 注入工具调用 ID                                        │
│                              │ Annotated[str, InjectedToolCallId()]                   │
│                              │                                                        │
│ InjectedStore                │ 注入 Store 实例                                        │
│                              │ Annotated[Any, InjectedStore()]                        │
│                              │                                                        │
│ InjectedState                │ 注入 Agent 状态                                        │
│                              │ Annotated[dict, InjectedState()]                       │
│                              │                                                        │
│ ToolRuntime                  │ 直接注入运行时上下文（无需 Annotated）                  │
│                              │ runtime: ToolRuntime                                   │
│                              │ 包含: state, context, store                            │
│                              │                                                        │
│ RunnableConfig               │ 注入运行时配置                                          │
│                              │ config: RunnableConfig                                 │
│                              │                                                        │
│ CallbackManagerForToolRun    │ 注入回调管理器                                          │
│                              │ run_manager: CallbackManagerForToolRun                 │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
```

**代码示例**：

```python
from typing import Annotated, Any
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedStore, InjectedState, ToolRuntime

@tool
def comprehensive_tool(
    # LLM 控制的参数（会发送给模型）
    query: str,
    limit: int = 10,

    # 运行时注入的参数（不发送给模型）
    tool_call_id: Annotated[str, InjectedToolCallId()],
    store: Annotated[Any, InjectedStore()],
    state: Annotated[dict, InjectedState()],
    runtime: ToolRuntime,  # 直接注入，无需 Annotated
) -> str:
    """综合示例工具"""
    # tool_call_id: 来自 AIMessage.tool_calls[i]["id"]
    # store: create_agent(store=...) 传入的 store
    # state: Agent 当前状态 {"messages": [...], ...}
    # runtime: 包含 state, context, store 的运行时对象
    return f"Query: {query}, Limit: {limit}"
```

---

## 2. 工具返回值可以是什么

### 2.1 返回值类型

根据 `base.py` 第 491-497 行和 `_format_output` 函数：

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              工具返回值类型                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                          response_format 参数决定处理方式
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
            ▼                                                   ▼
   ┌────────────────────┐                         ┌────────────────────────────┐
   │ "content" (默认)   │                         │ "content_and_artifact"      │
   └─────────┬──────────┘                         └──────────────┬─────────────┘
             │                                                   │
             ▼                                                   ▼
   ┌────────────────────┐                         ┌────────────────────────────┐
   │ 返回值 → content   │                         │ 返回 tuple[content, artifact]│
   │                    │                         │                            │
   │ 支持的类型:        │                         │ content: 发送给 LLM        │
   │ - str             │                         │ artifact: 保留在 ToolMessage │
   │ - list[str|dict]  │                         │           不发送给 LLM      │
   │ - dict            │                         │                            │
   │ - int/float/bool  │                         │ 示例:                       │
   │ - Pydantic Model  │                         │ return ("摘要", full_data) │
   │ - ToolMessage     │                         │                            │
   └────────────────────┘                         └────────────────────────────┘
```

### 2.2 返回值处理流程

```python
# 源码：libs/core/langchain_core/tools/base.py 第 1181-1210 行

def _format_output(
    content: Any,
    artifact: Any,
    tool_call_id: str | None,
    name: str,
    status: str,
) -> ToolOutputMixin | Any:
    """格式化工具输出为 ToolMessage"""

    # 1. 如果已经是 ToolOutputMixin（如 ToolMessage），直接返回
    if isinstance(content, ToolOutputMixin) or tool_call_id is None:
        return content

    # 2. 如果内容不是合法的消息内容类型，转为字符串
    if not _is_message_content_type(content):
        content = _stringify(content)

    # 3. 包装为 ToolMessage
    return ToolMessage(
        content,
        artifact=artifact,
        tool_call_id=tool_call_id,
        name=name,
        status=status,  # "success" 或 "error"
    )
```

### 2.3 ToolMessage 数据结构

```python
# 源码：libs/core/langchain_core/messages/tool.py

class ToolMessage(BaseMessage, ToolOutputMixin):
    """工具执行结果消息"""

    tool_call_id: str           # 关联的工具调用 ID
    type: Literal["tool"] = "tool"
    artifact: Any = None        # 不发给 LLM 的附加数据
    status: Literal["success", "error"] = "success"

    # content: 继承自 BaseMessage
    # - str: 文本内容
    # - list[str | dict]: 多模态内容块
```

### 2.4 返回值示例

```python
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# 示例 1: 返回字符串
@tool
def simple_tool(x: int) -> str:
    return f"结果是 {x}"
# 输出: ToolMessage(content="结果是 5", tool_call_id="xxx")

# 示例 2: 返回 dict（自动转为字符串）
@tool
def dict_tool(x: int) -> dict:
    return {"result": x, "doubled": x * 2}
# 输出: ToolMessage(content='{"result": 5, "doubled": 10}', ...)

# 示例 3: 返回 content + artifact
@tool(response_format="content_and_artifact")
def artifact_tool(query: str) -> tuple[str, dict]:
    full_data = {"results": [...], "metadata": {...}}
    summary = "找到 10 条结果"
    return (summary, full_data)
# 输出: ToolMessage(
#     content="找到 10 条结果",          # 发给 LLM
#     artifact={"results": [...], ...}  # 不发给 LLM
# )

# 示例 4: 直接返回 ToolMessage
@tool
def custom_message_tool(x: int, tool_call_id: Annotated[str, InjectedToolCallId()]) -> ToolMessage:
    return ToolMessage(
        content=f"结果: {x}",
        artifact={"raw": x},
        tool_call_id=tool_call_id,
        name="custom_message_tool"
    )
```

---

## 3. AI 如何与工具串联

### 3.1 完整流程图

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           AI 与工具的串联流程                                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  1. 用户输入                                                                            │
│                                                                                         │
│     user_input = "帮我搜索一下 Python 教程"                                             │
│                           │                                                             │
│                           ▼                                                             │
│     HumanMessage(content="帮我搜索一下 Python 教程")                                     │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  2. model.bind_tools() - 工具绑定                                                       │
│                                                                                         │
│     工具定义被转换为 JSON Schema 发送给 LLM:                                             │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ {                                                                               │ │
│     │   "type": "function",                                                           │ │
│     │   "function": {                                                                 │ │
│     │     "name": "search",                                                           │ │
│     │     "description": "搜索内容",                                                   │ │
│     │     "parameters": {                                                             │ │
│     │       "type": "object",                                                         │ │
│     │       "properties": {                                                           │ │
│     │         "query": {"type": "string", "description": "搜索关键词"},                │ │
│     │         "limit": {"type": "integer", "default": 10}                             │ │
│     │       },                                                                        │ │
│     │       "required": ["query"]                                                     │ │
│     │     }                                                                           │ │
│     │   }                                                                             │ │
│     │ }                                                                               │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│     注意：InjectedToolArg 标记的参数不会出现在 schema 中！                               │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  3. LLM 决策 - 返回 AIMessage                                                           │
│                                                                                         │
│     LLM 分析用户请求，决定调用工具:                                                       │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ AIMessage(                                                                      │ │
│     │   content="",                          # 通常为空或包含思考过程                   │ │
│     │   tool_calls=[                                                                  │ │
│     │     {                                                                           │ │
│     │       "type": "tool_call",                                                      │ │
│     │       "id": "call_abc123",             # 唯一标识符                              │ │
│     │       "name": "search",                # 工具名称                               │ │
│     │       "args": {                        # LLM 填充的参数                          │ │
│     │         "query": "Python 教程",                                                 │ │
│     │         "limit": 5                                                              │ │
│     │       }                                                                         │ │
│     │     }                                                                           │ │
│     │   ]                                                                             │ │
│     │ )                                                                               │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  4. Agent 路由判断                                                                      │
│                                                                                         │
│     # 源码: factory.py _make_model_to_tools_edge                                        │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ if len(last_ai_message.tool_calls) == 0:                                        │ │
│     │     return END          # 没有工具调用，结束循环                                  │ │
│     │                                                                                 │ │
│     │ pending_tool_calls = [c for c in last_ai_message.tool_calls ...]                │ │
│     │                                                                                 │ │
│     │ if pending_tool_calls:                                                          │ │
│     │     return [Send("tools", ToolCallWithContext(...)) for ...]                    │ │
│     │     # 有待处理的工具调用，发送到工具节点                                          │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  5. ToolNode 执行工具                                                                   │
│                                                                                         │
│     ToolNode 接收 ToolCallWithContext:                                                  │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ ToolCallWithContext = {                                                         │ │
│     │     "__type": "tool_call_with_context",                                         │ │
│     │     "tool_call": {                                                              │ │
│     │         "id": "call_abc123",                                                    │ │
│     │         "name": "search",                                                       │ │
│     │         "args": {"query": "Python 教程", "limit": 5}                            │ │
│     │     },                                                                          │ │
│     │     "state": {...当前 Agent 状态...}                                            │ │
│     │ }                                                                               │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│     执行过程:                                                                           │
│     1. 从 tool_call 提取 args                                                          │
│     2. 注入特殊参数 (InjectedStore, InjectedState, ToolRuntime 等)                      │
│     3. 调用工具函数                                                                     │
│     4. 格式化返回值为 ToolMessage                                                       │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  6. 返回 ToolMessage                                                                    │
│                                                                                         │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ ToolMessage(                                                                    │ │
│     │   content="找到以下 Python 教程:\n1. ...\n2. ...",                              │ │
│     │   tool_call_id="call_abc123",         # 关联到原始调用                           │ │
│     │   name="search",                                                                │ │
│     │   status="success",                                                             │ │
│     │   artifact=None                       # 可选的附加数据                           │ │
│     │ )                                                                               │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  7. 路由回 Model 节点                                                                   │
│                                                                                         │
│     # 源码: factory.py _make_tools_to_model_edge                                        │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ def tools_to_model(state):                                                      │ │
│     │     # 检查 return_direct                                                        │ │
│     │     if all(tool.return_direct for tool in executed_tools):                      │ │
│     │         return END                                                              │ │
│     │                                                                                 │ │
│     │     # 默认：继续循环，让模型处理工具结果                                          │ │
│     │     return "model"                                                              │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  8. LLM 处理工具结果                                                                    │
│                                                                                         │
│     消息列表现在包含:                                                                    │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ [                                                                               │ │
│     │   HumanMessage("帮我搜索一下 Python 教程"),                                      │ │
│     │   AIMessage(tool_calls=[...]),                                                  │ │
│     │   ToolMessage(content="找到以下...", tool_call_id="call_abc123")                │ │
│     │ ]                                                                               │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
│     LLM 根据工具结果生成最终回复:                                                        │
│     ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│     │ AIMessage(                                                                      │ │
│     │   content="我为您找到了以下 Python 教程:\n\n1. ...\n2. ...",                    │ │
│     │   tool_calls=[]                       # 没有更多工具调用                         │ │
│     │ )                                                                               │ │
│     └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  9. 结束循环                                                                            │
│                                                                                         │
│     tool_calls 为空 → 返回 END → 输出最终结果                                           │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 参数如何流转

### 4.1 参数流转详细图解

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              参数流转完整过程                                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                              【定义时】
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  @tool                                                                                  │
│  def search(                                                                            │
│      query: str,                          ← LLM 控制参数                                │
│      limit: int = 10,                     ← LLM 控制参数（有默认值）                    │
│      store: Annotated[Any, InjectedStore()],  ← 注入参数                                │
│      runtime: ToolRuntime,                ← 注入参数                                    │
│  ) -> str:                                                                              │
│      ...                                                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ @tool 装饰器处理
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  create_schema_from_function() - 生成参数 Schema                                        │
│                                                                                         │
│  源码: base.py 第 279-375 行                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ def create_schema_from_function(..., include_injected: bool = True):                ││
│  │     sig = inspect.signature(func)                                                   ││
│  │                                                                                     ││
│  │     for existing_param in existing_params:                                          ││
│  │         if not include_injected and _is_injected_arg_type(...):                     ││
│  │             filter_args_.append(existing_param)  # 过滤注入参数                      ││
│  │                                                                                     ││
│  │     return _create_subset_model(...)                                                ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  生成的 Schema（发给 LLM）:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ {                                                                                   ││
│  │   "properties": {                                                                   ││
│  │     "query": {"type": "string"},          ✓ 包含                                    ││
│  │     "limit": {"type": "integer", "default": 10}  ✓ 包含                             ││
│  │   },                                                                                ││
│  │   "required": ["query"]                                                             ││
│  │   // store 和 runtime 不在这里！                                                    ││
│  │ }                                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ LLM 调用
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  LLM 返回 tool_call                                                                     │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ tool_call = {                                                                       ││
│  │     "id": "call_abc123",                                                            ││
│  │     "name": "search",                                                               ││
│  │     "args": {                                                                       ││
│  │         "query": "Python 教程",          ← LLM 填充                                 ││
│  │         "limit": 5                       ← LLM 填充                                 ││
│  │         // 没有 store 和 runtime！                                                  ││
│  │     }                                                                               ││
│  │ }                                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ ToolNode 执行
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  _prep_run_args() - 准备执行参数                                                        │
│                                                                                         │
│  源码: base.py 第 1144-1178 行                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ def _prep_run_args(value, config, **kwargs):                                        ││
│  │     if _is_tool_call(value):                                                        ││
│  │         tool_call_id = value["id"]       # "call_abc123"                            ││
│  │         tool_input = value["args"].copy()  # {"query": "...", "limit": 5}           ││
│  │     return (tool_input, run_kwargs)                                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  此时 tool_input = {"query": "Python 教程", "limit": 5}                                │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ 注入参数
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  ToolNode / langgraph - 注入运行时参数                                                  │
│                                                                                         │
│  注入过程（发生在 langgraph.prebuilt.tool_node 中）:                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ # 对于每个工具参数:                                                                 ││
│  │ for param_name, param_type in get_type_hints(tool_func).items():                    ││
│  │     if _is_injected_arg_type(param_type, InjectedStore):                            ││
│  │         kwargs[param_name] = runtime.store                                          ││
│  │     elif _is_injected_arg_type(param_type, InjectedState):                          ││
│  │         kwargs[param_name] = state                                                  ││
│  │     elif _is_injected_arg_type(param_type, InjectedToolCallId):                     ││
│  │         kwargs[param_name] = tool_call_id                                           ││
│  │     elif param_type is ToolRuntime:  # 直接注入                                     ││
│  │         kwargs[param_name] = ToolRuntime(state=state, context=context, store=store) ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
│  合并后的完整参数:                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ kwargs = {                                                                          ││
│  │     "query": "Python 教程",              ← 来自 LLM                                 ││
│  │     "limit": 5,                          ← 来自 LLM                                 ││
│  │     "store": <InMemoryStore>,            ← 注入                                     ││
│  │     "runtime": ToolRuntime(state=..., context=..., store=...),  ← 注入              ││
│  │ }                                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ 执行工具函数
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  search(**kwargs) - 实际调用                                                            │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ def search(query, limit, store, runtime):                                           ││
│  │     # query = "Python 教程"                                                         ││
│  │     # limit = 5                                                                     ││
│  │     # store = <InMemoryStore>                                                       ││
│  │     # runtime = ToolRuntime(...)                                                    ││
│  │                                                                                     ││
│  │     results = do_search(query, limit)                                               ││
│  │     store.mset([("last_query", query)])  # 可以使用 store                           ││
│  │     return format_results(results)                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ 格式化输出
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  _format_output() - 包装为 ToolMessage                                                  │
│                                                                                         │
│  源码: base.py 第 1181-1210 行                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ return ToolMessage(                                                                 ││
│  │     content="找到以下结果...",           ← 工具返回值                               ││
│  │     artifact=None,                       ← 可选附加数据                             ││
│  │     tool_call_id="call_abc123",          ← 关联原始调用                             ││
│  │     name="search",                                                                  ││
│  │     status="success"                                                                ││
│  │ )                                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 关键数据结构总结

| 数据结构 | 位置 | 用途 |
|---------|------|------|
| `ToolCall` | `AIMessage.tool_calls[]` | LLM 请求调用工具 |
| `ToolMessage` | Agent State messages | 工具执行结果 |
| `ToolCallRequest` | 中间件 `wrap_tool_call` | 工具调用上下文 |
| `ToolRuntime` | 工具函数参数 | 运行时上下文访问 |

```python
# ToolCall 结构
ToolCall = {
    "type": "tool_call",
    "id": str,            # 唯一标识符
    "name": str,          # 工具名称
    "args": dict[str, Any]  # LLM 提供的参数
}

# ToolMessage 结构
ToolMessage = {
    "type": "tool",
    "content": str | list,  # 返回给 LLM 的内容
    "tool_call_id": str,    # 关联的 ToolCall.id
    "name": str,            # 工具名称
    "status": "success" | "error",
    "artifact": Any         # 不发给 LLM 的附加数据
}

# ToolRuntime 结构
ToolRuntime = {
    "state": AgentState,    # 当前 Agent 状态
    "context": dict,        # 用户上下文
    "store": BaseStore      # 持久化存储
}
```

