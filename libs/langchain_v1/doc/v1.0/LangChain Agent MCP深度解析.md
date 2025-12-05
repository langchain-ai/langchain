# MCP (Model Context Protocol) 深度解析

## 1. MCP 的底层数据流

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MCP 底层数据流                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │   用户输入       │
                         │                 │
                         │ "查询 MCP 规范   │
                         │  支持的协议"     │
                         └────────┬────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  1. 请求构建阶段 (Client → LLM Provider)                                                │
│                                                                                         │
│  ChatAnthropic/ChatOpenAI 构建 API 请求:                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ {                                                                                   ││
│  │   "model": "claude-sonnet-4-5-20250929",                                            ││
│  │   "messages": [...],                                                                ││
│  │   "mcp_servers": [                           ← MCP 服务器配置                        ││
│  │     {                                                                               ││
│  │       "type": "url",                                                                ││
│  │       "url": "https://mcp.deepwiki.com/mcp",                                        ││
│  │       "name": "deepwiki",                                                           ││
│  │       "tool_configuration": {"enabled": true, "allowed_tools": ["ask_question"]},   ││
│  │       "authorization_token": "xxx"                                                  ││
│  │     }                                                                               ││
│  │   ],                                                                                ││
│  │   "betas": ["mcp-client-2025-04-04"]         ← 启用 MCP beta 功能                   ││
│  │ }                                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  2. LLM Provider 内部处理                                                               │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                     ││
│  │  Claude/OpenAI Server                                                               ││
│  │         │                                                                           ││
│  │         │ 1. 解析用户问题                                                           ││
│  │         │ 2. 发现需要调用 MCP 工具                                                  ││
│  │         │ 3. 作为 MCP Client 连接 MCP Server                                        ││
│  │         │                                                                           ││
│  │         ▼                                                                           ││
│  │  ┌─────────────────┐          ┌─────────────────┐                                   ││
│  │  │ LLM 作为        │  HTTP    │   MCP Server    │                                   ││
│  │  │ MCP Client      │ ◄──────► │ (deepwiki)      │                                   ││
│  │  └─────────────────┘          └─────────────────┘                                   ││
│  │         │                                                                           ││
│  │         │ 4. 获取工具列表 (mcp_list_tools)                                          ││
│  │         │ 5. 调用工具 (mcp_call)                                                    ││
│  │         │ 6. 获取结果 (mcp_tool_result)                                             ││
│  │         │                                                                           ││
│  │         ▼                                                                           ││
│  │  生成包含 MCP 结果的响应                                                             ││
│  │                                                                                     ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  3. 响应返回阶段 (LLM Provider → Client)                                                │
│                                                                                         │
│  AIMessage.content 包含多种内容块:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ [                                                                                   ││
│  │   {"type": "text", "text": "让我查询 MCP 规范..."},                                 ││
│  │   {                                                                                 ││
│  │     "type": "mcp_tool_use",          ← MCP 工具调用请求                             ││
│  │     "id": "toolu_xxx",                                                              ││
│  │     "name": "ask_question",                                                         ││
│  │     "input": {"question": "..."},                                                   ││
│  │     "server_name": "deepwiki"                                                       ││
│  │   },                                                                                ││
│  │   {                                                                                 ││
│  │     "type": "mcp_tool_result",       ← MCP 工具执行结果                             ││
│  │     "tool_use_id": "toolu_xxx",                                                     ││
│  │     "content": "MCP 支持 stdio, HTTP, SSE..."                                       ││
│  │   },                                                                                ││
│  │   {"type": "text", "text": "根据查询结果，MCP 支持..."}                             ││
│  │ ]                                                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  4. LangChain 标准化处理                                                                │
│                                                                                         │
│  将 provider 特定格式转换为 LangChain 标准 content_blocks:                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │ v0 格式 (provider 原生):                                                            ││
│  │   mcp_tool_use → mcp_tool_use                                                       ││
│  │   mcp_tool_result → mcp_tool_result                                                 ││
│  │                                                                                     ││
│  │ v1 格式 (LangChain 标准):                                                           ││
│  │   mcp_tool_use → server_tool_call                                                   ││
│  │   mcp_tool_result → server_tool_result                                              ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. MCP 的关键设计

### 2.1 核心设计理念

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MCP 设计核心                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘

传统方式:                                    MCP 方式:
───────────                                 ───────────

┌─────────┐    定制集成    ┌─────────┐       ┌─────────┐   标准协议   ┌─────────┐
│  LLM 1  │ ◄───────────► │ Tool A  │       │  LLM 1  │              │ Tool A  │
└─────────┘               └─────────┘       └────┬────┘              └────┬────┘
                                                 │                        │
┌─────────┐    定制集成    ┌─────────┐            │                        │
│  LLM 2  │ ◄───────────► │ Tool B  │            │   ┌──────────────┐     │
└─────────┘               └─────────┘            └──►│ MCP Protocol │◄────┘
                                                     │  (标准化层)   │
┌─────────┐    定制集成    ┌─────────┐            ┌──►│              │◄────┐
│  LLM 3  │ ◄───────────► │ Tool C  │            │   └──────────────┘     │
└─────────┘               └─────────┘       ┌────┴────┐              ┌────┴────┐
                                            │  LLM 2  │              │ Tool B  │
问题:                                       └─────────┘              └─────────┘
- N 个 LLM × M 个工具 = N×M 种集成
- 每种 LLM 需要独立适配                      优势:
- 维护成本高                                 - 统一的工具发现和调用协议
                                            - LLM 和工具解耦
                                            - 一次实现，处处可用
```

### 2.2 分层架构

| 层级 | 名称 | 职责 | LangChain 实现 |
|-----|------|------|---------------|
| **应用层** | LLM 应用 | 用户交互、任务编排 | `create_agent`, 中间件 |
| **模型层** | LLM Provider | 理解意图、决定工具调用 | `ChatAnthropic`, `ChatOpenAI` |
| **协议层** | MCP 协议 | 标准化工具发现和调用 | Provider 内置支持 |
| **传输层** | HTTP/SSE/stdio | 数据传输 | MCP Server 配置 |
| **服务层** | MCP Server | 提供具体工具能力 | 第三方实现 |

---

## 3. MCP 的核心技术实现

### 3.1 LangChain 中的 MCP 支持

#### Anthropic 实现

根据 `libs/partners/anthropic/langchain_anthropic/chat_models.py`:

```python
class ChatAnthropic(BaseChatModel):
    # MCP 服务器配置
    mcp_servers: list[dict[str, Any]] | None = None
    """List of MCP servers to use for the request.

    Example: mcp_servers=[{
        "type": "url",
        "url": "https://mcp.example.com/mcp",
        "name": "example-mcp"
    }]

    This feature requires the beta header 'mcp-client-2025-11-20'
    """

    betas: list[str] | None = None
    """Beta features to enable. Example: betas=["mcp-client-2025-04-04"]"""
```

**请求构建流程** (源码 1853-1867 行):

```python
def _get_request_payload(self, ...):
    payload = {
        "model": self.model,
        "messages": formatted_messages,
        "mcp_servers": self.mcp_servers,  # ← MCP 服务器配置
        "betas": self.betas,
        ...
    }

    # 自动添加 MCP beta header
    if payload.get("mcp_servers"):
        required_beta = "mcp-client-2025-11-20"
        if payload["betas"]:
            if required_beta not in payload["betas"]:
                payload["betas"] = [*payload["betas"], required_beta]
        else:
            payload["betas"] = [required_beta]
```

#### OpenAI 实现

根据 `libs/partners/openai/langchain_openai/chat_models/base.py`:

```python
# OpenAI 将 MCP 作为 WellKnownTools
WellKnownTools = (
    "file_search",
    "web_search_preview",
    "web_search",
    "computer_use_preview",
    "code_interpreter",
    "mcp",              # ← MCP 工具类型
    "image_generation",
)
```

### 3.2 MCP 内容块类型

根据 `libs/core/langchain_core/messages/content.py`:

```python
# 服务端工具调用 (MCP 调用被转换为此类型)
class ServerToolCall(TypedDict):
    """Tool call that is executed server-side."""

    type: Literal["server_tool_call"]
    id: str                          # 调用标识符
    name: str                        # 工具名称 (如 "remote_mcp")
    args: dict[str, Any]             # 工具参数
    index: NotRequired[int | str]    # 流式索引
    extras: NotRequired[dict[str, Any]]  # 额外元数据

# 服务端工具结果
class ServerToolResult(TypedDict):
    """Result of a server-side tool call."""

    type: Literal["server_tool_result"]
    id: NotRequired[str]
    tool_call_id: str                # 关联的调用 ID
    status: Literal["success", "error"]
    output: NotRequired[Any]         # 工具输出
    index: NotRequired[int | str]
    extras: NotRequired[dict[str, Any]]
```

### 3.3 MCP 内容块转换

根据 `libs/core/langchain_core/messages/block_translators/openai.py`:

```python
# OpenAI mcp_call → LangChain server_tool_call
elif block_type == "mcp_call":
    mcp_call = {
        "type": "server_tool_call",
        "name": "remote_mcp",
        "id": block["id"],
    }
    if arguments := block.get("arguments"):
        try:
            mcp_call["args"] = json.loads(arguments)
        except json.JSONDecodeError:
            mcp_call["extras"] = {"arguments": arguments}

    # 保存工具名和服务器标签到 extras
    if "name" in block:
        mcp_call["extras"]["tool_name"] = block["name"]
    if "server_label" in block:
        mcp_call["extras"]["server_label"] = block["server_label"]

    yield cast("types.ServerToolCall", mcp_call)

    # 同时生成结果块
    mcp_result = {
        "type": "server_tool_result",
        "tool_call_id": block["id"],
    }
    if mcp_output := block.get("output"):
        mcp_result["output"] = mcp_output

    yield cast("types.ServerToolResult", mcp_result)
```

---

## 4. 回顾来看 MCP 协议的本质是什么？

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              MCP 协议的本质                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

本质: MCP 是一种 "远程工具调用协议"，让 LLM Provider 代替用户调用外部服务

                       传统工具调用                          MCP 工具调用
                       ─────────────                        ───────────────

    ┌─────────┐        ┌─────────┐        ┌─────────┐       ┌─────────┐
    │  User   │        │   LLM   │        │  User   │       │   LLM   │
    │   App   │        │Provider │        │   App   │       │Provider │
    └────┬────┘        └────┬────┘        └────┬────┘       └────┬────┘
         │  invoke()        │                   │  invoke()       │
         │ ─────────────►   │                   │ ────────────►   │
         │                  │                   │                 │
         │  tool_calls[]    │                   │                 │
         │ ◄─────────────   │                   │    ┌──────────────────┐
         │                  │                   │    │ Provider 内部    │
         │ 本地执行工具     │                   │    │ 连接 MCP Server │
         ├──────────────┐   │                   │    │ 执行工具调用    │
         │              │   │                   │    └──────────────────┘
         │ tool_results │   │                   │                 │
         │              │   │                   │  完整结果(含MCP) │
         │ 继续对话      │   │                   │ ◄───────────────│
         │ ─────────────►   │                   │                 │
         │                  │                   │   直接使用！     │
         │  最终响应       │                   │ ◄───────────────│
         │ ◄─────────────   │
         │                  │

  特点:                                        特点:
  - 用户 App 需要实现工具执行逻辑              - Provider 代理工具执行
  - 多轮对话完成工具调用                       - 单次调用返回完整结果
  - 本地工具                                   - 远程服务


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  MCP 核心价值                                                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  1. 简化客户端: 用户无需实现工具执行逻辑，Provider 代理执行                              │
│                                                                                         │
│  2. 单次往返: 不需要多轮对话，一次请求返回包含工具结果的完整响应                         │
│                                                                                         │
│  3. 标准化: 统一的协议让任何 MCP Server 都能被任何支持 MCP 的 LLM 使用                  │
│                                                                                         │
│  4. 安全性: MCP Server 可以有自己的认证机制，不暴露给最终用户                            │
│                                                                                         │
│  5. 能力扩展: 通过添加 MCP Server 即可扩展 LLM 能力，无需修改代码                        │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. MCP Client → MCP Server 实际做了什么

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        MCP Client ↔ MCP Server 交互详解                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                  MCP Client                              MCP Server
               (LLM Provider内部)                      (如 deepwiki)
                      │                                      │
    ──────────────────┼──────────────────────────────────────┼──────────────────────────
    阶段 1: 连接建立  │                                      │
                      │  1. HTTP/WebSocket 连接              │
                      │ ─────────────────────────────────►   │
                      │                                      │
                      │  2. 认证 (authorization_token)       │
                      │ ─────────────────────────────────►   │
                      │                                      │
                      │  3. 连接确认                          │
                      │ ◄─────────────────────────────────   │
                      │                                      │
    ──────────────────┼──────────────────────────────────────┼──────────────────────────
    阶段 2: 工具发现  │                                      │
                      │                                      │
                      │  4. tools/list (获取可用工具)        │
                      │ ─────────────────────────────────►   │
                      │                                      │
                      │  5. 返回工具列表                      │
                      │ ◄─────────────────────────────────   │
                      │  {                                   │
                      │    "tools": [                        │
                      │      {                               │
                      │        "name": "ask_question",       │
                      │        "description": "查询文档",     │
                      │        "inputSchema": {              │
                      │          "type": "object",           │
                      │          "properties": {             │
                      │            "question": {             │
                      │              "type": "string"        │
                      │            }                         │
                      │          }                           │
                      │        }                             │
                      │      }                               │
                      │    ]                                 │
                      │  }                                   │
                      │                                      │
    ──────────────────┼──────────────────────────────────────┼──────────────────────────
    阶段 3: 工具调用  │                                      │
                      │                                      │
                      │  6. tools/call (调用工具)            │
                      │ ─────────────────────────────────►   │
                      │  {                                   │
                      │    "name": "ask_question",           │
                      │    "arguments": {                    │
                      │      "question": "MCP 支持的协议"    │
                      │    }                                 │
                      │  }                                   │
                      │                                      │
                      │  7. Server 执行工具逻辑              │
                      │                                      │
                      │  8. 返回结果                          │
                      │ ◄─────────────────────────────────   │
                      │  {                                   │
                      │    "content": [                      │
                      │      {                               │
                      │        "type": "text",               │
                      │        "text": "MCP 支持 stdio..."   │
                      │      }                               │
                      │    ]                                 │
                      │  }                                   │
                      │                                      │
    ──────────────────┼──────────────────────────────────────┼──────────────────────────


┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  MCP Server 端实际执行                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  以 deepwiki MCP Server 为例:                                                           │
│                                                                                         │
│  1. 接收 ask_question 调用                                                              │
│  2. 解析 question 参数                                                                  │
│  3. 在 deepwiki 知识库中搜索                                                            │
│  4. 整理搜索结果                                                                        │
│  5. 格式化为 MCP 响应                                                                   │
│  6. 返回给 MCP Client                                                                   │
│                                                                                         │
│  MCP Server 可以:                                                                       │
│  - 访问数据库                                                                           │
│  - 调用外部 API                                                                         │
│  - 执行代码                                                                             │
│  - 操作文件系统                                                                         │
│  - 任何服务器端能力                                                                     │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 多个 MCP 如何驱动

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           多 MCP Server 驱动机制                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

配置多个 MCP Server:
───────────────────

mcp_servers = [
    {
        "type": "url",
        "url": "https://mcp.deepwiki.com/mcp",
        "name": "deepwiki",                          # 文档查询服务
        "tool_configuration": {
            "enabled": True,
            "allowed_tools": ["ask_question"]
        }
    },
    {
        "type": "url",
        "url": "https://mcp.weather.com/mcp",
        "name": "weather",                           # 天气查询服务
        "tool_configuration": {
            "enabled": True,
            "allowed_tools": ["get_weather", "get_forecast"]
        }
    },
    {
        "type": "url",
        "url": "https://mcp.calculator.com/mcp",
        "name": "calculator",                        # 计算服务
        "tool_configuration": {
            "enabled": True
        }
    }
]


LLM Provider 内部处理流程:
─────────────────────────

                         ┌─────────────────┐
                         │   LLM Provider  │
                         │  (Claude/GPT)   │
                         └────────┬────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │   MCP #1    │          │   MCP #2    │          │   MCP #3    │
   │  deepwiki   │          │   weather   │          │ calculator  │
   │             │          │             │          │             │
   │ 工具:       │          │ 工具:       │          │ 工具:       │
   │ ask_question│          │ get_weather │          │ add         │
   │             │          │ get_forecast│          │ multiply    │
   └─────────────┘          └─────────────┘          └─────────────┘


工具选择机制:
────────────

用户: "今天北京天气怎么样？然后帮我计算 25 + 37"

                    LLM 分析用户意图
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
   ┌────────────────────┐       ┌────────────────────┐
   │ 需要天气信息       │       │ 需要计算能力       │
   │ → 选择 weather MCP │       │ → 选择 calculator  │
   │ → 调用 get_weather │       │   MCP              │
   └────────────────────┘       │ → 调用 add         │
                                └────────────────────┘

响应中包含多个 MCP 调用结果:
───────────────────────────

AIMessage.content = [
    {"type": "text", "text": "让我帮你查询..."},

    # 天气 MCP 调用
    {"type": "mcp_tool_use", "name": "get_weather", "server_name": "weather", ...},
    {"type": "mcp_tool_result", "content": "北京今天晴，25°C...", ...},

    # 计算 MCP 调用
    {"type": "mcp_tool_use", "name": "add", "server_name": "calculator", ...},
    {"type": "mcp_tool_result", "content": "62", ...},

    {"type": "text", "text": "北京今天晴朗，25°C。25+37=62"}
]
```

---

## 7. MCP 的参数和返回值

### 7.1 MCP Server 配置参数

```python
# Anthropic MCP Server 配置结构
mcp_server = {
    # 必填字段
    "type": "url",                              # 连接类型: "url"
    "url": "https://mcp.example.com/mcp",       # MCP Server URL
    "name": "example_server",                   # 服务器名称标识

    # 可选字段
    "authorization_token": "Bearer xxx",        # 认证令牌
    "tool_configuration": {                     # 工具配置
        "enabled": True,                        # 是否启用
        "allowed_tools": ["tool1", "tool2"]     # 允许的工具列表
    }
}
```

### 7.2 MCP 工具调用数据结构

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        MCP 数据结构详解                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘

1. mcp_tool_use / server_tool_call (工具调用请求)
─────────────────────────────────────────────────

Anthropic 原生格式 (v0):
{
    "type": "mcp_tool_use",
    "id": "toolu_01ABC...",          # 调用唯一标识
    "name": "ask_question",          # 工具名称
    "input": {                       # 工具参数 (dict)
        "question": "MCP 是什么？"
    },
    "server_name": "deepwiki"        # MCP Server 名称
}

LangChain 标准格式 (v1):
{
    "type": "server_tool_call",
    "id": "toolu_01ABC...",
    "name": "remote_mcp",            # 统一为 remote_mcp
    "args": {                        # 参数
        "question": "MCP 是什么？"
    },
    "extras": {                      # 元数据
        "tool_name": "ask_question",
        "server_label": "deepwiki"
    }
}


2. mcp_tool_result / server_tool_result (工具执行结果)
─────────────────────────────────────────────────────

Anthropic 原生格式 (v0):
{
    "type": "mcp_tool_result",
    "tool_use_id": "toolu_01ABC...", # 关联的调用 ID
    "content": "MCP 是 Model Context Protocol...",
    "is_error": false                # 是否错误
}

LangChain 标准格式 (v1):
{
    "type": "server_tool_result",
    "tool_call_id": "toolu_01ABC...",
    "status": "success",             # "success" | "error"
    "output": "MCP 是 Model Context Protocol...",
    "extras": {
        "error": null                # 错误信息（如有）
    }
}


3. mcp_list_tools (工具列表)
────────────────────────────

OpenAI 格式:
{
    "type": "mcp_list_tools",
    "id": "list_01ABC...",
    "server_label": "deepwiki",
    "tools": [
        {
            "name": "ask_question",
            "description": "向知识库提问",
            "inputSchema": {...}
        }
    ]
}

LangChain 标准格式:
{
    "type": "server_tool_call",
    "name": "mcp_list_tools",
    "id": "list_01ABC...",
    "args": {},
    "extras": {
        "server_label": "deepwiki"
    }
}
+
{
    "type": "server_tool_result",
    "tool_call_id": "list_01ABC...",
    "status": "success",
    "output": [
        {"name": "ask_question", ...}
    ]
}
```

### 7.3 与 AI 每一步交互的数据

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        AI 交互数据流                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

步骤 1: 用户输入
─────────────────
HumanMessage(content="MCP 支持哪些传输协议？")

步骤 2: 发送给 LLM Provider
────────────────────────────
{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
        {"role": "user", "content": "MCP 支持哪些传输协议？"}
    ],
    "mcp_servers": [
        {"type": "url", "url": "https://mcp.deepwiki.com/mcp", "name": "deepwiki", ...}
    ],
    "betas": ["mcp-client-2025-04-04"]
}

步骤 3: Provider 内部 MCP 交互 (用户不可见)
────────────────────────────────────────────
Provider → MCP Server: tools/list
MCP Server → Provider: {tools: [{name: "ask_question", ...}]}

Provider → MCP Server: tools/call {name: "ask_question", arguments: {...}}
MCP Server → Provider: {content: [{type: "text", text: "MCP 支持..."}]}

步骤 4: LLM 返回 AIMessage
──────────────────────────
AIMessage(
    content=[
        {"type": "text", "text": "让我查询一下 MCP 规范..."},
        {
            "type": "server_tool_call",
            "name": "remote_mcp",
            "id": "toolu_xxx",
            "args": {"question": "MCP 支持的传输协议"},
            "extras": {"tool_name": "ask_question", "server_label": "deepwiki"}
        },
        {
            "type": "server_tool_result",
            "tool_call_id": "toolu_xxx",
            "status": "success",
            "output": "MCP 支持以下传输协议：\n1. stdio\n2. HTTP\n3. SSE..."
        },
        {"type": "text", "text": "根据查询结果，MCP 支持三种传输协议：..."}
    ],
    response_metadata={...}
)

步骤 5: 用户看到的最终输出
──────────────────────────
"根据查询结果，MCP 支持三种传输协议：
1. stdio (标准输入输出)
2. HTTP (可流式)
3. SSE (服务器发送事件)
..."
```

---

## 8. LangChain v1 如何使用 MCP

### 8.1 直接使用 Anthropic Remote MCP

```python
from langchain_anthropic import ChatAnthropic

# 配置 MCP 服务器
mcp_servers = [
    {
        "type": "url",
        "url": "https://mcp.deepwiki.com/mcp",
        "name": "deepwiki",
        "tool_configuration": {
            "enabled": True,
            "allowed_tools": ["ask_question"]
        },
        "authorization_token": "your_token"
    }
]

# 创建启用 MCP 的模型
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    betas=["mcp-client-2025-04-04"],  # 启用 MCP beta
    mcp_servers=mcp_servers,
    max_tokens=10_000,
)

# 直接调用 - MCP 工具调用在 Provider 端自动完成
response = llm.invoke(
    "What transport protocols does MCP support?"
)

# 响应中包含 MCP 调用结果
for block in response.content:
    print(block["type"], ":", block.get("text") or block.get("output"))
```

### 8.2 在 create_agent 中使用（需要 langchain-mcp-adapters）

```python
# 注意: langchain-mcp-adapters 是独立的包，不在 langchain_v1 核心中
# pip install langchain-mcp-adapters

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

async def main():
    # 初始化 MCP 客户端
    async with MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "streamable_http",
            "url": "https://weather-mcp.example.com/mcp",
        }
    }) as client:
        # 加载 MCP 工具为 LangChain 工具
        tools = await load_mcp_tools(client)

        # 创建 Agent
        agent = create_agent(
            model="openai:gpt-4o",
            tools=tools,  # MCP 工具作为普通工具使用
        )

        # 调用
        response = await agent.ainvoke({
            "messages": [{"role": "user", "content": "3+5等于多少？"}]
        })
```

### 8.3 当前 langchain_v1 的 MCP 支持状态

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     langchain_v1 MCP 支持现状                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘

支持方式 1: Remote MCP (Provider 代理)
──────────────────────────────────────
✅ ChatAnthropic: 通过 mcp_servers 参数支持
✅ ChatOpenAI: 通过 tools=[{"type": "mcp", ...}] 支持

特点:
- Provider 作为 MCP Client
- 用户只需配置，无需实现
- 单次调用返回完整结果


支持方式 2: Local MCP (Client 端)
─────────────────────────────────
需要 langchain-mcp-adapters 包

特点:
- 用户 App 作为 MCP Client
- 需要管理 MCP Server 连接
- 更灵活的控制


langchain_v1 核心提供:
─────────────────────
- 标准化 content_blocks (server_tool_call, server_tool_result)
- Provider 原生格式 ↔ LangChain 格式转换
- 流式支持
```

