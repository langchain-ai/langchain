# SummarizationMiddleware 完整流程解析

## 一、中间件概述

`SummarizationMiddleware` 是一个**before_model 钩子**中间件，在每次模型调用前检查对话历史长度，当达到阈值时自动对旧消息进行摘要压缩，以避免超出模型的上下文窗口限制。

---

## 二、输入输出定义

### 输入（Inputs）

| 输入 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `state["messages"]` | `list[AnyMessage]` | LangGraph State | 当前完整对话历史 |
| `runtime` | `Runtime` | LangGraph | 运行时上下文（未使用） |

### 输出（Outputs）

| 输出 | 类型 | 说明 |
|------|------|------|
| `None` | `None` | 不需要摘要时，返回 None，state 不变 |
| `{"messages": [...]}` | `dict` | 需要摘要时，返回新的消息列表，替换原有历史 |

**输出消息结构（摘要时）：**
```python
{
    "messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 删除所有旧消息
        HumanMessage("Here is a summary of..."),  # 摘要消息
        *preserved_messages,  # 保留的最近消息
    ]
}
```

---

## 三、初始化参数

```python
SummarizationMiddleware(
    model: str | BaseChatModel,           # 用于生成摘要的模型
    trigger: ContextSize | list[ContextSize] | None,  # 触发条件
    keep: ContextSize,                     # 保留策略
    token_counter: TokenCounter,           # token 计数器
    summary_prompt: str,                   # 摘要提示词模板
    trim_tokens_to_summarize: int | None,  # 摘要前裁剪限制
)
```

### trigger 触发条件类型

| 类型 | 格式 | 示例 | 说明 |
|------|------|------|------|
| `ContextFraction` | `("fraction", float)` | `("fraction", 0.8)` | 达到模型最大输入的 80% 时触发 |
| `ContextTokens` | `("tokens", int)` | `("tokens", 3000)` | 达到 3000 tokens 时触发 |
| `ContextMessages` | `("messages", int)` | `("messages", 50)` | 达到 50 条消息时触发 |

### keep 保留策略

与 trigger 格式相同，指定摘要后保留多少最近内容。

---

## 四、完整流程图

```
                              ┌─────────────────────────────┐
                              │      Agent 执行循环          │
                              └─────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           before_model(state, runtime)                               │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: 获取消息列表                                                           │  │
│  │                                                                                │  │
│  │   messages = state["messages"]                                                 │  │
│  │   _ensure_message_ids(messages)  # 确保每条消息有唯一 ID                        │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 2: 计算当前 token 数                                                      │  │
│  │                                                                                │  │
│  │   total_tokens = self.token_counter(messages)                                  │  │
│  │                                                                                │  │
│  │   # token_counter 默认使用 count_tokens_approximately                          │  │
│  │   # 如果是 Anthropic 模型，使用 chars_per_token=3.3 的优化参数                  │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 3: 判断是否需要摘要 _should_summarize(messages, total_tokens)             │  │
│  │                                                                                │  │
│  │   for kind, value in self._trigger_conditions:                                 │  │
│  │       if kind == "messages" and len(messages) >= value:                        │  │
│  │           return True  # 消息数达到阈值                                         │  │
│  │       if kind == "tokens" and total_tokens >= value:                           │  │
│  │           return True  # token 数达到阈值                                       │  │
│  │       if kind == "fraction":                                                   │  │
│  │           max_input = model.profile["max_input_tokens"]                        │  │
│  │           threshold = int(max_input * value)                                   │  │
│  │           if total_tokens >= threshold:                                        │  │
│  │               return True  # 达到模型容量的指定比例                             │  │
│  │   return False                                                                 │  │
│  │                                                                                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                          │
│                         ┌─────────────────┴─────────────────┐                        │
│                         │                                   │                        │
│                    False (不需要摘要)                   True (需要摘要)               │
│                         │                                   │                        │
│                         ▼                                   ▼                        │
│                  ┌──────────────┐           ┌────────────────────────────────────┐   │
│                  │ return None  │           │ Step 4: 确定截断点                  │   │
│                  │              │           │ cutoff_index = _determine_cutoff_  │   │
│                  │ state 不变化  │           │                    index(messages) │   │
│                  └──────────────┘           └────────────────────────────────────┘   │
│                                                             │                        │
│                                                             ▼                        │
│                                             ┌────────────────────────────────────┐   │
│                                             │ Step 5: 分割消息                    │   │
│                                             │                                    │   │
│                                             │ messages_to_summarize = [:cutoff]  │   │
│                                             │ preserved_messages = [cutoff:]     │   │
│                                             │                                    │   │
│                                             └────────────────────────────────────┘   │
│                                                             │                        │
│                                                             ▼                        │
│                                             ┌────────────────────────────────────┐   │
│                                             │ Step 6: 生成摘要                    │   │
│                                             │                                    │   │
│                                             │ summary = _create_summary(         │   │
│                                             │     messages_to_summarize          │   │
│                                             │ )                                  │   │
│                                             │                                    │   │
│                                             └────────────────────────────────────┘   │
│                                                             │                        │
│                                                             ▼                        │
│                                             ┌────────────────────────────────────┐   │
│                                             │ Step 7: 构建新消息列表              │   │
│                                             │                                    │   │
│                                             │ return {                           │   │
│                                             │   "messages": [                    │   │
│                                             │     RemoveMessage(REMOVE_ALL),     │   │
│                                             │     HumanMessage(summary),         │   │
│                                             │     *preserved_messages            │   │
│                                             │   ]                                │   │
│                                             │ }                                  │   │
│                                             │                                    │   │
│                                             └────────────────────────────────────┘   │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────────┐
                              │   LangGraph 更新 state      │
                              │   → 进入 model 节点         │
                              └─────────────────────────────┘
```

---

## 五、关键函数详解

### 5.1 `_determine_cutoff_index(messages)` - 确定截断点

```
输入: messages = [msg0, msg1, msg2, ..., msgN]
输出: cutoff_index (整数)

目的: 找到安全的切割位置，确保不会把 AI 的 tool_calls 和对应的 ToolMessage 分开

流程:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  kind, value = self.keep  # 例如 ("messages", 20)                       │
│                                                                         │
│  if kind in {"tokens", "fraction"}:                                     │
│      # 使用二分查找找到保留 N tokens 的截断点                            │
│      cutoff = _find_token_based_cutoff(messages)                        │
│  else:  # kind == "messages"                                            │
│      # 保留最近 N 条消息                                                 │
│      cutoff = _find_safe_cutoff(messages, value)                        │
│                                                                         │
│  # 关键：确保截断点不会分离 AIMessage(tool_calls) 和 ToolMessage        │
│  # 向前搜索找到安全的截断位置                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 `_is_safe_cutoff_point(messages, cutoff_index)` - 安全截断检查

```
目的: 检查在 cutoff_index 处切割是否会分离 AI/Tool 消息对

检查范围: cutoff_index ± 5 条消息内

逻辑:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  for i in range(cutoff - 5, cutoff + 5):                               │
│      if messages[i] 是 AIMessage 且有 tool_calls:                       │
│          提取所有 tool_call_ids                                         │
│          检查对应的 ToolMessage 是否在 cutoff 同一侧                     │
│          如果 AIMessage 在 cutoff 前，但 ToolMessage 在 cutoff 后:      │
│              return False  # 不安全！                                   │
│                                                                         │
│  return True  # 安全                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 `_create_summary(messages_to_summarize)` - 生成摘要

```
输入: 需要摘要的消息列表
输出: 摘要文本字符串

流程:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. 裁剪消息以适应摘要生成限制                                           │
│     trimmed = _trim_messages_for_summary(messages_to_summarize)         │
│     # 默认限制 4000 tokens                                              │
│                                                                         │
│  2. 使用摘要模型生成摘要                                                 │
│     response = self.model.invoke(                                       │
│         self.summary_prompt.format(messages=trimmed)                    │
│     )                                                                   │
│                                                                         │
│  3. 返回摘要文本                                                         │
│     return response.text.strip()                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 六、状态变化示例

### 摘要前 state["messages"]

```python
messages = [
    HumanMessage("你好，帮我分析一下市场"),              # msg 0
    AIMessage(tool_calls=[search_market]),              # msg 1
    ToolMessage("市场数据: ..."),                       # msg 2
    AIMessage("根据市场数据..."),                       # msg 3
    HumanMessage("再帮我看看竞争对手"),                  # msg 4
    AIMessage(tool_calls=[search_competitors]),         # msg 5
    ToolMessage("竞争对手数据: ..."),                   # msg 6
    AIMessage("竞争对手分析..."),                       # msg 7
    # ... 更多消息 ...
    HumanMessage("总结一下"),                           # msg 48
    AIMessage("好的，我来总结..."),                     # msg 49
    HumanMessage("还有什么建议？"),                     # msg 50  ← 触发摘要
]
# 假设 trigger=("messages", 50), keep=("messages", 20)
```

### 摘要后 state["messages"]

```python
messages = [
    # 摘要消息（包含 msg 0-30 的精华）
    HumanMessage("Here is a summary of the conversation to date:\n\n用户请求市场分析和竞争对手分析，AI 通过搜索工具获取了相关数据并进行了详细分析..."),
    
    # 保留的最近 20 条消息 (msg 31-50)
    HumanMessage("那利润情况呢？"),                     # 原 msg 31
    AIMessage(tool_calls=[get_profit_data]),           # 原 msg 32
    ToolMessage("利润数据: ..."),                      # 原 msg 33
    # ... 保留的消息 ...
    HumanMessage("总结一下"),                          # 原 msg 48
    AIMessage("好的，我来总结..."),                    # 原 msg 49
    HumanMessage("还有什么建议？"),                    # 原 msg 50
]
```

---

## 七、摘要提示词模板

```python
DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant 
context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract 
the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You must do your very best to extract and record all of the most important context.
You want to ensure that you don't repeat any actions you've already completed.
</instructions>

<messages>
Messages to summarize:
{messages}
</messages>
"""
```

---

## 八、与 model.profile 的关联

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  当 trigger 或 keep 使用 "fraction" 类型时:                             │
│                                                                         │
│  1. 从 model.profile 读取 max_input_tokens                              │
│     max_input = self.model.profile.get("max_input_tokens")              │
│                                                                         │
│  2. 计算实际阈值                                                         │
│     threshold = int(max_input * fraction)                               │
│                                                                         │
│  例如:                                                                   │
│  - model.profile["max_input_tokens"] = 128000 (GPT-4o)                  │
│  - trigger = ("fraction", 0.8)                                          │
│  - 实际触发阈值 = 128000 * 0.8 = 102400 tokens                          │
│                                                                         │
│  如果 profile 不可用，会在 __init__ 时抛出 ValueError                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 九、完整数据流总结

```
用户发送消息
       │
       ▼
┌──────────────────┐
│ state["messages"]│ ← 包含完整对话历史
└──────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│              SummarizationMiddleware.before_model                │
│                                                                  │
│  输入:                                                           │
│    - state["messages"]: [Human, AI, Tool, AI, Human, ...]       │
│    - runtime: LangGraph Runtime                                  │
│                                                                  │
│  处理:                                                           │
│    1. 计算 token 数量                                            │
│    2. 检查是否达到触发条件                                        │
│    3. 如果需要摘要:                                              │
│       - 找安全截断点（不分离 AI/Tool 对）                        │
│       - 调用 LLM 生成摘要                                        │
│       - 构建新消息列表                                           │
│                                                                  │
│  输出:                                                           │
│    - None（不需要摘要）                                          │
│    - {"messages": [RemoveAll, Summary, ...Recent]}（需要摘要）   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│ LangGraph 更新    │ ← 如果返回 dict，合并到 state
│ state["messages"]│
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   model 节点     │ ← 使用压缩后的消息调用 LLM
└──────────────────┘
```

---

## 十、使用示例

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",  # 用小模型生成摘要（省成本）
            trigger=[
                ("fraction", 0.8),       # 达到模型 80% 容量
                ("messages", 100),       # 或达到 100 条消息
            ],
            keep=("messages", 20),       # 保留最近 20 条消息
        ),
    ],
)
```
