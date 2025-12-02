# LangChain 聊天记忆持久化演示

本目录包含两个演示文件，展示如何使用 LangChain 的 checkpointer 机制来持久化存储和恢复聊天记忆。

## 文件说明

### 1. `demo_memory_simple.py` - 简化演示（推荐先看）
- **不依赖真实 API**，只展示核心概念
- 适合理解机制和概念
- 可以直接运行，无需配置

### 2. `demo_memory_persistence.py` - 完整演示
- **需要配置 API 密钥**才能运行
- 展示实际使用场景
- 包含多个实际示例

## 快速开始

### 运行简化演示

```bash
python demo_memory_simple.py
```

这个演示不需要任何配置，可以直接运行。

### 运行完整演示

1. **配置 API 密钥**（选择一种方式）：

   ```bash
   # Anthropic (推荐)
   export ANTHROPIC_API_KEY="your-api-key"
   
   # 或者 OpenAI
   export OPENAI_API_KEY="your-api-key"
   ```

2. **修改模型标识符**（如果需要）：

   在 `demo_memory_persistence.py` 中，找到所有 `model="anthropic:claude-3-5-sonnet-20241022"` 的地方，
   可以替换为：
   - `"openai:gpt-4"` (OpenAI)
   - `"openai:gpt-3.5-turbo"` (OpenAI)
   - 或其他支持的模型

3. **运行演示**：

   ```bash
   python demo_memory_persistence.py
   ```

## 核心概念

### Checkpointer（检查点保存器）

用于持久化单个对话线程的状态（聊天记忆）：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
```

### thread_id（线程 ID）

用于区分不同的对话线程：

```python
thread_config = {"configurable": {"thread_id": "conversation-1"}}
```

### 创建带记忆的 Agent

```python
from langchain.agents import create_agent

agent = create_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[...],
    checkpointer=checkpointer,  # 启用记忆持久化
)
```

### 使用记忆

```python
# 第一轮对话
result1 = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config=thread_config,
)

# 第二轮对话（自动恢复之前的记忆）
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我刚才说了什么？")]},
    config=thread_config,  # 相同的 thread_id
)
```

## 演示内容

### 演示 1: 内存 Checkpointer
- 使用 `MemorySaver` 保存对话状态
- 展示多轮对话的记忆保持
- 查看完整对话历史

### 演示 2: SQLite Checkpointer（如果可用）
- 使用 `SqliteSaver` 持久化到文件
- 数据在程序重启后仍然保留
- 适合生产环境使用

### 演示 3: 多个独立对话线程
- 使用不同的 `thread_id` 创建独立对话
- 展示线程之间的隔离性
- 每个线程有独立的对话历史

### 演示 4: 状态检查
- 如何查看保存的状态
- 如何获取检查点信息
- 状态恢复机制

### 演示 5: 状态管理高级用法
- 状态的生命周期
- 自动保存和恢复
- 最佳实践

## 存储后端选择

### MemorySaver（内存存储）
- ✅ 适合开发和测试
- ✅ 无需额外配置
- ❌ 程序重启后数据丢失

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

### SqliteSaver（SQLite 数据库）
- ✅ 数据持久化到文件
- ✅ 适合生产环境
- ✅ 程序重启后数据保留
- ⚠️ 需要额外依赖

```python
from langgraph.checkpoint.sqlite import SqliteSaver
# 内存数据库
checkpointer = SqliteSaver.from_conn_string(":memory:")
# 或持久化到文件
checkpointer = SqliteSaver.from_conn_string("chat_memory.db")
```

### PostgresSaver（PostgreSQL 数据库）
- ✅ 适合多进程/分布式环境
- ✅ 高性能
- ⚠️ 需要 PostgreSQL 数据库

```python
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

## 关键要点

1. **Checkpointer** 用于持久化单个对话线程的状态
2. **thread_id** 用于区分不同的对话
3. **状态会在每次调用后自动保存**
4. **使用相同的 thread_id 会自动恢复之前的对话历史**
5. **支持内存、SQLite、PostgreSQL 等多种存储后端**

## 常见问题

### Q: 如何清除某个对话的记忆？

A: 使用新的 `thread_id` 即可创建新的对话，或者删除对应的检查点。

### Q: 多个对话之间会互相影响吗？

A: 不会。每个 `thread_id` 对应完全独立的对话历史。

### Q: 如何查看保存的所有对话？

A: 可以通过 checkpointer 的 `list()` 方法（如果支持）或直接查询数据库。

### Q: 记忆会占用多少空间？

A: 取决于对话长度。可以使用 `SummarizationMiddleware` 来自动摘要旧消息以节省空间。

## 相关文档

- [LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph Checkpointing 文档](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

## 示例输出

运行 `demo_memory_simple.py` 会看到类似输出：

```
============================================================
简化演示：聊天记忆持久化的基本概念
============================================================

1. 创建了 MemorySaver checkpointer

2. 模拟第一轮对话状态：
   - 用户: 你好，我的名字是张三
   - 助手: 你好张三！很高兴认识你。

3. 状态已保存到 checkpointer（在实际使用中自动完成）

4. 模拟第二轮对话（使用相同的 thread_id）：
   - 系统会自动加载之前保存的状态
   - 新的消息会追加到现有消息列表

5. 完整对话历史：
   1. 用户: 你好，我的名字是张三
   2. 助手: 你好张三！很高兴认识你。
   3. 用户: 我刚才说了我的名字是什么？
   4. 助手: 你刚才说你的名字是张三。

   总共有 4 条消息
```

## 贡献

如果发现问题或有改进建议，欢迎提交 Issue 或 PR！

