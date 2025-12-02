"""简化版的聊天记忆持久化演示 - 不依赖真实 API。

这个版本展示了核心概念，可以用于理解机制，但不会实际调用 LLM。
"""

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


def demo_basic_concepts():
    """演示基本概念：如何使用 checkpointer 保存和恢复状态"""
    print("=" * 60)
    print("简化演示：聊天记忆持久化的基本概念")
    print("=" * 60)

    # 1. 创建 checkpointer
    checkpointer = MemorySaver()
    print("\n1. 创建了 MemorySaver checkpointer")

    # 2. 模拟保存对话状态
    thread_config = {"configurable": {"thread_id": "demo-thread"}}

    # 模拟第一轮对话的状态
    state1 = {
        "messages": [
            HumanMessage(content="你好，我的名字是张三"),
            AIMessage(content="你好张三！很高兴认识你。"),
        ]
    }

    print("\n2. 模拟第一轮对话状态：")
    print(f"   - 用户: {state1['messages'][0].content}")
    print(f"   - 助手: {state1['messages'][1].content}")

    # 在实际使用中，状态会通过 agent.invoke() 自动保存
    # 这里我们手动演示概念
    print("\n3. 状态已保存到 checkpointer（在实际使用中自动完成）")

    # 3. 模拟第二轮对话（恢复之前的状态）
    print("\n4. 模拟第二轮对话（使用相同的 thread_id）：")
    print("   - 系统会自动加载之前保存的状态")
    print("   - 新的消息会追加到现有消息列表")

    # 模拟完整的状态（包含之前的所有消息）
    state2 = {
        "messages": [
            HumanMessage(content="你好，我的名字是张三"),
            AIMessage(content="你好张三！很高兴认识你。"),
            HumanMessage(content="我刚才说了我的名字是什么？"),
            AIMessage(content="你刚才说你的名字是张三。"),
        ]
    }

    print("\n5. 完整对话历史：")
    for i, msg in enumerate(state2["messages"], 1):
        if isinstance(msg, HumanMessage):
            print(f"   {i}. 用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"   {i}. 助手: {msg.content}")

    print(f"\n   总共有 {len(state2['messages'])} 条消息")

    # 4. 演示多个线程
    print("\n6. 多个独立对话线程：")
    print("   - thread_id='conversation-1': 关于天气的对话")
    print("   - thread_id='conversation-2': 关于编程的对话")
    print("   - 每个线程的状态完全独立，互不影响")

    return checkpointer, thread_config


def demo_code_structure():
    """展示实际代码结构"""
    print("\n\n" + "=" * 60)
    print("实际代码结构示例")
    print("=" * 60)

    code_example = '''
# 1. 导入必要的模块
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# 2. 创建 checkpointer
checkpointer = MemorySaver()

# 3. 创建带记忆的 agent
agent = create_agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=[...],  # 你的工具列表
    checkpointer=checkpointer,  # 启用记忆持久化
)

# 4. 配置对话线程
thread_config = {"configurable": {"thread_id": "conversation-1"}}

# 5. 第一轮对话
result1 = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    config=thread_config,
)

# 6. 第二轮对话（自动恢复之前的记忆）
result2 = agent.invoke(
    {"messages": [HumanMessage(content="我刚才说了什么？")]},
    config=thread_config,  # 相同的 thread_id
)
# 此时 result2["messages"] 包含所有之前的消息
'''

    print(code_example)


def demo_key_concepts():
    """解释关键概念"""
    print("\n\n" + "=" * 60)
    print("关键概念说明")
    print("=" * 60)

    concepts = [
        {
            "概念": "Checkpointer",
            "说明": "用于持久化单个对话线程的状态（聊天记忆）",
            "用途": "保存和恢复对话历史",
        },
        {
            "概念": "thread_id",
            "说明": "对话线程的唯一标识符",
            "用途": "区分不同的对话，每个 thread_id 对应独立的对话历史",
        },
        {
            "概念": "AgentState.messages",
            "说明": "存储所有对话消息的列表",
            "用途": "通过 add_messages reducer 自动合并新旧消息",
        },
        {
            "概念": "MemorySaver",
            "说明": "内存存储实现（适合开发和测试）",
            "用途": "数据保存在内存中，程序重启后丢失",
        },
        {
            "概念": "SqliteSaver",
            "说明": "SQLite 数据库存储实现（适合生产环境）",
            "用途": "数据持久化到文件，程序重启后仍然保留",
        },
    ]

    for i, concept in enumerate(concepts, 1):
        print(f"\n{i}. {concept['概念']}")
        print(f"   说明: {concept['说明']}")
        print(f"   用途: {concept['用途']}")


def main():
    """运行简化演示"""
    demo_basic_concepts()
    demo_code_structure()
    demo_key_concepts()

    print("\n\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 查看 demo_memory_persistence.py 了解完整示例")
    print("2. 配置 API 密钥后运行完整示例")
    print("3. 尝试修改 thread_id 来创建多个独立对话")


if __name__ == "__main__":
    main()

