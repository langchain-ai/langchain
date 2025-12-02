"""演示 LangChain 聊天记忆的持久化存储和恢复。

本示例展示了如何使用 checkpointer 来：
1. 持久化对话状态到内存或文件
2. 恢复之前的对话历史
3. 管理多个独立的对话线程
4. 查看和操作保存的状态

使用方法：
    python demo_memory_persistence.py

环境要求：
    - 需要配置 ANTHROPIC_API_KEY 环境变量（如果使用 Anthropic 模型）
    - 或者修改代码使用其他模型（如 OpenAI、本地模型等）
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

# 尝试导入 SQLiteSaver，如果不可用则跳过相关演示
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("警告: SQLiteSaver 不可用，将跳过 SQLite 演示")


def demo_memory_checkpointer():
    """演示使用内存 checkpointer（适合开发和测试）"""
    print("=" * 60)
    print("演示 1: 使用内存 Checkpointer (MemorySaver)")
    print("=" * 60)

    # 创建一个简单的工具函数
    def get_user_info(name: str) -> str:
        """获取用户信息"""
        return f"用户 {name} 的信息：年龄 25，城市 北京"

    # 创建 checkpointer（内存存储）
    checkpointer = MemorySaver()

    # 创建带记忆的 agent
    agent = create_agent(
        model="anthropic:claude-3-5-sonnet-20241022",  # 使用字符串标识符
        tools=[get_user_info],
        system_prompt="你是一个友好的助手，会记住之前的对话内容。",
        checkpointer=checkpointer,  # 启用记忆持久化
    )

    # 配置对话线程 ID（用于区分不同的对话）
    thread_config = {"configurable": {"thread_id": "conversation-1"}}

    print("\n【第一轮对话】")
    print("-" * 60)
    user_message_1 = "你好，我的名字是张三"
    print(f"用户: {user_message_1}")

    result1 = agent.invoke(
        {"messages": [HumanMessage(content=user_message_1)]},
        config=thread_config,
    )

    ai_response_1 = result1["messages"][-1].content
    print(f"助手: {ai_response_1}")

    print("\n【第二轮对话】")
    print("-" * 60)
    user_message_2 = "我刚才说了我的名字是什么？"
    print(f"用户: {user_message_2}")

    # 注意：这里使用相同的 thread_config，agent 会自动加载之前的对话历史
    result2 = agent.invoke(
        {"messages": [HumanMessage(content=user_message_2)]},
        config=thread_config,
    )

    ai_response_2 = result2["messages"][-1].content
    print(f"助手: {ai_response_2}")

    # 查看完整的对话历史
    print("\n【完整对话历史】")
    print("-" * 60)
    for i, msg in enumerate(result2["messages"], 1):
        if isinstance(msg, HumanMessage):
            print(f"{i}. 用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{i}. 助手: {msg.content}")

    print(f"\n总共有 {len(result2['messages'])} 条消息")

    return agent, checkpointer, thread_config


def demo_sqlite_checkpointer():
    """演示使用 SQLite checkpointer（持久化到文件）"""
    if not SQLITE_AVAILABLE:
        print("\n\n" + "=" * 60)
        print("演示 2: 使用 SQLite Checkpointer (跳过 - 不可用)")
        print("=" * 60)
        print("SQLiteSaver 不可用，跳过此演示")
        return None, None, None

    print("\n\n" + "=" * 60)
    print("演示 2: 使用 SQLite Checkpointer (持久化到文件)")
    print("=" * 60)

    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    # 创建 SQLite checkpointer（持久化到文件）
    # 数据会保存在本地 SQLite 数据库中
    checkpointer = SqliteSaver.from_conn_string(":memory:")  # 内存数据库
    # 如果要持久化到文件，可以使用：
    # checkpointer = SqliteSaver.from_conn_string("chat_memory.db")

    agent = create_agent(
        model="anthropic:claude-3-5-sonnet-20241022",
        tools=[calculate],
        system_prompt="你是一个数学助手，会记住之前的计算历史。",
        checkpointer=checkpointer,
    )

    thread_config = {"configurable": {"thread_id": "math-conversation"}}

    print("\n【第一轮对话】")
    print("-" * 60)
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="帮我计算 10 + 20")]},
        config=thread_config,
    )
    print(f"用户: 帮我计算 10 + 20")
    print(f"助手: {result1['messages'][-1].content}")

    print("\n【第二轮对话】")
    print("-" * 60)
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="刚才的计算结果是多少？")]},
        config=thread_config,
    )
    print(f"用户: 刚才的计算结果是多少？")
    print(f"助手: {result2['messages'][-1].content}")

    print("\n【第三轮对话】")
    print("-" * 60)
    result3 = agent.invoke(
        {"messages": [HumanMessage(content="用刚才的结果乘以 2")]},
        config=thread_config,
    )
    print(f"用户: 用刚才的结果乘以 2")
    print(f"助手: {result3['messages'][-1].content}")

    return agent, checkpointer, thread_config


def demo_multiple_threads():
    """演示多个独立的对话线程"""
    print("\n\n" + "=" * 60)
    print("演示 3: 多个独立的对话线程")
    print("=" * 60)

    checkpointer = MemorySaver()

    agent = create_agent(
        model="anthropic:claude-3-5-sonnet-20241022",
        tools=[],
        system_prompt="你是一个助手。",
        checkpointer=checkpointer,
    )

    # 线程 1：关于天气的对话
    thread1_config = {"configurable": {"thread_id": "weather-thread"}}
    print("\n【线程 1: 天气对话】")
    print("-" * 60)
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="今天天气怎么样？")]},
        config=thread1_config,
    )
    print(f"用户: 今天天气怎么样？")
    print(f"助手: {result1['messages'][-1].content[:100]}...")

    # 线程 2：关于编程的对话（完全独立）
    thread2_config = {"configurable": {"thread_id": "coding-thread"}}
    print("\n【线程 2: 编程对话】")
    print("-" * 60)
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="Python 是什么？")]},
        config=thread2_config,
    )
    print(f"用户: Python 是什么？")
    print(f"助手: {result2['messages'][-1].content[:100]}...")

    # 继续线程 1 的对话（不会受到线程 2 的影响）
    print("\n【继续线程 1 的对话】")
    print("-" * 60)
    result3 = agent.invoke(
        {"messages": [HumanMessage(content="明天会下雨吗？")]},
        config=thread1_config,  # 使用线程 1 的配置
    )
    print(f"用户: 明天会下雨吗？")
    print(f"助手: {result3['messages'][-1].content[:100]}...")

    print("\n【说明】")
    print("-" * 60)
    print("线程 1 和线程 2 的对话历史完全独立，互不影响。")
    print(f"线程 1 有 {len(result3['messages'])} 条消息")
    print(f"线程 2 有 {len(result2['messages'])} 条消息")

    return agent, checkpointer


def demo_state_inspection(checkpointer: MemorySaver, thread_config: dict):
    """演示如何查看和检查保存的状态"""
    print("\n\n" + "=" * 60)
    print("演示 4: 查看和检查保存的状态")
    print("=" * 60)

    # 获取保存的检查点
    checkpoint = checkpointer.get(thread_config)
    if checkpoint:
        print("\n【检查点信息】")
        print("-" * 60)
        print(f"检查点 ID: {checkpoint['id']}")
        print(f"检查点版本: {checkpoint['channel_values']['messages'].get('versions', 'N/A')}")
        print(f"消息数量: {len(checkpoint['channel_values'].get('messages', []))}")

        # 列出所有检查点
        print("\n【所有检查点】")
        print("-" * 60)
        # 注意：MemorySaver 的 list 方法可能需要特定的实现
        print("检查点已保存（可以通过 checkpointer.get() 获取）")

    print("\n【状态恢复】")
    print("-" * 60)
    print("当使用相同的 thread_id 调用 agent 时，")
    print("状态会自动从 checkpointer 中恢复，无需手动操作。")


def demo_state_management():
    """演示状态管理的高级用法"""
    print("\n\n" + "=" * 60)
    print("演示 5: 状态管理高级用法")
    print("=" * 60)

    checkpointer = MemorySaver()

    agent = create_agent(
        model="anthropic:claude-3-5-sonnet-20241022",
        tools=[],
        checkpointer=checkpointer,
    )

    thread_id = "demo-thread"
    config = {"configurable": {"thread_id": thread_id}}

    # 第一轮对话
    print("\n【创建新对话】")
    print("-" * 60)
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="记住：我喜欢咖啡")]},
        config=config,
    )
    print(f"用户: 记住：我喜欢咖啡")
    print(f"助手: {result1['messages'][-1].content[:80]}...")

    # 验证记忆
    print("\n【验证记忆】")
    print("-" * 60)
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="我喜欢什么？")]},
        config=config,
    )
    print(f"用户: 我喜欢什么？")
    print(f"助手: {result2['messages'][-1].content[:80]}...")

    # 查看状态
    checkpoint = checkpointer.get(config)
    if checkpoint:
        print(f"\n当前状态包含 {len(checkpoint['channel_values'].get('messages', []))} 条消息")

    print("\n【说明】")
    print("-" * 60)
    print("1. 每次调用 agent.invoke() 时，状态会自动保存")
    print("2. 下次使用相同的 thread_id 时，状态会自动恢复")
    print("3. 不同的 thread_id 对应完全独立的对话历史")


def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("LangChain 聊天记忆持久化存储和恢复演示")
    print("=" * 60)

    try:
        # 演示 1: 内存 checkpointer
        agent1, checkpointer1, config1 = demo_memory_checkpointer()

        # 演示 2: SQLite checkpointer
        if SQLITE_AVAILABLE:
            agent2, checkpointer2, config2 = demo_sqlite_checkpointer()

        # 演示 3: 多线程
        agent3, checkpointer3 = demo_multiple_threads()

        # 演示 4: 状态检查
        demo_state_inspection(checkpointer1, config1)

        # 演示 5: 状态管理
        demo_state_management()

        print("\n\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        print("\n关键要点：")
        print("1. checkpointer 用于持久化单个对话线程的状态")
        print("2. thread_id 用于区分不同的对话")
        print("3. 状态会在每次调用后自动保存")
        print("4. 使用相同的 thread_id 会自动恢复之前的对话历史")
        print("5. 支持内存、SQLite、PostgreSQL 等多种存储后端")

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示：")
        print("1. 确保已安装所需的依赖")
        print("2. 确保已配置 API 密钥（如 ANTHROPIC_API_KEY）")
        print("3. 如果使用 SQLite，确保有写入权限")


if __name__ == "__main__":
    main()

