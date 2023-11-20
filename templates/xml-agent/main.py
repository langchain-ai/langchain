from xml_agent.agent import agent_executor

if __name__ == "__main__":
    question = "who won the womens world cup in 2023?"
    print(agent_executor.invoke({"question": question, "chat_history": []}))
