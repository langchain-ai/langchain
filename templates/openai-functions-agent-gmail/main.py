from openai_functions_agent.agent import agent_executor

if __name__ == "__main__":
    question = "Write a draft response to the most recent email in my inbox."
    print(agent_executor.invoke({"input": question, "chat_history": []}))
