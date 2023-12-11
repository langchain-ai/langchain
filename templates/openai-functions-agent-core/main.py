from openai_functions_agent.agent import agent_executor

if __name__ == "__main__":
    question = "what's the log of 2 to the eight? What's the square root of that?"
    print(agent_executor.invoke({"input": question, "chat_history": []}))
