from csv_agent.agent import agent_executor

if __name__ == "__main__":
    question = "who was in cabin c28?"
    print(agent_executor.invoke({"input": question}))
