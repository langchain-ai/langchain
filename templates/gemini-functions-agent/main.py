from gemini_functions_agent import agent_executor # from current folder 
# Issue Not solved 
if __name__ == "__main__":
    question = "who won the womens world cup in 2023?"
    print(agent_executor.invoke({"input": question, "chat_history": []}))  # noqa: T201 
