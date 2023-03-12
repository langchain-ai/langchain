from langchain.agents import AgentExecutor


def run_agent(agent: AgentExecutor, data: list):
    results = []
    for datapoint in data:
        try:
            results.append(agent(datapoint))
        except Exception:
            results.append("ERROR")
    return results
