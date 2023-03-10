def main():
    import os

    os.environ["LANGCHAIN_HANDLER"] = "langchain"

    ## Uncomment this if using hosted setup.

    os.environ["LANGCHAIN_ENDPOINT"] = "http://127.0.0.1:8000"

    ## Uncomment this if you want traces to be recorded to "my_session" instead of default.

    # os.environ["LANGCHAIN_SESSION"] = "my_session"

    ## Better to set this environment variable in the terminal
    ## Uncomment this if using hosted version. Replace "my_api_key" with your actual API Key.

    # os.environ["LANGCHAIN_API_KEY"] = "my_api_key"

    import langchain
    from langchain.agents import Tool, initialize_agent, load_tools
    from langchain.llms import OpenAI

    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )

    agent.run("What is 2 raised to .123243 power?")


if __name__ == "__main__":
    main()
