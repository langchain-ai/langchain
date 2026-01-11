def test_duplicate_middleware_name_last_wins():
    from langchain.agents.middleware import TodoListMiddleware
    from langchain.agents.factory import create_agent
    from langchain_core.language_models.fake import FakeListLLM

    llm = FakeListLLM(responses=["ok"])

    m1 = TodoListMiddleware(system_prompt="default")
    m2 = TodoListMiddleware(system_prompt="custom")

    agent = create_agent(
        model=llm,
        tools=[],
        middleware=[m1, m2],
    )

    todos = [m for m in agent.middleware if m.name == m1.name]
    assert len(todos) == 1
    assert todos[0].system_prompt == "custom"
