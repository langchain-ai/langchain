'''
LangChain Tools module.

This module provides the ``tool`` decorator and ``ToolRuntime`` for creating
custom tools that can be used with LangChain agents.

Examples:
    Basic Tool::

        from langchain.tools import tool
        from langchain.agents import create_agent

        @tool
        def search(query: str) -> str:
            """Search for information on the web.

            Args:
                query: The search query string.

            Returns:
                str: Search result text.
            """
            return f"Results for: {query}"

        @tool
        def get_weather(location: str) -> str:
            """Get weather information for a location.

            Args:
                location: City name or coordinates.

            Returns:
                str: Weather result text.
            """
            return f"Weather in {location}: Sunny, 72F"

        agent = create_agent(
            model="gpt-4o-mini",
            tools=[search, get_weather],
            system_prompt="You are a helpful assistant."
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "What is the weather in SF?"}]
        })

    Tool with Runtime Context::

        from dataclasses import dataclass
        from langchain.tools import tool, ToolRuntime
        from langchain.agents import create_agent

        @dataclass
        class UserContext:
            user_id: str

        USER_DATABASE = {
            "user_123": {"name": "Alice", "role": "admin"},
            "user_456": {"name": "Bob", "role": "user"},
        }

        @tool
        def get_user_info(runtime: ToolRuntime[UserContext]) -> str:
            """Get information about the current user.

            Args:
                runtime: Tool runtime containing the user context.

            Returns:
                str: User information text.
            """
            user_id = runtime.context.user_id
            if user_id in USER_DATABASE:
                user = USER_DATABASE[user_id]
                return f"User: {user['name']}, Role: {user['role']}"
            return "User not found"

        agent = create_agent(
            model="gpt-4o-mini",
            tools=[get_user_info],
            context_schema=UserContext,
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Who am I?"}]},
            context=UserContext(user_id="user_123")
        )

    Custom Tool Name and Description::

        from langchain.tools import tool

        @tool(
            "web_search",
            description=(
                "Search the web for current information. "
                "Use this for any factual queries."
            ),
        )
        def search(query: str) -> str:
            """Internal search implementation.

            Args:
                query: Query string.

            Returns:
                str: Search result text.
            """
            return f"Results for: {query}"

        print(search.name)  # Output: "web_search"
'''
