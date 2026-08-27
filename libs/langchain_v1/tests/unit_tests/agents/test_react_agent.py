# import dataclasses
# import inspect
# from types import UnionType
# from typing import (
#     Annotated,
#     Union,
# )

# import pytest
# from langchain_core.language_models import BaseChatModel
# from langchain_core.messages import (
#     AIMessage,
#     HumanMessage,
#     MessageLikeRepresentation,
#     RemoveMessage,
#     SystemMessage,
#     ToolCall,
#     ToolMessage,
# )
# from langchain_core.runnables import RunnableConfig, RunnableLambda
# from langchain_core.tools import BaseTool, InjectedToolCallId, ToolException
# from langchain_core.tools import tool as dec_tool
# from langgraph.checkpoint.base import BaseCheckpointSaver
# from langgraph.graph import START, MessagesState, StateGraph
# from langgraph.graph.message import REMOVE_ALL_MESSAGES
# from langgraph.runtime import Runtime
# from langgraph.store.base import BaseStore
# from langgraph.store.memory import InMemoryStore
# from langgraph.types import Command, Interrupt, interrupt
# from pydantic import BaseModel, Field
# from typing_extensions import TypedDict

# from langchain.agents import (
#     AgentState,
#     create_agent,
# )
# from langchain.tools import (
#     ToolNode,
#     InjectedState,
#     InjectedStore,
# )
# from langchain.tools.tool_node import (
#     _get_state_args,
#     _infer_handled_types,
# )

# from tests.unit_tests.agents.any_str import AnyStr
# from tests.unit_tests.agents.messages import _AnyIdHumanMessage, _AnyIdToolMessage
# from tests.unit_tests.agents.model import FakeToolCallingModel

# pytestmark = pytest.mark.anyio


# def test_no_prompt(sync_checkpointer: BaseCheckpointSaver) -> None:
#     model = FakeToolCallingModel()

#     agent = create_agent(
#         model,
#         [],
#         checkpointer=sync_checkpointer,
#     )
#     inputs = [HumanMessage("hi?")]
#     thread = {"configurable": {"thread_id": "123"}}
#     response = agent.invoke({"messages": inputs}, thread, debug=True)
#     expected_response = {"messages": [*inputs, AIMessage(content="hi?", id="0")]}
#     assert response == expected_response

#     saved = sync_checkpointer.get_tuple(thread)
#     assert saved is not None
#     assert saved.checkpoint["channel_values"] == {
#         "messages": [
#             _AnyIdHumanMessage(content="hi?"),
#             AIMessage(content="hi?", id="0"),
#         ],
#     }
#     assert saved.metadata == {
#         "parents": {},
#         "source": "loop",
#         "step": 1,
#     }
#     assert saved.pending_writes == []


# async def test_no_prompt_async(async_checkpointer: BaseCheckpointSaver) -> None:
#     model = FakeToolCallingModel()

#     agent = create_agent(model, [], checkpointer=async_checkpointer)
#     inputs = [HumanMessage("hi?")]
#     thread = {"configurable": {"thread_id": "123"}}
#     response = await agent.ainvoke({"messages": inputs}, thread, debug=True)
#     expected_response = {"messages": [*inputs, AIMessage(content="hi?", id="0")]}
#     assert response == expected_response

#     saved = await async_checkpointer.aget_tuple(thread)
#     assert saved is not None
#     assert saved.checkpoint["channel_values"] == {
#         "messages": [
#             _AnyIdHumanMessage(content="hi?"),
#             AIMessage(content="hi?", id="0"),
#         ],
#     }
#     assert saved.metadata == {
#         "parents": {},
#         "source": "loop",
#         "step": 1,
#     }
#     assert saved.pending_writes == []


# def test_system_message_prompt() -> None:
#     prompt = SystemMessage(content="Foo")
#     agent = create_agent(FakeToolCallingModel(), [], system_prompt=prompt)
#     inputs = [HumanMessage("hi?")]
#     response = agent.invoke({"messages": inputs})
#     expected_response = {"messages": [*inputs, AIMessage(content="Foo-hi?", id="0", tool_calls=[])]}
#     assert response == expected_response


# def test_string_prompt() -> None:
#     prompt = "Foo"
#     agent = create_agent(FakeToolCallingModel(), [], system_prompt=prompt)
#     inputs = [HumanMessage("hi?")]
#     response = agent.invoke({"messages": inputs})
#     expected_response = {"messages": [*inputs, AIMessage(content="Foo-hi?", id="0", tool_calls=[])]}
#     assert response == expected_response


# def test_callable_prompt() -> None:
#     def prompt(state):
#         modified_message = f"Bar {state['messages'][-1].content}"
#         return [HumanMessage(content=modified_message)]

#     agent = create_agent(FakeToolCallingModel(), [], system_prompt=prompt)
#     inputs = [HumanMessage("hi?")]
#     response = agent.invoke({"messages": inputs})
#     expected_response = {"messages": [*inputs, AIMessage(content="Bar hi?", id="0")]}
#     assert response == expected_response


# async def test_callable_prompt_async() -> None:
#     async def prompt(state):
#         modified_message = f"Bar {state['messages'][-1].content}"
#         return [HumanMessage(content=modified_message)]

#     agent = create_agent(FakeToolCallingModel(), [], system_prompt=prompt)
#     inputs = [HumanMessage("hi?")]
#     response = await agent.ainvoke({"messages": inputs})
#     expected_response = {"messages": [*inputs, AIMessage(content="Bar hi?", id="0")]}
#     assert response == expected_response


# def test_runnable_prompt() -> None:
#     prompt = RunnableLambda(
#         lambda state: [HumanMessage(content=f"Baz {state['messages'][-1].content}")]
#     )

#     agent = create_agent(FakeToolCallingModel(), [], system_prompt=prompt)
#     inputs = [HumanMessage("hi?")]
#     response = agent.invoke({"messages": inputs})
#     expected_response = {"messages": [*inputs, AIMessage(content="Baz hi?", id="0")]}
#     assert response == expected_response


# def test_prompt_with_store() -> None:
#     def add(a: int, b: int):
#         """Adds a and b"""
#         return a + b

#     in_memory_store = InMemoryStore()
#     in_memory_store.put(("memories", "1"), "user_name", {"data": "User name is Alice"})
#     in_memory_store.put(("memories", "2"), "user_name", {"data": "User name is Bob"})

#     def prompt(state, config, *, store):
#         user_id = config["configurable"]["user_id"]
#         system_str = store.get(("memories", user_id), "user_name").value["data"]
#         return [SystemMessage(system_str)] + state["messages"]

#     def prompt_no_store(state, config):
#         return SystemMessage("foo") + state["messages"]

#     model = FakeToolCallingModel()

#     # test state modifier that uses store works
#     agent = create_agent(
#         model,
#         [add],
#         prompt=prompt,
#         store=in_memory_store,
#     )
#     response = agent.invoke({"messages": [("user", "hi")]}, {"configurable": {"user_id": "1"}})
#     assert response["messages"][-1].content == "User name is Alice-hi"

#     # test state modifier that doesn't use store works
#     agent = create_agent(
#         model,
#         [add],
#         prompt=prompt_no_store,
#         store=in_memory_store,
#     )
#     response = agent.invoke({"messages": [("user", "hi")]}, {"configurable": {"user_id": "2"}})
#     assert response["messages"][-1].content == "foo-hi"


# async def test_prompt_with_store_async() -> None:
#     async def add(a: int, b: int):
#         """Adds a and b"""
#         return a + b

#     in_memory_store = InMemoryStore()
#     await in_memory_store.aput(("memories", "1"), "user_name", {"data": "User name is Alice"})
#     await in_memory_store.aput(("memories", "2"), "user_name", {"data": "User name is Bob"})

#     async def prompt(state, config, *, store):
#         user_id = config["configurable"]["user_id"]
#         system_str = (await store.aget(("memories", user_id), "user_name")).value["data"]
#         return [SystemMessage(system_str)] + state["messages"]

#     async def prompt_no_store(state, config):
#         return SystemMessage("foo") + state["messages"]

#     model = FakeToolCallingModel()

#     # test state modifier that uses store works
#     agent = create_agent(model, [add], system_prompt=prompt, store=in_memory_store)
#     response = await agent.ainvoke(
#         {"messages": [("user", "hi")]}, {"configurable": {"user_id": "1"}}
#     )
#     assert response["messages"][-1].content == "User name is Alice-hi"

#     # test state modifier that doesn't use store works
#     agent = create_agent(model, [add], system_prompt=prompt_no_store, store=in_memory_store)
#     response = await agent.ainvoke(
#         {"messages": [("user", "hi")]}, {"configurable": {"user_id": "2"}}
#     )
#     assert response["messages"][-1].content == "foo-hi"


# @pytest.mark.parametrize("tool_style", ["openai", "anthropic"])
# @pytest.mark.parametrize("include_builtin", [True, False])
# def test_model_with_tools(tool_style: str, include_builtin: bool) -> None:
#     model = FakeToolCallingModel(tool_style=tool_style)

#     @dec_tool
#     def tool1(some_val: int) -> str:
#         """Tool 1 docstring."""
#         return f"Tool 1: {some_val}"

#     @dec_tool
#     def tool2(some_val: int) -> str:
#         """Tool 2 docstring."""
#         return f"Tool 2: {some_val}"

#     tools: list[BaseTool | dict] = [tool1, tool2]
#     if include_builtin:
#         tools.append(
#             {
#                 "type": "mcp",
#                 "server_label": "atest_sever",
#                 "server_url": "https://some.mcp.somewhere.com/sse",
#                 "headers": {"foo": "bar"},
#                 "allowed_tools": [
#                     "mcp_tool_1",
#                     "set_active_account",
#                     "get_url_markdown",
#                     "get_url_screenshot",
#                 ],
#                 "require_approval": "never",
#             }
#         )
#     # check valid agent constructor
#     with pytest.raises(ValueError):
#         create_agent(
#             model.bind_tools(tools),
#             tools,
#         )


# # Test removed: _validate_chat_history function no longer exists
# # def test__validate_messages() -> None:
# #     pass


# def test__infer_handled_types() -> None:
#     def handle(e) -> str:  # type: ignore
#         return ""

#     def handle2(e: Exception) -> str:
#         return ""

#     def handle3(e: ValueError | ToolException) -> str:
#         return ""

#     def handle4(e: Union[ValueError, ToolException]) -> str:
#         return ""

#     class Handler:
#         def handle(self, e: ValueError) -> str:
#             return ""

#     handle5 = Handler().handle

#     def handle6(e: Union[Union[TypeError, ValueError], ToolException]) -> str:
#         return ""

#     expected: tuple = (Exception,)
#     actual = _infer_handled_types(handle)
#     assert expected == actual

#     expected = (Exception,)
#     actual = _infer_handled_types(handle2)
#     assert expected == actual

#     expected = (ValueError, ToolException)
#     actual = _infer_handled_types(handle3)
#     assert expected == actual

#     expected = (ValueError, ToolException)
#     actual = _infer_handled_types(handle4)
#     assert expected == actual

#     expected = (ValueError,)
#     actual = _infer_handled_types(handle5)
#     assert expected == actual

#     expected = (TypeError, ValueError, ToolException)
#     actual = _infer_handled_types(handle6)
#     assert expected == actual

#     with pytest.raises(ValueError):

#         def handler(e: str) -> str:
#             return ""

#         _infer_handled_types(handler)

#     with pytest.raises(ValueError):

#         def handler(e: list[Exception]) -> str:
#             return ""

#         _infer_handled_types(handler)

#     with pytest.raises(ValueError):

#         def handler(e: Union[str, int]) -> str:
#             return ""

#         _infer_handled_types(handler)


# def test_react_agent_with_structured_response() -> None:
#     class WeatherResponse(BaseModel):
#         temperature: float = Field(description="The temperature in fahrenheit")

#     tool_calls = [
#         [{"args": {}, "id": "1", "name": "get_weather"}],
#         [{"name": "WeatherResponse", "id": "2", "args": {"temperature": 75}}],
#     ]

#     def get_weather() -> str:
#         """Get the weather"""
#         return "The weather is sunny and 75°F."

#     expected_structured_response = WeatherResponse(temperature=75)
#     model = FakeToolCallingModel(
#         tool_calls=tool_calls, structured_response=expected_structured_response
#     )
#     agent = create_agent(
#         model,
#         [get_weather],
#         response_format=WeatherResponse,
#     )
#     response = agent.invoke({"messages": [HumanMessage("What's the weather?")]})
#     assert response["structured_response"] == expected_structured_response
#     assert len(response["messages"]) == 5

#     # Check message types in message history
#     msg_types = [m.type for m in response["messages"]]
#     assert msg_types == [
#         "human",  # "What's the weather?"
#         "ai",  # "What's the weather?"
#         "tool",  # "The weather is sunny and 75°F."
#         "ai",  # structured response
#         "tool",  # artificial tool message
#     ]

#     assert [m.content for m in response["messages"]] == [
#         "What's the weather?",
#         "What's the weather?",
#         "The weather is sunny and 75°F.",
#         "What's the weather?-What's the weather?-The weather is sunny and 75°F.",
#         "Returning structured response: {'temperature': 75.0}",
#     ]


# class CustomState(AgentState):
#     user_name: str


# def test_react_agent_update_state(
#     sync_checkpointer: BaseCheckpointSaver,
# ) -> None:
#     @dec_tool
#     def get_user_name(tool_call_id: Annotated[str, InjectedToolCallId]):
#         """Retrieve user name"""
#         user_name = interrupt("Please provider user name:")
#         return Command(
#             update={
#                 "user_name": user_name,
#                 "messages": [
#                     ToolMessage("Successfully retrieved user name", tool_call_id=tool_call_id)
#                 ],
#             }
#         )

#     def prompt(state: CustomState):
#         user_name = state.get("user_name")
#         if user_name is None:
#             return state["messages"]

#         system_msg = f"User name is {user_name}"
#         return [{"role": "system", "content": system_msg}] + state["messages"]

#     tool_calls = [[{"args": {}, "id": "1", "name": "get_user_name"}]]
#     model = FakeToolCallingModel(tool_calls=tool_calls)
#     agent = create_agent(
#         model,
#         [get_user_name],
#         state_schema=CustomState,
#         prompt=prompt,
#         checkpointer=sync_checkpointer,
#     )
#     config = {"configurable": {"thread_id": "1"}}
#     # Run until interrupted
#     agent.invoke({"messages": [("user", "what's my name")]}, config)
#     # supply the value for the interrupt
#     response = agent.invoke(Command(resume="Archibald"), config)
#     # confirm that the state was updated
#     assert response["user_name"] == "Archibald"
#     assert len(response["messages"]) == 4
#     tool_message: ToolMessage = response["messages"][-2]
#     assert tool_message.content == "Successfully retrieved user name"
#     assert tool_message.tool_call_id == "1"
#     assert tool_message.name == "get_user_name"


# def test_react_agent_parallel_tool_calls(
#     sync_checkpointer: BaseCheckpointSaver,
# ) -> None:
#     human_assistance_execution_count = 0

#     @dec_tool
#     def human_assistance(query: str) -> str:
#         """Request assistance from a human."""
#         nonlocal human_assistance_execution_count
#         human_response = interrupt({"query": query})
#         human_assistance_execution_count += 1
#         return human_response["data"]

#     get_weather_execution_count = 0

#     @dec_tool
#     def get_weather(location: str) -> str:
#         """Use this tool to get the weather."""
#         nonlocal get_weather_execution_count
#         get_weather_execution_count += 1
#         return "It's sunny!"

#     tool_calls = [
#         [
#             {"args": {"location": "sf"}, "id": "1", "name": "get_weather"},
#             {"args": {"query": "request help"}, "id": "2", "name": "human_assistance"},
#         ],
#         [],
#     ]
#     model = FakeToolCallingModel(tool_calls=tool_calls)
#     agent = create_agent(
#         model,
#         [human_assistance, get_weather],
#         checkpointer=sync_checkpointer,
#     )
#     config = {"configurable": {"thread_id": "1"}}
#     query = "Get user assistance and also check the weather"
#     message_types = []
#     for event in agent.stream({"messages": [("user", query)]}, config, stream_mode="values"):
#         if messages := event.get("messages"):
#             message_types.append([m.type for m in messages])

#     assert message_types == [
#         ["human"],
#         ["human", "ai"],
#         ["human", "ai", "tool"],
#     ]

#     # Resume
#     message_types = []
#     for event in agent.stream(Command(resume={"data": "Hello"}), config, stream_mode="values"):
#         if messages := event.get("messages"):
#             message_types.append([m.type for m in messages])

#     assert message_types == [
#         ["human", "ai"],
#         ["human", "ai", "tool", "tool"],
#         ["human", "ai", "tool", "tool", "ai"],
#     ]

#     assert human_assistance_execution_count == 1
#     assert get_weather_execution_count == 1


# class AgentStateExtraKey(AgentState):
#     foo: int


# def test_create_agent_inject_vars() -> None:
#     """Test that the agent can inject state and store into tool functions."""
#     store = InMemoryStore()
#     namespace = ("test",)
#     store.put(namespace, "test_key", {"bar": 3})

#     def tool1(
#         some_val: int,
#         state: Annotated[dict, InjectedState],
#         store: Annotated[BaseStore, InjectedStore()],
#     ) -> str:
#         """Tool 1 docstring."""
#         store_val = store.get(namespace, "test_key").value["bar"]
#         return some_val + state["foo"] + store_val

#     tool_call = {
#         "name": "tool1",
#         "args": {"some_val": 1},
#         "id": "some 0",
#         "type": "tool_call",
#     }
#     model = FakeToolCallingModel(tool_calls=[[tool_call], []])
#     agent = create_agent(
#         model,
#         ToolNode([tool1], handle_tool_errors=False),
#         state_schema=AgentStateExtraKey,
#         store=store,
#     )
#     result = agent.invoke({"messages": [{"role": "user", "content": "hi"}], "foo": 2})
#     assert result["messages"] == [
#         _AnyIdHumanMessage(content="hi"),
#         AIMessage(content="hi", tool_calls=[tool_call], id="0"),
#         _AnyIdToolMessage(content="6", name="tool1", tool_call_id="some 0"),
#         AIMessage("hi-hi-6", id="1"),
#     ]
#     assert result["foo"] == 2


# async def test_return_direct() -> None:
#     @dec_tool(return_direct=True)
#     def tool_return_direct(input: str) -> str:
#         """A tool that returns directly."""
#         return f"Direct result: {input}"

#     @dec_tool
#     def tool_normal(input: str) -> str:
#         """A normal tool."""
#         return f"Normal result: {input}"

#     first_tool_call = [
#         ToolCall(
#             name="tool_return_direct",
#             args={"input": "Test direct"},
#             id="1",
#         ),
#     ]
#     expected_ai = AIMessage(
#         content="Test direct",
#         id="0",
#         tool_calls=first_tool_call,
#     )
#     model = FakeToolCallingModel(tool_calls=[first_tool_call, []])
#     agent = create_agent(
#         model,
#         [tool_return_direct, tool_normal],
#     )

#     # Test direct return for tool_return_direct
#     result = agent.invoke({"messages": [HumanMessage(content="Test direct", id="hum0")]})
#     assert result["messages"] == [
#         HumanMessage(content="Test direct", id="hum0"),
#         expected_ai,
#         ToolMessage(
#             content="Direct result: Test direct",
#             name="tool_return_direct",
#             tool_call_id="1",
#             id=result["messages"][2].id,
#         ),
#     ]
#     second_tool_call = [
#         ToolCall(
#             name="tool_normal",
#             args={"input": "Test normal"},
#             id="2",
#         ),
#     ]
#     model = FakeToolCallingModel(tool_calls=[second_tool_call, []])
#     agent = create_agent(model, [tool_return_direct, tool_normal])
#     result = agent.invoke({"messages": [HumanMessage(content="Test normal", id="hum1")]})
#     assert result["messages"] == [
#         HumanMessage(content="Test normal", id="hum1"),
#         AIMessage(content="Test normal", id="0", tool_calls=second_tool_call),
#         ToolMessage(
#             content="Normal result: Test normal",
#             name="tool_normal",
#             tool_call_id="2",
#             id=result["messages"][2].id,
#         ),
#         AIMessage(content="Test normal-Test normal-Normal result: Test normal", id="1"),
#     ]

#     both_tool_calls = [
#         ToolCall(
#             name="tool_return_direct",
#             args={"input": "Test both direct"},
#             id="3",
#         ),
#         ToolCall(
#             name="tool_normal",
#             args={"input": "Test both normal"},
#             id="4",
#         ),
#     ]
#     model = FakeToolCallingModel(tool_calls=[both_tool_calls, []])
#     agent = create_agent(model, [tool_return_direct, tool_normal])
#     result = agent.invoke({"messages": [HumanMessage(content="Test both", id="hum2")]})
#     assert result["messages"] == [
#         HumanMessage(content="Test both", id="hum2"),
#         AIMessage(content="Test both", id="0", tool_calls=both_tool_calls),
#         ToolMessage(
#             content="Direct result: Test both direct",
#             name="tool_return_direct",
#             tool_call_id="3",
#             id=result["messages"][2].id,
#         ),
#         ToolMessage(
#             content="Normal result: Test both normal",
#             name="tool_normal",
#             tool_call_id="4",
#             id=result["messages"][3].id,
#         ),
#     ]


# def test__get_state_args() -> None:
#     class Schema1(BaseModel):
#         a: Annotated[str, InjectedState]

#     class Schema2(Schema1):
#         b: Annotated[int, InjectedState("bar")]

#     @dec_tool(args_schema=Schema2)
#     def foo(a: str, b: int) -> float:
#         """return"""
#         return 0.0

#     assert _get_state_args(foo) == {"a": None, "b": "bar"}


# def test_inspect_react() -> None:
#     model = FakeToolCallingModel(tool_calls=[])
#     agent = create_agent(model, [])
#     inspect.getclosurevars(agent.nodes["agent"].bound.func)


# def test_react_with_subgraph_tools(
#     sync_checkpointer: BaseCheckpointSaver,
# ) -> None:
#     class State(TypedDict):
#         a: int
#         b: int

#     class Output(TypedDict):
#         result: int

#     # Define the subgraphs
#     def add(state):
#         return {"result": state["a"] + state["b"]}

#     add_subgraph = (
#         StateGraph(State, output_schema=Output).add_node(add).add_edge(START, "add").compile()
#     )

#     def multiply(state):
#         return {"result": state["a"] * state["b"]}

#     multiply_subgraph = (
#         StateGraph(State, output_schema=Output)
#         .add_node(multiply)
#         .add_edge(START, "multiply")
#         .compile()
#     )

#     multiply_subgraph.invoke({"a": 2, "b": 3})

#     # Add subgraphs as tools

#     def addition(a: int, b: int):
#         """Add two numbers"""
#         return add_subgraph.invoke({"a": a, "b": b})["result"]

#     def multiplication(a: int, b: int):
#         """Multiply two numbers"""
#         return multiply_subgraph.invoke({"a": a, "b": b})["result"]

#     model = FakeToolCallingModel(
#         tool_calls=[
#             [
#                 {"args": {"a": 2, "b": 3}, "id": "1", "name": "addition"},
#                 {"args": {"a": 2, "b": 3}, "id": "2", "name": "multiplication"},
#             ],
#             [],
#         ]
#     )
#     tool_node = ToolNode([addition, multiplication], handle_tool_errors=False)
#     agent = create_agent(
#         model,
#         tool_node,
#         checkpointer=sync_checkpointer,
#     )
#     result = agent.invoke(
#         {"messages": [HumanMessage(content="What's 2 + 3 and 2 * 3?")]},
#         config={"configurable": {"thread_id": "1"}},
#     )
#     assert result["messages"] == [
#         _AnyIdHumanMessage(content="What's 2 + 3 and 2 * 3?"),
#         AIMessage(
#             content="What's 2 + 3 and 2 * 3?",
#             id="0",
#             tool_calls=[
#                 ToolCall(name="addition", args={"a": 2, "b": 3}, id="1"),
#                 ToolCall(name="multiplication", args={"a": 2, "b": 3}, id="2"),
#             ],
#         ),
#         ToolMessage(content="5", name="addition", tool_call_id="1", id=result["messages"][2].id),
#         ToolMessage(
#             content="6",
#             name="multiplication",
#             tool_call_id="2",
#             id=result["messages"][3].id,
#         ),
#         AIMessage(content="What's 2 + 3 and 2 * 3?-What's 2 + 3 and 2 * 3?-5-6", id="1"),
#     ]


# def test_react_agent_subgraph_streaming_sync() -> None:
#     """Test React agent streaming when used as a subgraph node sync version"""

#     @dec_tool
#     def get_weather(city: str) -> str:
#         """Get the weather of a city."""
#         return f"The weather of {city} is sunny."

#     # Create a React agent
#     model = FakeToolCallingModel(
#         tool_calls=[
#             [{"args": {"city": "Tokyo"}, "id": "1", "name": "get_weather"}],
#             [],
#         ]
#     )

#     agent = create_agent(
#         model,
#         tools=[get_weather],
#         prompt="You are a helpful travel assistant.",
#     )

#     # Create a subgraph that uses the React agent as a node
#     def react_agent_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
#         """Node that runs the React agent and collects streaming output."""
#         collected_content = ""

#         # Stream the agent output and collect content
#         for msg_chunk, _msg_metadata in agent.stream(
#             {"messages": [("user", state["messages"][-1].content)]},
#             config,
#             stream_mode="messages",
#         ):
#             if hasattr(msg_chunk, "content") and msg_chunk.content:
#                 collected_content += msg_chunk.content

#         return {"messages": [("assistant", collected_content)]}

#     # Create the main workflow with the React agent as a subgraph node
#     workflow = StateGraph(MessagesState)
#     workflow.add_node("react_agent", react_agent_node)
#     workflow.add_edge(START, "react_agent")
#     workflow.add_edge("react_agent", "__end__")
#     compiled_workflow = workflow.compile()

#     # Test the streaming functionality
#     result = compiled_workflow.invoke({"messages": [("user", "What is the weather in Tokyo?")]})

#     # Verify the result contains expected structure
#     assert len(result["messages"]) == 2
#     assert result["messages"][0].content == "What is the weather in Tokyo?"
#     assert "assistant" in str(result["messages"][1])

#     # Test streaming with subgraphs = True
#     result = compiled_workflow.invoke(
#         {"messages": [("user", "What is the weather in Tokyo?")]},
#         subgraphs=True,
#     )
#     assert len(result["messages"]) == 2

#     events = []
#     for event in compiled_workflow.stream(
#         {"messages": [("user", "What is the weather in Tokyo?")]},
#         stream_mode="messages",
#         subgraphs=False,
#     ):
#         events.append(event)

#     assert len(events) == 0

#     events = []
#     for event in compiled_workflow.stream(
#         {"messages": [("user", "What is the weather in Tokyo?")]},
#         stream_mode="messages",
#         subgraphs=True,
#     ):
#         events.append(event)

#     assert len(events) == 3
#     namespace, (msg, metadata) = events[0]
#     # FakeToolCallingModel returns a single AIMessage with tool calls
#     # The content of the AIMessage reflects the input message
#     assert msg.content.startswith("You are a helpful travel assistant")
#     namespace, (msg, metadata) = events[1]  # ToolMessage
#     assert msg.content.startswith("The weather of Tokyo is sunny.")


# async def test_react_agent_subgraph_streaming() -> None:
#     """Test React agent streaming when used as a subgraph node."""

#     @dec_tool
#     def get_weather(city: str) -> str:
#         """Get the weather of a city."""
#         return f"The weather of {city} is sunny."

#     # Create a React agent
#     model = FakeToolCallingModel(
#         tool_calls=[
#             [{"args": {"city": "Tokyo"}, "id": "1", "name": "get_weather"}],
#             [],
#         ]
#     )

#     agent = create_agent(
#         model,
#         tools=[get_weather],
#         prompt="You are a helpful travel assistant.",
#     )

#     # Create a subgraph that uses the React agent as a node
#     async def react_agent_node(state: MessagesState, config: RunnableConfig) -> MessagesState:
#         """Node that runs the React agent and collects streaming output."""
#         collected_content = ""

#         # Stream the agent output and collect content
#         async for msg_chunk, _msg_metadata in agent.astream(
#             {"messages": [("user", state["messages"][-1].content)]},
#             config,
#             stream_mode="messages",
#         ):
#             if hasattr(msg_chunk, "content") and msg_chunk.content:
#                 collected_content += msg_chunk.content

#         return {"messages": [("assistant", collected_content)]}

#     # Create the main workflow with the React agent as a subgraph node
#     workflow = StateGraph(MessagesState)
#     workflow.add_node("react_agent", react_agent_node)
#     workflow.add_edge(START, "react_agent")
#     workflow.add_edge("react_agent", "__end__")
#     compiled_workflow = workflow.compile()

#     # Test the streaming functionality
#     result = await compiled_workflow.ainvoke(
#         {"messages": [("user", "What is the weather in Tokyo?")]}
#     )

#     # Verify the result contains expected structure
#     assert len(result["messages"]) == 2
#     assert result["messages"][0].content == "What is the weather in Tokyo?"
#     assert "assistant" in str(result["messages"][1])

#     # Test streaming with subgraphs = True
#     result = await compiled_workflow.ainvoke(
#         {"messages": [("user", "What is the weather in Tokyo?")]},
#         subgraphs=True,
#     )
#     assert len(result["messages"]) == 2

#     events = []
#     async for event in compiled_workflow.astream(
#         {"messages": [("user", "What is the weather in Tokyo?")]},
#         stream_mode="messages",
#         subgraphs=False,
#     ):
#         events.append(event)

#     assert len(events) == 0

#     events = []
#     async for event in compiled_workflow.astream(
#         {"messages": [("user", "What is the weather in Tokyo?")]},
#         stream_mode="messages",
#         subgraphs=True,
#     ):
#         events.append(event)

#     assert len(events) == 3
#     namespace, (msg, metadata) = events[0]
#     # FakeToolCallingModel returns a single AIMessage with tool calls
#     # The content of the AIMessage reflects the input message
#     assert msg.content.startswith("You are a helpful travel assistant")
#     namespace, (msg, metadata) = events[1]  # ToolMessage
#     assert msg.content.startswith("The weather of Tokyo is sunny.")


# def test_tool_node_node_interrupt(
#     sync_checkpointer: BaseCheckpointSaver,
# ) -> None:
#     def tool_normal(some_val: int) -> str:
#         """Tool docstring."""
#         return "normal"

#     def tool_interrupt(some_val: int) -> str:
#         """Tool docstring."""
#         return interrupt("provide value for foo")

#     # test inside react agent
#     model = FakeToolCallingModel(
#         tool_calls=[
#             [
#                 ToolCall(name="tool_interrupt", args={"some_val": 0}, id="1"),
#                 ToolCall(name="tool_normal", args={"some_val": 1}, id="2"),
#             ],
#             [],
#         ]
#     )
#     config = {"configurable": {"thread_id": "1"}}
#     agent = create_agent(
#         model,
#         [tool_interrupt, tool_normal],
#         checkpointer=sync_checkpointer,
#     )
#     result = agent.invoke({"messages": [HumanMessage("hi?")]}, config)
#     expected_messages = [
#         _AnyIdHumanMessage(content="hi?"),
#         AIMessage(
#             content="hi?",
#             id="0",
#             tool_calls=[
#                 {
#                     "name": "tool_interrupt",
#                     "args": {"some_val": 0},
#                     "id": "1",
#                     "type": "tool_call",
#                 },
#                 {
#                     "name": "tool_normal",
#                     "args": {"some_val": 1},
#                     "id": "2",
#                     "type": "tool_call",
#                 },
#             ],
#         ),
#         _AnyIdToolMessage(content="normal", name="tool_normal", tool_call_id="2"),
#     ]
#     assert result["messages"] == expected_messages

#     state = agent.get_state(config)
#     assert state.next == ("tools",)
#     task = state.tasks[0]
#     assert task.name == "tools"
#     assert task.interrupts == (
#         Interrupt(
#             value="provide value for foo",
#             id=AnyStr(),
#         ),
#     )
