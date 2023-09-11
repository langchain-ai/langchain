# from typing import Mapping, Sequence, Optional, List
#
# from langchain.automaton.typedefs import MessageLike
# from langchain.schema.runnable import Runnable, RunnableConfig
#
#
# def step(
#     sates: Mapping[str, Runnable],
#     messages: Sequence[MessageLike],
#     *,
#     config: Optional[RunnableConfig] = None,
# ) -> List[MessageLike]:
#     """Step through the environment."""
#     last_messages = messages[-1] if messages else None
#
#     if not last_messages:
#         return []
#
#     match last_message:
#         case AIMessage():
#             return []
#         case AgentFinish():
#             return []
#         case HumanMessage():
#             return [RetrievalRequest(query=last_message.content)]
#         case RetrievalRequest():
#             return [self.retriever.invoke(last_message, config=config)]
#         case _:
#             return self.llm_program.invoke(messages, config=config)
#
#
#
# class Agent:
#     def step(
#         self, messages: Sequence[MessageLike], *, config: Optional[RunnableConfig]
#      ) -> List[MessageLike]:
#         """Take a single step with the agent."""
#
#