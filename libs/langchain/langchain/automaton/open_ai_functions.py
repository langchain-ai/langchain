from __future__ import annotations

import json
from typing import Sequence

from langchain.base_language import BaseLanguageModel
from langchain.runnables.openai_functions import OpenAIFunctionsRouter
from langchain.schema import BaseMessage, AIMessage
from langchain.schema.runnable import Runnable
from langchain.tools.base import BaseTool
from langchain.tools.convert_to_openai import format_tool_to_openai_function


# def create_action_taking_llm(
#     llm: BaseLanguageModel,
#     *,
#     tools: Sequence[BaseTool] = (),
#     stop: Sequence[str] | None = None,
#     invoke_function: bool = True,
# ) -> Runnable:
#     """A chain that can create an action.
#
#     Args:
#         llm: The language model to use.
#         tools: The tools to use.
#         stop: The stop tokens to use.
#         invoke_function: Whether to invoke the function.
#
#     Returns:
#         a segment of a runnable that take an action.
#     """
#
#     openai_funcs = [format_tool_to_openai_function(tool_) for tool_ in tools]
#
#     def _interpret_message(message: BaseMessage) -> ActingResult:
#         """Interpret a message."""
#         if (
#             isinstance(message, AIMessage)
#             and "function_call" in message.additional_kwargs
#         ):
#             if invoke_function:
#                 result = invoke_from_function.invoke(  # TODO: fixme using invoke
#                     message
#                 )
#             else:
#                 result = None
#             return {
#                 "message": message,
#                 "function_call": {
#                     "name": message.additional_kwargs["function_call"]["name"],
#                     "arguments": json.loads(
#                         message.additional_kwargs["function_call"]["arguments"]
#                     ),
#                     "result": result,
#                     # Check this works.
#                     # "result": message.additional_kwargs["function_call"]
#                     #           | invoke_from_function,
#                 },
#             }
#         else:
#             return {
#                 "message": message,
#                 "function_call": None,
#             }
#
#     invoke_from_function = OpenAIFunctionsRouter(
#         functions=openai_funcs,
#         runnables={
#             openai_func["name"]: tool_
#             for openai_func, tool_ in zip(openai_funcs, tools)
#         },
#     )
#
#     if stop:
#         _llm = llm.bind(stop=stop)
#     else:
#         _llm = llm
#
#     chain = _llm.bind(functions=openai_funcs) | _interpret_message
#     return chain
#
#
# def _interpret_message(message: BaseMessage) -> ActingResult:
#     """Interpret a message."""
#     if isinstance(message, AIMessage) and "function_call" in message.additional_kwargs:
#         raise NotImplementedError()
#         # if invoke_function:
#         #     result = invoke_from_function.invoke(message)  # TODO: fixme using invoke
#         # else:
#         #     result = None
#         # return {
#         #     "message": message,
#         #     "function_call": {
#         #         "name": message.additional_kwargs["function_call"]["name"],
#         #         "arguments": json.loads(
#         #             message.additional_kwargs["function_call"]["arguments"]
#         #         ),
#         #         "result": result,
#         #         # Check this works.
#         #         # "result": message.additional_kwargs["function_call"]
#         #         #           | invoke_from_function,
#         #     },
#         # }
#     else:
#         return {
#             "message": message,
#             "function_call": None,
#         }
