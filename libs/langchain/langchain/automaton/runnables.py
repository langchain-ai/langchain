"""Module contains useful runnables for agents."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

from langchain.automaton.typedefs import (
    FunctionCall,
    FunctionResult,
    MessageLike,
)
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import AIMessage, BaseMessage, PromptValue
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import BaseOutputParser
from langchain.schema.runnable import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
    patch_config,
)
from langchain.tools import BaseTool

T = TypeVar("T")


def _to_message(result: Union[BaseMessage, str]) -> BaseMessage:
    """Convert to a list of messages."""
    if isinstance(result, BaseMessage):
        return result
    elif isinstance(result, str):
        return AIMessage(content=result)
    else:
        raise NotImplementedError(f"Unsupported type {type(result)}")


def _to_list(element: Union[None, T, List[T]]) -> List[T]:
    """Convert to a sequence."""
    if element is None:
        return []
    elif isinstance(element, list):
        return element
    else:
        return [element]


def _to_runnable_parser(parser: Optional[BaseOutputParser]) -> Runnable:
    """Adapt a parser to a runnable."""
    if parser is None:
        # Then create a runnable that returns no messages
        return RunnableLambda(lambda *args, **kwargs: None)
    elif isinstance(parser, Runnable):
        return parser
    elif isinstance(parser, Callable):
        return RunnableLambda(parser)
    else:
        raise ValueError(f"Expected BaseOutputParser, got {parser}")


def _concatenate_head_and_tail(intermediate_input: Dict[str, Any]) -> List[BaseMessage]:
    """Concatenate head and tail into a single list."""
    head = _to_list(intermediate_input["head"])
    tail = _to_list(intermediate_input["tail"])
    return head + tail


def _apply_and_concat(
    head: Union[Runnable, Callable], tail: Union[Runnable, Callable]
) -> Runnable:
    """Apply head and tail and concatenate the results.

    Note: Probably generalize to _apply(funcs) and _concatenate runnables

    Args:
        head: A runnable or callable
        tail: A runnable or callable

    Returns:
        A runnable that applies head and tail and concatenates the results in order.
    """
    head_ = head if isinstance(head, Runnable) else RunnableLambda(head)
    tail_ = tail if isinstance(tail, Runnable) else RunnableLambda(tail)

    return (
        RunnableMap(
            steps={
                "head": head_,
                "tail": tail_,
            }
        )
        | _concatenate_head_and_tail
    )


# PUBLIC API


def create_tool_invoker(
    tools: Sequence[BaseTool],
) -> Runnable[MessageLike, Optional[FunctionResult]]:
    """See if possible to re-write with router

    TODO:
    * re-write with router
    * potentially remove hack replace MessageLike with FunctionCall, requires
      a branching runnable
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def func(
        function_call: MessageLike,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Optional[FunctionResult]:
        """A function that can invoke a tool using .run"""
        if not isinstance(
            function_call, FunctionCall
        ):  # TODO(Hack): Workaround lack of conditional apply
            return None
        try:
            tool = tools_by_name[function_call.name]
        except KeyError:
            raise AssertionError(f"No such tool: {function_call.name}")
        try:
            result = tool.invoke(
                function_call.named_arguments or {},
                patch_config(config, callbacks=run_manager.get_child()),
            )
            error = None
        except Exception as e:
            result = None
            error = repr(e) + repr(function_call.named_arguments)

        return FunctionResult(name=function_call.name, result=result, error=error)

    async def afunc(
        function_call: MessageLike,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Optional[FunctionResult]:
        """A function that can invoke a tool using .run"""
        if not isinstance(
            function_call, FunctionCall
        ):  # TODO(Hack): Workaround lack of conditional apply
            return None
        try:
            tool = tools_by_name[function_call.name]
        except KeyError:
            raise AssertionError(f"No such tool: {function_call.name}")
        try:
            result = await tool.ainvoke(
                function_call.named_arguments or {},
                patch_config(config, callbacks=run_manager.get_child()),
            )
            error = None
        except Exception as e:
            result = None
            error = repr(e) + repr(function_call.named_arguments)

        return FunctionResult(name=function_call.name, result=result, error=error)

    return RunnableLambda(func=func, afunc=afunc)


def create_llm_program(
    llm: BaseLanguageModel,
    prompt_generator: Union[
        Callable[
            [Sequence[MessageLike]], Union[str, PromptValue, Sequence[BaseMessage]]
        ],
        Runnable,
    ],
    *,
    tools: Optional[Sequence[BaseTool]] = None,
    stop: Optional[Sequence[str]] = None,
    parser: Union[
        Runnable[Union[BaseMessage, str], MessageLike],
        Callable[[Union[BaseMessage, str]], MessageLike],
        BaseOutputParser,
        None,
    ] = None,
    invoke_tools: bool = True,  # TODO(Eugene): Perhaps remove.
) -> Runnable[Sequence[MessageLike], List[MessageLike]]:
    """Create a runnable that provides a generalized interface to an LLM with actions.

    Args:
        llm: A language model
        prompt_generator: A function that takes a list of messages and returns a prompt
        tools: A list of tools to invoke
        stop: optional list of stop tokens
        parser: optional parser to apply to the output of the LLM
        invoke_tools: Whether to invoke tools on the output of the LLM

    Returns:
        A runnable that returns a list of messages
    """

    if not isinstance(prompt_generator, Runnable):
        _prompt_generator = RunnableLambda(prompt_generator)
    else:
        _prompt_generator = prompt_generator

    if stop:
        llm = llm.bind(stop=stop)

    # Add parser to the end of the chain and concatenate original llm output
    # with the parser output.
    # The parser is always created even if it is None, to make sure that
    # the _to_message adapter is always applied (regardless of the parser).
    _parser = _to_runnable_parser(parser)

    chain = _prompt_generator | llm | _apply_and_concat(_to_message, _parser)

    # Add tool invoker to the end of the chain.
    if invoke_tools and tools:
        tool_invoker = create_tool_invoker(tools)
        invoke_on_last = RunnableLambda(lambda msgs: msgs[-1]) | tool_invoker
        complete_chain = chain | _apply_and_concat(
            RunnablePassthrough(), invoke_on_last
        )
    else:
        complete_chain = chain

    return complete_chain
