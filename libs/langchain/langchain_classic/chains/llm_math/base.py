"""Chain that interprets a prompt and executes python code to do math."""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import ConfigDict, model_validator

from langchain_classic.chains.base import Chain
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.llm_math.prompt import PROMPT


@deprecated(
    since="0.2.13",
    message=(
        "This class is deprecated and will be removed in langchain 1.0. "
        "See API reference for replacement: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.llm_math.base.LLMMathChain.html"
    ),
    removal="1.0",
)
class LLMMathChain(Chain):
    """Chain that interprets a prompt and executes python code to do math.

    !!! note
        This class is deprecated. See below for a replacement implementation using
        LangGraph. The benefits of this implementation are:

        - Uses LLM tool calling features;
        - Support for both token-by-token and step-by-step streaming;
        - Support for checkpointing and memory of chat history;
        - Easier to modify or extend
            (e.g., with additional tools, structured responses, etc.)

        Install LangGraph with:

        ```bash
        pip install -U langgraph
        ```

        ```python
        import math
        from typing import Annotated, Sequence

        from langchain_core.messages import BaseMessage
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt.tool_node import ToolNode
        import numexpr
        from typing_extensions import TypedDict

        @tool
        def calculator(expression: str) -> str:
            \"\"\"Calculate expression using Python's numexpr library.

            Expression should be a single line mathematical expression
            that solves the problem.
        ```

    Examples:
                    "37593 * 67" for "37593 times 67"
                    "37593**(1/5)" for "37593^(1/5)"
                \"\"\"
                local_dict = {"pi": math.pi, "e": math.e}
                return str(
                    numexpr.evaluate(
                        expression.strip(),
                        global_dict={},  # restrict access to globals
                        local_dict=local_dict,  # add common mathematical functions
                    )
                )

            model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            tools = [calculator]
            model_with_tools = model.bind_tools(tools, tool_choice="any")

            class ChainState(TypedDict):
                \"\"\"LangGraph state.\"\"\"

                messages: Annotated[Sequence[BaseMessage], add_messages]

            async def acall_chain(state: ChainState, config: RunnableConfig):
                last_message = state["messages"][-1]
                response = await model_with_tools.ainvoke(state["messages"], config)
                return {"messages": [response]}

            async def acall_model(state: ChainState, config: RunnableConfig):
                response = await model.ainvoke(state["messages"], config)
                return {"messages": [response]}

            graph_builder = StateGraph(ChainState)
            graph_builder.add_node("call_tool", acall_chain)
            graph_builder.add_node("execute_tool", ToolNode(tools))
            graph_builder.add_node("call_model", acall_model)
            graph_builder.set_entry_point("call_tool")
            graph_builder.add_edge("call_tool", "execute_tool")
            graph_builder.add_edge("execute_tool", "call_model")
            graph_builder.add_edge("call_model", END)
            chain = graph_builder.compile()

        ```python
        example_query = "What is 551368 divided by 82"

        events = chain.astream(
            {"messages": [("user", example_query)]},
            stream_mode="values",
        )
        async for event in events:
            event["messages"][-1].pretty_print()
        ```

        ```txt
        ================================ Human Message =================================

        What is 551368 divided by 82
        ================================== Ai Message ==================================
        Tool Calls:
        calculator (call_MEiGXuJjJ7wGU4aOT86QuGJS)
        Call ID: call_MEiGXuJjJ7wGU4aOT86QuGJS
        Args:
            expression: 551368 / 82
        ================================= Tool Message =================================
        Name: calculator

        6724.0
        ================================== Ai Message ==================================

        551368 divided by 82 equals 6724.
        ```

    Example:
        ```python
        from langchain_classic.chains import LLMMathChain
        from langchain_community.llms import OpenAI

        llm_math = LLMMathChain.from_llm(OpenAI())
        ```
    """

    llm_chain: LLMChain
    llm: BaseLanguageModel | None = None
    """[Deprecated] LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated] Prompt to use to translate to python if necessary."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _raise_deprecation(cls, values: dict) -> Any:
        try:
            import numexpr  # noqa: F401
        except ImportError as e:
            msg = (
                "LLMMathChain requires the numexpr package. "
                "Please install it with `pip install numexpr`."
            )
            raise ImportError(msg) from e
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMMathChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method.",
                stacklevel=5,
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> list[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _evaluate_expression(self, expression: str) -> str:
        import numexpr

        try:
            local_dict = {"pi": math.pi, "e": math.e}
            output = str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                ),
            )
        except Exception as e:
            msg = (
                f'LLMMathChain._evaluate("{expression}") raised error: {e}.'
                " Please try again with a valid numerical expression"
            )
            raise ValueError(msg) from e

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)

    def _process_llm_result(
        self,
        llm_output: str,
        run_manager: CallbackManagerForChainRun,
    ) -> dict[str, str]:
        run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            msg = f"unknown format from LLM: {llm_output}"
            raise ValueError(msg)
        return {self.output_key: answer}

    async def _aprocess_llm_result(
        self,
        llm_output: str,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            msg = f"unknown format from LLM: {llm_output}"
            raise ValueError(msg)
        return {self.output_key: answer}

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
        self,
        inputs: dict[str, str],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_math_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> LLMMathChain:
        """Create a LLMMathChain from a language model.

        Args:
            llm: a language model
            prompt: a prompt template
            **kwargs: additional arguments
        """
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
