"""Base classes for LLM-powered router chains."""

from __future__ import annotations

from typing import Any, cast

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.utils.json import parse_and_check_json_markdown
from pydantic import model_validator
from typing_extensions import Self, override

from langchain_classic.chains import LLMChain
from langchain_classic.chains.router.base import RouterChain


@deprecated(
    since="0.2.12",
    removal="1.0",
    message=(
        "Use RunnableLambda to select from multiple prompt templates. See example "
        "in API reference: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.router.llm_router.LLMRouterChain.html"
    ),
)
class LLMRouterChain(RouterChain):
    """A router chain that uses an LLM chain to perform routing.

    This class is deprecated. See below for a replacement, which offers several
    benefits, including streaming and batch support.

    Below is an example implementation:

        ```python
        from operator import itemgetter
        from typing import Literal
        from typing_extensions import TypedDict

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o-mini")

        prompt_1 = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on animals."),
                ("human", "{query}"),
            ]
        )
        prompt_2 = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert on vegetables."),
                ("human", "{query}"),
            ]
        )

        chain_1 = prompt_1 | model | StrOutputParser()
        chain_2 = prompt_2 | model | StrOutputParser()

        route_system = "Route the user's query to either the animal "
        "or vegetable expert."
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", route_system),
                ("human", "{query}"),
            ]
        )


        class RouteQuery(TypedDict):
            \"\"\"Route query to destination.\"\"\"
            destination: Literal["animal", "vegetable"]


        route_chain = (
            route_prompt
            | model.with_structured_output(RouteQuery)
            | itemgetter("destination")
        )

        chain = {
            "destination": route_chain,  # "animal" or "vegetable"
            "query": lambda x: x["query"],  # pass through input query
        } | RunnableLambda(
            # if animal, chain_1. otherwise, chain_2.
            lambda x: chain_1 if x["destination"] == "animal" else chain_2,
        )

        chain.invoke({"query": "what color are carrots"})

        ```
    """

    llm_chain: LLMChain
    """LLM chain used to perform routing"""

    @model_validator(mode="after")
    def _validate_prompt(self) -> Self:
        prompt = self.llm_chain.prompt
        if prompt.output_parser is None:
            msg = (
                "LLMRouterChain requires base llm_chain prompt to have an output"
                " parser that converts LLM text output to a dictionary with keys"
                " 'destination' and 'next_inputs'. Received a prompt with no output"
                " parser."
            )
            raise ValueError(msg)
        return self

    @property
    def input_keys(self) -> list[str]:
        """Will be whatever keys the LLM chain prompt expects.

        :meta private:
        """
        return self.llm_chain.input_keys

    def _validate_outputs(self, outputs: dict[str, Any]) -> None:
        super()._validate_outputs(outputs)
        if not isinstance(outputs["next_inputs"], dict):
            raise ValueError  # noqa: TRY004

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()

        prediction = self.llm_chain.predict(callbacks=callbacks, **inputs)
        return cast(
            "dict[str, Any]",
            self.llm_chain.prompt.output_parser.parse(prediction),
        )

    async def _acall(
        self,
        inputs: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun | None = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        return cast(
            "dict[str, Any]",
            await self.llm_chain.apredict_and_parse(callbacks=callbacks, **inputs),
        )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate,
        **kwargs: Any,
    ) -> LLMRouterChain:
        """Convenience constructor."""
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)


class RouterOutputParser(BaseOutputParser[dict[str, str]]):
    """Parser for output of router chain in the multi-prompt chain."""

    default_destination: str = "DEFAULT"
    next_inputs_type: type = str
    next_inputs_inner_key: str = "input"

    @override
    def parse(self, text: str) -> dict[str, Any]:
        try:
            expected_keys = ["destination", "next_inputs"]
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if not isinstance(parsed["destination"], str):
                msg = "Expected 'destination' to be a string."
                raise TypeError(msg)
            if not isinstance(parsed["next_inputs"], self.next_inputs_type):
                msg = f"Expected 'next_inputs' to be {self.next_inputs_type}."
                raise TypeError(msg)
            parsed["next_inputs"] = {self.next_inputs_inner_key: parsed["next_inputs"]}
            if (
                parsed["destination"].strip().lower()
                == self.default_destination.lower()
            ):
                parsed["destination"] = None
            else:
                parsed["destination"] = parsed["destination"].strip()
        except Exception as e:
            msg = f"Parsing text\n{text}\n raised following error:\n{e}"
            raise OutputParserException(msg) from e
        return parsed
