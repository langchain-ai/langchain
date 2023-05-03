"""Use a single chain to route an input to one of multiple candidate chains."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Mapping, Optional, Tuple

from pydantic import Extra, Field, root_validator

from langchain import BasePromptTemplate, ConversationChain, OpenAI, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException


class RouterChain(Chain, ABC):
    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]


class MultiRouteChain(Chain):
    """Use a single chain to route an input to one of multiple candidate chains."""

    router_chain: RouterChain
    """Chain that routes inputs to destination chains."""
    destination_chains: Mapping[str, Chain]
    """Chains that return final answer to inputs."""
    default_chain: Chain
    """Default chain to use when routing fails."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.router_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return []

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        router_output = self.router_chain(inputs, callbacks=callbacks)
        destination = router_output["destination"]
        next_inputs = router_output["next_inputs"]
        _run_manager.on_text(destination + " " + next_inputs, verbose=self.verbose)
        if destination in self.destination_chains:
            return self.destination_chains[destination](
                next_inputs, callbacks=callbacks
            )
        else:
            return self.default_chain(next_inputs, callbacks=callbacks)


from langchain.chains.llm import LLMChain

# Multiprompt chain


class LLMRouterChain(RouterChain):
    llm_chain: LLMChain

    @root_validator()
    def validate_prompt(cls, values: dict) -> dict:
        prompt = values["llm_chain"].prompt
        if prompt.output_parser is None:
            raise ValueError(
                "LLMRouterChain requires base llm_chain prompt to have an output"
                " parser that converts LLM text output to a dictionary with keys"
                " 'destination' and 'next_inputs'. Received a prompt with no output"
                " parser."
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        return self.llm_chain.input_keys

    def _validate_outputs(self, outputs: Dict[str, str]) -> None:
        super()._validate_outputs(outputs)
        if not isinstance(outputs["next_inputs"], dict):
            raise ValueError

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        output = self.llm_chain.predict_and_parse(callbacks=callbacks, **inputs)
        return output

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: BasePromptTemplate, **kwargs: Any
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)


DEFAULT_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may also revise the original input if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ name of the prompt to use
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below.
REMEMBER: "next_inputs" can just be the original input if you don't think any \
modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
"""


class RouterOutputParser(BaseOutputParser[Dict[str, str]]):
    def parse(self, text: str) -> Dict[str, str]:
        try:
            expected_keys = ["destination", "next_inputs"]
            parsed = parse_json_markdown(text, expected_keys)
            if not isinstance(parsed["destination"], str):
                raise ValueError("Expected 'destination' to be a string.")
            if not isinstance(parsed["next_inputs"], str):
                raise ValueError("Expected 'next_inputs' to be a string.")
            parsed["next_inputs"] = {"input": parsed["next_inputs"]}
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )


class MultiPromptChain(MultiRouteChain):
    router_chain: LLMRouterChain
    destination_chains: Mapping[str, LLMChain]
    default_chain: LLMChain = Field(
        default_factory=lambda: ConversationChain(llm=OpenAI())
    )

    @classmethod
    def from_descriptions(
        cls,
        llm: BaseLanguageModel,
        destinations: Dict[str, Tuple[str, str]],
        **kwargs: Any,
    ) -> MultiPromptChain:
        """"""
        destinations_str = "\n".join(
            [
                f"{prompt_name}: {prompt_description}"
                for prompt_name, (
                    prompt_description,
                    _,
                ) in destinations.items()
            ]
        )
        template = DEFAULT_TEMPLATE.format(destinations=destinations_str)
        router_prompt = PromptTemplate(
            template=template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {
            prompt_name: LLMChain(
                llm=llm,
                prompt=PromptTemplate(template=prompt, input_variables=["input"]),
            )
            for prompt_name, (_, prompt) in destinations.items()
        }
        return cls(
            router_chain=router_chain, destination_chains=destination_chains, **kwargs
        )
