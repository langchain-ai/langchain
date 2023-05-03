"""Use a single chain to route an input to one of multiple candidate chains."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from langchain import ConversationChain, OpenAI, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException


class RouterOutputParser(BaseOutputParser[Dict[str, str]]):
    """Parser for output of router chain int he multi-prompt chain."""

    def parse(self, text: str) -> Dict[str, str]:
        try:
            expected_keys = ["destination", "next_inputs"]
            parsed = parse_json_markdown(text, expected_keys)
            for key in expected_keys:
                if not isinstance(parsed[key], str):
                    raise ValueError(f"Expected '{key}' to be a string.")
            parsed["next_inputs"] = {"input": parsed["next_inputs"]}
            return parsed
        except Exception as e:
            raise OutputParserException(
                f"Parsing text\n{text}\n raised following error:\n{e}"
            )


class MultiPromptChain(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst prompts."""

    router_chain: LLMRouterChain
    """Chain for deciding a destination chain and the input to it."""
    destination_chains: Mapping[str, LLMChain]
    """Name to chain map for candidate chains to route inputs to."""
    default_chain: LLMChain
    """Default chain to use when router doesn't map input to one of the destinations."""

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_names: List[str],
        prompt_descriptions: List[str],
        prompt_templates: List[str],
        default_chain: Optional[LLMChain] = None,
        **kwargs: Any,
    ) -> MultiPromptChain:
        """Convenience constructor for instantiating from destination prompts."""
        destinations_str = "\n".join(
            [
                f"{name}: {description}"
                for name, description in zip(prompt_names, prompt_descriptions)
            ]
        )
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {
            name: LLMChain(
                llm=llm,
                prompt=PromptTemplate(template=prompt, input_variables=["input"]),
            )
            for name, prompt in zip(prompt_names, prompt_templates)
        }
        default_chain = default_chain or ConversationChain(llm=OpenAI())
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            **kwargs,
        )
