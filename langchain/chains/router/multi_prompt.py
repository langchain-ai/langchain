"""Use a single chain to route an input to one of multiple candidate chains."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from pydantic import Field

from langchain import ConversationChain, OpenAI, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.llm import LLMRouterChain
from langchain.output_parsers.structured import parse_json_markdown
from langchain.schema import BaseOutputParser, OutputParserException

ROUTER_PROMPT_TEMPLATE = """\
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
    router_chain: LLMRouterChain
    destination_chains: Mapping[str, LLMChain]
    default_chain: LLMChain = Field(
        default_factory=lambda: ConversationChain(llm=OpenAI())
    )

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_names: List[str],
        prompt_descriptions: List[str],
        prompt_templates: List[str],
        **kwargs: Any,
    ) -> MultiPromptChain:
        destinations_str = "\n".join(
            [
                f"{name}: {description}"
                for name, description in zip(prompt_names, prompt_descriptions)
            ]
        )
        router_template = ROUTER_PROMPT_TEMPLATE.format(destinations=destinations_str)
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
        return cls(
            router_chain=router_chain, destination_chains=destination_chains, **kwargs
        )
