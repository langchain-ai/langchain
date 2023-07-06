"""
CPAL Chain and its subchains
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Type, Optional
import pydantic

from langchain.base_language import BaseLanguageModel
from langchain.prompts.prompt import PromptTemplate

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser

from langchain.experimental.chains.cpal.constants import Constant
from langchain.experimental.chains.cpal.models import (
    StoryModel,
    NarrativeModel,
    CausalModel,
    QueryModel,
    InterventionModel,
)
from langchain.experimental.chains.cpal.templates.univariate.narrative import (
    template as narrative_template,
)
from langchain.experimental.chains.cpal.templates.univariate.causal import (
    template as causal_template,
)
from langchain.experimental.chains.cpal.templates.univariate.intervention import (
    template as intervention_template,
)
from langchain.experimental.chains.cpal.templates.univariate.query import (
    template as query_template,
)


def make_prompt_template(template: str, data_model) -> PromptTemplate:
    parser = PydanticOutputParser(pydantic_object=data_model)
    return PromptTemplate(
        input_variables=[Constant.narrative_input.value],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


class _BaseStoryElementChain(Chain):
    chain: LLMChain
    data_model: Type[pydantic.BaseModel]
    input_key: str = Constant.narrative_input.value  #: :meta private:
    output_key: str = Constant.chain_answer.value  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys


class NarrativeChain(_BaseStoryElementChain):
    """Decompose the human's narrative into its story elements

    - causal model
    - query
    - intervention
    """

    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        template: str = narrative_template,
        data_model: Type[pydantic.BaseModel] = NarrativeModel,
        **kwargs: Any,
    ) -> NarrativeChain:
        return cls(
            chain=LLMChain(
                llm=llm,
                prompt=make_prompt_template(template=template, data_model=data_model),
            ),
            data_model=data_model,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        completion = self.chain.run(inputs[self.input_key])
        parser = PydanticOutputParser(pydantic_object=self.data_model)
        pydantic_data = parser.parse(completion)
        return {
            Constant.chain_data.value: pydantic_data,
            Constant.chain_answer.value: None,
        }


class CausalChain(_BaseStoryElementChain):
    """Translate narrative plot (causal mental model) into a graph"""

    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        template: str = causal_template,
        data_model: Type[pydantic.BaseModel] = CausalModel,
        **kwargs: Any,
    ) -> CausalChain:
        return cls(
            chain=LLMChain(
                llm=llm,
                prompt=make_prompt_template(template=template, data_model=data_model),
            ),
            data_model=data_model,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        completion = self.chain.run(inputs[self.input_key])
        parser = PydanticOutputParser(pydantic_object=self.data_model)
        pydantic_data = parser.parse(completion)
        return {
            Constant.chain_data.value: pydantic_data,
            Constant.chain_answer.value: None,
        }


class InterventionChain(_BaseStoryElementChain):
    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        template: str = intervention_template,
        data_model: Type[pydantic.BaseModel] = InterventionModel,
        **kwargs: Any,
    ) -> InterventionChain:
        return cls(
            chain=LLMChain(
                llm=llm,
                prompt=make_prompt_template(template=template, data_model=data_model),
            ),
            data_model=data_model,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        completion = self.chain.run(inputs[self.input_key])
        parser = PydanticOutputParser(pydantic_object=self.data_model)
        pydantic_data = parser.parse(completion)
        return {
            Constant.chain_data.value: pydantic_data,
            Constant.chain_answer.value: None,
        }


class QueryChain(_BaseStoryElementChain):
    # TODO: db_schema = inputs["db_schema"]  # add db schema to prompt

    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        template: str = query_template,
        data_model: Type[pydantic.BaseModel] = QueryModel,
        **kwargs: Any,
    ) -> QueryChain:
        return cls(
            chain=LLMChain(
                llm=llm,
                prompt=make_prompt_template(template=template, data_model=data_model),
            ),
            data_model=data_model,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        human_question = inputs[self.input_key]
        human_question = human_question.lower()
        completion = self.chain.run(human_question)
        parser = PydanticOutputParser(pydantic_object=self.data_model)
        pydantic_data = parser.parse(completion)
        return {
            Constant.chain_data.value: pydantic_data,
            Constant.chain_answer.value: None,
        }


class CPALChain(_BaseStoryElementChain):
    llm: BaseLanguageModel
    data_model: Type[pydantic.BaseModel] = StoryModel  # data class
    chain: Optional[LLMChain] = None
    narrative_chain: Optional[NarrativeChain] = None
    causal_chain: Optional[CausalChain] = None
    intervention_chain: Optional[InterventionChain] = None
    query_chain: Optional[QueryChain] = None
    prompt: str
    _story: StoryModel = pydantic.PrivateAttr(default=None)

    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        data_model: Type[pydantic.BaseModel] = StoryModel,
        **kwargs: Any,
    ) -> CPALChain:
        """Convenience method to initialize CPALChain with the component
        chains initialized from univariate prompts.
        """
        return cls(
            llm=llm,
            data_model=data_model,
            chain=None,
            narrative_chain=NarrativeChain.from_univariate_prompt(llm=llm),
            causal_chain=CausalChain.from_univariate_prompt(llm=llm),
            intervention_chain=InterventionChain.from_univariate_prompt(llm=llm),
            query_chain=QueryChain.from_univariate_prompt(llm=llm),
            prompt="univariate",
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if self.prompt == "univariate":
            if self.narrative_chain is None:
                self.narrative_chain = NarrativeChain.from_univariate_prompt(
                    llm=self.llm
                )
            if self.causal_chain is None:
                self.causal_chain = CausalChain.from_univariate_prompt(llm=self.llm)
            if self.intervention_chain is None:
                self.intervention_chain = InterventionChain.from_univariate_prompt(
                    llm=self.llm
                )
            if self.query_chain is None:
                self.query_chain = QueryChain.from_univariate_prompt(llm=self.llm)
        else:
            raise NotImplementedError(
                (
                    f"Prompt {self.prompt} not implemented."
                    "Only 'univariate' is implemented."
                )
            )

        # decompose human's narrative into three causal story elements
        narrative = self.narrative_chain(inputs[Constant.narrative_input.value])[
            Constant.chain_data.value
        ]

        story = StoryModel(
            causal_operations=self.causal_chain(narrative.story_plot)[
                Constant.chain_data.value
            ],
            intervention=self.intervention_chain(narrative.story_hypothetical)[
                Constant.chain_data.value
            ],
            query=self.query_chain(narrative.story_outcome_question)[
                Constant.chain_data.value
            ],
        )
        self._story = story

        def pretty_print_str(title: str, d: str) -> str:
            return title + "\n" + d

        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(
            pretty_print_str("story outcome data", story._outcome_table.to_string()),
            color="green",
            end="\n\n",
            verbose=self.verbose,
        )

        def pretty_print_dict(title: str, d: dict) -> str:
            return title + "\n" + json.dumps(d, indent=4)

        _run_manager.on_text(
            pretty_print_dict("query data", story.query.dict()),
            color="blue",
            end="\n\n",
            verbose=self.verbose,
        )
        if story.query._result_table.empty:
            # prevent piping bad data into subsequent chains
            raise ValueError(
                (
                    "unanswerable, query and outcome are incoherent\n"
                    "\n"
                    "outcome:\n"
                    f"{story._outcome_table}\n"
                    "query:\n"
                    f"{story.query.dict()}"
                )
            )
        else:
            # TODO: make reporting more robust using LLM
            answer = float(story.query._result_table.values[0][-1])
        output = {
            Constant.chain_data.value: story,
            self.output_key: answer,
            **kwargs,
        }
        return output
