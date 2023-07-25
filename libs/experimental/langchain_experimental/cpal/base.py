"""
CPAL Chain and its subchains
"""
from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Optional, Type

import pydantic
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate

from langchain_experimental.cpal.constants import Constant
from langchain_experimental.cpal.models import (
    CausalModel,
    InterventionModel,
    NarrativeModel,
    QueryModel,
    StoryModel,
)
from langchain_experimental.cpal.templates.univariate.causal import (
    template as causal_template,
)
from langchain_experimental.cpal.templates.univariate.intervention import (
    template as intervention_template,
)
from langchain_experimental.cpal.templates.univariate.narrative import (
    template as narrative_template,
)
from langchain_experimental.cpal.templates.univariate.query import (
    template as query_template,
)


class _BaseStoryElementChain(Chain):
    chain: LLMChain
    input_key: str = Constant.narrative_input.value  #: :meta private:
    output_key: str = Constant.chain_answer.value  #: :meta private:
    pydantic_model: ClassVar[
        Optional[Type[pydantic.BaseModel]]
    ] = None  #: :meta private:
    template: ClassVar[Optional[str]] = None  #: :meta private:

    @classmethod
    def parser(cls) -> PydanticOutputParser:
        """Parse LLM output into a pydantic object."""
        if cls.pydantic_model is None:
            raise NotImplementedError(
                f"pydantic_model not implemented for {cls.__name__}"
            )
        return PydanticOutputParser(pydantic_object=cls.pydantic_model)

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

    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> Any:
        return cls(
            chain=LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=[Constant.narrative_input.value],
                    template=kwargs.get("template", cls.template),
                    partial_variables={
                        "format_instructions": cls.parser().get_format_instructions()
                    },
                ),
            ),
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        completion = self.chain.run(inputs[self.input_key])
        pydantic_data = self.__class__.parser().parse(completion)
        return {
            Constant.chain_data.value: pydantic_data,
            Constant.chain_answer.value: None,
        }


class NarrativeChain(_BaseStoryElementChain):
    """Decompose the narrative into its story elements

    - causal model
    - query
    - intervention
    """

    pydantic_model: ClassVar[Type[pydantic.BaseModel]] = NarrativeModel
    template: ClassVar[str] = narrative_template


class CausalChain(_BaseStoryElementChain):
    """Translate the causal narrative into a stack of operations."""

    pydantic_model: ClassVar[Type[pydantic.BaseModel]] = CausalModel
    template: ClassVar[str] = causal_template


class InterventionChain(_BaseStoryElementChain):
    """Set the hypothetical conditions for the causal model."""

    pydantic_model: ClassVar[Type[pydantic.BaseModel]] = InterventionModel
    template: ClassVar[str] = intervention_template


class QueryChain(_BaseStoryElementChain):
    """Query the outcome table using SQL."""

    pydantic_model: ClassVar[Type[pydantic.BaseModel]] = QueryModel
    template: ClassVar[str] = query_template  # TODO: incl. table schema


class CPALChain(_BaseStoryElementChain):
    llm: BaseLanguageModel
    narrative_chain: Optional[NarrativeChain] = None
    causal_chain: Optional[CausalChain] = None
    intervention_chain: Optional[InterventionChain] = None
    query_chain: Optional[QueryChain] = None
    _story: StoryModel = pydantic.PrivateAttr(default=None)  # TODO: change name ?

    @classmethod
    def from_univariate_prompt(
        cls,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> CPALChain:
        """instantiation depends on component chains"""
        return cls(
            llm=llm,
            chain=LLMChain(
                llm=llm,
                prompt=PromptTemplate(
                    input_variables=["question", "query_result"],
                    template=(
                        "Summarize this answer '{query_result}' to this "
                        "question '{question}'? "
                    ),
                ),
            ),
            narrative_chain=NarrativeChain.from_univariate_prompt(llm=llm),
            causal_chain=CausalChain.from_univariate_prompt(llm=llm),
            intervention_chain=InterventionChain.from_univariate_prompt(llm=llm),
            query_chain=QueryChain.from_univariate_prompt(llm=llm),
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # instantiate component chains
        if self.narrative_chain is None:
            self.narrative_chain = NarrativeChain.from_univariate_prompt(llm=self.llm)
        if self.causal_chain is None:
            self.causal_chain = CausalChain.from_univariate_prompt(llm=self.llm)
        if self.intervention_chain is None:
            self.intervention_chain = InterventionChain.from_univariate_prompt(
                llm=self.llm
            )
        if self.query_chain is None:
            self.query_chain = QueryChain.from_univariate_prompt(llm=self.llm)

        # decompose narrative into three causal story elements
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
            query_result = float(story.query._result_table.values[0][-1])
            if False:
                """TODO: add this back in when demanded by composable chains"""
                reporting_chain = self.chain
                human_report = reporting_chain.run(
                    question=story.query.question, query_result=query_result
                )
                query_result = {
                    "query_result": query_result,
                    "human_report": human_report,
                }
        output = {
            Constant.chain_data.value: story,
            self.output_key: query_result,
            **kwargs,
        }
        return output

    def draw(self, **kwargs: Any) -> None:
        """
        CPAL chain can draw its resulting DAG.

        Usage in a jupyter notebook:

            >>> from IPython.display import SVG
            >>> cpal_chain.draw(path="graph.svg")
            >>> SVG('graph.svg')
        """
        self._story._networkx_wrapper.draw_graphviz(**kwargs)
