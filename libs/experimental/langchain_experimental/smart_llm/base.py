"""Chain for applying self-critique using the SmartGPT workflow."""
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.input import get_colored_text
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import LLMResult, PromptValue

from langchain_experimental.pydantic_v1 import Extra, root_validator


class SmartLLMChain(Chain):
    """
    Generalized implementation of SmartGPT (origin: https://youtu.be/wVzuvf9D9BU)

    A SmartLLMChain is an LLMChain that instead of simply passing the prompt to the LLM
    performs these 3 steps:
    1. Ideate: Pass the user prompt to an ideation LLM n_ideas times,
       each result is an "idea"
    2. Critique: Pass the ideas to a critique LLM which looks for flaws in the ideas
       & picks the best one
    3. Resolve: Pass the critique to a resolver LLM which improves upon the best idea
       & outputs only the (improved version of) the best output

    In total, SmartLLMChain pass will use n_ideas+2 LLM calls

    Note that SmartLLMChain will only improve results (compared to a basic LLMChain),
    when the underlying models have the capability for reflection, which smaller models
    often don't.

    Finally, a SmartLLMChain assumes that each underlying LLM outputs exactly 1 result.
    """

    class SmartLLMChainHistory:
        question: str = ""
        ideas: List[str] = []
        critique: str = ""

        @property
        def n_ideas(self) -> int:
            return len(self.ideas)

        def ideation_prompt_inputs(self) -> Dict[str, Any]:
            return {"question": self.question}

        def critique_prompt_inputs(self) -> Dict[str, Any]:
            return {
                "question": self.question,
                **{f"idea_{i+1}": idea for i, idea in enumerate(self.ideas)},
            }

        def resolve_prompt_inputs(self) -> Dict[str, Any]:
            return {
                "question": self.question,
                **{f"idea_{i+1}": idea for i, idea in enumerate(self.ideas)},
                "critique": self.critique,
            }

    prompt: BasePromptTemplate
    """Prompt object to use."""
    ideation_llm: Optional[BaseLanguageModel] = None
    """LLM to use in ideation step. If None given, 'llm' will be used."""
    critique_llm: Optional[BaseLanguageModel] = None
    """LLM to use in critique step. If None given, 'llm' will be used."""
    resolver_llm: Optional[BaseLanguageModel] = None
    """LLM to use in resolve step. If None given, 'llm' will be used."""
    llm: Optional[BaseLanguageModel] = None
    """LLM to use for each steps, if no specific llm for that step is given. """
    n_ideas: int = 3
    """Number of ideas to generate in idea step"""
    return_intermediate_steps: bool = False
    """Whether to return ideas and critique, in addition to resolution."""
    history: SmartLLMChainHistory = SmartLLMChainHistory()

    class Config:
        extra = Extra.forbid

    # TODO: move away from `root_validator` since it is deprecated in pydantic v2
    #       and causes mypy type-checking failures (hence the `type: ignore`)
    @root_validator  # type: ignore[call-overload]
    @classmethod
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure we have an LLM for each step."""
        llm = values.get("llm")
        ideation_llm = values.get("ideation_llm")
        critique_llm = values.get("critique_llm")
        resolver_llm = values.get("resolver_llm")

        if not llm and not ideation_llm:
            raise ValueError(
                "Either ideation_llm or llm needs to be given. Pass llm, "
                "if you want to use the same llm for all steps, or pass "
                "ideation_llm, critique_llm and resolver_llm if you want "
                "to use different llms for each step."
            )
        if not llm and not critique_llm:
            raise ValueError(
                "Either critique_llm or llm needs to be given. Pass llm, "
                "if you want to use the same llm for all steps, or pass "
                "ideation_llm, critique_llm and resolver_llm if you want "
                "to use different llms for each step."
            )
        if not llm and not resolver_llm:
            raise ValueError(
                "Either resolve_llm or llm needs to be given. Pass llm, "
                "if you want to use the same llm for all steps, or pass "
                "ideation_llm, critique_llm and resolver_llm if you want "
                "to use different llms for each step."
            )
        if llm and ideation_llm and critique_llm and resolver_llm:
            raise ValueError(
                "LLMs are given for each step (ideation_llm, critique_llm,"
                " resolver_llm), but backup LLM (llm) is also given, which"
                " would not be used."
            )
        return values

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        if self.return_intermediate_steps:
            return ["ideas", "critique", "resolution"]
        return ["resolution"]

    def prep_prompts(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[PromptValue, Optional[List[str]]]:
        """Prepare prompts from inputs."""
        stop = None
        if "stop" in inputs:
            stop = inputs["stop"]
        selected_inputs = {k: inputs[k] for k in self.prompt.input_variables}
        prompt = self.prompt.format_prompt(**selected_inputs)
        _colored_text = get_colored_text(prompt.to_string(), "green")
        _text = "Prompt after formatting:\n" + _colored_text
        if run_manager:
            run_manager.on_text(_text, end="\n", verbose=self.verbose)
        if "stop" in inputs and inputs["stop"] != stop:
            raise ValueError(
                "If `stop` is present in any inputs, should be present in all."
            )
        return prompt, stop

    def _call(
        self,
        input_list: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        prompt, stop = self.prep_prompts(input_list, run_manager=run_manager)
        self.history.question = prompt.to_string()
        ideas = self._ideate(stop, run_manager)
        self.history.ideas = ideas
        critique = self._critique(stop, run_manager)
        self.history.critique = critique
        resolution = self._resolve(stop, run_manager)
        if self.return_intermediate_steps:
            return {"ideas": ideas, "critique": critique, "resolution": resolution}
        return {"resolution": resolution}

    def _get_text_from_llm_result(self, result: LLMResult, step: str) -> str:
        """Between steps, only the LLM result text is passed, not the LLMResult object.
        This function extracts the text from an LLMResult."""
        if len(result.generations) != 1:
            raise ValueError(
                f"In SmartLLM the LLM result in step {step} is not "
                "exactly 1 element. This should never happen"
            )
        if len(result.generations[0]) != 1:
            raise ValueError(
                f"In SmartLLM the LLM in step {step} returned more than "
                "1 output. SmartLLM only works with LLMs returning "
                "exactly 1 output."
            )
        return result.generations[0][0].text

    def get_prompt_strings(
        self, stage: str
    ) -> List[Tuple[Type[BaseMessagePromptTemplate], str]]:
        role_strings: List[Tuple[Type[BaseMessagePromptTemplate], str]] = []
        role_strings.append(
            (
                HumanMessagePromptTemplate,
                "Question: {question}\nAnswer: Let's work this out in a step by "
                "step way to be sure we have the right answer:",
            )
        )
        if stage == "ideation":
            return role_strings
        role_strings.extend(
            [
                *[
                    (
                        AIMessagePromptTemplate,
                        "Idea " + str(i + 1) + ": {idea_" + str(i + 1) + "}",
                    )
                    for i in range(self.n_ideas)
                ],
                (
                    HumanMessagePromptTemplate,
                    "You are a researcher tasked with investigating the "
                    f"{self.n_ideas} response options provided. List the flaws and "
                    "faulty logic of each answer options. Let'w work this out in a step"
                    " by step way to be sure we have all the errors:",
                ),
            ]
        )
        if stage == "critique":
            return role_strings
        role_strings.extend(
            [
                (AIMessagePromptTemplate, "Critique: {critique}"),
                (
                    HumanMessagePromptTemplate,
                    "You are a resolved tasked with 1) finding which of "
                    f"the {self.n_ideas} answer options the researcher thought was  "
                    "best,2) improving that answer and 3) printing the answer in full. "
                    "Don't output anything for step 1 or 2, only the full answer in 3. "
                    "Let's work this out in a step by step way to be sure we have "
                    "the right answer:",
                ),
            ]
        )
        if stage == "resolve":
            return role_strings
        raise ValueError(
            "stage should be either 'ideation', 'critique' or 'resolve',"
            f" but it is '{stage}'. This should never happen."
        )

    def ideation_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_strings(self.get_prompt_strings("ideation"))

    def critique_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_strings(self.get_prompt_strings("critique"))

    def resolve_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_strings(self.get_prompt_strings("resolve"))

    def _ideate(
        self,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[str]:
        """Generate n_ideas ideas as response to user prompt."""
        llm = self.ideation_llm if self.ideation_llm else self.llm
        prompt = self.ideation_prompt().format_prompt(
            **self.history.ideation_prompt_inputs()
        )
        callbacks = run_manager.get_child() if run_manager else None
        if llm:
            ideas = [
                self._get_text_from_llm_result(
                    llm.generate_prompt([prompt], stop, callbacks),
                    step="ideate",
                )
                for _ in range(self.n_ideas)
            ]
            for i, idea in enumerate(ideas):
                _colored_text = get_colored_text(idea, "blue")
                _text = f"Idea {i+1}:\n" + _colored_text
                if run_manager:
                    run_manager.on_text(_text, end="\n", verbose=self.verbose)
            return ideas
        else:
            raise ValueError("llm is none, which should never happen")

    def _critique(
        self,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> str:
        """Critique each of the ideas from ideation stage & select best one."""
        llm = self.critique_llm if self.critique_llm else self.llm
        prompt = self.critique_prompt().format_prompt(
            **self.history.critique_prompt_inputs()
        )
        callbacks = run_manager.handlers if run_manager else None
        if llm:
            critique = self._get_text_from_llm_result(
                llm.generate_prompt([prompt], stop, callbacks), step="critique"
            )
            _colored_text = get_colored_text(critique, "yellow")
            _text = "Critique:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            return critique
        else:
            raise ValueError("llm is none, which should never happen")

    def _resolve(
        self,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> str:
        """Improve upon the best idea as chosen in critique step & return it."""
        llm = self.resolver_llm if self.resolver_llm else self.llm
        prompt = self.resolve_prompt().format_prompt(
            **self.history.resolve_prompt_inputs()
        )
        callbacks = run_manager.handlers if run_manager else None
        if llm:
            resolution = self._get_text_from_llm_result(
                llm.generate_prompt([prompt], stop, callbacks), step="resolve"
            )
            _colored_text = get_colored_text(resolution, "green")
            _text = "Resolution:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            return resolution
        else:
            raise ValueError("llm is none, which should never happen")
