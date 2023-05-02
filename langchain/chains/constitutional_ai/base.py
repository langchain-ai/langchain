"""Chain for applying constitutional principles to the outputs of another chain."""
from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.constitutional_ai.principles import PRINCIPLES
from langchain.chains.constitutional_ai.prompts import CRITIQUE_PROMPT, REVISION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate


class ConstitutionalChain(Chain):
    """Chain for applying constitutional principles.

    Example:
        .. code-block:: python

            from langchain.llms import OpenAI
            from langchain.chains import LLMChain, ConstitutionalChain
            from langchain.chains.constitutional_ai.models \
                import ConstitutionalPrinciple

            llm = OpenAI()

            qa_prompt = PromptTemplate(
                template="Q: {question} A:",
                input_variables=["question"],
            )
            qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

            constitutional_chain = ConstitutionalChain.from_llm(
                llm=llm,
                chain=qa_chain,
                constitutional_principles=[
                    ConstitutionalPrinciple(
                        critique_request="Tell if this answer is good.",
                        revision_request="Give a better answer.",
                    )
                ],
            )

            constitutional_chain.run(question="What is the meaning of life?")
    """

    chain: LLMChain
    constitutional_principles: List[ConstitutionalPrinciple]
    critique_chain: LLMChain
    revision_chain: LLMChain
    return_intermediate_steps: bool = False

    @classmethod
    def get_principles(
        cls, names: Optional[List[str]] = None
    ) -> List[ConstitutionalPrinciple]:
        if names is None:
            return list(PRINCIPLES.values())
        else:
            return [PRINCIPLES[name] for name in names]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        chain: LLMChain,
        critique_prompt: BasePromptTemplate = CRITIQUE_PROMPT,
        revision_prompt: BasePromptTemplate = REVISION_PROMPT,
        **kwargs: Any,
    ) -> "ConstitutionalChain":
        """Create a chain from an LLM."""
        critique_chain = LLMChain(llm=llm, prompt=critique_prompt)
        revision_chain = LLMChain(llm=llm, prompt=revision_prompt)
        return cls(
            chain=chain,
            critique_chain=critique_chain,
            revision_chain=revision_chain,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        if self.return_intermediate_steps:
            return ["output", "critiques_and_revisions", "initial_output"]
        return ["output"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        response = self.chain.run(**inputs)
        initial_response = response
        input_prompt = self.chain.prompt.format(**inputs)

        _run_manager.on_text(
            text="Initial response: " + response + "\n\n",
            verbose=self.verbose,
            color="yellow",
        )
        critiques_and_revisions = []
        for constitutional_principle in self.constitutional_principles:
            # Do critique

            raw_critique = self.critique_chain.run(
                input_prompt=input_prompt,
                output_from_model=response,
                critique_request=constitutional_principle.critique_request,
                callbacks=_run_manager.get_child(),
            )
            critique = self._parse_critique(
                output_string=raw_critique,
            ).strip()

            # if the critique contains "No critique needed", then we're done
            # in this case, initial_output is the same as output,
            # but we'll keep it for consistency
            if "no critique needed" in critique.lower():
                critiques_and_revisions.append((critique, ""))
                continue

            # Do revision

            revision = self.revision_chain.run(
                input_prompt=input_prompt,
                output_from_model=response,
                critique_request=constitutional_principle.critique_request,
                critique=critique,
                revision_request=constitutional_principle.revision_request,
                callbacks=_run_manager.get_child(),
            ).strip()
            response = revision
            critiques_and_revisions.append((critique, revision))

            _run_manager.on_text(
                text=f"Applying {constitutional_principle.name}..." + "\n\n",
                verbose=self.verbose,
                color="green",
            )

            _run_manager.on_text(
                text="Critique: " + critique + "\n\n",
                verbose=self.verbose,
                color="blue",
            )

            _run_manager.on_text(
                text="Updated response: " + revision + "\n\n",
                verbose=self.verbose,
                color="yellow",
            )

        final_output: Dict[str, Any] = {"output": response}
        if self.return_intermediate_steps:
            final_output["initial_output"] = initial_response
            final_output["critiques_and_revisions"] = critiques_and_revisions
        return final_output

    @staticmethod
    def _parse_critique(output_string: str) -> str:
        if "Revision request:" not in output_string:
            return output_string
        output_string = output_string.split("Revision request:")[0]
        if "\n\n" in output_string:
            output_string = output_string.split("\n\n")[0]
        return output_string
