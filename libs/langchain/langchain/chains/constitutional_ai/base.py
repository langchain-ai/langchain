"""Chain for applying constitutional principles to the outputs of another chain."""

from typing import Any, Optional

from langchain_core._api import deprecated
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate

from langchain.chains.base import Chain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.chains.constitutional_ai.principles import PRINCIPLES
from langchain.chains.constitutional_ai.prompts import CRITIQUE_PROMPT, REVISION_PROMPT
from langchain.chains.llm import LLMChain


@deprecated(
    since="0.2.13",
    message=(
        "This class is deprecated and will be removed in langchain 1.0. "
        "See API reference for replacement: "
        "https://api.python.langchain.com/en/latest/chains/langchain.chains.constitutional_ai.base.ConstitutionalChain.html"  # noqa: E501
    ),
    removal="1.0",
)
class ConstitutionalChain(Chain):
    """Chain for applying constitutional principles.

    Note: this class is deprecated. See below for a replacement implementation
        using LangGraph. The benefits of this implementation are:

        - Uses LLM tool calling features instead of parsing string responses;
        - Support for both token-by-token and step-by-step streaming;
        - Support for checkpointing and memory of chat history;
        - Easier to modify or extend (e.g., with additional tools, structured responses, etc.)

        Install LangGraph with:

        .. code-block:: bash

            pip install -U langgraph

        .. code-block:: python

            from typing import List, Optional, Tuple

            from langchain.chains.constitutional_ai.prompts import (
                CRITIQUE_PROMPT,
                REVISION_PROMPT,
            )
            from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from langgraph.graph import END, START, StateGraph
            from typing_extensions import Annotated, TypedDict

            llm = ChatOpenAI(model="gpt-4o-mini")

            class Critique(TypedDict):
                \"\"\"Generate a critique, if needed.\"\"\"
                critique_needed: Annotated[bool, ..., "Whether or not a critique is needed."]
                critique: Annotated[str, ..., "If needed, the critique."]

            critique_prompt = ChatPromptTemplate.from_template(
                "Critique this response according to the critique request. "
                "If no critique is needed, specify that.\\n\\n"
                "Query: {query}\\n\\n"
                "Response: {response}\\n\\n"
                "Critique request: {critique_request}"
            )

            revision_prompt = ChatPromptTemplate.from_template(
                "Revise this response according to the critique and reivsion request.\\n\\n"
                "Query: {query}\\n\\n"
                "Response: {response}\\n\\n"
                "Critique request: {critique_request}\\n\\n"
                "Critique: {critique}\\n\\n"
                "If the critique does not identify anything worth changing, ignore the "
                "revision request and return 'No revisions needed'. If the critique "
                "does identify something worth changing, revise the response based on "
                "the revision request.\\n\\n"
                "Revision Request: {revision_request}"
            )

            chain = llm | StrOutputParser()
            critique_chain = critique_prompt | llm.with_structured_output(Critique)
            revision_chain = revision_prompt | llm | StrOutputParser()


            class State(TypedDict):
                query: str
                constitutional_principles: List[ConstitutionalPrinciple]
                initial_response: str
                critiques_and_revisions: List[Tuple[str, str]]
                response: str


            async def generate_response(state: State):
                \"\"\"Generate initial response.\"\"\"
                response = await chain.ainvoke(state["query"])
                return {"response": response, "initial_response": response}

            async def critique_and_revise(state: State):
                \"\"\"Critique and revise response according to principles.\"\"\"
                critiques_and_revisions = []
                response = state["initial_response"]
                for principle in state["constitutional_principles"]:
                    critique = await critique_chain.ainvoke(
                        {
                            "query": state["query"],
                            "response": response,
                            "critique_request": principle.critique_request,
                        }
                    )
                    if critique["critique_needed"]:
                        revision = await revision_chain.ainvoke(
                            {
                                "query": state["query"],
                                "response": response,
                                "critique_request": principle.critique_request,
                                "critique": critique["critique"],
                                "revision_request": principle.revision_request,
                            }
                        )
                        response = revision
                        critiques_and_revisions.append((critique["critique"], revision))
                    else:
                        critiques_and_revisions.append((critique["critique"], ""))
                return {
                    "critiques_and_revisions": critiques_and_revisions,
                    "response": response,
                }

            graph = StateGraph(State)
            graph.add_node("generate_response", generate_response)
            graph.add_node("critique_and_revise", critique_and_revise)

            graph.add_edge(START, "generate_response")
            graph.add_edge("generate_response", "critique_and_revise")
            graph.add_edge("critique_and_revise", END)
            app = graph.compile()

        .. code-block:: python

            constitutional_principles=[
                ConstitutionalPrinciple(
                    critique_request="Tell if this answer is good.",
                    revision_request="Give a better answer.",
                )
            ]

            query = "What is the meaning of life? Answer in 10 words or fewer."

            async for step in app.astream(
                {"query": query, "constitutional_principles": constitutional_principles},
                stream_mode="values",
            ):
                subset = ["initial_response", "critiques_and_revisions", "response"]
                print({k: v for k, v in step.items() if k in subset})

    Example:
        .. code-block:: python

            from langchain_community.llms import OpenAI
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
    """  # noqa: E501

    chain: LLMChain
    constitutional_principles: list[ConstitutionalPrinciple]
    critique_chain: LLMChain
    revision_chain: LLMChain
    return_intermediate_steps: bool = False

    @classmethod
    def get_principles(
        cls, names: Optional[list[str]] = None
    ) -> list[ConstitutionalPrinciple]:
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
    def input_keys(self) -> list[str]:
        """Input keys."""
        return self.chain.input_keys

    @property
    def output_keys(self) -> list[str]:
        """Output keys."""
        if self.return_intermediate_steps:
            return ["output", "critiques_and_revisions", "initial_output"]
        return ["output"]

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        response = self.chain.run(
            **inputs,
            callbacks=_run_manager.get_child("original"),
        )
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
                callbacks=_run_manager.get_child("critique"),
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
                callbacks=_run_manager.get_child("revision"),
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

        final_output: dict[str, Any] = {"output": response}
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
