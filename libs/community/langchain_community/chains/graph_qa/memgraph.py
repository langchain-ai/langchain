"""Question answering over a graph."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable
from pydantic import Field

from langchain_community.chains.graph_qa.prompts import (
    MEMGRAPH_GENERATION_PROMPT,
    MEMGRAPH_QA_PROMPT,
)
from langchain_community.graphs.memgraph_graph import MemgraphGraph

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

FUNCTION_RESPONSE_SYSTEM = """You are an assistant that helps to form nice and human 
understandable answers based on the provided information from tools.
Do not add any other information that wasn't present in the tools, and use 
very concise style in interpreting results!
"""


def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text


def get_function_response(
    question: str, context: List[Dict[str, Any]]
) -> List[BaseMessage]:
    TOOL_ID = "call_H7fABDuzEau48T10Qn0Lsh0D"
    messages = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": TOOL_ID,
                        "function": {
                            "arguments": '{"question":"' + question + '"}',
                            "name": "GetInformation",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(content=str(context), tool_call_id=TOOL_ID),
    ]
    return messages


class MemgraphQAChain(Chain):
    """Chain for question-answering against a graph by generating Cypher statements.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    graph: MemgraphGraph = Field(exclude=True)
    cypher_generation_chain: Runnable
    qa_chain: Runnable
    graph_schema: str
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    top_k: int = 10
    """Number of results to return from the query"""
    return_intermediate_steps: bool = False
    """Whether or not to return the intermediate steps along with the final answer."""
    return_direct: bool = False
    """Optional cypher validation tool"""
    use_function_response: bool = False
    """Whether to wrap the database context as tool/function response"""
    allow_dangerous_requests: bool = False
    """Forced user opt-in to acknowledge that the chain can make dangerous requests.
    
    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )

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

    @property
    def _chain_type(self) -> str:
        return "graph_cypher_chain"

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        *,
        qa_prompt: Optional[BasePromptTemplate] = None,
        cypher_prompt: Optional[BasePromptTemplate] = None,
        cypher_llm: Optional[BaseLanguageModel] = None,
        qa_llm: Optional[Union[BaseLanguageModel, Any]] = None,
        qa_llm_kwargs: Optional[Dict[str, Any]] = None,
        cypher_llm_kwargs: Optional[Dict[str, Any]] = None,
        use_function_response: bool = False,
        function_response_system: str = FUNCTION_RESPONSE_SYSTEM,
        **kwargs: Any,
    ) -> MemgraphQAChain:
        """Initialize from LLM."""

        if not cypher_llm and not llm:
            raise ValueError("Either `llm` or `cypher_llm` parameters must be provided")
        if not qa_llm and not llm:
            raise ValueError("Either `llm` or `qa_llm` parameters must be provided")
        if cypher_llm and qa_llm and llm:
            raise ValueError(
                "You can specify up to two of 'cypher_llm', 'qa_llm'"
                ", and 'llm', but not all three simultaneously."
            )
        if cypher_prompt and cypher_llm_kwargs:
            raise ValueError(
                "Specifying cypher_prompt and cypher_llm_kwargs together is"
                " not allowed. Please pass prompt via cypher_llm_kwargs."
            )
        if qa_prompt and qa_llm_kwargs:
            raise ValueError(
                "Specifying qa_prompt and qa_llm_kwargs together is"
                " not allowed. Please pass prompt via qa_llm_kwargs."
            )
        use_qa_llm_kwargs = qa_llm_kwargs if qa_llm_kwargs is not None else {}
        use_cypher_llm_kwargs = (
            cypher_llm_kwargs if cypher_llm_kwargs is not None else {}
        )
        if "prompt" not in use_qa_llm_kwargs:
            use_qa_llm_kwargs["prompt"] = (
                qa_prompt if qa_prompt is not None else MEMGRAPH_QA_PROMPT
            )
        if "prompt" not in use_cypher_llm_kwargs:
            use_cypher_llm_kwargs["prompt"] = (
                cypher_prompt
                if cypher_prompt is not None
                else MEMGRAPH_GENERATION_PROMPT
            )

        qa_llm = qa_llm or llm
        if use_function_response:
            try:
                qa_llm.bind_tools({})  # type: ignore[union-attr]
                response_prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=function_response_system),
                        HumanMessagePromptTemplate.from_template("{question}"),
                        MessagesPlaceholder(variable_name="function_response"),
                    ]
                )
                qa_chain = response_prompt | qa_llm | StrOutputParser()  # type: ignore
            except (NotImplementedError, AttributeError):
                raise ValueError("Provided LLM does not support native tools/functions")
        else:
            qa_chain = use_qa_llm_kwargs["prompt"] | qa_llm | StrOutputParser()  # type: ignore

        prompt = use_cypher_llm_kwargs["prompt"]
        llm_to_use = cypher_llm if cypher_llm is not None else llm

        if prompt is not None and llm_to_use is not None:
            cypher_generation_chain = prompt | llm_to_use | StrOutputParser()  # type: ignore[arg-type]
        else:
            raise ValueError(
                "Missing required components for the cypher generation chain: "
                "'prompt' or 'llm'"
            )

        graph_schema = kwargs["graph"].get_schema

        return cls(
            graph_schema=graph_schema,
            qa_chain=qa_chain,
            cypher_generation_chain=cypher_generation_chain,
            use_function_response=use_function_response,
            **kwargs,
        )

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement, use it to look up in db and answer question."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self.input_key]
        args = {
            "question": question,
            "schema": self.graph_schema,
        }
        args.update(inputs)

        intermediate_steps: List = []

        generated_cypher = self.cypher_generation_chain.invoke(
            args, callbacks=callbacks
        )
        # Extract Cypher code if it is wrapped in backticks
        generated_cypher = extract_cypher(generated_cypher)

        _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_cypher, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_cypher})

        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_cypher:
            context = self.graph.query(generated_cypher)[: self.top_k]
        else:
            context = []

        if self.return_direct:
            result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )

            intermediate_steps.append({"context": context})
            if self.use_function_response:
                function_response = get_function_response(question, context)
                result = self.qa_chain.invoke(  # type: ignore
                    {"question": question, "function_response": function_response},
                )
            else:
                result = self.qa_chain.invoke(  # type: ignore
                    {"question": question, "context": context},
                    callbacks=callbacks,
                )

        chain_result: Dict[str, Any] = {"result": result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
