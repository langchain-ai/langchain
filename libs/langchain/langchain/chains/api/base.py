"""Chain that makes API calls and summarizes the responses to answer a question."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from langchain_core._api import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import Field, model_validator
from typing_extensions import Self

from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain


def _extract_scheme_and_domain(url: str) -> Tuple[str, str]:
    """Extract the scheme + domain from a given URL.

    Args:
        url (str): The input URL.

    Returns:
        return a 2-tuple of scheme and domain
    """
    parsed_uri = urlparse(url)
    return parsed_uri.scheme, parsed_uri.netloc


def _check_in_allowed_domain(url: str, limit_to_domains: Sequence[str]) -> bool:
    """Check if a URL is in the allowed domains.

    Args:
        url (str): The input URL.
        limit_to_domains (Sequence[str]): The allowed domains.

    Returns:
        bool: True if the URL is in the allowed domains, False otherwise.
    """
    scheme, domain = _extract_scheme_and_domain(url)

    for allowed_domain in limit_to_domains:
        allowed_scheme, allowed_domain = _extract_scheme_and_domain(allowed_domain)
        if scheme == allowed_scheme and domain == allowed_domain:
            return True
    return False


try:
    from langchain_community.utilities.requests import TextRequestsWrapper

    @deprecated(
        since="0.2.13",
        message=(
            "This class is deprecated and will be removed in langchain 1.0. "
            "See API reference for replacement: "
            "https://api.python.langchain.com/en/latest/chains/langchain.chains.api.base.APIChain.html"  # noqa: E501
        ),
        removal="1.0",
    )
    class APIChain(Chain):
        """Chain that makes API calls and summarizes the responses to answer a question.

        *Security Note*: This API chain uses the requests toolkit
            to make GET, POST, PATCH, PUT, and DELETE requests to an API.

            Exercise care in who is allowed to use this chain. If exposing
            to end users, consider that users will be able to make arbitrary
            requests on behalf of the server hosting the code. For example,
            users could ask the server to make a request to a private API
            that is only accessible from the server.

            Control access to who can submit issue requests using this toolkit and
            what network access it has.

            See https://python.langchain.com/docs/security for more information.

        Note: this class is deprecated. See below for a replacement implementation
        using LangGraph. The benefits of this implementation are:

        - Uses LLM tool calling features to encourage properly-formatted API requests;
        - Support for both token-by-token and step-by-step streaming;
        - Support for checkpointing and memory of chat history;
        - Easier to modify or extend (e.g., with additional tools, structured responses, etc.)

        Install LangGraph with:

        .. code-block:: bash

            pip install -U langgraph

        .. code-block:: python

            from typing import Annotated, Sequence
            from typing_extensions import TypedDict

            from langchain.chains.api.prompt import API_URL_PROMPT
            from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
            from langchain_community.utilities.requests import TextRequestsWrapper
            from langchain_core.messages import BaseMessage
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
            from langchain_core.runnables import RunnableConfig
            from langgraph.graph import END, StateGraph
            from langgraph.graph.message import add_messages
            from langgraph.prebuilt.tool_node import ToolNode

            # NOTE: There are inherent risks in giving models discretion
            # to execute real-world actions. We must "opt-in" to these
            # risks by setting allow_dangerous_request=True to use these tools.
            # This can be dangerous for calling unwanted requests. Please make
            # sure your custom OpenAPI spec (yaml) is safe and that permissions
            # associated with the tools are narrowly-scoped.
            ALLOW_DANGEROUS_REQUESTS = True

            # Subset of spec for https://jsonplaceholder.typicode.com
            api_spec = \"\"\"
            openapi: 3.0.0
            info:
              title: JSONPlaceholder API
              version: 1.0.0
            servers:
              - url: https://jsonplaceholder.typicode.com
            paths:
              /posts:
                get:
                  summary: Get posts
                  parameters: &id001
                    - name: _limit
                      in: query
                      required: false
                      schema:
                        type: integer
                      example: 2
                      description: Limit the number of results
            \"\"\"

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            toolkit = RequestsToolkit(
                requests_wrapper=TextRequestsWrapper(headers={}),  # no auth required
                allow_dangerous_requests=ALLOW_DANGEROUS_REQUESTS,
            )
            tools = toolkit.get_tools()

            api_request_chain = (
                API_URL_PROMPT.partial(api_docs=api_spec)
                | llm.bind_tools(tools, tool_choice="any")
            )

            class ChainState(TypedDict):
                \"\"\"LangGraph state.\"\"\"

                messages: Annotated[Sequence[BaseMessage], add_messages]


            async def acall_request_chain(state: ChainState, config: RunnableConfig):
                last_message = state["messages"][-1]
                response = await api_request_chain.ainvoke(
                    {"question": last_message.content}, config
                )
                return {"messages": [response]}

            async def acall_model(state: ChainState, config: RunnableConfig):
                response = await llm.ainvoke(state["messages"], config)
                return {"messages": [response]}

            graph_builder = StateGraph(ChainState)
            graph_builder.add_node("call_tool", acall_request_chain)
            graph_builder.add_node("execute_tool", ToolNode(tools))
            graph_builder.add_node("call_model", acall_model)
            graph_builder.set_entry_point("call_tool")
            graph_builder.add_edge("call_tool", "execute_tool")
            graph_builder.add_edge("execute_tool", "call_model")
            graph_builder.add_edge("call_model", END)
            chain = graph_builder.compile()

        .. code-block:: python

            example_query = "Fetch the top two posts. What are their titles?"

            events = chain.astream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            async for event in events:
                event["messages"][-1].pretty_print()
        """  # noqa: E501

        api_request_chain: LLMChain
        api_answer_chain: LLMChain
        requests_wrapper: TextRequestsWrapper = Field(exclude=True)
        api_docs: str
        question_key: str = "question"  #: :meta private:
        output_key: str = "output"  #: :meta private:
        limit_to_domains: Optional[Sequence[str]] = Field(
            default_factory=list  # type: ignore
        )
        """Use to limit the domains that can be accessed by the API chain.
        
        * For example, to limit to just the domain `https://www.example.com`, set
            `limit_to_domains=["https://www.example.com"]`.
            
        * The default value is an empty tuple, which means that no domains are
          allowed by default. By design this will raise an error on instantiation.
        * Use a None if you want to allow all domains by default -- this is not
          recommended for security reasons, as it would allow malicious users to
          make requests to arbitrary URLS including internal APIs accessible from
          the server.
        """

        @property
        def input_keys(self) -> List[str]:
            """Expect input key.

            :meta private:
            """
            return [self.question_key]

        @property
        def output_keys(self) -> List[str]:
            """Expect output key.

            :meta private:
            """
            return [self.output_key]

        @model_validator(mode="after")
        def validate_api_request_prompt(self) -> Self:
            """Check that api request prompt expects the right variables."""
            input_vars = self.api_request_chain.prompt.input_variables
            expected_vars = {"question", "api_docs"}
            if set(input_vars) != expected_vars:
                raise ValueError(
                    f"Input variables should be {expected_vars}, got {input_vars}"
                )
            return self

        @model_validator(mode="before")
        @classmethod
        def validate_limit_to_domains(cls, values: Dict) -> Any:
            """Check that allowed domains are valid."""
            # This check must be a pre=True check, so that a default of None
            # won't be set to limit_to_domains if it's not provided.
            if "limit_to_domains" not in values:
                raise ValueError(
                    "You must specify a list of domains to limit access using "
                    "`limit_to_domains`"
                )
            if (
                not values["limit_to_domains"]
                and values["limit_to_domains"] is not None
            ):
                raise ValueError(
                    "Please provide a list of domains to limit access using "
                    "`limit_to_domains`."
                )
            return values

        @model_validator(mode="after")
        def validate_api_answer_prompt(self) -> Self:
            """Check that api answer prompt expects the right variables."""
            input_vars = self.api_answer_chain.prompt.input_variables
            expected_vars = {"question", "api_docs", "api_url", "api_response"}
            if set(input_vars) != expected_vars:
                raise ValueError(
                    f"Input variables should be {expected_vars}, got {input_vars}"
                )
            return self

        def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Dict[str, str]:
            _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
            question = inputs[self.question_key]
            api_url = self.api_request_chain.predict(
                question=question,
                api_docs=self.api_docs,
                callbacks=_run_manager.get_child(),
            )
            _run_manager.on_text(api_url, color="green", end="\n", verbose=self.verbose)
            api_url = api_url.strip()
            if self.limit_to_domains and not _check_in_allowed_domain(
                api_url, self.limit_to_domains
            ):
                raise ValueError(
                    f"{api_url} is not in the allowed domains: {self.limit_to_domains}"
                )
            api_response = self.requests_wrapper.get(api_url)
            _run_manager.on_text(
                str(api_response), color="yellow", end="\n", verbose=self.verbose
            )
            answer = self.api_answer_chain.predict(
                question=question,
                api_docs=self.api_docs,
                api_url=api_url,
                api_response=api_response,
                callbacks=_run_manager.get_child(),
            )
            return {self.output_key: answer}

        async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        ) -> Dict[str, str]:
            _run_manager = (
                run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
            )
            question = inputs[self.question_key]
            api_url = await self.api_request_chain.apredict(
                question=question,
                api_docs=self.api_docs,
                callbacks=_run_manager.get_child(),
            )
            await _run_manager.on_text(
                api_url, color="green", end="\n", verbose=self.verbose
            )
            api_url = api_url.strip()
            if self.limit_to_domains and not _check_in_allowed_domain(
                api_url, self.limit_to_domains
            ):
                raise ValueError(
                    f"{api_url} is not in the allowed domains: {self.limit_to_domains}"
                )
            api_response = await self.requests_wrapper.aget(api_url)
            await _run_manager.on_text(
                str(api_response), color="yellow", end="\n", verbose=self.verbose
            )
            answer = await self.api_answer_chain.apredict(
                question=question,
                api_docs=self.api_docs,
                api_url=api_url,
                api_response=api_response,
                callbacks=_run_manager.get_child(),
            )
            return {self.output_key: answer}

        @classmethod
        def from_llm_and_api_docs(
            cls,
            llm: BaseLanguageModel,
            api_docs: str,
            headers: Optional[dict] = None,
            api_url_prompt: BasePromptTemplate = API_URL_PROMPT,
            api_response_prompt: BasePromptTemplate = API_RESPONSE_PROMPT,
            limit_to_domains: Optional[Sequence[str]] = tuple(),
            **kwargs: Any,
        ) -> APIChain:
            """Load chain from just an LLM and the api docs."""
            get_request_chain = LLMChain(llm=llm, prompt=api_url_prompt)
            requests_wrapper = TextRequestsWrapper(headers=headers)
            get_answer_chain = LLMChain(llm=llm, prompt=api_response_prompt)
            return cls(
                api_request_chain=get_request_chain,
                api_answer_chain=get_answer_chain,
                requests_wrapper=requests_wrapper,
                api_docs=api_docs,
                limit_to_domains=limit_to_domains,
                **kwargs,
            )

        @property
        def _chain_type(self) -> str:
            return "api_chain"

except ImportError:

    class APIChain:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "To use the APIChain, you must install the langchain_community package."
                "pip install langchain_community"
            )
