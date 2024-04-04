"""UpTrain's Callback Handler."""
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)
from uuid import UUID

from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages.base import BaseMessage
from uptrain import APIClient, EvalLLM, Evals, Settings


def import_uptrain() -> Any:
    try:
        import uptrain
    except ImportError as e:
        raise ImportError(
            "To use the UpTrainCallbackHandler, you need the"
            "`uptrain` package. Please install it with"
            "`pip install uptrain`.",
            e
        )

    return uptrain

class UpTrainDataSchema:
    """UpTrain Data Schema"""

    def __init__(self, project_name_prefix: str) -> None:
        """Initialize the UpTrain data schema."""
        # For tracking project name and results
        self.project_name_prefix: str = project_name_prefix
        self.uptrain_results: DefaultDict[str, Any] = defaultdict(list)

        # For tracking event types
        self.eval_types: Set[str] = set()

        ## RAG
        self.query: str = ""
        self.context: str = ""
        self.response: str = ""

        ## CONTEXT CONCISENESS
        self.old_context: list[str] = []
        self.new_context: list[str] = []
        self.context_conciseness_run_id: str = ""

        # MULTI QUERY
        self.multi_queries: List[str] = []
        self.multi_query_run_id: str = ""
        self.multi_query_daugher_run_id: str = ""

class UpTrainCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to uptrain.

    Args:
        BaseCallbackHandler (_type_): _description_
    """

    REPO_URL: str = "https://github.com/uptrain-ai/uptrain"
    ISSUES_URL: str = f"{REPO_URL}/issues"
    DOCS_URL: str = "https://docs.uptrain.ai"

    def __init__(
            self,
            project_name_prefix: str = "langchain",
            key_type: str = "openai",
            api_key: str = "sk-****************",
            ) -> None:
        """Initializes the `UpTrainCallbackHandler`."""
        super().__init__()

        import_uptrain()

        # Set uptrain variables
        self.schema = UpTrainDataSchema(project_name_prefix=project_name_prefix)
        self.first_score_printed_flag = False


        # Based on whether the user enters an UpTrain API key or an OpenAI API key, the client is initialized
        # If both are entered, the UpTrain API key is used
        if key_type == "uptrain":
            settings = Settings(uptrain_access_token=api_key)
            self.uptrain_client = APIClient(settings=settings)
        elif key_type == "openai":
            settings = Settings(openai_api_key=api_key, evaluate_locally=False)
            self.uptrain_client = EvalLLM(settings=settings)
        else:
            raise ValueError("Invalid key type: Must be 'uptrain' or 'openai'")

            
        
    def uptrain_evaluate(
        self,
        project_name: str,
        data: List[Dict[str, str]],
        checks: List[str],
    ) -> None:
        """Run an evaluation on the UpTrain server using UpTrain client."""
        if self.uptrain_client.__class__.__name__ == "APIClient":
            uptrain_result = self.uptrain_client.log_and_evaluate(
                project_name=project_name,
                data=data,
                checks=checks,
            )
        else:
            uptrain_result = self.uptrain_client.evaluate(
                data=data,
                checks=checks,
            )
        self.schema.uptrain_results[project_name].append(uptrain_result)

        score_name_map = {
            "score_context_relevance": "Context Relevance Score",
            "score_factual_accuracy": "Factual Accuracy Score",
            "score_response_completeness": "Response Completeness Score",
            "score_sub_query_completeness": "Sub Query Completeness Score",
            "score_context_reranking": "Context Reranking Score",
            "score_context_conciseness": "Context Conciseness Score",
            "score_multi_query_accuracy": "Multi Query Accuracy Score",
        }

        for row in uptrain_result:
            columns = list(row.keys())
            for column in columns:
                if column == "question":
                    print(f"\nQuestion: {row[column]}")
                    self.first_score_printed_flag = False
                elif column == "response":
                    print(f"Response: {row[column]}")
                    self.first_score_printed_flag = False
                elif column == "variants":
                    print(f"Multi Queries:")
                    for variant in row[column]:
                        print(f"  {variant}")
                    self.first_score_printed_flag = False
                elif column.startswith("score"):
                    if not self.first_score_printed_flag:
                        print()
                        self.first_score_printed_flag = True
                    if column in score_name_map:
                        print(f"{score_name_map[column]}: {row[column]}")
                    else:
                        print(f"{column}: {row[column]}")
            print()

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Store the prompts"""
        pass
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log records to uptrain when an LLM ends."""
        self.schema.response = response.generations[0][0].text
        if "qa_rag" in self.schema.eval_types and parent_run_id != self.schema.multi_query_daugher_run_id:
            data = [
                {
                    "question": self.schema.query,
                    "context": self.schema.context,
                    "response": self.schema.response,
                }
            ]

            self.uptrain_evaluate(
                project_name=f"{self.schema.project_name_prefix}",
                data=data,
                checks=[
                    Evals.CONTEXT_RELEVANCE,
                    Evals.FACTUAL_ACCURACY,
                    Evals.RESPONSE_COMPLETENESS,
                ],
            )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        pass


    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:        
        """Do nothing when chain starts"""
        if parent_run_id == self.schema.multi_query_run_id:
            self.schema.multi_query_daugher_run_id = run_id
        if isinstance(inputs, dict) and set(inputs.keys()) == {"context", "question"}:
            self.schema.eval_types.add("qa_rag")

            context = ""
            if isinstance(inputs["context"], Document):
                context = inputs["context"].page_content
            elif isinstance(inputs["context"], list):
                for doc in inputs["context"]:
                    context += doc.page_content + "\n"
            elif isinstance(inputs["context"], str):
                context = inputs["context"]
            self.schema.context = context
            self.schema.query = inputs["question"]
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when chain ends."""
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain outputs an error."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when tool outputs an error."""
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
        pass

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if "contextual_compression" in serialized["id"]:
            self.schema.eval_types.add("contextual_compression")
            self.schema.query = query
            self.schema.context_conciseness_run_id = run_id

        if "multi_query" in serialized["id"]:
            self.schema.eval_types.add("multi_query")
            self.schema.multi_query_run_id = run_id
            self.schema.query = query
        elif "multi_query" in self.schema.eval_types:
            self.schema.multi_queries.append(query)


    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        if run_id == self.schema.multi_query_run_id:
            data = [
                {
                    "question": self.schema.query,
                    "variants": self.schema.multi_queries,
                }
            ]

            self.uptrain_evaluate(
                project_name=f"{self.schema.project_name_prefix}",
                data=data,
                checks=[
                    Evals.MULTI_QUERY_ACCURACY
                ],
            )
        if "contextual_compression" in self.schema.eval_types:
            # Get the original context
            if parent_run_id == self.schema.context_conciseness_run_id:
                for doc in documents:
                    self.schema.old_context.append(doc.page_content)
            elif run_id == self.schema.context_conciseness_run_id:
                for doc in documents:
                    self.schema.new_context.append(doc.page_content)
                context = "\n".join(
                    [
                        f"{index}. {string}"
                        for index, string in enumerate(self.schema.old_context, start=1)
                    ]
                )
                reranked_context = "\n".join(
                    [
                        f"{index}. {string}"
                        for index, string in enumerate(self.schema.new_context, start=1)
                    ]
                )
                data = [
                    {
                        "question": self.schema.query,
                        "context": context,
                        "concise_context": reranked_context,
                        "reranked_context": reranked_context,
                    }
                ]
                self.uptrain_evaluate(
                    project_name=f"{self.schema.project_name_prefix}_context_conciseness",
                    data=data,
                    checks=[
                        Evals.CONTEXT_CONCISENESS,
                        Evals.CONTEXT_RERANKING,
                    ],
                )
