from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_community.callbacks.utils import import_pandas


class ArizeCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to Arize."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        SPACE_KEY: Optional[str] = None,
        API_KEY: Optional[str] = None,
    ) -> None:
        """Initialize callback handler."""

        super().__init__()
        self.model_id = model_id
        self.model_version = model_version
        self.space_key = SPACE_KEY
        self.api_key = API_KEY
        self.prompt_records: List[str] = []
        self.response_records: List[str] = []
        self.prediction_ids: List[str] = []
        self.pred_timestamps: List[int] = []
        self.response_embeddings: List[float] = []
        self.prompt_embeddings: List[float] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.step = 0

        from arize.pandas.embeddings import EmbeddingGenerator, UseCases
        from arize.pandas.logger import Client

        self.generator = EmbeddingGenerator.from_use_case(
            use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
            model_name="distilbert-base-uncased",
            tokenizer_max_length=512,
            batch_size=256,
        )
        self.arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)
        if SPACE_KEY == "SPACE_KEY" or API_KEY == "API_KEY":
            raise ValueError("❌ CHANGE SPACE AND API KEYS")
        else:
            print("✅ Arize client setup done! Now you can start using Arize!")  # noqa: T201

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        for prompt in prompts:
            self.prompt_records.append(prompt.replace("\n", ""))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pd = import_pandas()
        from arize.utils.types import (
            EmbeddingColumnNames,
            Environments,
            ModelTypes,
            Schema,
        )

        # Safe check if 'llm_output' and 'token_usage' exist
        if response.llm_output and "token_usage" in response.llm_output:
            self.prompt_tokens = response.llm_output["token_usage"].get(
                "prompt_tokens", 0
            )
            self.total_tokens = response.llm_output["token_usage"].get(
                "total_tokens", 0
            )
            self.completion_tokens = response.llm_output["token_usage"].get(
                "completion_tokens", 0
            )
        else:
            self.prompt_tokens = (
                self.total_tokens
            ) = self.completion_tokens = 0  # assign default value

        for generations in response.generations:
            for generation in generations:
                prompt = self.prompt_records[self.step]
                self.step = self.step + 1
                prompt_embedding = pd.Series(
                    self.generator.generate_embeddings(
                        text_col=pd.Series(prompt.replace("\n", " "))
                    ).reset_index(drop=True)
                )

                # Assigning text to response_text instead of response
                response_text = generation.text.replace("\n", " ")
                response_embedding = pd.Series(
                    self.generator.generate_embeddings(
                        text_col=pd.Series(generation.text.replace("\n", " "))
                    ).reset_index(drop=True)
                )
                pred_timestamp = datetime.now().timestamp()

                # Define the columns and data
                columns = [
                    "prediction_ts",
                    "response",
                    "prompt",
                    "response_vector",
                    "prompt_vector",
                    "prompt_token",
                    "completion_token",
                    "total_token",
                ]
                data = [
                    [
                        pred_timestamp,
                        response_text,
                        prompt,
                        response_embedding[0],
                        prompt_embedding[0],
                        self.prompt_tokens,
                        self.total_tokens,
                        self.completion_tokens,
                    ]
                ]

                # Create the DataFrame
                df = pd.DataFrame(data, columns=columns)

                # Declare prompt and response columns
                prompt_columns = EmbeddingColumnNames(
                    vector_column_name="prompt_vector", data_column_name="prompt"
                )

                response_columns = EmbeddingColumnNames(
                    vector_column_name="response_vector", data_column_name="response"
                )

                schema = Schema(
                    timestamp_column_name="prediction_ts",
                    tag_column_names=[
                        "prompt_token",
                        "completion_token",
                        "total_token",
                    ],
                    prompt_column_names=prompt_columns,
                    response_column_names=response_columns,
                )

                response_from_arize = self.arize_client.log(
                    dataframe=df,
                    schema=schema,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    model_type=ModelTypes.GENERATIVE_LLM,
                    environment=Environments.PRODUCTION,
                )
                if response_from_arize.status_code == 200:
                    print("✅ Successfully logged data to Arize!")  # noqa: T201
                else:
                    print(f'❌ Logging failed "{response_from_arize.text}"')  # noqa: T201

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing."""
        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing."""
        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        pass

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        pass
