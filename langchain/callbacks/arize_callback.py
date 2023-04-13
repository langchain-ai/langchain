# Import the necessary packages for ingestion
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class ArizeCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to Arize platform."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        SPACE_KEY: Optional[str] = None,
        API_KEY: Optional[str] = None,
    ) -> None:
        """Initialize callback handler."""

        super().__init__()

        # Set the model_id and model_version for the Arize monitoring.
        self.model_id = model_id
        self.model_version = model_version

        # Set the SPACE_KEY and API_KEY for the Arize client.
        self.space_key = SPACE_KEY
        self.api_key = API_KEY

        # Initialize empty lists to store the prompt/response pairs
        # and other necessary data.
        self.prompt_records: List = []
        self.response_records: List = []
        self.prediction_ids: List = []
        self.pred_timestamps: List = []
        self.response_embeddings: List = []
        self.prompt_embeddings: List = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        from arize.api import Client
        from arize.pandas.embeddings import EmbeddingGenerator, UseCases

        # Create an embedding generator for generating embeddings
        # from prompts and responses.
        self.generator = EmbeddingGenerator.from_use_case(
            use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
            model_name="distilbert-base-uncased",
            tokenizer_max_length=512,
            batch_size=256,
        )

        # Create an Arize client and check if the SPACE_KEY and API_KEY
        # are not set to the default values.
        self.arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)
        if SPACE_KEY == "SPACE_KEY" or API_KEY == "API_KEY":
            raise ValueError("❌ CHANGE SPACE AND API KEYS")
        else:
            print("✅ Arize client setup done! Now you can start using Arize!")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Record the prompts when an LLM starts."""

        for prompt in prompts:
            self.prompt_records.append(prompt.replace("\n", " "))

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing when a new token is generated."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log data to Arize when an LLM ends."""

        from arize.utils.types import Embedding, Environments, ModelTypes

        # Record token usage of the LLM
        if response.llm_output is not None:
            self.prompt_tokens = response.llm_output["token_usage"]["prompt_tokens"]
            self.total_tokens = response.llm_output["token_usage"]["total_tokens"]
            self.completion_tokens = response.llm_output["token_usage"][
                "completion_tokens"
            ]
        i = 0

        # Go through each prompt response pair and generate embeddings as
        # well as timestamp and prediction ids
        for generations in response.generations:
            for generation in generations:
                prompt = self.prompt_records[i]
                prompt_embedding = pd.Series(
                    self.generator.generate_embeddings(
                        text_col=pd.Series(prompt.replace("\n", " "))
                    ).reset_index(drop=True)
                )
                generated_text = generation.text.replace("\n", " ")
                response_embedding = pd.Series(
                    self.generator.generate_embeddings(
                        text_col=pd.Series(generation.text.replace("\n", " "))
                    ).reset_index(drop=True)
                )
                pred_id = str(uuid.uuid4())

                # Define embedding features for Arize ingestion
                embedding_features = {
                    "prompt_embedding": Embedding(
                        vector=pd.Series(prompt_embedding[0]), data=prompt
                    ),
                    "response_embedding": Embedding(
                        vector=pd.Series(response_embedding[0]), data=generated_text
                    ),
                }
                tags = {
                    "Prompt Tokens": self.prompt_tokens,
                    "Completion Tokens": self.completion_tokens,
                    "Total Tokens": self.total_tokens,
                }

                # Log each prompt response data into arize
                future = self.arize_client.log(
                    prediction_id=pred_id,
                    tags=tags,
                    prediction_label="1",
                    model_id=self.model_id,
                    model_type=ModelTypes.SCORE_CATEGORICAL,
                    model_version=self.model_version,
                    environment=Environments.PRODUCTION,
                    embedding_features=embedding_features,
                )

                result = future.result()
                if result.status_code == 200:
                    print("✅ Successfully logged data to Arize!")
                else:
                    print(
                        f"❌ Logging failed with status code {result.status_code}"
                        f' and message "{result.text}"'
                    )

                i = i + 1

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Do nothing when LLM outputs an error."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain starts."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Do nothing when LLM chain ends."""
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
