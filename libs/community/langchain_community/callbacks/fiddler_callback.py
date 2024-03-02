import time
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_community.callbacks.utils import import_pandas

# Define constants

# LLMResult keys
TOKEN_USAGE = "token_usage"
TOTAL_TOKENS = "total_tokens"
PROMPT_TOKENS = "prompt_tokens"
COMPLETION_TOKENS = "completion_tokens"
RUN_ID = "run_id"
MODEL_NAME = "model_name"

# Default values
DEFAULT_MAX_TOKEN = 65536
DEFAULT_MAX_DURATION = 120

# Fiddler specific constants
PROMPT = "prompt"
RESPONSE = "response"
DURATION = "duration"

# Define a dataset dictionary
_dataset_dict = {
    PROMPT: ["fiddler"] * 10,
    RESPONSE: ["fiddler"] * 10,
    MODEL_NAME: ["fiddler"] * 10,
    RUN_ID: ["123e4567-e89b-12d3-a456-426614174000"] * 10,
    TOTAL_TOKENS: [0, DEFAULT_MAX_TOKEN] * 5,
    PROMPT_TOKENS: [0, DEFAULT_MAX_TOKEN] * 5,
    COMPLETION_TOKENS: [0, DEFAULT_MAX_TOKEN] * 5,
    DURATION: [1, DEFAULT_MAX_DURATION] * 5,
}


def import_fiddler() -> Any:
    """Import the fiddler python package and raise an error if it is not installed."""
    try:
        import fiddler  # noqa: F401
    except ImportError:
        raise ImportError(
            "To use fiddler callback handler you need to have `fiddler-client`"
            "package installed. Please install it with `pip install fiddler-client`"
        )
    return fiddler


# First, define custom callback handler implementations
class FiddlerCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        url: str,
        org: str,
        project: str,
        model: str,
        api_key: str,
    ) -> None:
        """
        Initialize Fiddler callback handler.

        Args:
            url: Fiddler URL (e.g. https://demo.fiddler.ai).
                Make sure to include the protocol (http/https).
            org: Fiddler organization id
            project: Fiddler project name to publish events to
            model: Fiddler model name to publish events to
            api_key: Fiddler authentication token
        """
        super().__init__()
        # Initialize Fiddler client and other necessary properties
        self.fdl = import_fiddler()
        self.pd = import_pandas()

        self.url = url
        self.org = org
        self.project = project
        self.model = model
        self.api_key = api_key
        self._df = self.pd.DataFrame(_dataset_dict)

        self.run_id_prompts: Dict[str, List[str]] = {}
        self.run_id_starttime: Dict[str, int] = {}

        # Initialize Fiddler client here
        self.fiddler_client = self.fdl.FiddlerApi(url, org_id=org, auth_token=api_key)

        if self.project not in self.fiddler_client.get_project_names():
            print(  # noqa: T201
                f"adding project {self.project}." "This only has to be done once."
            )
            try:
                self.fiddler_client.add_project(self.project)
            except Exception as e:
                print(  # noqa: T201
                    f"Error adding project {self.project}:"
                    "{e}. Fiddler integration will not work."
                )
                raise e

        dataset_info = self.fdl.DatasetInfo.from_dataframe(
            self._df, max_inferred_cardinality=0
        )
        if self.model not in self.fiddler_client.get_dataset_names(self.project):
            print(  # noqa: T201
                f"adding dataset {self.model} to project {self.project}."
                "This only has to be done once."
            )
            try:
                self.fiddler_client.upload_dataset(
                    project_id=self.project,
                    dataset_id=self.model,
                    dataset={"train": self._df},
                    info=dataset_info,
                )
            except Exception as e:
                print(  # noqa: T201
                    f"Error adding dataset {self.model}: {e}."
                    "Fiddler integration will not work."
                )
                raise e

        model_info = self.fdl.ModelInfo.from_dataset_info(
            dataset_info=dataset_info,
            dataset_id="train",
            model_task=self.fdl.ModelTask.LLM,
            features=[PROMPT, RESPONSE],
            metadata_cols=[
                RUN_ID,
                TOTAL_TOKENS,
                PROMPT_TOKENS,
                COMPLETION_TOKENS,
                MODEL_NAME,
            ],
            custom_features=self.custom_features,
        )

        if self.model not in self.fiddler_client.get_model_names(self.project):
            print(  # noqa: T201
                f"adding model {self.model} to project {self.project}."
                "This only has to be done once."  # noqa: T201
            )
            try:
                self.fiddler_client.add_model(
                    project_id=self.project,
                    dataset_id=self.model,
                    model_id=self.model,
                    model_info=model_info,
                )
            except Exception as e:
                print(  # noqa: T201
                    f"Error adding model {self.model}: {e}."
                    "Fiddler integration will not work."  # noqa: T201
                )
                raise e

    @property
    def custom_features(self) -> list:
        """
        Define custom features for the model to automatically enrich the data with.
        Here, we enable the following enrichments:
        - Automatic Embedding generation for prompt and response
        - Text Statistics such as:
            - Automated Readability Index
            - Coleman Liau Index
            - Dale Chall Readability Score
            - Difficult Words
            - Flesch Reading Ease
            - Flesch Kincaid Grade
            - Gunning Fog
            - Linsear Write Formula
        - PII - Personal Identifiable Information
        - Sentiment Analysis

        """

        return [
            self.fdl.Enrichment(
                name="Prompt Embedding",
                enrichment="embedding",
                columns=[PROMPT],
            ),
            self.fdl.TextEmbedding(
                name="Prompt CF",
                source_column=PROMPT,
                column="Prompt Embedding",
            ),
            self.fdl.Enrichment(
                name="Response Embedding",
                enrichment="embedding",
                columns=[RESPONSE],
            ),
            self.fdl.TextEmbedding(
                name="Response CF",
                source_column=RESPONSE,
                column="Response Embedding",
            ),
            self.fdl.Enrichment(
                name="Text Statistics",
                enrichment="textstat",
                columns=[PROMPT, RESPONSE],
                config={
                    "statistics": [
                        "automated_readability_index",
                        "coleman_liau_index",
                        "dale_chall_readability_score",
                        "difficult_words",
                        "flesch_reading_ease",
                        "flesch_kincaid_grade",
                        "gunning_fog",
                        "linsear_write_formula",
                    ]
                },
            ),
            self.fdl.Enrichment(
                name="PII",
                enrichment="pii",
                columns=[PROMPT, RESPONSE],
            ),
            self.fdl.Enrichment(
                name="Sentiment",
                enrichment="sentiment",
                columns=[PROMPT, RESPONSE],
            ),
        ]

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        run_id = kwargs[RUN_ID]
        self.run_id_prompts[run_id] = prompts
        self.run_id_starttime[run_id] = int(time.time())

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        flattened_llmresult = response.flatten()
        token_usage_dict = {}
        run_id = kwargs[RUN_ID]
        run_duration = self.run_id_starttime[run_id] - int(time.time())
        prompt_responses = []
        model_name = ""

        if isinstance(response.llm_output, dict):
            if TOKEN_USAGE in response.llm_output:
                token_usage_dict = response.llm_output[TOKEN_USAGE]
            if MODEL_NAME in response.llm_output:
                model_name = response.llm_output[MODEL_NAME]

        for llmresult in flattened_llmresult:
            prompt_responses.append(llmresult.generations[0][0].text)

        df = self.pd.DataFrame(
            {
                PROMPT: self.run_id_prompts[run_id],
                RESPONSE: prompt_responses,
            }
        )

        if TOTAL_TOKENS in token_usage_dict:
            df[PROMPT_TOKENS] = int(token_usage_dict[TOTAL_TOKENS])

        if PROMPT_TOKENS in token_usage_dict:
            df[TOTAL_TOKENS] = int(token_usage_dict[PROMPT_TOKENS])

        if COMPLETION_TOKENS in token_usage_dict:
            df[COMPLETION_TOKENS] = token_usage_dict[COMPLETION_TOKENS]

        df[MODEL_NAME] = model_name
        df[RUN_ID] = str(run_id)
        df[DURATION] = run_duration

        try:
            self.fiddler_client.publish_events_batch(self.project, self.model, df)
        except Exception as e:
            print(f"Error publishing events to fiddler: {e}. continuing...")  # noqa: T201
