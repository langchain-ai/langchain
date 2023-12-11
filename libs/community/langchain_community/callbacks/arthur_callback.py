"""ArthurAI's Callback Handler."""
from __future__ import annotations

import os
import uuid
from collections import defaultdict
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, Optional

import numpy as np
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

if TYPE_CHECKING:
    import arthurai
    from arthurai.core.models import ArthurModel

PROMPT_TOKENS = "prompt_tokens"
COMPLETION_TOKENS = "completion_tokens"
TOKEN_USAGE = "token_usage"
FINISH_REASON = "finish_reason"
DURATION = "duration"


def _lazy_load_arthur() -> arthurai:
    """Lazy load Arthur."""
    try:
        import arthurai
    except ImportError as e:
        raise ImportError(
            "To use the ArthurCallbackHandler you need the"
            " `arthurai` package. Please install it with"
            " `pip install arthurai`.",
            e,
        )

    return arthurai


class ArthurCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs to Arthur platform.

    Arthur helps enterprise teams optimize model operations
    and performance at scale. The Arthur API tracks model
    performance, explainability, and fairness across tabular,
    NLP, and CV models. Our API is model- and platform-agnostic,
    and continuously scales with complex and dynamic enterprise needs.
    To learn more about Arthur, visit our website at
    https://www.arthur.ai/ or read the Arthur docs at
    https://docs.arthur.ai/
    """

    def __init__(
        self,
        arthur_model: ArthurModel,
    ) -> None:
        """Initialize callback handler."""
        super().__init__()
        arthurai = _lazy_load_arthur()
        Stage = arthurai.common.constants.Stage
        ValueType = arthurai.common.constants.ValueType
        self.arthur_model = arthur_model
        # save the attributes of this model to be used when preparing
        # inferences to log to Arthur in on_llm_end()
        self.attr_names = set([a.name for a in self.arthur_model.get_attributes()])
        self.input_attr = [
            x
            for x in self.arthur_model.get_attributes()
            if x.stage == Stage.ModelPipelineInput
            and x.value_type == ValueType.Unstructured_Text
        ][0].name
        self.output_attr = [
            x
            for x in self.arthur_model.get_attributes()
            if x.stage == Stage.PredictedValue
            and x.value_type == ValueType.Unstructured_Text
        ][0].name
        self.token_likelihood_attr = None
        if (
            len(
                [
                    x
                    for x in self.arthur_model.get_attributes()
                    if x.value_type == ValueType.TokenLikelihoods
                ]
            )
            > 0
        ):
            self.token_likelihood_attr = [
                x
                for x in self.arthur_model.get_attributes()
                if x.value_type == ValueType.TokenLikelihoods
            ][0].name

        self.run_map: DefaultDict[str, Any] = defaultdict(dict)

    @classmethod
    def from_credentials(
        cls,
        model_id: str,
        arthur_url: Optional[str] = "https://app.arthur.ai",
        arthur_login: Optional[str] = None,
        arthur_password: Optional[str] = None,
    ) -> ArthurCallbackHandler:
        """Initialize callback handler from Arthur credentials.

        Args:
            model_id (str): The ID of the arthur model to log to.
            arthur_url (str, optional): The URL of the Arthur instance to log to.
                Defaults to "https://app.arthur.ai".
            arthur_login (str, optional): The login to use to connect to Arthur.
                Defaults to None.
            arthur_password (str, optional): The password to use to connect to
                Arthur. Defaults to None.

        Returns:
            ArthurCallbackHandler: The initialized callback handler.
        """
        arthurai = _lazy_load_arthur()
        ArthurAI = arthurai.ArthurAI
        ResponseClientError = arthurai.common.exceptions.ResponseClientError

        # connect to Arthur
        if arthur_login is None:
            try:
                arthur_api_key = os.environ["ARTHUR_API_KEY"]
            except KeyError:
                raise ValueError(
                    "No Arthur authentication provided. Either give"
                    " a login to the ArthurCallbackHandler"
                    " or set an ARTHUR_API_KEY as an environment variable."
                )
            arthur = ArthurAI(url=arthur_url, access_key=arthur_api_key)
        else:
            if arthur_password is None:
                arthur = ArthurAI(url=arthur_url, login=arthur_login)
            else:
                arthur = ArthurAI(
                    url=arthur_url, login=arthur_login, password=arthur_password
                )
        # get model from Arthur by the provided model ID
        try:
            arthur_model = arthur.get_model(model_id)
        except ResponseClientError:
            raise ValueError(
                f"Was unable to retrieve model with id {model_id} from Arthur."
                " Make sure the ID corresponds to a model that is currently"
                " registered with your Arthur account."
            )
        return cls(arthur_model)

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """On LLM start, save the input prompts"""
        run_id = kwargs["run_id"]
        self.run_map[run_id]["input_texts"] = prompts
        self.run_map[run_id]["start_time"] = time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """On LLM end, send data to Arthur."""
        try:
            import pytz  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "Could not import pytz. Please install it with 'pip install pytz'."
            ) from e

        run_id = kwargs["run_id"]

        # get the run params from this run ID,
        # or raise an error if this run ID has no corresponding metadata in self.run_map
        try:
            run_map_data = self.run_map[run_id]
        except KeyError as e:
            raise KeyError(
                "This function has been called with a run_id"
                " that was never registered in on_llm_start()."
                " Restart and try running the LLM again"
            ) from e

        # mark the duration time between on_llm_start() and on_llm_end()
        time_from_start_to_end = time() - run_map_data["start_time"]

        # create inferences to log to Arthur
        inferences = []
        for i, generations in enumerate(response.generations):
            for generation in generations:
                inference = {
                    "partner_inference_id": str(uuid.uuid4()),
                    "inference_timestamp": datetime.now(tz=pytz.UTC),
                    self.input_attr: run_map_data["input_texts"][i],
                    self.output_attr: generation.text,
                }

                if generation.generation_info is not None:
                    # add finish reason to the inference
                    # if generation info contains a finish reason and
                    # if the ArthurModel was registered to monitor finish_reason
                    if (
                        FINISH_REASON in generation.generation_info
                        and FINISH_REASON in self.attr_names
                    ):
                        inference[FINISH_REASON] = generation.generation_info[
                            FINISH_REASON
                        ]

                    # add token likelihoods data to the inference if the ArthurModel
                    # was registered to monitor token likelihoods
                    logprobs_data = generation.generation_info["logprobs"]
                    if (
                        logprobs_data is not None
                        and self.token_likelihood_attr is not None
                    ):
                        logprobs = logprobs_data["top_logprobs"]
                        likelihoods = [
                            {k: np.exp(v) for k, v in logprobs[i].items()}
                            for i in range(len(logprobs))
                        ]
                        inference[self.token_likelihood_attr] = likelihoods

                # add token usage counts to the inference if the
                # ArthurModel was registered to monitor token usage
                if (
                    isinstance(response.llm_output, dict)
                    and TOKEN_USAGE in response.llm_output
                ):
                    token_usage = response.llm_output[TOKEN_USAGE]
                    if (
                        PROMPT_TOKENS in token_usage
                        and PROMPT_TOKENS in self.attr_names
                    ):
                        inference[PROMPT_TOKENS] = token_usage[PROMPT_TOKENS]
                    if (
                        COMPLETION_TOKENS in token_usage
                        and COMPLETION_TOKENS in self.attr_names
                    ):
                        inference[COMPLETION_TOKENS] = token_usage[COMPLETION_TOKENS]

                # add inference duration to the inference if the ArthurModel
                # was registered to monitor inference duration
                if DURATION in self.attr_names:
                    inference[DURATION] = time_from_start_to_end

                inferences.append(inference)

        # send inferences to arthur
        self.arthur_model.send_inferences(inferences)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """On chain start, do nothing."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """On chain end, do nothing."""

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when LLM outputs an error."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """On new token, pass."""

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when LLM chain outputs an error."""

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Do nothing when agent takes a specific action."""

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Do nothing when tool outputs an error."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Do nothing"""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Do nothing"""
