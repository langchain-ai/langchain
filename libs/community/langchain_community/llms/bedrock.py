from __future__ import annotations

import asyncio
import json
import warnings
from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env

from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.anthropic import (
    get_num_tokens_anthropic,
    get_token_ids_anthropic,
)

if TYPE_CHECKING:
    from botocore.config import Config

AMAZON_BEDROCK_TRACE_KEY = "amazon-bedrock-trace"
GUARDRAILS_BODY_KEY = "amazon-bedrock-guardrailAssessment"
HUMAN_PROMPT = "\n\nHuman:"
ASSISTANT_PROMPT = "\n\nAssistant:"
ALTERNATION_ERROR = (
    "Error: Prompt must alternate between '\n\nHuman:' and '\n\nAssistant:'."
)


def _add_newlines_before_ha(input_text: str) -> str:
    new_text = input_text
    for word in ["Human:", "Assistant:"]:
        new_text = new_text.replace(word, "\n\n" + word)
        for i in range(2):
            new_text = new_text.replace("\n\n\n" + word, "\n\n" + word)
    return new_text


def _human_assistant_format(input_text: str) -> str:
    if input_text.count("Human:") == 0 or (
        input_text.find("Human:") > input_text.find("Assistant:")
        and "Assistant:" in input_text
    ):
        input_text = HUMAN_PROMPT + " " + input_text  # SILENT CORRECTION
    if input_text.count("Assistant:") == 0:
        input_text = input_text + ASSISTANT_PROMPT  # SILENT CORRECTION
    if input_text[: len("Human:")] == "Human:":
        input_text = "\n\n" + input_text
    input_text = _add_newlines_before_ha(input_text)
    count = 0
    # track alternation
    for i in range(len(input_text)):
        if input_text[i : i + len(HUMAN_PROMPT)] == HUMAN_PROMPT:
            if count % 2 == 0:
                count += 1
            else:
                warnings.warn(ALTERNATION_ERROR + f" Received {input_text}")
        if input_text[i : i + len(ASSISTANT_PROMPT)] == ASSISTANT_PROMPT:
            if count % 2 == 1:
                count += 1
            else:
                warnings.warn(ALTERNATION_ERROR + f" Received {input_text}")

    if count % 2 == 1:  # Only saw Human, no Assistant
        input_text = input_text + ASSISTANT_PROMPT  # SILENT CORRECTION

    return input_text


class LLMInputOutputAdapter:
    """Adapter class to prepare the inputs from Langchain to a format
    that LLM model expects.

    It also provides helper function to extract
    the generated text from the model response."""

    provider_to_output_key_map = {
        "anthropic": "completion",
        "amazon": "outputText",
        "cohere": "text",
        "meta": "generation",
    }

    @classmethod
    def prepare_input(
        cls, provider: str, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        input_body = {**model_kwargs}
        if provider == "anthropic":
            input_body["prompt"] = _human_assistant_format(prompt)
        elif provider in ("ai21", "cohere", "meta"):
            input_body["prompt"] = prompt
        elif provider == "amazon":
            input_body = dict()
            input_body["inputText"] = prompt
            input_body["textGenerationConfig"] = {**model_kwargs}
        else:
            input_body["inputText"] = prompt

        if provider == "anthropic" and "max_tokens_to_sample" not in input_body:
            input_body["max_tokens_to_sample"] = 256

        return input_body

    @classmethod
    def prepare_output(cls, provider: str, response: Any) -> dict:
        if provider == "anthropic":
            response_body = json.loads(response.get("body").read().decode())
            text = response_body.get("completion")
        else:
            response_body = json.loads(response.get("body").read())

            if provider == "ai21":
                text = response_body.get("completions")[0].get("data").get("text")
            elif provider == "cohere":
                text = response_body.get("generations")[0].get("text")
            elif provider == "meta":
                text = response_body.get("generation")
            else:
                text = response_body.get("results")[0].get("outputText")

        return {
            "text": text,
            "body": response_body,
        }

    @classmethod
    def prepare_output_stream(
        cls, provider: str, response: Any, stop: Optional[List[str]] = None
    ) -> Iterator[GenerationChunk]:
        stream = response.get("body")

        if not stream:
            return

        output_key = cls.provider_to_output_key_map.get(provider, None)

        if not output_key:
            raise ValueError(
                f"Unknown streaming response output key for provider: {provider}"
            )

        for event in stream:
            chunk = event.get("chunk")
            if not chunk:
                continue

            chunk_obj = json.loads(chunk.get("bytes").decode())

            if provider == "cohere" and (
                chunk_obj["is_finished"] or chunk_obj[output_key] == "<EOS_TOKEN>"
            ):
                return
                # chunk obj format varies with provider
            yield GenerationChunk(
                text=chunk_obj[output_key],
                generation_info={
                    GUARDRAILS_BODY_KEY: chunk_obj.get(GUARDRAILS_BODY_KEY)
                    if GUARDRAILS_BODY_KEY in chunk_obj
                    else None,
                },
            )

    @classmethod
    async def aprepare_output_stream(
        cls, provider: str, response: Any, stop: Optional[List[str]] = None
    ) -> AsyncIterator[GenerationChunk]:
        stream = response.get("body")

        if not stream:
            return

        output_key = cls.provider_to_output_key_map.get(provider, None)

        if not output_key:
            raise ValueError(
                f"Unknown streaming response output key for provider: {provider}"
            )

        for event in stream:
            chunk = event.get("chunk")
            if not chunk:
                continue

            chunk_obj = json.loads(chunk.get("bytes").decode())

            if provider == "cohere" and (
                chunk_obj["is_finished"] or chunk_obj[output_key] == "<EOS_TOKEN>"
            ):
                return

            yield GenerationChunk(text=chunk_obj[output_key])


class BedrockBase(BaseModel, ABC):
    """Base class for Bedrock models."""

    client: Any = Field(exclude=True)  #: :meta private:

    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = Field(default=None, exclude=True)
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    config: Optional[Config] = None
    """An optional botocore.config.Config instance to pass to the client."""

    provider: Optional[str] = None
    """The model provider, e.g., amazon, cohere, ai21, etc. When not supplied, provider
    is extracted from the first part of the model_id e.g. 'amazon' in 
    'amazon.titan-text-express-v1'. This value should be provided for model ids that do
    not have the provider in them, e.g., custom and provisioned models that have an ARN
    associated with them."""

    model_id: str
    """Id of the model to call, e.g., amazon.titan-text-express-v1, this is
    equivalent to the modelId property in the list-foundation-models api. For custom and
    provisioned models, an ARN value is expected."""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_url: Optional[str] = None
    """Needed if you don't want to default to us-east-1 endpoint"""

    streaming: bool = False
    """Whether to stream the results."""

    provider_stop_sequence_key_name_map: Mapping[str, str] = {
        "anthropic": "stop_sequences",
        "amazon": "stopSequences",
        "ai21": "stop_sequences",
        "cohere": "stop_sequences",
    }

    guardrails: Optional[Mapping[str, Any]] = {
        "id": None,
        "version": None,
        "trace": False,
    }
    """
    An optional dictionary to configure guardrails for Bedrock.

    This field 'guardrails' consists of two keys: 'id' and 'version',
    which should be strings, but are initialized to None. It's used to
    determine if specific guardrails are enabled and properly set.

    Type:
        Optional[Mapping[str, str]]: A mapping with 'id' and 'version' keys.

    Example:
    llm = Bedrock(model_id="<model_id>", client=<bedrock_client>,
                  model_kwargs={},
                  guardrails={
                        "id": "<guardrail_id>",
                        "version": "<guardrail_version>"})

    To enable tracing for guardrails, set the 'trace' key to True and pass a callback handler to the
    'run_manager' parameter of the 'generate', '_call' methods.

    Example:
    llm = Bedrock(model_id="<model_id>", client=<bedrock_client>,
                  model_kwargs={},
                  guardrails={
                        "id": "<guardrail_id>",
                        "version": "<guardrail_version>",
                        "trace": True},
                callbacks=[BedrockAsyncCallbackHandler()])

    [https://python.langchain.com/docs/modules/callbacks/] for more information on callback handlers.

    class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
        async def on_llm_error(
            self,
            error: BaseException,
            **kwargs: Any,
        ) -> Any:
            reason = kwargs.get("reason")
            if reason == "GUARDRAIL_INTERVENED":
                ...Logic to handle guardrail intervention...
    """  # noqa: E501

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:
            import boto3

            if values["credentials_profile_name"] is not None:
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                # use default credentials
                session = boto3.Session()

            values["region_name"] = get_from_dict_or_env(
                values,
                "region_name",
                "AWS_DEFAULT_REGION",
                default=session.region_name,
            )

            client_params = {}
            if values["region_name"]:
                client_params["region_name"] = values["region_name"]
            if values["endpoint_url"]:
                client_params["endpoint_url"] = values["endpoint_url"]
            if values["config"]:
                client_params["config"] = values["config"]

            values["client"] = session.client("bedrock-runtime", **client_params)

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_provider(self) -> str:
        if self.provider:
            return self.provider
        if self.model_id.startswith("arn"):
            raise ValueError(
                "Model provider should be supplied when passing a model ARN as "
                "model_id"
            )

        return self.model_id.split(".")[0]

    @property
    def _model_is_anthropic(self) -> bool:
        return self._get_provider() == "anthropic"

    @property
    def _guardrails_enabled(self) -> bool:
        """
        Determines if guardrails are enabled and correctly configured.
        Checks if 'guardrails' is a dictionary with non-empty 'id' and 'version' keys.
        Checks if 'guardrails.trace' is true.

        Returns:
            bool: True if guardrails are correctly configured, False otherwise.
        Raises:
            TypeError: If 'guardrails' lacks 'id' or 'version' keys.
        """
        try:
            return (
                isinstance(self.guardrails, dict)
                and bool(self.guardrails["id"])
                and bool(self.guardrails["version"])
            )

        except KeyError as e:
            raise TypeError(
                "Guardrails must be a dictionary with 'id' and 'version' keys."
            ) from e

    def _get_guardrails_canonical(self) -> Dict[str, Any]:
        """
        The canonical way to pass in guardrails to the bedrock service
        adheres to the following format:

        "amazon-bedrock-guardrailDetails": {
            "guardrailId": "string",
            "guardrailVersion": "string"
        }
        """
        return {
            "amazon-bedrock-guardrailDetails": {
                "guardrailId": self.guardrails.get("id"),  # type: ignore[union-attr]
                "guardrailVersion": self.guardrails.get("version"),  # type: ignore[union-attr]
            }
        }

    def _prepare_input_and_invoke(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        _model_kwargs = self.model_kwargs or {}

        provider = self._get_provider()
        params = {**_model_kwargs, **kwargs}
        if self._guardrails_enabled:
            params.update(self._get_guardrails_canonical())
        input_body = LLMInputOutputAdapter.prepare_input(provider, prompt, params)
        body = json.dumps(input_body)
        accept = "application/json"
        contentType = "application/json"

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": accept,
            "contentType": contentType,
        }

        if self._guardrails_enabled:
            request_options["guardrail"] = "ENABLED"
            if self.guardrails.get("trace"):  # type: ignore[union-attr]
                request_options["trace"] = "ENABLED"

        try:
            response = self.client.invoke_model(**request_options)

            text, body = LLMInputOutputAdapter.prepare_output(
                provider, response
            ).values()

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        # Verify and raise a callback error if any intervention occurs or a signal is
        # sent from a Bedrock service,
        # such as when guardrails are triggered.
        services_trace = self._get_bedrock_services_signal(body)  # type: ignore[arg-type]

        if services_trace.get("signal") and run_manager is not None:
            run_manager.on_llm_error(
                Exception(
                    f"Error raised by bedrock service: {services_trace.get('reason')}"
                ),
                **services_trace,
            )

        return text

    def _get_bedrock_services_signal(self, body: dict) -> dict:
        """
        This function checks the response body for an interrupt flag or message that indicates
        whether any of the Bedrock services have intervened in the processing flow. It is
        primarily used to identify modifications or interruptions imposed by these services
        during the request-response cycle with a Large Language Model (LLM).
        """  # noqa: E501

        if (
            self._guardrails_enabled
            and self.guardrails.get("trace")  # type: ignore[union-attr]
            and self._is_guardrails_intervention(body)
        ):
            return {
                "signal": True,
                "reason": "GUARDRAIL_INTERVENED",
                "trace": body.get(AMAZON_BEDROCK_TRACE_KEY),
            }

        return {
            "signal": False,
            "reason": None,
            "trace": None,
        }

    def _is_guardrails_intervention(self, body: dict) -> bool:
        return body.get(GUARDRAILS_BODY_KEY) == "GUARDRAIL_INTERVENED"

    def _prepare_input_and_invoke_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        _model_kwargs = self.model_kwargs or {}
        provider = self._get_provider()

        if stop:
            if provider not in self.provider_stop_sequence_key_name_map:
                raise ValueError(
                    f"Stop sequence key name for {provider} is not supported."
                )

            # stop sequence from _generate() overrides
            # stop sequences in the class attribute
            _model_kwargs[self.provider_stop_sequence_key_name_map.get(provider)] = stop

        if provider == "cohere":
            _model_kwargs["stream"] = True

        params = {**_model_kwargs, **kwargs}

        if self._guardrails_enabled:
            params.update(self._get_guardrails_canonical())

        input_body = LLMInputOutputAdapter.prepare_input(provider, prompt, params)
        body = json.dumps(input_body)

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": "application/json",
            "contentType": "application/json",
        }

        if self._guardrails_enabled:
            request_options["guardrail"] = "ENABLED"
            if self.guardrails.get("trace"):  # type: ignore[union-attr]
                request_options["trace"] = "ENABLED"

        try:
            response = self.client.invoke_model_with_response_stream(**request_options)

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        for chunk in LLMInputOutputAdapter.prepare_output_stream(
            provider, response, stop
        ):
            yield chunk
            # verify and raise callback error if any middleware intervened
            self._get_bedrock_services_signal(chunk.generation_info)  # type: ignore[arg-type]

            if run_manager is not None:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _aprepare_input_and_invoke_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        _model_kwargs = self.model_kwargs or {}
        provider = self._get_provider()

        if stop:
            if provider not in self.provider_stop_sequence_key_name_map:
                raise ValueError(
                    f"Stop sequence key name for {provider} is not supported."
                )
            _model_kwargs[self.provider_stop_sequence_key_name_map.get(provider)] = stop

        if provider == "cohere":
            _model_kwargs["stream"] = True

        params = {**_model_kwargs, **kwargs}
        input_body = LLMInputOutputAdapter.prepare_input(provider, prompt, params)
        body = json.dumps(input_body)

        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.client.invoke_model_with_response_stream(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            ),
        )

        async for chunk in LLMInputOutputAdapter.aprepare_output_stream(
            provider, response, stop
        ):
            yield chunk
            if run_manager is not None and asyncio.iscoroutinefunction(
                run_manager.on_llm_new_token
            ):
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            elif run_manager is not None:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)  # type: ignore[unused-coroutine]


class Bedrock(LLM, BedrockBase):
    """Bedrock models.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Bedrock service.
    """

    """
    Example:
        .. code-block:: python

            from bedrock_langchain.bedrock_llm import BedrockLLM

            llm = BedrockLLM(
                credentials_profile_name="default",
                model_id="amazon.titan-text-express-v1",
                streaming=True
            )

    """

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "amazon_bedrock"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "bedrock"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.region_name:
            attributes["region_name"] = self.region_name

        return attributes

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Bedrock service with streaming.

        Args:
            prompt (str): The prompt to pass into the model
            stop (Optional[List[str]], optional): Stop sequences. These will
                override any stop sequences in the `model_kwargs` attribute.
                Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): Callback
                run managers used to process the output. Defaults to None.

        Returns:
            Iterator[GenerationChunk]: Generator that yields the streamed responses.

        Yields:
            Iterator[GenerationChunk]: Responses from the model.
        """
        return self._prepare_input_and_invoke_stream(
            prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Bedrock service model.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = llm("Tell me a joke.")
        """

        if self.streaming:
            completion = ""
            for chunk in self._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                completion += chunk.text
            return completion

        return self._prepare_input_and_invoke(
            prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[GenerationChunk, None]:
        """Call out to Bedrock service with streaming.

        Args:
            prompt (str): The prompt to pass into the model
            stop (Optional[List[str]], optional): Stop sequences. These will
                override any stop sequences in the `model_kwargs` attribute.
                Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun], optional): Callback
                run managers used to process the output. Defaults to None.

        Yields:
            AsyncGenerator[GenerationChunk, None]: Generator that asynchronously yields
            the streamed responses.
        """
        async for chunk in self._aprepare_input_and_invoke_stream(
            prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Bedrock service model.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = await llm._acall("Tell me a joke.")
        """

        if not self.streaming:
            raise ValueError("Streaming must be set to True for async operations. ")

        chunks = [
            chunk.text
            async for chunk in self._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            )
        ]
        return "".join(chunks)

    def get_num_tokens(self, text: str) -> int:
        if self._model_is_anthropic:
            return get_num_tokens_anthropic(text)
        else:
            return super().get_num_tokens(text)

    def get_token_ids(self, text: str) -> List[int]:
        if self._model_is_anthropic:
            return get_token_ids_anthropic(text)
        else:
            return super().get_token_ids(text)
