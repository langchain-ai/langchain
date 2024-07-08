import logging
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

from ibm_watsonx_ai import Credentials  # type: ignore
from ibm_watsonx_ai.foundation_models import Model, ModelInference  # type: ignore
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

logger = logging.getLogger(__name__)


class WatsonxLLM(BaseLLM):
    """
    IBM WatsonX.ai large language models.

    To use, you should have ``langchain_ibm`` python package installed,
    and the environment variable ``WATSONX_APIKEY`` set with your API key, or pass
    it as a named parameter to the constructor.


    Example:
        .. code-block:: python

            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
            parameters = {
                GenTextParamsMetaNames.DECODING_METHOD: "sample",
                GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
                GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
                GenTextParamsMetaNames.TEMPERATURE: 0.5,
                GenTextParamsMetaNames.TOP_K: 50,
                GenTextParamsMetaNames.TOP_P: 1,
            }

            from langchain_ibm import WatsonxLLM
            watsonx_llm = WatsonxLLM(
                model_id="google/flan-ul2",
                url="https://us-south.ml.cloud.ibm.com",
                apikey="*****",
                project_id="*****",
                params=parameters,
            )
    """

    model_id: str = ""
    """Type of model to use."""

    deployment_id: str = ""
    """Type of deployed model to use."""

    project_id: str = ""
    """ID of the Watson Studio project."""

    space_id: str = ""
    """ID of the Watson Studio space."""

    url: Optional[SecretStr] = None
    """Url to Watson Machine Learning or CPD instance"""

    apikey: Optional[SecretStr] = None
    """Apikey to Watson Machine Learning or CPD instance"""

    token: Optional[SecretStr] = None
    """Token to CPD instance"""

    password: Optional[SecretStr] = None
    """Password to CPD instance"""

    username: Optional[SecretStr] = None
    """Username to CPD instance"""

    instance_id: Optional[SecretStr] = None
    """Instance_id of CPD instance"""

    version: Optional[SecretStr] = None
    """Version of CPD instance"""

    params: Optional[dict] = None
    """Model parameters to use during generate requests."""

    verify: Union[str, bool, None] = None
    """User can pass as verify one of following:
        the path to a CA_BUNDLE file
        the path of directory with certificates of trusted CAs
        True - default path to truststore will be taken
        False - no verification will be made"""

    streaming: bool = False
    """ Whether to stream the results or not. """

    watsonx_model: ModelInference = Field(default=None, exclude=True)  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example:
            {
                "url": "WATSONX_URL",
                "apikey": "WATSONX_APIKEY",
                "token": "WATSONX_TOKEN",
                "password": "WATSONX_PASSWORD",
                "username": "WATSONX_USERNAME",
                "instance_id": "WATSONX_INSTANCE_ID",
            }
        """
        return {
            "url": "WATSONX_URL",
            "apikey": "WATSONX_APIKEY",
            "token": "WATSONX_TOKEN",
            "password": "WATSONX_PASSWORD",
            "username": "WATSONX_USERNAME",
            "instance_id": "WATSONX_INSTANCE_ID",
        }

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that credentials and python package exists in environment."""
        if isinstance(values.get("watsonx_model"), (ModelInference, Model)):
            values["model_id"] = getattr(values["watsonx_model"], "model_id")
            values["deployment_id"] = getattr(
                values["watsonx_model"], "deployment_id", ""
            )
            values["project_id"] = getattr(
                getattr(values["watsonx_model"], "_client"),
                "default_project_id",
            )
            values["space_id"] = getattr(
                getattr(values["watsonx_model"], "_client"), "default_space_id"
            )
            values["params"] = getattr(values["watsonx_model"], "params")
        else:
            values["url"] = convert_to_secret_str(
                get_from_dict_or_env(values, "url", "WATSONX_URL")
            )
            if "cloud.ibm.com" in values.get("url", "").get_secret_value():
                values["apikey"] = convert_to_secret_str(
                    get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                )
            else:
                if (
                    not values["token"]
                    and "WATSONX_TOKEN" not in os.environ
                    and not values["password"]
                    and "WATSONX_PASSWORD" not in os.environ
                    and not values["apikey"]
                    and "WATSONX_APIKEY" not in os.environ
                ):
                    raise ValueError(
                        "Did not find 'token', 'password' or 'apikey',"
                        " please add an environment variable"
                        " `WATSONX_TOKEN`, 'WATSONX_PASSWORD' or 'WATSONX_APIKEY' "
                        "which contains it,"
                        " or pass 'token', 'password' or 'apikey'"
                        " as a named parameter."
                    )
                elif values["token"] or "WATSONX_TOKEN" in os.environ:
                    values["token"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "token", "WATSONX_TOKEN")
                    )
                elif values["password"] or "WATSONX_PASSWORD" in os.environ:
                    values["password"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "password", "WATSONX_PASSWORD")
                    )
                    values["username"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                    )
                elif values["apikey"] or "WATSONX_APIKEY" in os.environ:
                    values["apikey"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "apikey", "WATSONX_APIKEY")
                    )
                    values["username"] = convert_to_secret_str(
                        get_from_dict_or_env(values, "username", "WATSONX_USERNAME")
                    )
                if not values["instance_id"] or "WATSONX_INSTANCE_ID" not in os.environ:
                    values["instance_id"] = convert_to_secret_str(
                        get_from_dict_or_env(
                            values, "instance_id", "WATSONX_INSTANCE_ID"
                        )
                    )
            credentials = Credentials(
                url=values["url"].get_secret_value() if values["url"] else None,
                api_key=values["apikey"].get_secret_value()
                if values["apikey"]
                else None,
                token=values["token"].get_secret_value() if values["token"] else None,
                password=values["password"].get_secret_value()
                if values["password"]
                else None,
                username=values["username"].get_secret_value()
                if values["username"]
                else None,
                instance_id=values["instance_id"].get_secret_value()
                if values["instance_id"]
                else None,
                version=values["version"].get_secret_value()
                if values["version"]
                else None,
                verify=values["verify"],
            )

            watsonx_model = ModelInference(
                model_id=values["model_id"],
                deployment_id=values["deployment_id"],
                credentials=credentials,
                params=values["params"],
                project_id=values["project_id"],
                space_id=values["space_id"],
            )
            values["watsonx_model"] = watsonx_model

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
            "params": self.params,
            "project_id": self.project_id,
            "space_id": self.space_id,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "IBM watsonx.ai"

    @staticmethod
    def _extract_token_usage(
        response: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if response is None:
            return {"generated_token_count": 0, "input_token_count": 0}

        input_token_count = 0
        generated_token_count = 0

        def get_count_value(key: str, result: Dict[str, Any]) -> int:
            return result.get(key, 0) or 0

        for res in response:
            results = res.get("results")
            if results:
                input_token_count += get_count_value("input_token_count", results[0])
                generated_token_count += get_count_value(
                    "generated_token_count", results[0]
                )

        return {
            "generated_token_count": generated_token_count,
            "input_token_count": input_token_count,
        }

    def _get_chat_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        params = {**self.params} if self.params else {}
        params = params | {**kwargs.get("params", {})}
        if stop is not None:
            if params and "stop_sequences" in params:
                raise ValueError(
                    "`stop_sequences` found in both the input and default params."
                )
            params = (params or {}) | {"stop_sequences": stop}
        return params

    def _create_llm_result(self, response: List[dict]) -> LLMResult:
        """Create the LLMResult from the choices and prompts."""
        generations = []
        for res in response:
            results = res.get("results")
            if results:
                finish_reason = results[0].get("stop_reason")
                gen = Generation(
                    text=results[0].get("generated_text"),
                    generation_info={"finish_reason": finish_reason},
                )
                generations.append([gen])
        final_token_usage = self._extract_token_usage(response)
        llm_output = {
            "token_usage": final_token_usage,
            "model_id": self.model_id,
            "deployment_id": self.deployment_id,
        }
        return LLMResult(generations=generations, llm_output=llm_output)

    def _stream_response_to_generation_chunk(
        self,
        stream_response: Dict[str, Any],
    ) -> GenerationChunk:
        """Convert a stream response to a generation chunk."""
        if not stream_response["results"]:
            return GenerationChunk(text="")
        return GenerationChunk(
            text=stream_response["results"][0]["generated_text"],
            generation_info=dict(
                finish_reason=stream_response["results"][0].get("stop_reason", None),
                llm_output={
                    "model_id": self.model_id,
                    "deployment_id": self.deployment_id,
                },
            ),
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the IBM watsonx.ai inference endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python

                response = watsonx_llm.invoke("What is a molecule")
        """
        result = self._generate(
            prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs
        )
        return result.generations[0][0].text

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call the IBM watsonx.ai inference endpoint which then generate the response.
        Args:
            prompts: List of strings (prompts) to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The full LLMResult output.
        Example:
            .. code-block:: python

                response = watsonx_llm.generate(["What is a molecule"])
        """
        params = self._get_chat_params(stop=stop, **kwargs)
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            if len(prompts) > 1:
                raise ValueError(
                    f"WatsonxLLM currently only supports single prompt, got {prompts}"
                )
            generation = GenerationChunk(text="")
            stream_iter = self._stream(
                prompts[0], stop=stop, run_manager=run_manager, **kwargs
            )
            for chunk in stream_iter:
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            if isinstance(generation.generation_info, dict):
                llm_output = generation.generation_info.pop("llm_output")
                return LLMResult(generations=[[generation]], llm_output=llm_output)
            return LLMResult(generations=[[generation]])
        else:
            response = self.watsonx_model.generate(
                prompt=prompts, **(kwargs | {"params": params})
            )
            return self._create_llm_result(response)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call the IBM watsonx.ai inference endpoint which then streams the response.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The iterator which yields generation chunks.
        Example:
            .. code-block:: python

                response = watsonx_llm.stream("What is a molecule")
                for chunk in response:
                    print(chunk, end='')
        """
        params = self._get_chat_params(stop=stop, **kwargs)
        for stream_resp in self.watsonx_model.generate_text_stream(
            prompt=prompt, raw_response=True, **(kwargs | {"params": params})
        ):
            if not isinstance(stream_resp, dict):
                stream_resp = stream_resp.dict()
            chunk = self._stream_response_to_generation_chunk(stream_resp)

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def get_num_tokens(self, text: str) -> int:
        response = self.watsonx_model.tokenize(text, return_tokens=False)
        return response["result"]["token_count"]

    def get_token_ids(self, text: str) -> List[int]:
        raise NotImplementedError("API does not support returning token ids.")
