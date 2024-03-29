from typing import Any, Dict, List, Mapping, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Field, SecretStr


class Predibase(LLM):
    """Use your Predibase models with Langchain.

    To use, you should have the ``predibase`` python package installed,
    and have your Predibase API key.
    """

    model: str
    predibase_api_key: SecretStr
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    default_options_for_generation: dict = Field(
        {
            "max_new_tokens": 256,
            "temperature": 0.1,
        },
        const=True,
    )

    @property
    def _llm_type(self) -> str:
        return "predibase"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        # TODO: <Alex>ALEX</Alex>
        # options: Optional[Dict[str, Union[str, float]]] = None,
        # TODO: <Alex>ALEX</Alex>
        # TODO: <Alex>ALEX</Alex>
        **kwargs,
        # TODO: <Alex>ALEX</Alex>
    ) -> str:
        try:
            # TODO: <Alex>ALEX</Alex>
            print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] PREDIBASE_API_KEY:\n{self.predibase_api_key} ; TYPE: {str(type(self.predibase_api_key))}')
            print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] PREDIBASE_API_KEY.GET_SECRET_VALUE:\n{self.predibase_api_key.get_secret_value()} ; TYPE: {str(type(self.predibase_api_key.get_secret_value()))}')
            print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] MODEL:\n{self.model} ; TYPE: {str(type(self.model))}')
            from predibase.pql import get_session
            from predibase.pql.api import Session
            # TODO: <Alex>ALEX</Alex>
            from predibase import PredibaseClient
            # TODO: <Alex>ALEX</Alex>
            from predibase.resource.llm.interface import LLMDeployment
            from predibase.resource.llm.response import GeneratedResponse
            # TODO: <Alex>ALEX</Alex>

            # TODO: <Alex>ALEX</Alex>
            # session: Session = get_session(token=self.predibase_api_key.get_secret_value(), gateway="https://api.staging.predibase.com/v1", serving_endpoint="serving.staging.predibase.com")
            session: Session = get_session(token=self.predibase_api_key.get_secret_value(), gateway="https://api.app.predibase.com/v1", serving_endpoint="serving.app.predibase.com")
            # TODO: <Alex>ALEX</Alex>
            # TODO: <Alex>ALEX</Alex>
            # pc: PredibaseClient = PredibaseClient(token=self.predibase_api_key.get_secret_value())
            # TODO: <Alex>ALEX</Alex>
            # TODO: <Alex>ALEX</Alex>
            pc: PredibaseClient = PredibaseClient(session=session)
            # pc: PredibaseClient = PredibaseClient(token=self.predibase_api_key.get_secret_value(), gateway="https://api.staging.predibase.com/v1")
            # a = pc.list_llm_deployments(active_only=False, print_as_table=True)
            # print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] LLM_DEPLOYMENTS:\n{a} ; TYPE: {str(type(a))}')
            # b = self.model in a
            # print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] MODEL_IN_LLM_DEPLOYMENTS:\n{b} ; TYPE: {str(type(b))}')
            # TODO: <Alex>ALEX</Alex>
        except ImportError as e:
            raise ImportError(
                "Could not import Predibase Python package. "
                "Please install it with `pip install predibase`."
            ) from e
        except ValueError as e:
            raise ValueError("Your API key is not correct. Please try again") from e
        # TODO: <Alex>ALEX</Alex>
        # # load model and version
        # results = pc.prompt(prompt, model_name=self.model)
        # TODO: <Alex>ALEX</Alex>
        options: Dict[str, Union[str, float]] = kwargs or self.default_options_for_generation
        print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] OPTIONS:\n{options} ; TYPE: {str(type(options))}')
        base_llm_deployment: LLMDeployment = pc.LLM(uri=f"pb://deployments/{self.model}")
        result: GeneratedResponse = base_llm_deployment.generate(
            prompt=prompt,
            options=options,
        )
        print(f'\n[ALEX_TEST] [LangChain::Community::LLMs::Predibase] RESULT.RESPONSE:\n{result.response} ; TYPE: {str(type(result.response))}')
        return result.response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_kwargs": self.model_kwargs},
        }
