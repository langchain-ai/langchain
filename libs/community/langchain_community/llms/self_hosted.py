import logging
from typing import Any, List, Mapping, Optional
import runhouse as rh

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.utils import enforce_stop_tokens

logger = logging.getLogger(__name__)


class ModelPipeline:
    def __init__(self, model_id, **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.tokenizer, self.model, self.curr_pipeline, self.task = None, None, None, model_kwargs.get("task", None)

    def load_model(self, hf_token) -> Any:
        """
        Accepts a huggingface model_id and returns a pipeline for the task.
        Sent to the remote hardware and being executed there, as part of the rh.Module(LangchainLLMModelPipeline).
        """
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
        from transformers import pipeline as hf_pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,
                                                       token=hf_token)

        try:
            self.model = AutoModel.from_pretrained(self.model_id,
                                                   token=hf_token)
        except RuntimeError:
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_id,
                                                              token=hf_token)

        try:
            curr_pipeline = hf_pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                token=hf_token
            )

            self.curr_pipeline = curr_pipeline

        except RuntimeError:
            from sentence_transformers import SentenceTransformer
            self.curr_pipeline = SentenceTransformer(self.model_id)

    def interface_fn(
            self,
            pipeline: Any,
            prompt: str,
            *args: Any,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        """Inference function to send to the remote hardware.

        Accepts a pipeline callable (or, more likely,
        a key pointing to the model on the cluster's object store)
        and returns text predictions for each document
        in the batch.
        """
        text = self.curr_pipeline(prompt, *args, **kwargs)
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text


class SelfHostedPipeline(LLM):
    """ A generic Model inference on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another
    cloud like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Example for custom pipeline and inference functions:
        .. code-block:: python

            from langchain_community.llms import SelfHostedPipeline
            import runhouse as rh

            class LoadingModel:
                def load_model(self, **args, **kwargs):
                    # define your model loading function

                # this is the prediction function
                def interface_fn(self, **args, **kwargs):
                    # define your interface_fn

            gpu = rh.ondemand_cluster(name="rh-a10x", instance_type="g5.2xlarge")
            my_env = rh.env(reqs=["transformers", "torch"])

            llm = SelfHostedHuggingFaceLLM(pipline_cls=LoadingModel, hardware=gpu, env=model_env)
            # if you will not provide a 'pipline_cls', the default 'pipline_cls' will be used.

    """
    load_fn_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model load function."""
    hardware: rh.Cluster
    """The remote hardware the model will run on"""
    env: rh.Env
    """The env with the necessary requirements for the model execution """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True

    def __init__(self, pipline_cls: Any = ModelPipeline, **kwargs: Any):
        """Init the pipeline with an auxiliary function.

        The load function must be in global scope to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function.
        """
        super().__init__(**kwargs)
        gpu, model_env = kwargs.get("hardware"), kwargs.get("env")
        model_id, task = kwargs.get("model_id"), kwargs.get("task")
        ModelPipeline_remote = rh.module(pipline_cls).to(system=gpu, env=model_env)
        self.ModelPipeline_remote_instance = ModelPipeline_remote(model_id=model_id, task=task)
        _load_fn_kwargs = self.load_fn_kwargs or {}
        hf_token = ModelPipeline_remote.env.secrets[0].values.get("token")
        self.ModelPipeline_remote_instance.load_model.remote(hf_token=hf_token, **_load_fn_kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"hardware": self.hardware},
        }

    @property
    def _llm_type(self) -> str:
        return "self_hosted_llm"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.ModelPipeline_remote_instance.interface_fn(
            prompt=prompt, stop=stop, **kwargs
        )