import runhouse as rh
import logging
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.self_hosted import SelfHostedPipeline

DEFAULT_MODEL_ID = "google/gemma-2b-it"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text2text-generation", "text-generation", "summarization")

logger = logging.getLogger(__name__)


class LangchainLLMModelPipeline:
    DEFAULT_MODEL_ID = "google/gemma-2b-it"
    DEFAULT_TASK = "text-generation"
    VALID_TASKS = ("text2text-generation", "text-generation", "summarization")

    def __init__(self, model_id=DEFAULT_MODEL_ID, task=DEFAULT_TASK, **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs, self.task = model_id, model_kwargs, task
        self.tokenizer, self.model, self.curr_pipeline = None, None, None

    def load_model(self, hf_token) -> Any:
        """Inference function to send to the remote hardware.

        Accepts a huggingface model_id and returns a pipeline for the task.
        """
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GemmaForCausalLM
        from transformers import pipeline as hf_pipeline
        import torch

        _model_kwargs = self.model_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id,
                                                       torch_dtype=torch.float16,
                                                       device_map=torch.device("cpu"),
                                                       token=hf_token,
                                                       **_model_kwargs)

        self.model = GemmaForCausalLM.from_pretrained(self.model_id,
                                                       torch_dtype=torch.float16,
                                                       device_map=torch.device("cpu"),
                                                       token=hf_token,
                                                       **_model_kwargs)

        # try:
        #
        #     if self.task == "text-generation":
        #         self.model = AutoModelForCausalLM.from_pretrained(self.model_id,
        #                                                           torch_dtype=torch.float16,
        #                                                           device_map=torch.device("cpu"),
        #                                                           token=hf_token,
        #                                                           **_model_kwargs)
        #     elif self.task in ("text2text-generation", "summarization"):
        #         self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id,
        #                                                            torch_dtype=torch.float16,
        #                                                            token=hf_token,
        #                                                            device_map=torch.device("cpu"),
        #                                                            **_model_kwargs)
        #     else:
        #         raise ValueError(
        #             f"Got invalid task {self.task}, "
        #             f"currently only {self.VALID_TASKS} are supported"
        #         )
        # except ImportError as e:
        #     raise ValueError(
        #         f"Could not load the {self.task} model due to missing dependencies."
        #     ) from e

        curr_pipeline = hf_pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            token=hf_token,
            model_kwargs=_model_kwargs
        )
        if curr_pipeline.task not in self.VALID_TASKS:
            raise ValueError(
                f"Got invalid task {curr_pipeline.task}, "
                f"currently only {self.VALID_TASKS} are supported"
            )
        self.curr_pipeline = curr_pipeline

    def interface_fn(
            self,
            prompt: str,
            *args: Any,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> str:
        """Inference function to send to the remote hardware.

        Accepts a Hugging Face pipeline (or more likely,
        a key pointing to such a pipeline on the cluster's object store)
        and returns generated text.
        """
        from langchain_community.llms.utils import enforce_stop_tokens
        response = self.curr_pipeline(prompt, *args, **kwargs)
        if self.curr_pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt):]
        elif self.curr_pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        elif self.curr_pipeline.task == "summarization":
            text = response[0]["summary_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.curr_pipeline.task}, "
                f"currently only {self.VALID_TASKS} are supported"
            )
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        print(f'text is {response}')
        return text


class SelfHostedHuggingFaceLLM(SelfHostedPipeline):
    """HuggingFace Pipeline API to run on self-hosted remote hardware.

    Supported hardware includes auto-launched instances on AWS, GCP, Azure,
    and Lambda, as well as servers specified
    by IP address and SSH credentials (such as on-prem, or another cloud
    like Paperspace, Coreweave, etc.).

    To use, you should have the ``runhouse`` python package installed.

    Only supports `text-generation`, `text2text-generation` and `summarization` for now.

    Example using from_model_id:
        .. code-block:: python

            from langchain_community.llms import SelfHostedHuggingFaceLLM
            import runhouse as rh
            gpu = rh.cluster(name="rh-a10x", instance_type="g5.2xlarge")
            model_env = rh.env(reqs=["transformers", "torch"])
            hf = SelfHostedHuggingFaceLLM(
                model_id="google/gemma-2b-it", task="text2text-generation").to(gpu, env=model_env)

    Example passing fn that generates a pipeline (bc the pipeline is not serializable):
        .. code-block:: python

            from langchain_community.llms import SelfHostedHuggingFaceLLM
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import runhouse as rh

            def get_pipeline():
                model_id = "gpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                pipe = pipeline(
                    "text-generation", model=model, tokenizer=tokenizer
                )
                return pipe
            load_pipeline_remote = rh.function(fn=get_pipeline).to(gpu, env=model_env)
            hf = SelfHostedHuggingFaceLLM(
                model_load_fn=get_pipeline, model_id="gpt2").to(gpu, env=model_env)
    """

    model_id: str = DEFAULT_MODEL_ID
    """Hugging Face model_id to load the model."""
    task: str = DEFAULT_TASK
    """Hugging Face task ("text-generation", "text2text-generation" or "summarization")."""
    device: int = 0
    """Device to use for inference. -1 for CPU, 0 for GPU, 1 for second GPU, etc."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    hardware: rh.Cluster
    """The remote hardware the model will run on"""
    env: rh.Env
    """The env with the necessary requirements for the model execution """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        # This configuration is necessary for Pydantic to use the alias
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        allow_dangerous_deserialization = True

    def __init__(self, **kwargs: Any):
        """Construct the pipeline remotely using an auxiliary function.

        The load function needs to be importable to be imported
        and run on the server, i.e. in a module and not a REPL or closure.
        Then, initialize the remote inference function.
        """
        load_fn_kwargs = {
            "model_id": kwargs.get("model_id", DEFAULT_MODEL_ID),
            "task": kwargs.get("task", DEFAULT_TASK),
            "device": kwargs.get("device", 0),
            "load_fn_kwargs": kwargs.get("model_kwargs", None),
            "hardware": kwargs.get("hardware"),
            "env": kwargs.get("env")
        }
        super().__init__(pipline_cls=LangchainLLMModelPipeline, **load_fn_kwargs)
        # self.client = LangchainLLMModelPipeline_remote.interface_fn

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs}
        }

    @property
    def _llm_type(self) -> str:
        return "selfhosted_huggingface_pipeline"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        kwargs["max_length"] = 3072
        return self.ModelPipeline_remote_instance.interface_fn.remote(
            prompt=prompt, stop=stop, **kwargs
        )
