import logging
from typing import Any, List, Mapping, Optional

import runhouse as rh
from langchain_core.pydantic_v1 import Extra

from langchain_community.llms.self_hosted import SelfHostedPipeline

logger = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "google/gemma-2b-it"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text-generation", "text2text-generation", "summarization")


class LangchainLLMModelPipeline:
    def __init__(self, model_id=DEFAULT_MODEL_ID, task=DEFAULT_TASK, **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs, self.task = model_id, model_kwargs, task
        self.tokenizer, self.model, self.curr_pipeline = None, None, None

    def load_model(self, hf_token) -> Any:
        """
        Accepts a huggingface model_id and returns a pipeline for the task.
        Sent to the remote hardware and being executed there,
        as part of the rh.Module(LangchainLLMModelPipeline).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline

        if self.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {self.task}, "
                f"currently only {VALID_TASKS} are supported"
            )

        _model_kwargs = self.model_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=torch.device("cpu"),
            token=hf_token,
            **_model_kwargs,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map=torch.device("cpu"),
            token=hf_token,
            **_model_kwargs,
        )

        curr_pipeline = hf_pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            token=hf_token,
            model_kwargs=_model_kwargs,
        )
        if curr_pipeline.task not in VALID_TASKS:
            raise ValueError(
                f"Got invalid task {curr_pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )

        self.curr_pipeline = curr_pipeline

    def interface_fn(
        self,
        prompt: str,
        *args: Any,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        The prediction function of the model.
        Accepts a Hugging Face pipeline (or more likely,
        a key pointing to such a pipeline on the cluster's object store)
        and returns generated text.
        Sent to the remote hardware and being executed there, as part of the
        rh.Module(LangchainLLMModelPipeline).
        """
        from langchain_community.llms.utils import enforce_stop_tokens

        response = self.curr_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            num_return_sequences=1,
            add_special_tokens=True,
            *args,
            **kwargs,
        )
        if self.curr_pipeline.task in ["text2text-generation", "text-generation"]:
            text = response[0]["generated_text"][len(prompt):]
        elif self.curr_pipeline.task == "summarization":
            text = response[0]["summary_text"][len(prompt):]
        else:
            raise ValueError(
                f"Got invalid task {self.curr_pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
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
            model_env = rh.env(reqs=["transformers",
                                    "torch",
                                    "accelerate",
                                    "huggingface-hub"],
                               secrets=["huggingface"]
                               # need for downloading google/gemma-2b-it).to(system=gpu)
            hf = SelfHostedHuggingFaceLLM(
                model_id="google/gemma-2b-it",
                task="text2text-generation",
                hardware=gpu,
                env=model_env)

    """

    model_id: str = DEFAULT_MODEL_ID
    """Hugging Face model_id to load the model."""
    task: str = DEFAULT_TASK
    """Hugging Face task ("text-generation", "text2text-generation" or 
    "summarization")."""
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
        arbitrary_types_allowed = True

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
            "env": kwargs.get("env"),
        }
        super().__init__(pipline_cls=LangchainLLMModelPipeline, **load_fn_kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_id": self.model_id}, **{"model_kwargs": self.model_kwargs}}

    @property
    def _llm_type(self) -> str:
        return "selfhosted_huggingface_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        return self.ModelPipeline_remote_instance.interface_fn(
            prompt=prompt, stop=stop, **kwargs
        )
