import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_model(
    task: str, backend: str, model_id: str, **_model_kwargs: Optional[dict]
) -> Any:
    try:
        from transformers import (  # type: ignore[import]
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
        )

    except ImportError:
        raise ValueError(
            "Could not import transformers python package. "
            "Please install it with `pip install transformers`."
        )
    try:
        if task == "text-generation":
            if backend == "openvino":
                try:
                    from optimum.intel.openvino import (  # type: ignore[import]
                        OVModelForCausalLM,
                    )

                except ImportError:
                    raise ValueError(
                        "Could not import optimum-intel python package. "
                        "Please install it with: "
                        "pip install 'optimum[openvino,nncf]' "
                    )
                try:
                    # use local model
                    model = OVModelForCausalLM.from_pretrained(
                        model_id, **_model_kwargs
                    )

                except Exception:
                    # use remote model
                    model = OVModelForCausalLM.from_pretrained(
                        model_id, export=True, **_model_kwargs
                    )
                return model
            elif backend == "ipex":
                try:
                    import torch
                    from optimum.intel.ipex import (  # type: ignore[import]
                        IPEXModelForCausalLM,
                    )
                except ImportError:
                    raise ValueError(
                        "Could not import optimum-intel python package. "
                        "Please install it with: "
                        "pip install 'optimum[ipex]' "
                        "or follow installation instructions from: "
                        " https://github.com/rbrugaro/optimum-intel "
                    )
                try:
                    # use TorchScript model
                    config = AutoConfig.from_pretrained(model_id)
                    export = not getattr(config, "torchscript", False)
                except RuntimeError:
                    logger.warning(
                        "We will use IPEXModel with export=True to export the model"
                    )
                    export = True
                model = IPEXModelForCausalLM.from_pretrained(
                    model_id,
                    export=export,
                    **_model_kwargs,
                    torch_dtype=torch.bfloat16,  # keep or remove the dtype????
                )
                return model
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **_model_kwargs)
                return model

        elif task in ("text2text-generation", "summarization", "translation"):
            if backend == "openvino":
                try:
                    from optimum.intel.openvino import OVModelForSeq2SeqLM

                except ImportError:
                    raise ValueError(
                        "Could not import optimum-intel python package. "
                        "Please install it with: "
                        "pip install 'optimum[openvino,nncf]' "
                    )
                try:
                    # use local model
                    model = OVModelForSeq2SeqLM.from_pretrained(
                        model_id, **_model_kwargs
                    )

                except Exception:
                    # use remote model
                    model = OVModelForSeq2SeqLM.from_pretrained(
                        model_id, export=True, **_model_kwargs
                    )
                return model
            else:
                if backend == "ipex":
                    logger.warning("IPEX backend is not supported for this task.")
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_id, **_model_kwargs
                    )
                return model
        else:
            raise ValueError(f"Got invalid task {task}, " f"currently not supported")
    except ImportError as e:
        raise ValueError(
            f"Could not load the {task} model due to missing dependencies."
        ) from e
