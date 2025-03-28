try:
    from transformers import (  # type: ignore[import]
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

except ImportError:
    raise ValueError(
        "Could not import transformers python package. "
        "Please install it with `pip install transformers`."
    )


def use_hpu_model_device(model_kwargs: dict) -> None:
    """check if the model is using the hpu device."""
    return model_kwargs.get("device") == "hpu"


# HuggingFacePipeline usage
def get_gaudi_auto_model_for_causal_lm(model_id: str) -> AutoModelForCausalLM:
    """get the model for causal lm."""
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    from optimum.habana.utils import set_seed
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    import torch

    # Tweak generation so that it runs faster on Gaudi
    adapt_transformers_to_gaudi()
    set_seed(27)

    model_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_dtype)
    model = model.eval().to("hpu")
    model = wrap_in_hpu_graph(model)

    return model


def get_gaudi_auto_model_for_seq2seq_lm(model_id: str) -> AutoModelForSeq2SeqLM:
    """get the model for seq2seq lm."""
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    from optimum.habana.utils import set_seed
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    import torch

    # Tweak generation so that it runs faster on Gaudi
    adapt_transformers_to_gaudi()
    set_seed(27)

    model_dtype = torch.bfloat16

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=model_dtype)
    model = model.eval().to("hpu")
    model = wrap_in_hpu_graph(model)

    return model
