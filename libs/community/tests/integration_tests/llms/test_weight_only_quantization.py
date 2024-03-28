"""Test HuggingFace Pipeline wrapper."""

from langchain_community.llms.weight_only_quantization import WeightOnlyQuantPipeline

model_id = "google/flan-t5-large"


def test_weight_only_quantization_with_config() -> None:
    """Test valid call to HuggingFace text2text model."""
    from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig

    conf = WeightOnlyQuantConfig(weight_dtype="nf4")
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="text2text-generation", quantization_config=conf
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_weight_only_quantization_4bit() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="text2text-generation", load_in_4bit=True
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_weight_only_quantization_8bit() -> None:
    """Test valid call to HuggingFace text2text model."""
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="text2text-generation", load_in_8bit=True
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def test_init_with_pipeline() -> None:
    """Test initialization with a HF pipeline."""
    from intel_extension_for_transformers.transformers import AutoModelForSeq2SeqLM
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, load_in_4bit=True, use_llm_runtime=False
    )
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = WeightOnlyQuantPipeline(pipeline=pipe)
    output = llm("Say foo:")
    assert isinstance(output, str)


def text_weight_only_pipeline_summarization() -> None:
    """Test valid call to HuggingFace summarization model."""
    from intel_extension_for_transformers.transformers import WeightOnlyQuantConfig

    conf = WeightOnlyQuantConfig()
    llm = WeightOnlyQuantPipeline.from_model_id(
        model_id=model_id, task="summarization", quantization_config=conf
    )
    output = llm("Say foo:")
    assert isinstance(output, str)
