import pytest
from unittest.mock import Mock, patch
from langchain.callbacks import ArthurCallbackHandler
from langchain.llms.openai import BaseOpenAI
from langchain.schema import LLMResult, Generation


@pytest.fixture
def mock_arthur_client():
    client = Mock()
    return client


@pytest.fixture
def mock_arthur_model():
    try:
        from arthurai.core.attributes import ArthurAttribute
        from arthurai.common.constants import Stage, ValueType
    except ImportError:
        raise ImportError("To run the unit test for the Arthur integration with the ArthurCallbackHandler you need the `arthurai` package.\
                           Please install it with `pip install arthurai`.")
    arthur_model = Mock()
    arthur_model.get_attributes = lambda _ : [
        ArthurAttribute(name="input_text", stage=Stage.ModelPipelineInput, value_type=ValueType.Unstructured_Text),
        ArthurAttribute(name="output_text", stage=Stage.PredictedValue, value_type=ValueType.Unstructured_Text),
        ArthurAttribute(name="output_likelihoods", stage=Stage.PredictedValue, value_type=ValueType.TokenLikelihoods),
    ]
    arthur_model.send_inferences = lambda d : None
    return arthur_model


@pytest.fixture
def handler() -> ArthurCallbackHandler:
    return ArthurCallbackHandler()


def test_on_llm_end(handler: ArthurCallbackHandler) -> None:
    """Tests that the ArthurCallbackHandler can call on_llm_end() without errors 

    We use a response with mock data as well as patching a mock arthur client and mock arthur model to be used in on_llm_end()

    """
    with patch('langchain.callbacks.arthur_callback.ArthurAI', return_value=mock_arthur_client):
        with patch('langchain.callbacks.arthur_callback.ArthurAI.get_model', return_value=mock_arthur_model):
            response = LLMResult(
                generations=[Generation(
                    text="generated text", 
                    generation_info={
                        "logprobs" : {
                            "top_logprobs" : {"a" : -5, "b" : -4, "c" : -3}
                        }
                    })],
                llm_output={
                    "token_usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                    "model_name": BaseOpenAI.__fields__["model_name"].default,
                },
            )
            handler.on_llm_end(response)
