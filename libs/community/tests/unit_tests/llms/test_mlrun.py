import os
import pathlib
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Dict, Union

import mlrun
import pytest
from langchain_core.language_models.llms import LLM
from libs.community.langchain_community.llms.mlrun import MLRun
from mlrun.serving.v2_serving import V2ModelServer

import langchain_community.llms
# Some default models to test, can be changed to any model, we prefer small ones
_HUGGINGFACE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
_OPENAI_MODEL = "gpt-3.5-turbo-instruct"
_OLLAMA_MODEL = "qwen:0.5b"
_OLLAMA_DELETE_MODEL_POST_TEST = False

# Until the full model server will be available in an official mlrun release
# we will use the following server to test the Mlrun class
class LangChainModelServer(V2ModelServer):
    """
    LangChain Model serving class, short testing version
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        llm: Union[str, LLM] = None,
        init_kwargs: Dict[str, Any] = None,
        generation_kwargs: Dict[str, Any] = None,
        name: str = "",
        model_path: str = "",
    ):
        """
        Initialize a serving class for general llm usage.
        :param context:           The mlrun context to use.
        :param llm:               The llm object itself in case of local usage or the
                                    name of the llm.
        :param init_kwargs:       The initialization arguments to use while initializing
                                    the llm.
        :param generation_kwargs: The generation arguments to use while generating text.
        :param name:              The name of this server to be initialized.
        :param model_path:        Not in use. When adding a model pass any string value
        """
        super().__init__(name=name, context=context, model_path=model_path)
        self.llm = llm
        self.init_kwargs = init_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}
        self.model = None

    def load(self):
        """
        loads the model.
        """
        # If the llm is a string (or not given, then we take default model),
        # load the llm from langchain.
        model_class = getattr(langchain_community.llms, self.llm)
        if self.init_method:
            self.model = getattr(model_class, self.init_method)(**self.init_kwargs)
        else:
            self.model = model_class(**self.init_kwargs)

    def predict(
        self,
        request: Dict[str, Any],
    ) -> str:
        """
        Predict the output of the model, can use the following usages
        """
        inputs = request.get("inputs", [])
        usage = request.get("usage", "predict")
        generation_kwargs = (
            request.get("generation_kwargs", None) or self.generation_kwargs
        )
        # Both predict and invoke are the using the same method,
        # invoke has more generation options to comply with langchain's invoke method.
        if usage == "predict":
            return self.model.invoke(input=inputs[0], config=generation_kwargs)
        elif usage == "invoke":
            config = request.get("config", None)
            stop = request.get("stop", None)
            answer = self.model.invoke(
                input=inputs[0], config=config, stop=stop, **generation_kwargs
            )
            return answer
        elif usage == "batch":
            config = request.get("config", None)
            return_exceptions = request.get("return_exceptions", None)
            return self.model.batch(
                inputs=inputs,
                config=config,
                return_exceptions=return_exceptions,
                **generation_kwargs,
            )


@pytest.mark.requires("mlrun", "transformers")
def test_huggingface():
    """
    Test the langchain model server with a huggingface model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    prompt = "How far is the moon"
    question_len = len(prompt.split(" "))
    max_tokens = 10
    model_id = _HUGGINGFACE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_tokens
    )
    # Create a project
    project = mlrun.get_or_create_project(
        name="huggingface-model-server-example", context="./"
    )
    # Create a serving function
    serving_func = project.set_function(
        func=__file__,
        name="huggingface-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    # Add the hf pipeline to the serving function
    serving_func.add_model(
        "huggingface-langchain-model",
        class_name="LangChainModelServer",
        llm="HuggingFacePipeline",
        init_kwargs={"pipeline": pipe},
        model_path=".",
    )
    # Create a mock server instead of deploying it
    server = serving_func.to_mock_server()
    # Initialize the Mlrun class with the server and the model name
    # and test the functions with various inputs and parameters
    llm = MLRun(server, "huggingface-langchain-model")
    invoke_result = llm.invoke("how old are you?")
    assert invoke_result
    invoke_result_params = llm.invoke(
        "how old are you?",
        stop=["<eos>"],
        generation_kwargs={"return_full_text": False},
    )
    assert invoke_result_params
    assert len(invoke_result_params.lstrip().split(" ")) <= max_tokens + question_len
    batch_result = llm.batch(["how old are you?", "what is your name?"])
    assert batch_result
    batch_result_params = llm.batch(
        ["how old are you?", "what is your name?"],
        generation_kwargs={"return_full_text": False},
    )
    assert batch_result_params


def skip_openai() -> bool:
    """
    Check if the OpenAI API credentials are set
    """
    return not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL")


@pytest.mark.requires("mlrun")
@pytest.mark.skipif(skip_openai(), reason="OpenAI API credentials not set")
def test_openai():
    """
    Test the langchain model server with an openai model
    """
    prompt = "How far is the moon"
    question_len = len(prompt.split(" "))
    max_tokens = 10
    # Create a project
    project = mlrun.get_or_create_project(
        name="openai-model-server-example", context="./"
    )
    # Create a serving function
    serving_func = project.set_function(
        func=__file__,
        name="openai-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    # Add the openai model to the serving function
    serving_func.add_model(
        "openai-langchain-model",
        llm="OpenAI",
        class_name="LangChainModelServer",
        init_kwargs={"model": _OPENAI_MODEL},
        model_path=".",
    )
    # Create a mock server instead of deploying it
    server = serving_func.to_mock_server()
    # Initialize the Mlrun class with the server and the model name
    # and test the functions with various inputs and parameters
    llm = MLRun(server, "openai-langchain-model")
    invoke_result = llm.invoke(prompt)
    assert invoke_result
    invoke_result_params = llm.invoke(
        prompt, generation_kwargs={"max_tokens": max_tokens}
    )
    assert invoke_result_params
    assert len(invoke_result_params.split(" ")) <= max_tokens + question_len
    batch_result = llm.batch([prompt, "what is your name?"])
    assert batch_result


def ollama_check_skip() -> bool:
    """
    Check if ollama is installed
    """
    try:
        result = subprocess.run(["ollama", "--help"], stdout=PIPE)
    except Exception:
        return True
    return result.returncode != 0


@pytest.fixture
def ollama_fixture() -> None:
    """
    Do the setup and cleanup for the ollama test
    """
    # pull or make sure the model is available
    subprocess.run(["ollama", "pull", _OLLAMA_MODEL], stdout=PIPE)

    # start the ollama server
    ollama_serve_process = Popen(["ollama", "serve"], stdout=PIPE)

    yield

    ollama_serve_process.kill()
    # delete the model after the test if requested
    global _OLLAMA_DELETE_MODEL_POST_TEST
    if _OLLAMA_DELETE_MODEL_POST_TEST:
        subprocess.run(["ollama", "rm", _OLLAMA_MODEL], stdout=PIPE)


@pytest.mark.requires("mlrun")
@pytest.mark.skipif(ollama_check_skip(), reason="Ollama not installed")
def test_ollama(ollama_fixture: Any) -> None:
    """
    Test the langchain model server with an ollama model
    """
    prompt = "How far is the moon"
    question_len = len(prompt.split(" "))
    max_tokens = 10
    # Create a project
    project = mlrun.get_or_create_project(
        name="ollama-model-server-example", context="./"
    )
    # Create a serving function
    serving_func = project.set_function(
        func=__file__,
        name="ollama-langchain-model-server",
        kind="serving",
        image="mlrun/mlrun",
    )
    # Add the ollama model to the serving function
    serving_func.add_model(
        "ollama-langchain-model",
        llm="Ollama",
        class_name="LangChainModelServer",
        init_kwargs={"model": _OLLAMA_MODEL},
        model_path=".",
    )
    # Create a mock server instead of deploying it
    # and test the functions with various inputs and parameters
    server = serving_func.to_mock_server()
    llm = MLRun(server, "ollama-langchain-model")
    invoke_result = llm.invoke(prompt)
    assert invoke_result
    invoke_result_params = llm.invoke(
        prompt, generation_kwargs={"num_predict": max_tokens}, stop=["<eos>"]
    )
    assert invoke_result_params
    assert len(invoke_result_params.split(" ")) <= max_tokens + question_len
    batch_result = llm.batch([prompt, "what is your name?"])
    assert batch_result
    batch_result_params = llm.batch(
        [prompt, "what is your name?"],
        generation_kwargs={"num_predict": max_tokens},
    )
    assert batch_result_params
