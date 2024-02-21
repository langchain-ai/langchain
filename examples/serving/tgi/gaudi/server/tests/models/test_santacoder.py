import pytest

from text_generation_server.pb import generate_pb2
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.models.santacoder import SantaCoder


@pytest.fixture(scope="session")
def default_santacoder():
    return SantaCoder("bigcode/santacoder")


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="def",
        prefill_logprobs=True,
        truncate=100,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_fim_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="<fim-prefix>def<fim-suffix>world<fim-middle>",
        prefill_logprobs=True,
        truncate=100,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_fim_pb_batch(default_fim_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_fim_pb_request], size=1)


@pytest.mark.skip
def test_santacoder_generate_token_completion(default_santacoder, default_pb_batch):
    batch = CausalLMBatch.from_pb(
        default_pb_batch,
        default_santacoder.tokenizer,
        default_santacoder.dtype,
        default_santacoder.device,
    )
    next_batch = batch

    for _ in range(batch.stopping_criterias[0].max_new_tokens - 1):
        generations, next_batch = default_santacoder.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_santacoder.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert generations[0].generated_text.text == " test_get_all_users_with_"
    assert generations[0].request_id == batch.requests[0].id
    assert (
        generations[0].generated_text.generated_tokens
        == batch.stopping_criterias[0].max_new_tokens
    )


@pytest.mark.skip
def test_fim_santacoder_generate_token_completion(
    default_santacoder, default_fim_pb_batch
):
    batch = CausalLMBatch.from_pb(
        default_fim_pb_batch,
        default_santacoder.tokenizer,
        default_santacoder.dtype,
        default_santacoder.device,
    )
    next_batch = batch

    for _ in range(batch.stopping_criterias[0].max_new_tokens - 1):
        generations, next_batch = default_santacoder.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_santacoder.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert (
        generations[0].generated_text.text
        == """ineProperty(exports, "__esModule", { value"""
    )
    assert generations[0].request_id == batch.requests[0].id
    assert (
        generations[0].generated_text.generated_tokens
        == batch.stopping_criterias[0].max_new_tokens
    )
