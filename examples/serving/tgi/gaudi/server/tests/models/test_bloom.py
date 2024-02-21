import pytest
import torch

from copy import copy
from transformers import AutoTokenizer

from text_generation_server.pb import generate_pb2
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.utils import weight_hub_files, download_weights
from text_generation_server.models.bloom import BloomCausalLMBatch, BLOOMSharded


@pytest.fixture(scope="session")
def default_bloom():
    model_id = "bigscience/bloom-560m"
    revision = "main"
    filenames = weight_hub_files(model_id, revision, ".safetensors")
    download_weights(filenames, model_id, revision)
    return BLOOMSharded(model_id)


@pytest.fixture(scope="session")
def bloom_560m_tokenizer():
    return AutoTokenizer.from_pretrained("bigscience/bloom-560m", padding_side="left")


@pytest.fixture
def default_pb_request(default_pb_parameters, default_pb_stop_parameters):
    return generate_pb2.Request(
        id=0,
        inputs="Test",
        prefill_logprobs=True,
        truncate=100,
        parameters=default_pb_parameters,
        stopping_parameters=default_pb_stop_parameters,
    )


@pytest.fixture
def default_pb_batch(default_pb_request):
    return generate_pb2.Batch(id=0, requests=[default_pb_request], size=1)


@pytest.fixture
def default_bloom_batch(default_pb_batch, bloom_560m_tokenizer):
    return BloomCausalLMBatch.from_pb(
        default_pb_batch, bloom_560m_tokenizer, torch.float32, torch.device("cpu")
    )


@pytest.fixture
def default_multi_requests_bloom_batch(default_pb_request, bloom_560m_tokenizer):
    req_0 = copy(default_pb_request)
    req_0.id = 1
    req_1 = default_pb_request
    req_1.id = 2
    req_1.stopping_parameters.max_new_tokens = 5

    batch_pb = generate_pb2.Batch(id=0, requests=[req_0, req_1], size=2)
    return BloomCausalLMBatch.from_pb(
        batch_pb, bloom_560m_tokenizer, torch.float32, torch.device("cpu")
    )


def test_batch_from_pb(default_pb_batch, default_bloom_batch):
    batch = default_bloom_batch

    assert batch.batch_id == default_pb_batch.id
    assert batch.requests == default_pb_batch.requests

    assert len(batch.input_ids) == default_pb_batch.size
    assert batch.input_ids[0][-1] == 10264
    assert torch.all(batch.input_ids[0][:-1] == 3)

    assert batch.attention_mask[0][0] == 1
    assert torch.all(batch.attention_mask[0][1:] == 0)

    assert batch.past_key_values is None

    assert all(
        [
            torch.equal(input_ids, all_input_ids[:, 0])
            for input_ids, all_input_ids in zip(batch.input_ids, batch.all_input_ids)
        ]
    )

    assert batch.input_lengths == [1]

    assert len(batch) == default_pb_batch.size
    assert len(batch.next_token_choosers) == len(batch.stopping_criterias) == len(batch)

    assert batch.max_input_length == batch.input_lengths[0]


def test_batch_concatenate_no_prefill(default_bloom_batch):
    with pytest.raises(ValueError):
        BloomCausalLMBatch.concatenate([default_bloom_batch, default_bloom_batch])


def test_causal_lm_batch_type(default_bloom):
    assert default_bloom.batch_type == BloomCausalLMBatch


def test_causal_lm_generate_token(default_bloom, default_bloom_batch):
    sequence_length = len(default_bloom_batch.all_input_ids[0])
    generations, next_batch = default_bloom.generate_token(default_bloom_batch)

    assert len(generations) == len(default_bloom_batch)
    assert isinstance(next_batch, CausalLMBatch)
    assert not next_batch.keys_head_dim_last

    assert len(next_batch.all_input_ids) == len(next_batch)
    assert len(next_batch.all_input_ids[0]) == sequence_length + 1
    assert len(next_batch.attention_mask[0]) == 11
    assert torch.all(next_batch.all_input_ids[0][-2:] == 10264)
    assert torch.all(next_batch.all_input_ids[0][:-2] == 3)

    assert torch.all(next_batch.attention_mask[0][:2] == 1)
    assert torch.all(next_batch.attention_mask[0][2:] == 0)

    assert next_batch.input_ids.shape == (len(next_batch), 1)
    assert next_batch.input_ids[0, 0] == 10264

    assert next_batch.input_lengths == [2]
    assert next_batch.max_input_length == next_batch.input_lengths[0]

    assert next_batch.past_key_values is not None
    assert all(
        [p[0].shape == (16, 64, sequence_length) for p in next_batch.past_key_values]
    )
    assert all(
        [p[1].shape == (16, sequence_length, 64) for p in next_batch.past_key_values]
    )
    assert all([generation.generated_text is None for generation in generations])
    assert all([len(generation.prefill_tokens) == 1 for generation in generations])
    assert all([generation.token_id.item() == 10264 for generation in generations])
    assert all([generation.token_text == "Test" for generation in generations])
    assert generations[0].request_id == 0


def test_causal_lm_generate_token_completion(default_bloom, default_bloom_batch):
    next_batch = default_bloom_batch
    for _ in range(default_bloom_batch.stopping_criterias[0].max_new_tokens - 1):
        generations, next_batch = default_bloom.generate_token(next_batch)
        assert len(generations) == len(default_bloom_batch)

    generations, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert (
        generations[0].generated_text.text == "TestTestTestTestTestTestTestTestTestTest"
    )
    assert generations[0].request_id == default_bloom_batch.requests[0].id
    assert (
        generations[0].generated_text.generated_tokens
        == default_bloom_batch.stopping_criterias[0].max_new_tokens
    )


def test_causal_lm_generate_token_completion_multi(
    default_bloom, default_multi_requests_bloom_batch
):
    next_batch = default_multi_requests_bloom_batch

    for i in range(
        default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens - 1
    ):
        generations, next_batch = default_bloom.generate_token(next_batch)
        assert len(generations) == len(default_multi_requests_bloom_batch)

    generations, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is not None

    assert len(generations) == 2
    assert generations[1].generated_text.text == "TestTestTestTestTest"
    assert (
        generations[1].request_id == default_multi_requests_bloom_batch.requests[1].id
    )
    assert (
        generations[1].generated_text.generated_tokens
        == default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
    )
    # Copy stopping_criterias before filtering
    stopping_criterias = default_multi_requests_bloom_batch.stopping_criterias.copy()

    next_batch = next_batch.filter([next_batch.requests[0].id])

    for _ in range(
        stopping_criterias[0].max_new_tokens - stopping_criterias[1].max_new_tokens - 1
    ):
        generations, next_batch = default_bloom.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert (
        generations[0].generated_text.text == "TestTestTestTestTestTestTestTestTestTest"
    )
    assert (
        generations[0].request_id == default_multi_requests_bloom_batch.requests[0].id
    )
    assert (
        generations[0].generated_text.generated_tokens
        == default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
    )


def test_batch_concatenate(
    default_bloom, default_bloom_batch, default_multi_requests_bloom_batch
):
    next_batch_0 = default_bloom_batch
    _, next_batch_0 = default_bloom.generate_token(next_batch_0)
    _, next_batch_0 = default_bloom.generate_token(next_batch_0)

    next_batch_1 = default_multi_requests_bloom_batch
    _, next_batch_1 = default_bloom.generate_token(next_batch_1)

    # Clone past_key_values before concatenating to compare after,
    # because they are removed from the concatenated batches
    next_batch_0_past_key_values = [
        (k.clone(), v.clone()) for (k, v) in next_batch_0.past_key_values
    ]
    next_batch_1_past_key_values = [
        (k.clone(), v.clone()) for (k, v) in next_batch_1.past_key_values
    ]

    next_batch = BloomCausalLMBatch.concatenate([next_batch_0, next_batch_1])

    assert torch.equal(next_batch.all_input_ids[0], next_batch_0.all_input_ids[0])
    assert torch.equal(next_batch.all_input_ids[1], next_batch_1.all_input_ids[0])
    assert torch.equal(next_batch.all_input_ids[2], next_batch_1.all_input_ids[1])

    assert torch.all(
        next_batch.attention_mask[0, : -next_batch.padding_right_offset] == 1
    )
    assert torch.all(
        next_batch.attention_mask[1:, 1 : -next_batch.padding_right_offset] == 1
    )
    assert torch.all(next_batch.attention_mask[1:, 3:] == 0)

    assert next_batch.batch_id == 0
    assert torch.all(next_batch.input_ids == 10264)

    assert next_batch.input_lengths == [3, 2, 2]
    assert next_batch.max_input_length == 3

    assert next_batch.requests[0] == next_batch_0.requests[0]
    assert next_batch.requests[1:] == next_batch_1.requests

    assert next_batch.next_token_choosers[0] == next_batch_0.next_token_choosers[0]
    assert next_batch.next_token_choosers[1:] == next_batch_1.next_token_choosers

    assert next_batch.stopping_criterias[0] == next_batch_0.stopping_criterias[0]
    assert next_batch.stopping_criterias[1:] == next_batch_1.stopping_criterias

    assert next_batch.past_key_values is not None
    assert all([p[0].shape == (3, 16, 64, 2) for p in next_batch.past_key_values])
    assert all([p[1].shape == (3, 16, 2, 64) for p in next_batch.past_key_values])

    for i, past in enumerate(next_batch.past_key_values):
        assert torch.equal(next_batch_0_past_key_values[i][0][:, :, -2:], past[0][0])
        assert torch.equal(
            next_batch_1_past_key_values[i][0][:, :, -1:],
            past[0][1:, :, :, -1].reshape(-1, 64, 1),
        )

        assert torch.equal(next_batch_0_past_key_values[i][1][:, -2:, :], past[1][0])
        assert torch.equal(
            next_batch_1_past_key_values[i][1][:, -1:, :],
            past[1][1:, :, -1, :].reshape(-1, 1, 64),
        )

    for _ in range(
        default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens - 2
    ):
        generations, next_batch = default_bloom.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is not None

    assert len(generations) == 3
    assert generations[2].generated_text.text == "TestTestTestTestTest"
    assert (
        generations[2].request_id == default_multi_requests_bloom_batch.requests[1].id
    )
    assert (
        generations[2].generated_text.generated_tokens
        == default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
    )

    next_batch = next_batch.filter(
        [next_batch.requests[0].id, next_batch.requests[1].id]
    )

    for _ in range(
        default_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
        - 2
    ):
        generations, next_batch = default_bloom.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is not None

    assert len(generations) == 2
    assert (
        generations[0].generated_text.text == "TestTestTestTestTestTestTestTestTestTest"
    )
    assert generations[0].request_id == default_bloom_batch.requests[0].id
    assert (
        generations[0].generated_text.generated_tokens
        == default_bloom_batch.stopping_criterias[0].max_new_tokens
    )

    next_batch = next_batch.filter([next_batch.requests[1].id])

    for _ in range(
        default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_bloom_batch.stopping_criterias[0].max_new_tokens
        - default_multi_requests_bloom_batch.stopping_criterias[1].max_new_tokens
        - 4
    ):
        generations, next_batch = default_bloom.generate_token(next_batch)
        assert len(generations) == len(next_batch)

    generations, next_batch = default_bloom.generate_token(next_batch)
    assert next_batch is None

    assert len(generations) == 1
    assert (
        generations[0].generated_text.text == "TestTestTestTestTestTestTestTestTestTest"
    )
    assert (
        generations[0].request_id == default_multi_requests_bloom_batch.requests[0].id
    )
    assert (
        generations[0].generated_text.generated_tokens
        == default_multi_requests_bloom_batch.stopping_criterias[0].max_new_tokens
    )
