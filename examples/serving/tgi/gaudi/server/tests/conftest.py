import pytest

from text_generation_server.pb import generate_pb2


@pytest.fixture
def default_pb_parameters():
    return generate_pb2.NextTokenChooserParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False,
    )


@pytest.fixture
def default_pb_stop_parameters():
    return generate_pb2.StoppingCriteriaParameters(stop_sequences=[], max_new_tokens=10)
