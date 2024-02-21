import pytest
import torch

from transformers import AutoTokenizer

from text_generation_server.models import Model


def get_test_model():
    class TestModel(Model):
        def batch_type(self):
            raise NotImplementedError

        def generate_token(self, batch):
            raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained("huggingface/llama-7b")

    model = TestModel(
        torch.nn.Linear(1, 1), tokenizer, False, torch.float32, torch.device("cpu")
    )
    return model


@pytest.mark.private
def test_decode_streaming_english_spaces():
    model = get_test_model()
    truth = "Hello here, this is a simple test"
    all_input_ids = [15043, 1244, 29892, 445, 338, 263, 2560, 1243]
    assert (
        all_input_ids == model.tokenizer(truth, add_special_tokens=False)["input_ids"]
    )

    decoded_text = ""
    offset = 0
    token_offset = 0
    for i in range(len(all_input_ids)):
        text, offset, token_offset = model.decode_token(
            all_input_ids[: i + 1], offset, token_offset
        )
        decoded_text += text

    assert decoded_text == truth


@pytest.mark.private
def test_decode_streaming_chinese_utf8():
    model = get_test_model()
    truth = "我很感谢你的热情"
    all_input_ids = [
        30672,
        232,
        193,
        139,
        233,
        135,
        162,
        235,
        179,
        165,
        30919,
        30210,
        234,
        134,
        176,
        30993,
    ]

    decoded_text = ""
    offset = 0
    token_offset = 0
    for i in range(len(all_input_ids)):
        text, offset, token_offset = model.decode_token(
            all_input_ids[: i + 1], offset, token_offset
        )
        decoded_text += text

    assert decoded_text == truth
