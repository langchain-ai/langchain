import sys

sys.path.append("..")

import rl_chain.slates_chain as slates
from test_utils import MockEncoder

import pytest

encoded_keyword = "[encoded]"
encoded_text = "[ e n c o d e d ] "


def test_slate_text_creation_no_label_no_emb():
    named_actions = {"prefix": ["0", "1"], "context": ["bla"], "suffix": ["0", "1"]}
    expected = """slates shared  |\nslates action 0 |Action 0\nslates action 0 |Action 1\nslates action 1 |Action bla\nslates action 2 |Action 0\nslates action 2 |Action 1\nslates slot  |\nslates slot  |\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder()
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on={}
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected


def _str(embedding):
    return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])


def test_slate_text_creation_no_label_w_emb():
    action00 = "0"
    action01 = "1"
    action10 = "bla"
    action20 = "0"
    action21 = "1"
    encoded_action00 = _str(encoded_keyword + action00)
    encoded_action01 = _str(encoded_keyword + action01)
    encoded_action10 = _str(encoded_keyword + action10)
    encoded_action20 = _str(encoded_keyword + action20)
    encoded_action21 = _str(encoded_keyword + action21)

    named_actions = {
        "prefix": slates.base.Embed(["0", "1"]),
        "context": slates.base.Embed(["bla"]),
        "suffix": slates.base.Embed(["0", "1"]),
    }
    expected = f"""slates shared  |\nslates action 0 |Action {encoded_action00}\nslates action 0 |Action {encoded_action01}\nslates action 1 |Action {encoded_action10}\nslates action 2 |Action {encoded_action20}\nslates action 2 |Action {encoded_action21}\nslates slot  |\nslates slot  |\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder(model=MockEncoder())
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on={}
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected


def test_slate_text_create_no_label_w_embed_and_keep():
    action00 = "0"
    action01 = "1"
    action10 = "bla"
    action20 = "0"
    action21 = "1"
    encoded_action00 = _str(encoded_keyword + action00)
    encoded_action01 = _str(encoded_keyword + action01)
    encoded_action10 = _str(encoded_keyword + action10)
    encoded_action20 = _str(encoded_keyword + action20)
    encoded_action21 = _str(encoded_keyword + action21)

    named_actions = {
        "prefix": slates.base.EmbedAndKeep(["0", "1"]),
        "context": slates.base.EmbedAndKeep(["bla"]),
        "suffix": slates.base.EmbedAndKeep(["0", "1"]),
    }
    expected = f"""slates shared  |\nslates action 0 |Action {action00 + " " + encoded_action00}\nslates action 0 |Action {action01 + " " + encoded_action01}\nslates action 1 |Action {action10 + " " + encoded_action10}\nslates action 2 |Action {action20 + " " + encoded_action20}\nslates action 2 |Action {action21 + " " + encoded_action21}\nslates slot  |\nslates slot  |\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder(model=MockEncoder())
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on={}
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected


def test_slates_raw_features_underscored():
    action00 = "this is a long action 0"
    action01 = "this is a long action 1"
    action00_underscored = action00.replace(" ", "_")
    action01_underscored = action01.replace(" ", "_")
    encoded_action00 = _str(encoded_keyword + action00)
    encoded_action01 = _str(encoded_keyword + action01)

    ctx_str = "this is a long context"
    ctx_str_underscored = ctx_str.replace(" ", "_")
    encoded_ctx_str = encoded_text + " ".join(char for char in ctx_str)

    # No Embeddings
    named_actions = {"prefix": [action00, action01]}
    context = {"context": ctx_str}
    expected_no_embed = f"""slates shared  |context {ctx_str_underscored} \nslates action 0 |Action {action00_underscored}\nslates action 0 |Action {action01_underscored}\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder(model=MockEncoder())
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected_no_embed

    # Just embeddings
    named_actions = {"prefix": slates.base.Embed([action00, action01])}
    context = {"context": slates.base.Embed(ctx_str)}
    expected_embed = f"""slates shared  |context {encoded_ctx_str} \nslates action 0 |Action {encoded_action00}\nslates action 0 |Action {encoded_action01}\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder(model=MockEncoder())
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected_embed

    # Embeddings and raw features
    named_actions = {"prefix": slates.base.EmbedAndKeep([action00, action01])}
    context = {"context": slates.base.EmbedAndKeep(ctx_str)}
    expected_embed_and_keep = f"""slates shared  |context {ctx_str_underscored + " " + encoded_ctx_str} \nslates action 0 |Action {action00_underscored + " " + encoded_action00}\nslates action 0 |Action {action01_underscored + " " + encoded_action01}\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder(model=MockEncoder())
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected_embed_and_keep
