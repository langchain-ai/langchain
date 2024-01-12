import pytest
from test_utils import MockEncoder

import langchain_experimental.rl_chain.base as rl_chain
import langchain_experimental.rl_chain.pick_best_chain as pick_best_chain

encoded_keyword = "[encoded]"


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_missing_context_throws() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_action = {"action": ["0", "1", "2"]}
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_action, based_on={}
    )
    with pytest.raises(ValueError):
        feature_embedder.format(event)


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_missing_actions_throws() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from={}, based_on={"context": "context"}
    )
    with pytest.raises(ValueError):
        feature_embedder.format(event)


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_no_label_no_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": ["0", "1", "2"]}
    expected = """shared |context context \n|action1 0 \n|action1 1 \n|action1 2 """
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on={"context": "context"}
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_w_label_no_score_no_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": ["0", "1", "2"]}
    expected = """shared |context context \n|action1 0 \n|action1 1 \n|action1 2 """
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0)
    event = pick_best_chain.PickBestEvent(
        inputs={},
        to_select_from=named_actions,
        based_on={"context": "context"},
        selected=selected,
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_w_full_label_no_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": ["0", "1", "2"]}
    expected = (
        """shared |context context \n0:-0.0:1.0 |action1 0 \n|action1 1 \n|action1 2 """
    )
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={},
        to_select_from=named_actions,
        based_on={"context": "context"},
        selected=selected,
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_w_full_label_w_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = rl_chain.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"
    encoded_ctx_str_1 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_1))

    named_actions = {"action1": rl_chain.Embed([str1, str2, str3])}
    context = {"context": rl_chain.Embed(ctx_str_1)}
    expected = f"""shared |context {encoded_ctx_str_1} \n0:-0.0:1.0 |action1 {encoded_str1} \n|action1 {encoded_str2} \n|action1 {encoded_str3} """  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_w_full_label_w_embed_and_keep() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = rl_chain.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"
    encoded_ctx_str_1 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_1))

    named_actions = {"action1": rl_chain.EmbedAndKeep([str1, str2, str3])}
    context = {"context": rl_chain.EmbedAndKeep(ctx_str_1)}
    expected = f"""shared |context {ctx_str_1 + " " + encoded_ctx_str_1} \n0:-0.0:1.0 |action1 {str1 + " " + encoded_str1} \n|action1 {str2 + " " + encoded_str2} \n|action1 {str3 + " " + encoded_str3} """  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_no_label_no_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": [{"a": "0", "b": "0"}, "1", "2"]}
    context = {"context1": "context1", "context2": "context2"}
    expected = """shared |context1 context1 |context2 context2 \n|a 0 |b 0 \n|action1 1 \n|action1 2 """  # noqa: E501
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_w_label_no_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": [{"a": "0", "b": "0"}, "1", "2"]}
    context = {"context1": "context1", "context2": "context2"}
    expected = """shared |context1 context1 |context2 context2 \n|a 0 |b 0 \n|action1 1 \n|action1 2 """  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_w_full_label_no_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    named_actions = {"action1": [{"a": "0", "b": "0"}, "1", "2"]}
    context = {"context1": "context1", "context2": "context2"}
    expected = """shared |context1 context1 |context2 context2 \n0:-0.0:1.0 |a 0 |b 0 \n|action1 1 \n|action1 2 """  # noqa: E501
    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_w_full_label_w_full_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = rl_chain.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"
    encoded_ctx_str_1 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_1))
    encoded_ctx_str_2 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_2))

    named_actions = {"action1": rl_chain.Embed([{"a": str1, "b": str1}, str2, str3])}
    context = {
        "context1": rl_chain.Embed(ctx_str_1),
        "context2": rl_chain.Embed(ctx_str_2),
    }
    expected = f"""shared |context1 {encoded_ctx_str_1} |context2 {encoded_ctx_str_2} \n0:-0.0:1.0 |a {encoded_str1} |b {encoded_str1} \n|action1 {encoded_str2} \n|action1 {encoded_str3} """  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_w_full_label_w_full_embed_and_keep() -> (
    None
):
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = rl_chain.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"
    encoded_ctx_str_1 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_1))
    encoded_ctx_str_2 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_2))

    named_actions = {
        "action1": rl_chain.EmbedAndKeep([{"a": str1, "b": str1}, str2, str3])
    }
    context = {
        "context1": rl_chain.EmbedAndKeep(ctx_str_1),
        "context2": rl_chain.EmbedAndKeep(ctx_str_2),
    }
    expected = f"""shared |context1 {ctx_str_1 + " " + encoded_ctx_str_1} |context2 {ctx_str_2 + " " + encoded_ctx_str_2} \n0:-0.0:1.0 |a {str1 + " " + encoded_str1} |b {str1 + " " + encoded_str1} \n|action1 {str2 + " " + encoded_str2} \n|action1 {str3 + " " + encoded_str3} """  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_w_full_label_w_partial_emb() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"
    encoded_ctx_str_2 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_2))

    named_actions = {
        "action1": [
            {"a": str1, "b": rl_chain.Embed(str1)},
            str2,
            rl_chain.Embed(str3),
        ]
    }
    context = {"context1": ctx_str_1, "context2": rl_chain.Embed(ctx_str_2)}
    expected = f"""shared |context1 {ctx_str_1} |context2 {encoded_ctx_str_2} \n0:-0.0:1.0 |a {str1} |b {encoded_str1} \n|action1 {str2} \n|action1 {encoded_str3} """  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_pickbest_textembedder_more_namespaces_w_full_label_w_partial_emakeep() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"
    encoded_ctx_str_2 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_2))

    named_actions = {
        "action1": [
            {"a": str1, "b": rl_chain.EmbedAndKeep(str1)},
            str2,
            rl_chain.EmbedAndKeep(str3),
        ]
    }
    context = {
        "context1": ctx_str_1,
        "context2": rl_chain.EmbedAndKeep(ctx_str_2),
    }
    expected = f"""shared |context1 {ctx_str_1} |context2 {ctx_str_2 + " " + encoded_ctx_str_2} \n0:-0.0:1.0 |a {str1} |b {str1 + " " + encoded_str1} \n|action1 {str2} \n|action1 {str3 + " " + encoded_str3} """  # noqa: E501

    selected = pick_best_chain.PickBestSelected(index=0, probability=1.0, score=0.0)
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context, selected=selected
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected


@pytest.mark.requires("vowpal_wabbit_next")
def test_raw_features_underscored() -> None:
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    str1 = "this is a long string"
    str1_underscored = str1.replace(" ", "_")
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))

    ctx_str = "this is a long context"
    ctx_str_underscored = ctx_str.replace(" ", "_")
    encoded_ctx_str = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str))

    # No embeddings
    named_actions = {"action": [str1]}
    context = {"context": ctx_str}
    expected_no_embed = (
        f"""shared |context {ctx_str_underscored} \n|action {str1_underscored} """
    )
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected_no_embed

    # Just embeddings
    named_actions = {"action": rl_chain.Embed([str1])}
    context = {"context": rl_chain.Embed(ctx_str)}
    expected_embed = f"""shared |context {encoded_ctx_str} \n|action {encoded_str1} """
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected_embed

    # Embeddings and raw features
    named_actions = {"action": rl_chain.EmbedAndKeep([str1])}
    context = {"context": rl_chain.EmbedAndKeep(ctx_str)}
    expected_embed_and_keep = f"""shared |context {ctx_str_underscored + " " + encoded_ctx_str} \n|action {str1_underscored + " " + encoded_str1} """  # noqa: E501
    event = pick_best_chain.PickBestEvent(
        inputs={}, to_select_from=named_actions, based_on=context
    )
    vw_ex_str = feature_embedder.format(event)
    assert vw_ex_str == expected_embed_and_keep
