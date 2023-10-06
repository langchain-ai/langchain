from typing import Any, Dict

import pytest
from langchain.chat_models import FakeListChatModel
from langchain.prompts.prompt import PromptTemplate
from test_utils import MockEncoder, MockEncoderReturnsList

import langchain_experimental.rl_chain.base as rl_chain
import langchain_experimental.rl_chain.pick_best_chain as pick_best_chain

encoded_keyword = "[encoded]"


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def setup() -> tuple:
    _PROMPT_TEMPLATE = """This is a dummy prompt that will be ignored by the fake llm"""
    PROMPT = PromptTemplate(input_variables=[], template=_PROMPT_TEMPLATE)

    llm = FakeListChatModel(responses=["hey"])
    return llm, PROMPT


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_multiple_ToSelectFrom_throws() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    with pytest.raises(ValueError):
        chain.run(
            User=rl_chain.BasedOn("Context"),
            action=rl_chain.ToSelectFrom(actions),
            another_action=rl_chain.ToSelectFrom(actions),
        )


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_missing_basedOn_from_throws() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    with pytest.raises(ValueError):
        chain.run(action=rl_chain.ToSelectFrom(actions))


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_ToSelectFrom_not_a_list_throws() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = {"actions": ["0", "1", "2"]}
    with pytest.raises(ValueError):
        chain.run(
            User=rl_chain.BasedOn("Context"),
            action=rl_chain.ToSelectFrom(actions),
        )


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_update_with_delayed_score_with_auto_validator_throws() -> None:
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=rl_chain.AutoSelectionScorer(llm=auto_val_llm),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 3.0  # type: ignore
    with pytest.raises(RuntimeError):
        chain.update_with_delayed_score(
            chain_response=response, score=100  # type: ignore
        )


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_update_with_delayed_score_force() -> None:
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=rl_chain.AutoSelectionScorer(llm=auto_val_llm),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 3.0  # type: ignore
    chain.update_with_delayed_score(
        chain_response=response, score=100, force_score=True  # type: ignore
    )
    assert selection_metadata.selected.score == 100.0  # type: ignore


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_update_with_delayed_score() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=None,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score is None  # type: ignore
    chain.update_with_delayed_score(chain_response=response, score=100)  # type: ignore
    assert selection_metadata.selected.score == 100.0  # type: ignore


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_user_defined_scorer() -> None:
    llm, PROMPT = setup()

    class CustomSelectionScorer(rl_chain.SelectionScorer):
        def score_response(
            self,
            inputs: Dict[str, Any],
            llm_response: str,
            event: pick_best_chain.PickBestEvent,
        ) -> float:
            score = 200
            return score

    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=CustomSelectionScorer(),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 200.0  # type: ignore


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_everything_embedded() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder, auto_embed=False
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = rl_chain.stringify_embedding(list(encoded_keyword + str1))
    encoded_str2 = rl_chain.stringify_embedding(list(encoded_keyword + str2))
    encoded_str3 = rl_chain.stringify_embedding(list(encoded_keyword + str3))

    ctx_str_1 = "context1"

    encoded_ctx_str_1 = rl_chain.stringify_embedding(list(encoded_keyword + ctx_str_1))

    expected = f"""shared |User {ctx_str_1 + " " + encoded_ctx_str_1} \n|action {str1 + " " + encoded_str1} \n|action {str2 + " " + encoded_str2} \n|action {str3 + " " + encoded_str3} """  # noqa

    actions = [str1, str2, str3]

    response = chain.run(
        User=rl_chain.EmbedAndKeep(rl_chain.BasedOn(ctx_str_1)),
        action=rl_chain.EmbedAndKeep(rl_chain.ToSelectFrom(actions)),
    )
    selection_metadata = response["selection_metadata"]  # type: ignore
    vw_str = feature_embedder.format(selection_metadata)  # type: ignore
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_auto_embedder_is_off() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    ctx_str_1 = "context1"

    expected = f"""shared |User {ctx_str_1} \n|action {str1} \n|action {str2} \n|action {str3} """  # noqa

    actions = [str1, str2, str3]

    response = chain.run(
        User=pick_best_chain.base.BasedOn(ctx_str_1),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    selection_metadata = response["selection_metadata"]  # type: ignore
    vw_str = feature_embedder.format(selection_metadata)  # type: ignore
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_w_embeddings_off() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=False, model=MockEncoder()
    )
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder, auto_embed=False
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    ctx_str_1 = "context1"

    expected = f"""shared |User {ctx_str_1} \n|action {str1} \n|action {str2} \n|action {str3} """  # noqa

    actions = [str1, str2, str3]

    response = chain.run(
        User=rl_chain.BasedOn(ctx_str_1),
        action=rl_chain.ToSelectFrom(actions),
    )
    selection_metadata = response["selection_metadata"]  # type: ignore
    vw_str = feature_embedder.format(selection_metadata)  # type: ignore
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_w_embeddings_on() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=True, model=MockEncoderReturnsList()
    )
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder, auto_embed=True
    )

    str1 = "0"
    str2 = "1"
    ctx_str_1 = "context1"
    dot_prod = "dotprod 0:5.0"  # dot prod of [1.0, 2.0] and [1.0, 2.0]

    expected = f"""shared |User {ctx_str_1} |@ User={ctx_str_1}\n|action {str1} |# action={str1} |{dot_prod}\n|action {str2} |# action={str2} |{dot_prod}"""  # noqa

    actions = [str1, str2]

    response = chain.run(
        User=rl_chain.BasedOn(ctx_str_1),
        action=rl_chain.ToSelectFrom(actions),
    )
    selection_metadata = response["selection_metadata"]  # type: ignore
    vw_str = feature_embedder.format(selection_metadata)  # type: ignore
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_embeddings_mixed_w_explicit_user_embeddings() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(
        auto_embed=True, model=MockEncoderReturnsList()
    )
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder, auto_embed=True
    )

    str1 = "0"
    str2 = "1"
    encoded_str2 = rl_chain.stringify_embedding([1.0, 2.0])
    ctx_str_1 = "context1"
    ctx_str_2 = "context2"
    encoded_ctx_str_1 = rl_chain.stringify_embedding([1.0, 2.0])
    dot_prod = "dotprod 0:5.0 1:5.0"  # dot prod of [1.0, 2.0] and [1.0, 2.0]

    expected = f"""shared |User {encoded_ctx_str_1} |@ User={encoded_ctx_str_1} |User2 {ctx_str_2} |@ User2={ctx_str_2}\n|action {str1} |# action={str1} |{dot_prod}\n|action {encoded_str2} |# action={encoded_str2} |{dot_prod}"""  # noqa

    actions = [str1, rl_chain.Embed(str2)]

    response = chain.run(
        User=rl_chain.BasedOn(rl_chain.Embed(ctx_str_1)),
        User2=rl_chain.BasedOn(ctx_str_2),
        action=rl_chain.ToSelectFrom(actions),
    )
    selection_metadata = response["selection_metadata"]  # type: ignore
    vw_str = feature_embedder.format(selection_metadata)  # type: ignore
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_no_scorer_specified() -> None:
    _, PROMPT = setup()
    chain_llm = FakeListChatModel(responses=["hey", "100"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=chain_llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 100.0  # type: ignore


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_explicitly_no_scorer() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=None,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score is None  # type: ignore


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_auto_scorer_with_user_defined_llm() -> None:
    llm, PROMPT = setup()
    scorer_llm = FakeListChatModel(responses=["300"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=rl_chain.AutoSelectionScorer(llm=scorer_llm),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "hey"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 300.0  # type: ignore


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_calling_chain_w_reserved_inputs_throws() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    with pytest.raises(ValueError):
        chain.run(
            User=rl_chain.BasedOn("Context"),
            rl_chain_selected_based_on=rl_chain.ToSelectFrom(["0", "1", "2"]),
        )

    with pytest.raises(ValueError):
        chain.run(
            User=rl_chain.BasedOn("Context"),
            rl_chain_selected=rl_chain.ToSelectFrom(["0", "1", "2"]),
        )


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_activate_and_deactivate_scorer() -> None:
    _, PROMPT = setup()
    llm = FakeListChatModel(responses=["hey1", "hey2", "hey3"])
    scorer_llm = FakeListChatModel(responses=["300", "400"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=pick_best_chain.base.AutoSelectionScorer(llm=scorer_llm),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(
            auto_embed=False, model=MockEncoder()
        ),
    )
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "hey1"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 300.0  # type: ignore

    chain.deactivate_selection_scorer()
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(["0", "1", "2"]),
    )
    assert response["response"] == "hey2"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score is None  # type: ignore

    chain.activate_selection_scorer()
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(["0", "1", "2"]),
    )
    assert response["response"] == "hey3"  # type: ignore
    selection_metadata = response["selection_metadata"]  # type: ignore
    assert selection_metadata.selected.score == 400.0  # type: ignore
