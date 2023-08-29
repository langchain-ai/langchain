from typing import Any, Dict

import pytest
from test_utils import MockEncoder

import langchain.chains.rl_chain.base as rl_chain
import langchain.chains.rl_chain.pick_best_chain as pick_best_chain
from langchain.chat_models import FakeListChatModel
from langchain.prompts.prompt import PromptTemplate

encoded_text = "[ e n c o d e d ] "


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
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
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
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
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
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
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
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 3.0
    with pytest.raises(RuntimeError):
        chain.update_with_delayed_score(event=selection_metadata, score=100)


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_update_with_delayed_score_force() -> None:
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=rl_chain.AutoSelectionScorer(llm=auto_val_llm),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 3.0
    chain.update_with_delayed_score(
        event=selection_metadata, score=100, force_score=True
    )
    assert selection_metadata.selected.score == 100.0


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_update_with_delayed_score() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=None,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score is None
    chain.update_with_delayed_score(event=selection_metadata, score=100)
    assert selection_metadata.selected.score == 100.0


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_user_defined_scorer() -> None:
    llm, PROMPT = setup()

    class CustomSelectionScorer(rl_chain.SelectionScorer):
        def score_response(self, inputs: Dict[str, Any], llm_response: str) -> float:
            score = 200
            return score

    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=CustomSelectionScorer(),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 200.0


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_embeddings() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder())
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = encoded_text + " ".join(char for char in str1)
    encoded_str2 = encoded_text + " ".join(char for char in str2)
    encoded_str3 = encoded_text + " ".join(char for char in str3)

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"

    encoded_ctx_str_1 = encoded_text + " ".join(char for char in ctx_str_1)
    encoded_text + " ".join(char for char in ctx_str_2)

    expected = f"""shared |User {ctx_str_1 + " " + encoded_ctx_str_1} \n|action {str1 + " " + encoded_str1} \n|action {str2 + " " + encoded_str2} \n|action {str3 + " " + encoded_str3} """  # noqa

    actions = [str1, str2, str3]

    response = chain.run(
        User=rl_chain.BasedOn(ctx_str_1),
        action=rl_chain.ToSelectFrom(actions),
    )
    selection_metadata = response["selection_metadata"]
    vw_str = feature_embedder.format(selection_metadata)
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_embeddings_off() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder())
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
    selection_metadata = response["selection_metadata"]
    vw_str = feature_embedder.format(selection_metadata)
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_embeddings_mixed_w_explicit_user_embeddings() -> None:
    llm, PROMPT = setup()
    feature_embedder = pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder())
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, feature_embedder=feature_embedder
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = encoded_text + " ".join(char for char in str1)
    encoded_str2 = encoded_text + " ".join(char for char in str2)
    encoded_str3 = encoded_text + " ".join(char for char in str3)

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"

    encoded_ctx_str_1 = encoded_text + " ".join(char for char in ctx_str_1)
    encoded_ctx_str_2 = encoded_text + " ".join(char for char in ctx_str_2)

    expected = f"""shared |User {encoded_ctx_str_1} |User2 {ctx_str_2 + " " + encoded_ctx_str_2} \n|action {str1 + " " + encoded_str1} \n|action {str2 + " " + encoded_str2} \n|action {encoded_str3} """  # noqa

    actions = [str1, str2, rl_chain.Embed(str3)]

    response = chain.run(
        User=rl_chain.BasedOn(rl_chain.Embed(ctx_str_1)),
        User2=rl_chain.BasedOn(ctx_str_2),
        action=rl_chain.ToSelectFrom(actions),
    )
    selection_metadata = response["selection_metadata"]
    vw_str = feature_embedder.format(selection_metadata)
    assert vw_str == expected


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_default_no_scorer_specified() -> None:
    _, PROMPT = setup()
    chain_llm = FakeListChatModel(responses=[100])
    chain = pick_best_chain.PickBest.from_llm(
        llm=chain_llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "100"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 100.0


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_explicitly_no_scorer() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=None,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score is None


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_auto_scorer_with_user_defined_llm() -> None:
    llm, PROMPT = setup()
    scorer_llm = FakeListChatModel(responses=[300])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=rl_chain.AutoSelectionScorer(llm=scorer_llm),
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
    )
    response = chain.run(
        User=rl_chain.BasedOn("Context"),
        action=rl_chain.ToSelectFrom(["0", "1", "2"]),
    )
    # chain llm used for both basic prompt and for scoring
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 300.0


@pytest.mark.requires("vowpal_wabbit_next", "sentence_transformers")
def test_calling_chain_w_reserved_inputs_throws() -> None:
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        feature_embedder=pick_best_chain.PickBestFeatureEmbedder(model=MockEncoder()),
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
