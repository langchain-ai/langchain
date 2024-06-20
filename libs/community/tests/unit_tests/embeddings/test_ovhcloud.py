import pytest

from langchain_community.embeddings.ovhcloud import OVHCloudEmbeddings


def test_ovhcloud_correct_instantiation() -> None:
    llm = OVHCloudEmbeddings(model_name="multilingual-e5-base")
    assert isinstance(llm, OVHCloudEmbeddings)


def test_ovhcloud_empty_model_name_should_raise_error() -> None:
    with pytest.raises(ValueError):
        OVHCloudEmbeddings(model_name="")


def test_ovhcloud_empty_region_should_raise_error() -> None:
    with pytest.raises(ValueError):
        OVHCloudEmbeddings(model_name="multilingual-e5-base", region="")


def test_ovhcloud_empty_access_token_should_not_raise_error() -> None:
    llm = OVHCloudEmbeddings(
        model_name="multilingual-e5-base", region="kepler", access_token=""
    )
    assert isinstance(llm, OVHCloudEmbeddings)
