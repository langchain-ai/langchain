"""Test transform chain."""
from typing import Dict

import pytest

from langchain.chains.transform import TransformChain


def dummy_transform(inputs: Dict[str, str]) -> Dict[str, str]:
    """Transform a dummy input for tests."""
    outputs = inputs
    outputs["greeting"] = f"{inputs['first_name']} {inputs['last_name']} says hello"
    del outputs["first_name"]
    del outputs["last_name"]
    return outputs


def test_transform_chain() -> None:
    """Test basic transform chain."""
    transform_chain = TransformChain(
        input_variables=["first_name", "last_name"],
        output_variables=["greeting"],
        transform=dummy_transform,
    )
    input_dict = {"first_name": "Leroy", "last_name": "Jenkins"}
    response = transform_chain(input_dict)
    expected_response = {"greeting": "Leroy Jenkins says hello"}
    assert response == expected_response


def test_transform_chain_bad_inputs() -> None:
    """Test basic transform chain."""
    transform_chain = TransformChain(
        input_variables=["first_name", "last_name"],
        output_variables=["greeting"],
        transform=dummy_transform,
    )
    input_dict = {"name": "Leroy", "last_name": "Jenkins"}
    with pytest.raises(ValueError):
        _ = transform_chain(input_dict)
