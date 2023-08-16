import pytest as pytest
from langchain.data_generation.synthetic import generate_synthetic, agenerate_synthetic

examples = [
    # all examples must be in format of dict with a key example -> value the example itself
    {
        "example": "Patient ID: 123456, Patient Name: John Doe, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"
    },
    {
        "example": "Patient ID: 789012, Patient Name: Jane Smith, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"
    },
]


@pytest.mark.requires("openai")
async def test_generate_synthetic():
    synthetic_results = generate_synthetic(examples, "medical_billing", runs=10)
    assert len(synthetic_results) == 10
    for row in synthetic_results:
        assert len(row) > 0
        assert isinstance(row, (str,))
    print(synthetic_results)


@pytest.mark.requires("openai")
@pytest.mark.asyncio
async def test_agenerate_synthetic():
    synthetic_results = await agenerate_synthetic(examples, "medical_billing", runs=10)
    assert len(synthetic_results) == 10
    for row in synthetic_results:
        assert len(row) > 0
        assert isinstance(row, (str,))
    print(synthetic_results)
