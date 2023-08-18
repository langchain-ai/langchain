import examples as examples
import pytest
from pydantic.types import conlist

from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.data_generation.base import SyntheticDataGenerator
from langchain.data_generation.openai import create_openai_data_generator, OPENAI_TEMPLATE
from langchain.data_generation.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX, DEFAULT_PROMPT
from pydantic import BaseModel


# Define the desired output schema for individual medical billing record
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: str
    total_charge: float
    insurance_claim_amount: float


examples = [
    {
        "example": "Patient ID: 123456, Patient Name: John Doe, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"},
    {
        "example": "Patient ID: 789012, Patient Name: Johnson Smith, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"},
]

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)


@pytest.fixture(scope="function")
def synthetic_data_generator():
    return create_openai_data_generator(
        output_schema=MedicalBilling,
        llm=ChatOpenAI(temperature=1),  # replace with your LLM instance
        prompt=prompt_template
    )


@pytest.mark.requires("openai")
def test_generate_synthetic(synthetic_data_generator: SyntheticDataGenerator):
    synthetic_results = synthetic_data_generator.generate("medical_billing", runs=10)
    assert len(synthetic_results) == 10
    for row in synthetic_results:
        assert isinstance(row, MedicalBilling)
    print(synthetic_results)


@pytest.mark.requires("openai")
@pytest.mark.asyncio
async def test_agenerate_synthetic(synthetic_data_generator: SyntheticDataGenerator):
    synthetic_results = await synthetic_data_generator.agenerate("medical_billing", runs=10)
    assert len(synthetic_results) == 10
    for row in synthetic_results:
        assert isinstance(row, MedicalBilling)
    print(synthetic_results)
