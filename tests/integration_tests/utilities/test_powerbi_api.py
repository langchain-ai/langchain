"""Integration test for POWERBI API Wrapper."""
import pytest

from langchain.utilities.powerbi import PowerBIDataset
from langchain.utils import get_from_env


def azure_installed() -> bool:
    try:
        from azure.core.credentials import TokenCredential  # noqa: F401
        from azure.identity import DefaultAzureCredential  # noqa: F401

        return True
    except Exception as e:
        print(f"azure not installed, skipping test {e}")
        return False


@pytest.mark.skipif(not azure_installed(), reason="requires azure package")
def test_daxquery() -> None:
    from azure.identity import DefaultAzureCredential

    DATASET_ID = get_from_env("", "POWERBI_DATASET_ID")
    TABLE_NAME = get_from_env("", "POWERBI_TABLE_NAME")
    NUM_ROWS = get_from_env("", "POWERBI_NUMROWS")

    powerbi = PowerBIDataset(
        dataset_id=DATASET_ID,
        table_names=[TABLE_NAME],
        credential=DefaultAzureCredential(),
    )

    output = powerbi.run(f'EVALUATE ROW("RowCount", COUNTROWS({TABLE_NAME}))')
    numrows = str(output["results"][0]["tables"][0]["rows"][0]["[RowCount]"])

    assert NUM_ROWS == numrows
