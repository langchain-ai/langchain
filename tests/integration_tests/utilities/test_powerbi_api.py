"""Integration test for POWERBI API Wrapper."""
from langchain.utilities.powerbi import PowerBIDataset
from azure.identity import DefaultAzureCredential
from langchain.utils import get_from_env

def test_daxquery():

    DATASET_ID = get_from_env("","POWERBI_DATASET_ID")
    TABLE_NAME = get_from_env("", "POWERBI_TABLE_NAME")
    NUM_ROWS = get_from_env("", "POWERBI_NUMROWS")


    powerbi=PowerBIDataset(
        dataset_id=DATASET_ID,
        table_names=[TABLE_NAME],
        credential=DefaultAzureCredential()
        )

    output = powerbi.run(f"EVALUATE ROW(\"RowCount\", COUNTROWS({TABLE_NAME}))")
    numrows = str(output["results"][0]["tables"][0]["rows"][0]["[RowCount]"])

    assert NUM_ROWS == numrows
    



