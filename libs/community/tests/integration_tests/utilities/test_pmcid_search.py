"""Integration test for Pmcid search Wrapper."""
import pytest
from langchain_community.utilities import PmcIDyRun


lxml = pytest.importorskip("lxml")


def test_run_success(api_client: PmcIDyRun) -> None:
    """Test that returns the full-text of pmc paper"""

    PMCID_string = "PMC2671642"
    output = api_client.run(PMCID_string)
    print(output)
    test_string = (  # the title of pmc paper
        "Comprehensive genomic characterization defines human glioblastoma genes and core pathways"
    )
    assert test_string in output


def test_run_returns_no_result(api_client: PmcIDyRun) -> None:
    """Test that gives no result. This pmcid has no free full text to extract"""

    output = api_client.run("PMC10043565")
    print(output)
    assert "Error: Unable to retrieve XML data for PMC ID. The XML of the PMC10043565 might not be available." == output


def test_wrong_id(api_client: PmcIDyRun) -> None:
    """Test that pmcid is wrong or not exist"""

    output = api_client.run("PMC100000")
    assert "Error: Unable to retrieve XML data for PMC ID. The XML of the PMC100000 might not be available." == output
