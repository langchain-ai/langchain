"""Integration test for Pmcid search Wrapper."""

import pytest

from langchain_community.utilities import PmcIDyRun

xmltodict = pytest.importorskip("xmltodict")


@pytest.fixture
def api_client() -> PmcIDyRun:
    return PmcIDyRun()  # type: ignore[call-arg]

@pytest.mark.requires('indra', 'lxml', 'bs4', 'xmltodict')
def test_run_success(api_client: PmcIDyRun) -> None:
    """Test that returns the full-text of pmc paper"""

    PMCID_string = "PMC2671642"
    output = api_client.run(PMCID_string)
    print(output)
    test_string = (  # the title of pmc paper
        "Comprehensive genomic characterization defines human glioblastoma genes and core pathways"
    )
    assert test_string in output

@pytest.mark.requires('indra', 'lxml', 'bs4', 'xmltodict')
def test_run_returns_no_result(api_client: PmcIDyRun) -> None:
    """Test that gives no result. This pmcid has no free full text to extract"""

    output = api_client.run("PMC10043565")
    print(output)
    assert "Error read and extract pmcid PMC10043565" == output

@pytest.mark.requires('indra', 'lxml', 'bs4', 'xmltodict')
def test_wrong_id(api_client: PmcIDyRun) -> None:
    """Test that pmcid is wrong or not exist"""

    output = api_client.run("PMC100000")
    assert "An error occurred (404)" == output
