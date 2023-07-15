import logging
import os

from langchain.docstore.document import Document
from langchain.document_loaders import RocksetLoader

logger = logging.getLogger(__name__)


def test_sql_query() -> None:
    import rockset

    assert os.environ.get("ROCKSET_API_KEY") is not None
    assert os.environ.get("ROCKSET_REGION") is not None

    api_key = os.environ.get("ROCKSET_API_KEY")
    region = os.environ.get("ROCKSET_REGION")
    if region == "use1a1":
        host = rockset.Regions.use1a1
    elif region == "usw2a1":
        host = rockset.Regions.usw2a1
    elif region == "euc1a1":
        host = rockset.Regions.euc1a1
    elif region == "dev":
        host = rockset.DevRegions.usw2a1
    else:
        logger.warning(
            "Using ROCKSET_REGION:%s as it is.. \
            You should know what you're doing...",
            region,
        )

        host = region

    client = rockset.RocksetClient(host, api_key)

    col_1 = "Rockset is a real-time analytics database which enables queries on massive, semi-structured data without operational burden. Rockset is serverless and fully managed. It offloads the work of managing configuration, cluster provisioning, denormalization, and shard / index management. Rockset is also SOC 2 Type II compliant and offers encryption at rest and in flight, securing and protecting any sensitive data. Most teams can ingest data into Rockset and start executing queries in less than 15 minutes."  # noqa: E501
    col_2 = 2
    col_3 = "e903e069-b0b5-4b80-95e2-86471b41f55f"
    id = 7320132

    """Run a simple SQL query query"""
    loader = RocksetLoader(
        client,
        rockset.models.QueryRequestSql(
            query=(
                f"SELECT '{col_1}' AS col_1, {col_2} AS col_2, '{col_3}' AS col_3,"
                f" {id} AS id"
            )
        ),
        ["col_1"],
        metadata_keys=["col_2", "col_3", "id"],
    )

    output = loader.load()

    assert len(output) == 1
    assert isinstance(output[0], Document)
    assert output[0].page_content == col_1
    assert output[0].metadata == {"col_2": col_2, "col_3": col_3, "id": id}
