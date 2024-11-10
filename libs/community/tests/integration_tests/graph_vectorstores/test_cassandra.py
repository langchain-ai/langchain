"""Test of Apache Cassandra graph vector g_store class `CassandraGraphVectorStore`"""

import json
import os
import random
from contextlib import contextmanager
from typing import Any, Generator, Iterable, List, Optional

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.base import Node
from langchain_community.graph_vectorstores.links import (
    METADATA_LINKS_KEY,
    Link,
    add_links,
)
from tests.integration_tests.cache.fake_embeddings import (
    AngularTwoDimensionalEmbeddings,
    FakeEmbeddings,
)

TEST_KEYSPACE = "graph_test_keyspace"


class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals


@pytest.fixture
def embedding_d2() -> Embeddings:
    return ParserEmbeddings(dimension=2)


class EarthEmbeddings(Embeddings):
    def get_vector_near(self, value: float) -> List[float]:
        base_point = [value, (1 - value**2) ** 0.5]
        fluctuation = random.random() / 100.0
        return [base_point[0] + fluctuation, base_point[1] - fluctuation]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        words = set(text.lower().split())
        if "earth" in words:
            vector = self.get_vector_near(0.9)
        elif {"planet", "world", "globe", "sphere"}.intersection(words):
            vector = self.get_vector_near(0.8)
        else:
            vector = self.get_vector_near(0.1)
        return vector


def _result_ids(docs: Iterable[Document]) -> List[Optional[str]]:
    return [doc.id for doc in docs]


@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    """
    This is a set of Documents to pre-populate a graph vector store,
    with entries placed in a certain way.

    Space of the entries (under Euclidean similarity):

                      A0    (*)
        ....        AL   AR       <....
        :              |              :
        :              |  ^           :
        v              |  .           v
                       |   :
       TR              |   :          BL
    T0   --------------x--------------   B0
       TL              |   :          BR
                       |   :
                       |  .
                       | .
                       |
                    FL   FR
                      F0

    the query point is meant to be at (*).
    the A are bidirectionally with B
    the A are outgoing to T
    the A are incoming from F
    The links are like: L with L, 0 with 0 and R with R.
    """

    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BL", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    return docs_a + docs_b + docs_f + docs_t


class CassandraSession:
    table_name: str
    session: Any

    def __init__(self, table_name: str, session: Any):
        self.table_name = table_name
        self.session = session


@contextmanager
def get_cassandra_session(
    table_name: str, drop: bool = True
) -> Generator[CassandraSession, None, None]:
    """Initialize the Cassandra cluster and session"""
    from cassandra.cluster import Cluster

    if "CASSANDRA_CONTACT_POINTS" in os.environ:
        contact_points = [
            cp.strip()
            for cp in os.environ["CASSANDRA_CONTACT_POINTS"].split(",")
            if cp.strip()
        ]
    else:
        contact_points = None

    cluster = Cluster(contact_points)
    session = cluster.connect()

    try:
        session.execute(
            (
                f"CREATE KEYSPACE IF NOT EXISTS {TEST_KEYSPACE}"
                " WITH replication = "
                "{'class': 'SimpleStrategy', 'replication_factor': 1}"
            )
        )
        if drop:
            session.execute(f"DROP TABLE IF EXISTS {TEST_KEYSPACE}.{table_name}")

        # Yield the session for usage
        yield CassandraSession(table_name=table_name, session=session)
    finally:
        # Ensure proper shutdown/cleanup of resources
        session.shutdown()
        cluster.shutdown()


@pytest.fixture(scope="function")
def graph_vector_store_angular(
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphVectorStore, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphVectorStore(
            embedding=AngularTwoDimensionalEmbeddings(),
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_vector_store_earth(
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphVectorStore, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphVectorStore(
            embedding=EarthEmbeddings(),
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_vector_store_fake(
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphVectorStore, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphVectorStore(
            embedding=FakeEmbeddings(),
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def graph_vector_store_d2(
    embedding_d2: Embeddings,
    table_name: str = "graph_test_table",
) -> Generator[CassandraGraphVectorStore, None, None]:
    with get_cassandra_session(table_name=table_name) as session:
        yield CassandraGraphVectorStore(
            embedding=embedding_d2,
            session=session.session,
            keyspace=TEST_KEYSPACE,
            table_name=session.table_name,
        )


@pytest.fixture(scope="function")
def populated_graph_vector_store_d2(
    graph_vector_store_d2: CassandraGraphVectorStore,
    graph_vector_store_docs: list[Document],
) -> Generator[CassandraGraphVectorStore, None, None]:
    graph_vector_store_d2.add_documents(graph_vector_store_docs)
    yield graph_vector_store_d2


def test_mmr_traversal(graph_vector_store_angular: CassandraGraphVectorStore) -> None:
    """ Test end to end construction and MMR search.
    The embedding function used here ensures `texts` become
    the following vectors on a circle (numbered v0 through v3):

           ______ v2
          /      \
         /        |  v1
    v3  |     .    | query
         |        /  v0
          |______/                 (N.B. very crude drawing)

    With fetch_k==2 and k==2, when query is at (1, ),
    one expects that v2 and v0 are returned (in some order)
    because v1 is "too close" to v0 (and v0 is closer than v1)).

    Both v2 and v3 are reachable via edges from v0, so once it is
    selected, those are both considered.
    """
    v0 = Node(
        id="v0",
        text="-0.124",
        links=[
            Link.outgoing(kind="explicit", tag="link"),
        ],
    )
    v1 = Node(
        id="v1",
        text="+0.127",
    )
    v2 = Node(
        id="v2",
        text="+0.25",
        links=[
            Link.incoming(kind="explicit", tag="link"),
        ],
    )
    v3 = Node(
        id="v3",
        text="+1.0",
        links=[
            Link.incoming(kind="explicit", tag="link"),
        ],
    )

    g_store = graph_vector_store_angular
    g_store.add_nodes([v0, v1, v2, v3])

    results = g_store.mmr_traversal_search("0.0", k=2, fetch_k=2)
    assert _result_ids(results) == ["v0", "v2"]

    # With max depth 0, no edges are traversed, so this doesn't reach v2 or v3.
    # So it ends up picking "v1" even though it's similar to "v0".
    results = g_store.mmr_traversal_search("0.0", k=2, fetch_k=2, depth=0)
    assert _result_ids(results) == ["v0", "v1"]

    # With max depth 0 but higher `fetch_k`, we encounter v2
    results = g_store.mmr_traversal_search("0.0", k=2, fetch_k=3, depth=0)
    assert _result_ids(results) == ["v0", "v2"]

    # v0 score is .46, v2 score is 0.16 so it won't be chosen.
    results = g_store.mmr_traversal_search("0.0", k=2, score_threshold=0.2)
    assert _result_ids(results) == ["v0"]

    # with k=4 we should get all of the documents.
    results = g_store.mmr_traversal_search("0.0", k=4)
    assert _result_ids(results) == ["v0", "v2", "v1", "v3"]


def test_write_retrieve_keywords(
    graph_vector_store_earth: CassandraGraphVectorStore,
) -> None:
    greetings = Node(
        id="greetings",
        text="Typical Greetings",
        links=[
            Link.incoming(kind="parent", tag="parent"),
        ],
    )

    node1 = Node(
        id="doc1",
        text="Hello World",
        links=[
            Link.outgoing(kind="parent", tag="parent"),
            Link.bidir(kind="kw", tag="greeting"),
            Link.bidir(kind="kw", tag="world"),
        ],
    )

    node2 = Node(
        id="doc2",
        text="Hello Earth",
        links=[
            Link.outgoing(kind="parent", tag="parent"),
            Link.bidir(kind="kw", tag="greeting"),
            Link.bidir(kind="kw", tag="earth"),
        ],
    )

    g_store = graph_vector_store_earth
    g_store.add_nodes(nodes=[greetings, node1, node2])

    # Doc2 is more similar, but World and Earth are similar enough that doc1 also
    # shows up.
    results: Iterable[Document] = g_store.similarity_search("Earth", k=2)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = g_store.similarity_search("Earth", k=1)
    assert _result_ids(results) == ["doc2"]

    results = g_store.traversal_search("Earth", k=2, depth=0)
    assert _result_ids(results) == ["doc2", "doc1"]

    results = g_store.traversal_search("Earth", k=2, depth=1)
    assert _result_ids(results) == ["doc2", "doc1", "greetings"]

    # K=1 only pulls in doc2 (Hello Earth)
    results = g_store.traversal_search("Earth", k=1, depth=0)
    assert _result_ids(results) == ["doc2"]

    # K=1 only pulls in doc2 (Hello Earth). Depth=1 traverses to parent and via
    # keyword edge.
    results = g_store.traversal_search("Earth", k=1, depth=1)
    assert set(_result_ids(results)) == {"doc2", "doc1", "greetings"}


def test_metadata(graph_vector_store_fake: CassandraGraphVectorStore) -> None:
    doc_a = Node(
        id="a",
        text="A",
        metadata={"other": "some other field"},
        links=[
            Link.incoming(kind="hyperlink", tag="http://a"),
            Link.bidir(kind="other", tag="foo"),
        ],
    )

    g_store = graph_vector_store_fake
    g_store.add_nodes([doc_a])
    results = g_store.similarity_search("A")
    assert len(results) == 1
    assert results[0].id == "a"
    metadata = results[0].metadata
    assert metadata["other"] == "some other field"
    assert set(metadata[METADATA_LINKS_KEY]) == {
        Link.incoming(kind="hyperlink", tag="http://a"),
        Link.bidir(kind="other", tag="foo"),
    }


class TestCassandraGraphVectorStore:
    def test_gvs_similarity_search_sync(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector g_store."""
        g_store = populated_graph_vector_store_d2
        ss_response = g_store.similarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        ss_by_v_response = g_store.similarity_search_by_vector(embedding=[2, 10], k=2)
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]

    async def test_gvs_similarity_search_async(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ss_response = await g_store.asimilarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        ss_by_v_response = await g_store.asimilarity_search_by_vector(
            embedding=[2, 10], k=2
        )
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]

    def test_gvs_traversal_search_sync(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ts_response = g_store.traversal_search(query="[2, 10]", k=2, depth=2)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in ts_response}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = g_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"]
            for doc in retriever.get_relevant_documents(query="[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    async def test_gvs_traversal_search_async(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        ts_labels = set()
        async for doc in g_store.atraversal_search(query="[2, 10]", k=2, depth=2):
            ts_labels.add(doc.metadata["label"])
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = g_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"]
            for doc in await retriever.aget_relevant_documents(query="[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    def test_gvs_mmr_traversal_search_sync(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_response = g_store.mmr_traversal_search(
            query="[2, 10]",
            k=2,
            depth=2,
            fetch_k=1,
            adjacent_k=2,
            lambda_mult=0.1,
        )
        # TODO: can this rightfully be a list (or must it be a set)?
        mt_labels = {doc.metadata["label"] for doc in mt_response}
        assert mt_labels == {"AR", "BR"}

    async def test_gvs_mmr_traversal_search_async(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """MMR Graph traversal search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_labels = set()
        async for doc in g_store.ammr_traversal_search(
            query="[2, 10]",
            k=2,
            depth=2,
            fetch_k=1,
            adjacent_k=2,
            lambda_mult=0.1,
        ):
            mt_labels.add(doc.metadata["label"])
        # TODO: can this rightfully be a list (or must it be a set)?
        assert mt_labels == {"AR", "BR"}

    def test_gvs_metadata_search_sync(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Metadata search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_response = g_store.metadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"

    async def test_gvs_metadata_search_async(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Metadata search on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        mt_response = await g_store.ametadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links: set[Link] = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"

    def test_gvs_get_by_document_id_sync(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        doc = g_store.get_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = g_store.get_by_document_id(document_id="invalid")
        assert invalid_doc is None

    async def test_gvs_get_by_document_id_async(
        self,
        populated_graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        """Get by document_id on a graph vector store."""
        g_store = populated_graph_vector_store_d2
        doc = await g_store.aget_by_document_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = await g_store.aget_by_document_id(document_id="invalid")
        assert invalid_doc is None

    def test_gvs_from_texts(
        self,
        graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        g_store = graph_vector_store_d2
        g_store.add_texts(
            texts=["[1, 2]"],
            metadatas=[{"md": 1}],
            ids=["x_id"],
        )

        hits = g_store.similarity_search("[2, 1]", k=2)
        assert len(hits) == 1
        assert hits[0].page_content == "[1, 2]"
        assert hits[0].id == "x_id"
        # there may be more re:graph structure.
        assert hits[0].metadata["md"] == "1.0"

    def test_gvs_from_documents_containing_ids(
        self,
        graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        the_document = Document(
            page_content="[1, 2]",
            metadata={"md": 1},
            id="x_id",
        )
        g_store = graph_vector_store_d2
        g_store.add_documents([the_document])
        hits = g_store.similarity_search("[2, 1]", k=2)
        assert len(hits) == 1
        assert hits[0].page_content == "[1, 2]"
        assert hits[0].id == "x_id"
        # there may be more re:graph structure.
        assert hits[0].metadata["md"] == "1.0"

    def test_gvs_add_nodes_sync(
        self,
        *,
        graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        links0 = [
            Link(kind="kA", direction="out", tag="tA"),
            Link(kind="kB", direction="bidir", tag="tB"),
        ]
        links1 = [
            Link(kind="kC", direction="in", tag="tC"),
        ]
        nodes = [
            Node(id="id0", text="[1, 0]", metadata={"m": 0}, links=links0),
            Node(text="[-1, 0]", metadata={"m": 1}, links=links1),
        ]
        graph_vector_store_d2.add_nodes(nodes)
        hits = graph_vector_store_d2.similarity_search_by_vector([0.9, 0.1])
        assert len(hits) == 2
        assert hits[0].id == "id0"
        assert hits[0].page_content == "[1, 0]"
        md0 = hits[0].metadata
        assert md0["m"] == "0.0"
        assert any(isinstance(v, set) for k, v in md0.items() if k != "m")

        assert hits[1].id != "id0"
        assert hits[1].page_content == "[-1, 0]"
        md1 = hits[1].metadata
        assert md1["m"] == "1.0"
        assert any(isinstance(v, set) for k, v in md1.items() if k != "m")

    async def test_gvs_add_nodes_async(
        self,
        *,
        graph_vector_store_d2: CassandraGraphVectorStore,
    ) -> None:
        links0 = [
            Link(kind="kA", direction="out", tag="tA"),
            Link(kind="kB", direction="bidir", tag="tB"),
        ]
        links1 = [
            Link(kind="kC", direction="in", tag="tC"),
        ]
        nodes = [
            Node(id="id0", text="[1, 0]", metadata={"m": 0}, links=links0),
            Node(text="[-1, 0]", metadata={"m": 1}, links=links1),
        ]
        async for _ in graph_vector_store_d2.aadd_nodes(nodes):
            pass

        hits = await graph_vector_store_d2.asimilarity_search_by_vector([0.9, 0.1])
        assert len(hits) == 2
        assert hits[0].id == "id0"
        assert hits[0].page_content == "[1, 0]"
        md0 = hits[0].metadata
        assert md0["m"] == "0.0"
        assert any(isinstance(v, set) for k, v in md0.items() if k != "m")
        assert hits[1].id != "id0"
        assert hits[1].page_content == "[-1, 0]"
        md1 = hits[1].metadata
        assert md1["m"] == "1.0"
        assert any(isinstance(v, set) for k, v in md1.items() if k != "m")
