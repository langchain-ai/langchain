"""Apache Cassandra DB graph vector store integration."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from langchain_core._api import beta
from langchain_core.documents import Document
from typing_extensions import override

from langchain_community.graph_vectorstores.base import GraphVectorStore, Node
from langchain_community.graph_vectorstores.links import METADATA_LINKS_KEY, Link
from langchain_community.graph_vectorstores.mmr_helper import MmrHelper
from langchain_community.utilities.cassandra import SetupMode
from langchain_community.vectorstores.cassandra import Cassandra as CassandraVectorStore

CGVST = TypeVar("CGVST", bound="CassandraGraphVectorStore")

if TYPE_CHECKING:
    from cassandra.cluster import Session
    from langchain_core.embeddings import Embeddings


logger = logging.getLogger(__name__)


class AdjacentNode:
    id: str
    links: list[Link]
    embedding: list[float]

    def __init__(self, node: Node, embedding: list[float]) -> None:
        """Create an Adjacent Node."""
        self.id = node.id or ""
        self.links = node.links
        self.embedding = embedding


def _serialize_links(links: list[Link]) -> str:
    class SetAndLinkEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:  # noqa: ANN401
            if not isinstance(obj, type) and is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, Iterable):
                return list(obj)

            # Let the base class default method raise the TypeError
            return super().default(obj)

    return json.dumps(links, cls=SetAndLinkEncoder)


def _deserialize_links(json_blob: str | None) -> set[Link]:
    return {
        Link(kind=link["kind"], direction=link["direction"], tag=link["tag"])
        for link in cast(list[dict[str, Any]], json.loads(json_blob or "[]"))
    }


def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"


def _metadata_link_value() -> str:
    return "link"


def _doc_to_node(doc: Document) -> Node:
    metadata = doc.metadata.copy()
    links = _deserialize_links(metadata.get(METADATA_LINKS_KEY))
    metadata[METADATA_LINKS_KEY] = links

    return Node(
        id=doc.id,
        text=doc.page_content,
        metadata=metadata,
        links=list(links),
    )


def _incoming_links(node: Node | AdjacentNode) -> set[Link]:
    return {link for link in node.links if link.direction in ["in", "bidir"]}


def _outgoing_links(node: Node | AdjacentNode) -> set[Link]:
    return {link for link in node.links if link.direction in ["out", "bidir"]}


@beta()
class CassandraGraphVectorStore(GraphVectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        session: Session | None = None,
        keyspace: str | None = None,
        table_name: str = "",
        ttl_seconds: int | None = None,
        *,
        body_index_options: list[tuple[str, Any]] | None = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        metadata_deny_list: Optional[list[str]] = None,
    ) -> None:
        """Apache Cassandra(R) for graph-vector-store workloads.

        To use it, you need a recent installation of the `cassio` library
        and a Cassandra cluster / Astra DB instance supporting vector capabilities.

        Example:
            .. code-block:: python

                    from langchain_community.graph_vectorstores import
                        CassandraGraphVectorStore
                    from langchain_openai import OpenAIEmbeddings

                    embeddings = OpenAIEmbeddings()
                    session = ...             # create your Cassandra session object
                    keyspace = 'my_keyspace'  # the keyspace should exist already
                    table_name = 'my_graph_vector_store'
                    vectorstore = CassandraGraphVectorStore(
                        embeddings,
                        session,
                        keyspace,
                        table_name,
                    )

        Args:
            embedding: Embedding function to use.
            session: Cassandra driver session. If not provided, it is resolved from
                cassio.
            keyspace: Cassandra keyspace. If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            setup_mode: mode used to create the Cassandra table (SYNC,
                ASYNC or OFF).
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.
        """
        self.embedding = embedding

        if metadata_deny_list is None:
            metadata_deny_list = []
        metadata_deny_list.append(METADATA_LINKS_KEY)

        self.vector_store = CassandraVectorStore(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            setup_mode=setup_mode,
            metadata_indexing=("deny_list", metadata_deny_list),
        )

        store_session: Session = self.vector_store.session

        self._insert_node = store_session.prepare(
            f"""
            INSERT INTO {keyspace}.{table_name} (
                row_id, body_blob, vector, attributes_blob, metadata_s
            ) VALUES (?, ?, ?, ?, ?)
            """  # noqa: S608
        )

    @property
    @override
    def embeddings(self) -> Embeddings | None:
        return self.embedding

    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        if outgoing_link is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[_metadata_link_key(link=outgoing_link)] = _metadata_link_value()
        return metadata_filter

    def _restore_links(self, doc: Document) -> Document:
        """Restores the links in the document by deserializing them from metadata.

        Args:
            doc: A single Document

        Returns:
            The same Document with restored links.
        """
        links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
        doc.metadata[METADATA_LINKS_KEY] = links
        # TODO: Could this be skipped if we put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link_key in [
            _metadata_link_key(link=link)
            for link in links
            if link.direction in ["in", "bidir"]
        ]:
            if incoming_link_key in doc.metadata:
                del doc.metadata[incoming_link_key]

        return doc

    def _get_node_metadata_for_insertion(self, node: Node) -> dict[str, Any]:
        metadata = node.metadata.copy()
        metadata[METADATA_LINKS_KEY] = _serialize_links(node.links)
        # TODO: Could we could put these metadata entries
        # only in the searchable `metadata_s` column?
        for incoming_link in _incoming_links(node=node):
            metadata[_metadata_link_key(link=incoming_link)] = _metadata_link_value()
        return metadata

    def _get_docs_for_insertion(
        self, nodes: Iterable[Node]
    ) -> tuple[list[Document], list[str]]:
        docs = []
        ids = []
        for node in nodes:
            node_id = secrets.token_hex(8) if not node.id else node.id

            doc = Document(
                page_content=node.text,
                metadata=self._get_node_metadata_for_insertion(node=node),
                id=node_id,
            )
            docs.append(doc)
            ids.append(node_id)
        return (docs, ids)

    @override
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        (docs, ids) = self._get_docs_for_insertion(nodes=nodes)
        return self.vector_store.add_documents(docs, ids=ids)

    @override
    async def aadd_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        (docs, ids) = self._get_docs_for_insertion(nodes=nodes)
        for inserted_id in await self.vector_store.aadd_documents(docs, ids=ids):
            yield inserted_id

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        return [
            self._restore_links(doc)
            for doc in self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        return [
            self._restore_links(doc)
            for doc in await self.vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the query vector.
        """
        return [
            self._restore_links(doc)
            for doc in self.vector_store.similarity_search_by_vector(
                embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    @override
    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
            **kwargs: Additional arguments are ignored.

        Returns:
            The list of Documents most similar to the query vector.
        """
        return [
            self._restore_links(doc)
            for doc in await self.vector_store.asimilarity_search_by_vector(
                embedding,
                k=k,
                filter=filter,
                **kwargs,
            )
        ]

    def metadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        """Get documents via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        return [
            self._restore_links(doc)
            for doc in self.vector_store.metadata_search(
                filter=filter or {},
                n=n,
            )
        ]

    async def ametadata_search(
        self,
        filter: dict[str, Any] | None = None,  # noqa: A002
        n: int = 5,
    ) -> Iterable[Document]:
        """Get documents via a metadata search.

        Args:
            filter: the metadata to query for.
            n: the maximum number of documents to return.
        """
        return [
            self._restore_links(doc)
            for doc in await self.vector_store.ametadata_search(
                filter=filter or {},
                n=n,
            )
        ]

    def get_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        doc = self.vector_store.get_by_document_id(document_id=document_id)
        return self._restore_links(doc) if doc is not None else None

    async def aget_by_document_id(self, document_id: str) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.
        """
        doc = await self.vector_store.aget_by_document_id(document_id=document_id)
        return self._restore_links(doc) if doc is not None else None

    def get_node(self, node_id: str) -> Node | None:
        """Retrieve a single node from the store, given its ID.

        Args:
            node_id: The node ID

        Returns:
            The the node if it exists. Otherwise None.
        """
        doc = self.vector_store.get_by_document_id(document_id=node_id)
        if doc is None:
            return None
        return _doc_to_node(doc=doc)

    @override
    async def ammr_traversal_search(  # noqa: C901
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        query_embedding = self.embedding.embed_query(query)
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )

        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()
        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        async def fetch_neighborhood(neighborhood: Sequence[str]) -> None:
            nonlocal outgoing_links_map, visited_links, retrieved_docs

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = await self._get_outgoing_links(neighborhood)

            # Call `self._get_adjacent` to fetch the candidates.
            adjacent_nodes = await self._get_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                filter=filter,
                retrieved_docs=retrieved_docs,
            )

            new_candidates: dict[str, list[float]] = {}
            for adjacent_node in adjacent_nodes:
                if adjacent_node.id not in outgoing_links_map:
                    outgoing_links_map[adjacent_node.id] = _outgoing_links(
                        node=adjacent_node
                    )
                    new_candidates[adjacent_node.id] = adjacent_node.embedding
            helper.add_candidates(new_candidates)

        async def fetch_initial_candidates() -> None:
            nonlocal outgoing_links_map, visited_links, retrieved_docs

            results = (
                await self.vector_store.asimilarity_search_with_embedding_id_by_vector(
                    embedding=query_embedding,
                    k=fetch_k,
                    filter=filter,
                )
            )

            candidates: dict[str, list[float]] = {}
            for doc, embedding, doc_id in results:
                if doc_id not in retrieved_docs:
                    retrieved_docs[doc_id] = doc

                if doc_id not in outgoing_links_map:
                    node = _doc_to_node(doc)
                    outgoing_links_map[doc_id] = _outgoing_links(node=node)
                    candidates[doc_id] = embedding
            helper.add_candidates(candidates)

        if initial_roots:
            await fetch_neighborhood(initial_roots)
        if fetch_k > 0:
            await fetch_initial_candidates()

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next nodes would not exceed the depth limit, find the
                # adjacent nodes.

                # Find the links linked to from the selected ID.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the nodes with incoming links from those links.
                adjacent_nodes = await self._get_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    filter=filter,
                    retrieved_docs=retrieved_docs,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                new_candidates = {}
                for adjacent_node in adjacent_nodes:
                    if adjacent_node.id not in outgoing_links_map:
                        outgoing_links_map[adjacent_node.id] = _outgoing_links(
                            node=adjacent_node
                        )
                        new_candidates[adjacent_node.id] = adjacent_node.embedding
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth
                helper.add_candidates(new_candidates)

        for doc_id, similarity_score, mmr_score in zip(
            helper.selected_ids,
            helper.selected_similarity_scores,
            helper.selected_mmr_scores,
        ):
            if doc_id in retrieved_docs:
                doc = self._restore_links(retrieved_docs[doc_id])
                doc.metadata["similarity_score"] = similarity_score
                doc.metadata["mmr_score"] = mmr_score
                yield doc
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    @override
    def mmr_traversal_search(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """

        async def collect_docs() -> Iterable[Document]:
            async_iter = self.ammr_traversal_search(
                query=query,
                initial_roots=initial_roots,
                k=k,
                depth=depth,
                fetch_k=fetch_k,
                adjacent_k=adjacent_k,
                lambda_mult=lambda_mult,
                score_threshold=score_threshold,
                filter=filter,
                **kwargs,
            )
            return [doc async for doc in async_iter]

        return asyncio.run(collect_docs())

    @override
    async def atraversal_search(  # noqa: C901
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """
        # Depth 0:
        #   Query for `k` nodes similar to the question.
        #   Retrieve `content_id` and `outgoing_links()`.
        #
        # Depth 1:
        #   Query for nodes that have an incoming link in the `outgoing_links()` set.
        #   Combine node IDs.
        #   Query for `outgoing_links()` of those "new" node IDs.
        #
        # ...

        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[Link, int] = {}

        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        async def visit_nodes(d: int, docs: Iterable[Document]) -> None:
            """Recursively visit nodes and their outgoing links."""
            nonlocal visited_ids, visited_links, retrieved_docs

            # Iterate over nodes, tracking the *new* outgoing links for this
            # depth. These are links that are either new, or newly discovered at a
            # lower depth.
            outgoing_links: set[Link] = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    # If this node is at a closer depth, update visited_ids
                    if d <= visited_ids.get(doc.id, depth):
                        visited_ids[doc.id] = d

                        # If we can continue traversing from this node,
                        if d < depth:
                            node = _doc_to_node(doc=doc)
                            # Record any new (or newly discovered at a lower depth)
                            # links to the set to traverse.
                            for link in _outgoing_links(node=node):
                                if d <= visited_links.get(link, depth):
                                    # Record that we'll query this link at the
                                    # given depth, so we don't fetch it again
                                    # (unless we find it an earlier depth)
                                    visited_links[link] = d
                                    outgoing_links.add(link)

            if outgoing_links:
                metadata_search_tasks = []
                for outgoing_link in outgoing_links:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_link=outgoing_link,
                    )
                    metadata_search_tasks.append(
                        asyncio.create_task(
                            self.vector_store.ametadata_search(
                                filter=metadata_filter, n=1000
                            )
                        )
                    )
                results = await asyncio.gather(*metadata_search_tasks)

                # Visit targets concurrently
                visit_target_tasks = [
                    visit_targets(d=d + 1, docs=docs) for docs in results
                ]
                await asyncio.gather(*visit_target_tasks)

        async def visit_targets(d: int, docs: Iterable[Document]) -> None:
            """Visit target nodes retrieved from outgoing links."""
            nonlocal visited_ids, retrieved_docs

            new_ids_at_next_depth = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    if d <= visited_ids.get(doc.id, depth):
                        new_ids_at_next_depth.add(doc.id)

            if new_ids_at_next_depth:
                visit_node_tasks = [
                    visit_nodes(d=d, docs=[retrieved_docs[doc_id]])
                    for doc_id in new_ids_at_next_depth
                    if doc_id in retrieved_docs
                ]

                fetch_tasks = [
                    asyncio.create_task(
                        self.vector_store.aget_by_document_id(document_id=doc_id)
                    )
                    for doc_id in new_ids_at_next_depth
                    if doc_id not in retrieved_docs
                ]

                new_docs: list[Document | None] = await asyncio.gather(*fetch_tasks)

                visit_node_tasks.extend(
                    visit_nodes(d=d, docs=[new_doc])
                    for new_doc in new_docs
                    if new_doc is not None
                )

                await asyncio.gather(*visit_node_tasks)

        # Start the traversal
        initial_docs = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )
        await visit_nodes(d=0, docs=initial_docs)

        for doc_id in visited_ids:
            if doc_id in retrieved_docs:
                yield self._restore_links(retrieved_docs[doc_id])
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    @override
    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.
        """

        async def collect_docs() -> Iterable[Document]:
            async_iter = self.atraversal_search(
                query=query,
                k=k,
                depth=depth,
                filter=filter,
                **kwargs,
            )
            return [doc async for doc in async_iter]

        return asyncio.run(collect_docs())

    async def _get_outgoing_links(self, source_ids: Iterable[str]) -> set[Link]:
        """Return the set of outgoing links for the given source IDs asynchronously.

        Args:
            source_ids: The IDs of the source nodes to retrieve outgoing links for.

        Returns:
            A set of `Link` objects representing the outgoing links from the source
            nodes.
        """
        links = set()

        # Create coroutine objects without scheduling them yet
        coroutines = [
            self.vector_store.aget_by_document_id(document_id=source_id)
            for source_id in source_ids
        ]

        # Schedule and await all coroutines
        docs = await asyncio.gather(*coroutines)

        for doc in docs:
            if doc is not None:
                node = _doc_to_node(doc=doc)
                links.update(_outgoing_links(node=node))

        return links

    async def _get_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        retrieved_docs: dict[str, Document],
        k_per_link: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> Iterable[AdjacentNode]:
        """Return the target nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            retrieved_docs: A cache of retrieved docs. This will be added to.
            k_per_link: The number of target nodes to fetch for each link.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        targets: dict[str, AdjacentNode] = {}

        tasks = []
        for link in links:
            metadata_filter = self._get_metadata_filter(
                metadata=filter,
                outgoing_link=link,
            )

            tasks.append(
                self.vector_store.asimilarity_search_with_embedding_id_by_vector(
                    embedding=query_embedding,
                    k=k_per_link or 10,
                    filter=metadata_filter,
                )
            )

        results = await asyncio.gather(*tasks)

        for result in results:
            for doc, embedding, doc_id in result:
                if doc_id not in retrieved_docs:
                    retrieved_docs[doc_id] = doc
                if doc_id not in targets:
                    node = _doc_to_node(doc=doc)
                    targets[doc_id] = AdjacentNode(node=node, embedding=embedding)

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()

    @staticmethod
    def _build_docs_from_texts(
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Document]:
        docs: List[Document] = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
            )
            if metadatas is not None:
                doc.metadata = metadatas[i]
            if ids is not None:
                doc.id = ids[i]
            docs.append(doc)
        return docs

    @classmethod
    def from_texts(
        cls: Type[CGVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the texts.
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.

        Returns:
            a CassandraGraphVectorStore.
        """
        docs = cls._build_docs_from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        return cls.from_documents(
            documents=docs,
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls: Type[CGVST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from raw texts.

        Args:
            texts: Texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the texts.
            ttl_seconds: Optional time-to-live for the added texts.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.

        Returns:
            a CassandraGraphVectorStore.
        """
        docs = cls._build_docs_from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        return await cls.afrom_documents(
            documents=docs,
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )

    @staticmethod
    def _add_ids_to_docs(
        docs: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[Document]:
        if ids is not None:
            for doc, doc_id in zip(docs, ids):
                doc.id = doc_id
        return docs

    @classmethod
    def from_documents(
        cls: Type[CGVST],
        documents: List[Document],
        embedding: Embeddings,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the vectorstore.
            embedding: Embedding function to use.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the documents.
            ttl_seconds: Optional time-to-live for the added documents.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.

        Returns:
            a CassandraGraphVectorStore.
        """
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )
        store.add_documents(documents=cls._add_ids_to_docs(docs=documents, ids=ids))
        return store

    @classmethod
    async def afrom_documents(
        cls: Type[CGVST],
        documents: List[Document],
        embedding: Embeddings,
        *,
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        table_name: str = "",
        ids: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        body_index_options: Optional[List[Tuple[str, Any]]] = None,
        metadata_deny_list: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> CGVST:
        """Create a CassandraGraphVectorStore from a document list.

        Args:
            documents: Documents to add to the vectorstore.
            embedding: Embedding function to use.
            session: Cassandra driver session.
                If not provided, it is resolved from cassio.
            keyspace: Cassandra key space.
                If not provided, it is resolved from cassio.
            table_name: Cassandra table (required).
            ids: Optional list of IDs associated with the documents.
            ttl_seconds: Optional time-to-live for the added documents.
            body_index_options: Optional options used to create the body index.
                Eg. body_index_options = [cassio.table.cql.STANDARD_ANALYZER]
            metadata_deny_list: Optional list of metadata keys to not index.
                i.e. to fine-tune which of the metadata fields are indexed.
                Note: if you plan to have massive unique text metadata entries,
                consider not indexing them for performance
                (and to overcome max-length limitations).
                Note: the `metadata_indexing` parameter from
                langchain_community.utilities.cassandra.Cassandra is not
                exposed since CassandraGraphVectorStore only supports the
                deny_list option.


        Returns:
            a CassandraGraphVectorStore.
        """
        store = cls(
            embedding=embedding,
            session=session,
            keyspace=keyspace,
            table_name=table_name,
            ttl_seconds=ttl_seconds,
            setup_mode=SetupMode.ASYNC,
            body_index_options=body_index_options,
            metadata_deny_list=metadata_deny_list,
            **kwargs,
        )
        await store.aadd_documents(
            documents=cls._add_ids_to_docs(docs=documents, ids=ids)
        )
        return store
