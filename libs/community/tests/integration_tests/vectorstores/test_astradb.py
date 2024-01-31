"""
Test of Astra DB vector store class `AstraDB`

Required to run this test:
    - a recent `astrapy` Python package available
    - an Astra DB instance;
    - the two environment variables set:
        export ASTRA_DB_API_ENDPOINT="https://<DB-ID>-us-east1.apps.astra.datastax.com"
        export ASTRA_DB_APPLICATION_TOKEN="AstraCS:........."
    - optionally this as well (otherwise defaults are used):
        export ASTRA_DB_KEYSPACE="my_keyspace"
"""

import json
import math
import os
from typing import Iterable, List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import AstraDB

# Ad-hoc embedding classes:


class SomeEmbeddings(Embeddings):
    """
    Turn a sentence into an embedding vector in some way.
    Not important how. It is deterministic is all that counts.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        unnormed0 = [ord(c) for c in text[: self.dimension]]
        unnormed = (unnormed0 + [1] + [0] * (self.dimension - 1 - len(unnormed0)))[
            : self.dimension
        ]
        norm = sum(x * x for x in unnormed) ** 0.5
        normed = [x / norm for x in unnormed]
        return normed

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


class ParserEmbeddings(Embeddings):
    """
    Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(txt) for txt in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            vals = json.loads(text)
            assert len(vals) == self.dimension
            return vals
        except Exception:
            print(f'[ParserEmbeddings] Returning a moot vector for "{text}"')
            return [0.0] * self.dimension

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


def _has_env_vars() -> bool:
    return all(
        [
            "ASTRA_DB_APPLICATION_TOKEN" in os.environ,
            "ASTRA_DB_API_ENDPOINT" in os.environ,
        ]
    )


@pytest.fixture(scope="function")
def store_someemb() -> Iterable[AstraDB]:
    emb = SomeEmbeddings(dimension=2)
    v_store = AstraDB(
        embedding=emb,
        collection_name="lc_test_s",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield v_store
    v_store.delete_collection()


@pytest.fixture(scope="function")
def store_parseremb() -> Iterable[AstraDB]:
    emb = ParserEmbeddings(dimension=2)
    v_store = AstraDB(
        embedding=emb,
        collection_name="lc_test_p",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
        namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
    )
    yield v_store
    v_store.delete_collection()


@pytest.mark.requires("astrapy")
@pytest.mark.skipif(not _has_env_vars(), reason="Missing Astra DB env. vars")
class TestAstraDB:
    def test_astradb_vectorstore_create_delete(self) -> None:
        """Create and delete."""
        emb = SomeEmbeddings(dimension=2)
        # creation by passing the connection secrets
        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_1",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store.delete_collection()
        # Creation by passing a ready-made astrapy client:
        from astrapy.db import AstraDB as LibAstraDB

        astra_db_client = LibAstraDB(
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store_2 = AstraDB(
            embedding=emb,
            collection_name="lc_test_2",
            astra_db_client=astra_db_client,
        )
        v_store_2.delete_collection()

    async def test_astradb_vectorstore_create_delete_async(self) -> None:
        """Create and delete."""
        emb = SomeEmbeddings(dimension=2)
        # creation by passing the connection secrets
        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_1_async",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        await v_store.adelete_collection()
        # Creation by passing a ready-made astrapy client:
        from astrapy.db import AsyncAstraDB

        astra_db_client = AsyncAstraDB(
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store_2 = AstraDB(
            embedding=emb,
            collection_name="lc_test_2_async",
            async_astra_db_client=astra_db_client,
        )
        await v_store_2.adelete_collection()

    def test_astradb_vectorstore_pre_delete_collection(self) -> None:
        """Create and delete."""
        emb = SomeEmbeddings(dimension=2)
        # creation by passing the connection secrets

        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_pre_del",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        try:
            v_store.add_texts(
                texts=["aa"],
                metadatas=[
                    {"k": "a", "ord": 0},
                ],
                ids=["a"],
            )
            res1 = v_store.similarity_search("aa", k=5)
            assert len(res1) == 1
            v_store = AstraDB(
                embedding=emb,
                pre_delete_collection=True,
                collection_name="lc_test_pre_del",
                token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
                api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
                namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
            )
            res1 = v_store.similarity_search("aa", k=5)
            assert len(res1) == 0
        finally:
            v_store.delete_collection()

    async def test_astradb_vectorstore_pre_delete_collection_async(self) -> None:
        """Create and delete."""
        emb = SomeEmbeddings(dimension=2)
        # creation by passing the connection secrets

        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_pre_del_async",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        try:
            await v_store.aadd_texts(
                texts=["aa"],
                metadatas=[
                    {"k": "a", "ord": 0},
                ],
                ids=["a"],
            )
            res1 = await v_store.asimilarity_search("aa", k=5)
            assert len(res1) == 1
            v_store = AstraDB(
                embedding=emb,
                pre_delete_collection=True,
                collection_name="lc_test_pre_del_async",
                token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
                api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
                namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
            )
            res1 = await v_store.asimilarity_search("aa", k=5)
            assert len(res1) == 0
        finally:
            await v_store.adelete_collection()

    def test_astradb_vectorstore_from_x(self) -> None:
        """from_texts and from_documents methods."""
        emb = SomeEmbeddings(dimension=2)
        # from_texts
        v_store = AstraDB.from_texts(
            texts=["Hi", "Ho"],
            embedding=emb,
            collection_name="lc_test_ft",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        try:
            assert v_store.similarity_search("Ho", k=1)[0].page_content == "Ho"
        finally:
            v_store.delete_collection()

        # from_documents
        v_store_2 = AstraDB.from_documents(
            [
                Document(page_content="Hee"),
                Document(page_content="Hoi"),
            ],
            embedding=emb,
            collection_name="lc_test_fd",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        try:
            assert v_store_2.similarity_search("Hoi", k=1)[0].page_content == "Hoi"
        finally:
            v_store_2.delete_collection()

    async def test_astradb_vectorstore_from_x_async(self) -> None:
        """from_texts and from_documents methods."""
        emb = SomeEmbeddings(dimension=2)
        # from_texts
        v_store = await AstraDB.afrom_texts(
            texts=["Hi", "Ho"],
            embedding=emb,
            collection_name="lc_test_ft_async",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        try:
            assert (await v_store.asimilarity_search("Ho", k=1))[0].page_content == "Ho"
        finally:
            await v_store.adelete_collection()

        # from_documents
        v_store_2 = await AstraDB.afrom_documents(
            [
                Document(page_content="Hee"),
                Document(page_content="Hoi"),
            ],
            embedding=emb,
            collection_name="lc_test_fd_async",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        try:
            assert (await v_store_2.asimilarity_search("Hoi", k=1))[
                0
            ].page_content == "Hoi"
        finally:
            await v_store_2.adelete_collection()

    def test_astradb_vectorstore_crud(self, store_someemb: AstraDB) -> None:
        """Basic add/delete/update behaviour."""
        res0 = store_someemb.similarity_search("Abc", k=2)
        assert res0 == []
        # write and check again
        store_someemb.add_texts(
            texts=["aa", "bb", "cc"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = store_someemb.similarity_search("Abc", k=5)
        assert {doc.page_content for doc in res1} == {"aa", "bb", "cc"}
        # partial overwrite and count total entries
        store_someemb.add_texts(
            texts=["cc", "dd"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        res2 = store_someemb.similarity_search("Abc", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = store_someemb.similarity_search_with_score_id(
            query="cc", k=1, filter={"k": "c_new"}
        )
        print(str(res3))
        doc3, score3, id3 = res3[0]
        assert doc3.page_content == "cc"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert score3 > 0.999  # leaving some leeway for approximations...
        assert id3 == "c"
        # delete and count again
        del1_res = store_someemb.delete(["b"])
        assert del1_res is True
        del2_res = store_someemb.delete(["a", "c", "Z!"])
        assert del2_res is True  # a non-existing ID was supplied
        assert len(store_someemb.similarity_search("xy", k=10)) == 1
        # clear store
        store_someemb.clear()
        assert store_someemb.similarity_search("Abc", k=2) == []
        # add_documents with "ids" arg passthrough
        store_someemb.add_documents(
            [
                Document(page_content="vv", metadata={"k": "v", "ord": 204}),
                Document(page_content="ww", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(store_someemb.similarity_search("xy", k=10)) == 2
        res4 = store_someemb.similarity_search("ww", k=1, filter={"k": "w"})
        assert res4[0].metadata["ord"] == 205

    async def test_astradb_vectorstore_crud_async(self, store_someemb: AstraDB) -> None:
        """Basic add/delete/update behaviour."""
        res0 = await store_someemb.asimilarity_search("Abc", k=2)
        assert res0 == []
        # write and check again
        await store_someemb.aadd_texts(
            texts=["aa", "bb", "cc"],
            metadatas=[
                {"k": "a", "ord": 0},
                {"k": "b", "ord": 1},
                {"k": "c", "ord": 2},
            ],
            ids=["a", "b", "c"],
        )
        res1 = await store_someemb.asimilarity_search("Abc", k=5)
        assert {doc.page_content for doc in res1} == {"aa", "bb", "cc"}
        # partial overwrite and count total entries
        await store_someemb.aadd_texts(
            texts=["cc", "dd"],
            metadatas=[
                {"k": "c_new", "ord": 102},
                {"k": "d_new", "ord": 103},
            ],
            ids=["c", "d"],
        )
        res2 = await store_someemb.asimilarity_search("Abc", k=10)
        assert len(res2) == 4
        # pick one that was just updated and check its metadata
        res3 = await store_someemb.asimilarity_search_with_score_id(
            query="cc", k=1, filter={"k": "c_new"}
        )
        print(str(res3))
        doc3, score3, id3 = res3[0]
        assert doc3.page_content == "cc"
        assert doc3.metadata == {"k": "c_new", "ord": 102}
        assert score3 > 0.999  # leaving some leeway for approximations...
        assert id3 == "c"
        # delete and count again
        del1_res = await store_someemb.adelete(["b"])
        assert del1_res is True
        del2_res = await store_someemb.adelete(["a", "c", "Z!"])
        assert del2_res is False  # a non-existing ID was supplied
        assert len(await store_someemb.asimilarity_search("xy", k=10)) == 1
        # clear store
        await store_someemb.aclear()
        assert await store_someemb.asimilarity_search("Abc", k=2) == []
        # add_documents with "ids" arg passthrough
        await store_someemb.aadd_documents(
            [
                Document(page_content="vv", metadata={"k": "v", "ord": 204}),
                Document(page_content="ww", metadata={"k": "w", "ord": 205}),
            ],
            ids=["v", "w"],
        )
        assert len(await store_someemb.asimilarity_search("xy", k=10)) == 2
        res4 = await store_someemb.asimilarity_search("ww", k=1, filter={"k": "w"})
        assert res4[0].metadata["ord"] == 205

    @staticmethod
    def _v_from_i(i: int, N: int) -> str:
        angle = 2 * math.pi * i / N
        vector = [math.cos(angle), math.sin(angle)]
        return json.dumps(vector)

    def test_astradb_vectorstore_mmr(self, store_parseremb: AstraDB) -> None:
        """
        MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        """
        i_vals = [0, 4, 5, 13]
        N_val = 20
        store_parseremb.add_texts(
            [self._v_from_i(i, N_val) for i in i_vals],
            metadatas=[{"i": i} for i in i_vals],
        )
        res1 = store_parseremb.max_marginal_relevance_search(
            self._v_from_i(3, N_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    async def test_astradb_vectorstore_mmr_async(
        self, store_parseremb: AstraDB
    ) -> None:
        """
        MMR testing. We work on the unit circle with angle multiples
        of 2*pi/20 and prepare a store with known vectors for a controlled
        MMR outcome.
        """
        i_vals = [0, 4, 5, 13]
        N_val = 20
        await store_parseremb.aadd_texts(
            [self._v_from_i(i, N_val) for i in i_vals],
            metadatas=[{"i": i} for i in i_vals],
        )
        res1 = await store_parseremb.amax_marginal_relevance_search(
            self._v_from_i(3, N_val),
            k=2,
            fetch_k=3,
        )
        res_i_vals = {doc.metadata["i"] for doc in res1}
        assert res_i_vals == {0, 4}

    def test_astradb_vectorstore_metadata(self, store_someemb: AstraDB) -> None:
        """Metadata filtering."""
        store_someemb.add_documents(
            [
                Document(
                    page_content="q",
                    metadata={"ord": ord("q"), "group": "consonant"},
                ),
                Document(
                    page_content="w",
                    metadata={"ord": ord("w"), "group": "consonant"},
                ),
                Document(
                    page_content="r",
                    metadata={"ord": ord("r"), "group": "consonant"},
                ),
                Document(
                    page_content="e",
                    metadata={"ord": ord("e"), "group": "vowel"},
                ),
                Document(
                    page_content="i",
                    metadata={"ord": ord("i"), "group": "vowel"},
                ),
                Document(
                    page_content="o",
                    metadata={"ord": ord("o"), "group": "vowel"},
                ),
            ]
        )
        # no filters
        res0 = store_someemb.similarity_search("x", k=10)
        assert {doc.page_content for doc in res0} == set("qwreio")
        # single filter
        res1 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"group": "vowel"},
        )
        assert {doc.page_content for doc in res1} == set("eio")
        # multiple filters
        res2 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"group": "consonant", "ord": ord("q")},
        )
        assert {doc.page_content for doc in res2} == set("q")
        # excessive filters
        res3 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"group": "consonant", "ord": ord("q"), "case": "upper"},
        )
        assert res3 == []
        # filter with logical operator
        res4 = store_someemb.similarity_search(
            "x",
            k=10,
            filter={"$or": [{"ord": ord("q")}, {"ord": ord("r")}]},
        )
        assert {doc.page_content for doc in res4} == {"q", "r"}

    def test_astradb_vectorstore_similarity_scale(
        self, store_parseremb: AstraDB
    ) -> None:
        """Scale of the similarity scores."""
        store_parseremb.add_texts(
            texts=[
                json.dumps([1, 1]),
                json.dumps([-1, -1]),
            ],
            ids=["near", "far"],
        )
        res1 = store_parseremb.similarity_search_with_score(
            json.dumps([0.5, 0.5]),
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < 0.001 and abs(sco_far) < 0.001

    async def test_astradb_vectorstore_similarity_scale_async(
        self, store_parseremb: AstraDB
    ) -> None:
        """Scale of the similarity scores."""
        await store_parseremb.aadd_texts(
            texts=[
                json.dumps([1, 1]),
                json.dumps([-1, -1]),
            ],
            ids=["near", "far"],
        )
        res1 = await store_parseremb.asimilarity_search_with_score(
            json.dumps([0.5, 0.5]),
            k=2,
        )
        scores = [sco for _, sco in res1]
        sco_near, sco_far = scores
        assert abs(1 - sco_near) < 0.001 and abs(sco_far) < 0.001

    def test_astradb_vectorstore_massive_delete(self, store_someemb: AstraDB) -> None:
        """Larger-scale bulk deletes."""
        M = 50
        texts = [str(i + 1 / 7.0) for i in range(2 * M)]
        ids0 = ["doc_%i" % i for i in range(M)]
        ids1 = ["doc_%i" % (i + M) for i in range(M)]
        ids = ids0 + ids1
        store_someemb.add_texts(texts=texts, ids=ids)
        # deleting a bunch of these
        del_res0 = store_someemb.delete(ids0)
        assert del_res0 is True
        # deleting the rest plus a fake one
        del_res1 = store_someemb.delete(ids1 + ["ghost!"])
        assert del_res1 is True  # ensure no error
        # nothing left
        assert store_someemb.similarity_search("x", k=2 * M) == []

    def test_astradb_vectorstore_drop(self) -> None:
        """behaviour of 'delete_collection'."""
        collection_name = "lc_test_d"
        emb = SomeEmbeddings(dimension=2)
        v_store = AstraDB(
            embedding=emb,
            collection_name=collection_name,
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store.add_texts(["huh"])
        assert len(v_store.similarity_search("hah", k=10)) == 1
        # another instance pointing to the same collection on DB
        v_store_kenny = AstraDB(
            embedding=emb,
            collection_name=collection_name,
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
        )
        v_store_kenny.delete_collection()
        # dropped on DB, but 'v_store' should have no clue:
        with pytest.raises(ValueError):
            _ = v_store.similarity_search("hah", k=10)

    def test_astradb_vectorstore_custom_params(self) -> None:
        """Custom batch size and concurrency params."""
        emb = SomeEmbeddings(dimension=2)
        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_c",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
            batch_size=17,
            bulk_insert_batch_concurrency=13,
            bulk_insert_overwrite_concurrency=7,
            bulk_delete_concurrency=19,
        )
        try:
            # add_texts
            N = 50
            texts = [str(i + 1 / 7.0) for i in range(N)]
            ids = ["doc_%i" % i for i in range(N)]
            v_store.add_texts(texts=texts, ids=ids)
            v_store.add_texts(
                texts=texts,
                ids=ids,
                batch_size=19,
                batch_concurrency=7,
                overwrite_concurrency=13,
            )
            #
            _ = v_store.delete(ids[: N // 2])
            _ = v_store.delete(ids[N // 2 :], concurrency=23)
            #
        finally:
            v_store.delete_collection()

    async def test_astradb_vectorstore_custom_params_async(self) -> None:
        """Custom batch size and concurrency params."""
        emb = SomeEmbeddings(dimension=2)
        v_store = AstraDB(
            embedding=emb,
            collection_name="lc_test_c_async",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
            batch_size=17,
            bulk_insert_batch_concurrency=13,
            bulk_insert_overwrite_concurrency=7,
            bulk_delete_concurrency=19,
        )
        try:
            # add_texts
            N = 50
            texts = [str(i + 1 / 7.0) for i in range(N)]
            ids = ["doc_%i" % i for i in range(N)]
            await v_store.aadd_texts(texts=texts, ids=ids)
            await v_store.aadd_texts(
                texts=texts,
                ids=ids,
                batch_size=19,
                batch_concurrency=7,
                overwrite_concurrency=13,
            )
            #
            await v_store.adelete(ids[: N // 2])
            await v_store.adelete(ids[N // 2 :], concurrency=23)
            #
        finally:
            await v_store.adelete_collection()

    def test_astradb_vectorstore_metrics(self) -> None:
        """
        Different choices of similarity metric.
        Both stores (with "cosine" and "euclidea" metrics) contain these two:
            - a vector slightly rotated w.r.t query vector
            - a vector which is a long multiple of query vector
        so, which one is "the closest one" depends on the metric.
        """
        emb = ParserEmbeddings(dimension=2)
        isq2 = 0.5**0.5
        isa = 0.7
        isb = (1.0 - isa * isa) ** 0.5
        texts = [
            json.dumps([isa, isb]),
            json.dumps([10 * isq2, 10 * isq2]),
        ]
        ids = [
            "rotated",
            "scaled",
        ]
        query_text = json.dumps([isq2, isq2])
        # creation, population, query - cosine
        vstore_cos = AstraDB(
            embedding=emb,
            collection_name="lc_test_m_c",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
            metric="cosine",
        )
        try:
            vstore_cos.add_texts(
                texts=texts,
                ids=ids,
            )
            _, _, id_from_cos = vstore_cos.similarity_search_with_score_id(
                query_text,
                k=1,
            )[0]
            assert id_from_cos == "scaled"
        finally:
            vstore_cos.delete_collection()
        # creation, population, query - euclidean

        vstore_euc = AstraDB(
            embedding=emb,
            collection_name="lc_test_m_e",
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
            namespace=os.environ.get("ASTRA_DB_KEYSPACE"),
            metric="euclidean",
        )
        try:
            vstore_euc.add_texts(
                texts=texts,
                ids=ids,
            )
            _, _, id_from_euc = vstore_euc.similarity_search_with_score_id(
                query_text,
                k=1,
            )[0]
            assert id_from_euc == "rotated"
        finally:
            vstore_euc.delete_collection()
