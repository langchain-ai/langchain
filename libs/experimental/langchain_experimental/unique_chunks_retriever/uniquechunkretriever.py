from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.documents import Document


class UniqueChunkRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def optimize(
        self,
        queries_list: List[str],
        k: int = 4,
        **kwargs: Any,
    ) -> List[List[Document]]:
        """Return unique k chunks for each query (in order).

        Args:
            queries_list: list of input text queries
            k: Number of Documents to return for each query. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Document Lists, each of length k
        """
        try:
            from mip import BINARY, Model, OptimizationStatus, maximize, xsum
        except ImportError as e:
            raise ImportError(
                "Unable to import mip, please install with `pip install -U mip`"
            ) from e
        ss_result = {}
        retrieved_k = k * 5
        for query in queries_list:
            ss_result[query] = (
                self.vector_store.similarity_search_with_relevance_scores(
                    query, k=retrieved_k, **kwargs
                )
            )

        col = len(queries_list)
        C = []
        index_to_id = {}
        id_to_index = {}
        id_to_doc = {}
        index = 0
        for j, query in enumerate(ss_result):
            for tupl in ss_result[query]:
                if tupl[0].metadata["UID"] in id_to_index:
                    C[id_to_index[tupl[0].metadata["UID"]]][j] = tupl[1]
                else:
                    index_to_id[index] = tupl[0].metadata["UID"]
                    id_to_index[tupl[0].metadata["UID"]] = index
                    id_to_doc[tupl[0].metadata["UID"]] = tupl[0]
                    C.append([-1] * col)
                    C[index][j] = tupl[1]
                    index += 1

        # GAP optimization
        mip_model = Model()
        x = {}
        row = len(C)
        assert k * col <= row, "Not enough unique chunks retrieved"

        for i in range(row):
            for j in range(col):
                x[i, j] = mip_model.add_var(
                    var_type=BINARY, name="x({},{})".format(i, j)
                )

        mip_model.objective = maximize(xsum(C[i][j] * x[i, j] for i, j in x))
        for j in range(col):
            mip_model.add_constr(xsum(x[i, j] for i in range(row)) == k)

        for i in range(row):
            mip_model.add_constr(xsum(x[i, j] for j in range(col)) <= 1)

        status = mip_model.optimize()
        assert status == OptimizationStatus(0), "No solution was found"

        sol = [[None for _ in range(col)] for _ in range(row)]
        for i, j in x:
            sol[i][j] = int(x[i, j].x)

        output = []
        for j in range(col):
            current_list = []
            for i in range(row):
                if sol[i][j] == 1:
                    current_list.append(id_to_doc[index_to_id[i]])
            output.append(current_list)

        return output
