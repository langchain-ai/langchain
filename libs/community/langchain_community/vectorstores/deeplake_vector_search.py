import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
from langchain_core.embeddings import Embeddings

try:
    import deeplake

    _DEEPLAKE_INSTALLED = True
except ImportError:
    _DEEPLAKE_INSTALLED = False


class DeepLakeVectorStore:
    def __init__(
        self,
        path: str,
        embedding_function: Optional[Embeddings] = None,
        read_only: bool = False,
        token: Optional[str] = None,
        exec_option: Optional[str] = None,
        verbose: bool = False,
        runtime: Optional[Dict] = None,
        index_params: Optional[Dict[str, Union[int, str]]] = None,
        **kwargs: Any,
    ):
        if _DEEPLAKE_INSTALLED is False:
            raise ImportError(
                "Could not import deeplake python package. "
                "Please install it with `pip install deeplake[enterprise]`."
            )
        self.path = path
        self.embedding_function = embedding_function
        self.read_only = read_only
        self.token = token
        self.exec_options = exec_option
        self.verbose = verbose
        self.runtime = runtime
        self.index_params = index_params
        self.kwargs = kwargs
        if read_only:
            try:
                self.ds = deeplake.open_read_only(self.path, self.token)
            except Exception as e:
                try:
                    self.ds = deeplake.query(
                        f"select * from {self.path}", token=self.token
                    )
                except Exception:
                    raise e
        else:
            try:
                self.ds = deeplake.open(self.path, self.token)
            except deeplake.LogNotexistsError:
                self.__create_dataset()

    def tensors(self) -> list[str]:
        return [c.name for c in self.ds.schema.columns]

    def add(
        self,
        text: List[str],
        metadata: Optional[List[dict]],
        embedding_data: Iterable[str],
        embedding_tensor: str,
        embedding_function: Optional[Callable],
        return_ids: bool,
        **tensors: Any,
    ) -> Optional[list[str]]:
        if embedding_function is not None:
            embedding_data = embedding_function(text)
        if embedding_tensor is None:
            embedding_tensor = "embedding"
        _id = (
            tensors["id"]
            if "id" in tensors
            else [str(uuid.uuid1()) for _ in range(len(text))]
        )
        self.ds.append(
            {
                "text": text,
                "metadata": metadata,
                "id": _id,
                embedding_tensor: np.array(embedding_data),
            }
        )
        self.ds.commit()
        if return_ids:
            return _id
        else:
            return None

    def search_tql(self, query: str, exec_options: Optional[str]) -> Dict[str, Any]:
        view = self.ds.query(query)
        return self.__view_to_docs(view)

    def search(
        self,
        embedding: Union[str, List[float]],
        k: int,
        distance_metric: str,
        filter: Optional[Dict[str, Any]],
        exec_option: Optional[str],
        return_tensors: List[str],
        deep_memory: Optional[bool],
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        if query is None and embedding is None:
            raise ValueError(
                "Both `embedding` and `query` were specified."
                " Please specify either one or the other."
            )
        if query is not None:
            return self.search_tql(query, exec_option)

        if isinstance(embedding, str):
            if self.embedding_function is None:
                raise ValueError(
                    "embedding_function is required when embedding is a string"
                )
            embedding = self.embedding_function.embed_documents([embedding])[0]
        emb_str = ", ".join([str(e) for e in embedding])

        column_list = " * " if not return_tensors else ", ".join(return_tensors)

        metric = self.__metric_to_function(distance_metric)
        order_by = " ASC "
        if metric == "cosine_similarity":
            order_by = " DESC "
        dp = f"(embedding, ARRAY[{emb_str}])"
        column_list += f", {metric}{dp} as score"
        query = f"SELECT {column_list} ORDER BY {metric}{dp} {order_by} LIMIT {k}"
        view = self.ds.query(query)
        return self.__view_to_docs(view)

    def delete(self, ids: List[str], filter: Dict[str, Any], delete_all: bool) -> None:
        raise NotImplementedError

    def dataset(self) -> Any:
        return self.ds

    def __view_to_docs(self, view: Any) -> Dict[str, Any]:
        docs = {}
        tenors = [(c.name, str(c.dtype)) for c in view.schema.columns]
        for name, type in tenors:
            if type == "dict":
                docs[name] = [i.to_dict() for i in view[name][:]]
            else:
                try:
                    docs[name] = view[name][:].tolist()
                except AttributeError:
                    docs[name] = view[name][:]
        return docs

    def __metric_to_function(self, metric: str) -> str:
        if metric is None or metric == "cosine" or metric == "cosine_similarity":
            return "cosine_similarity"
        elif metric == "l2" or metric == "l2_norm":
            return "l2_norm"
        else:
            raise ValueError(
                f"Unknown metric: {metric}, should be one of "
                "['cosine', 'cosine_similarity', 'l2', 'l2_norm']"
            )

    def __create_dataset(self) -> None:
        if self.embedding_function is None:
            raise ValueError("embedding_function is required to create a new dataset")
        emb_size = len(self.embedding_function.embed_documents(["test"])[0])
        self.ds = deeplake.create(self.path, self.token)
        self.ds.add_column("text", deeplake.types.Text("inverted"))
        self.ds.add_column("metadata", deeplake.types.Dict())
        self.ds.add_column("embedding", deeplake.types.Embedding(size=emb_size))
        self.ds.add_column("id", deeplake.types.Text)
        self.ds.commit()
