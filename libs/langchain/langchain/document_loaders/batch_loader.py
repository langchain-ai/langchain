import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, TypeVar

from langchain.document_loaders.base import BaseLoader
from langchain.schema.document import Document

T = TypeVar("T")


def _make_iterator_list(
    iterable: Iterable[T],
    args_length: int,
    show_progress: bool = False,
) -> List[T]:
    """Make an iterator from an iterable. If show_progress is True, will
    return a tqdm iterator.

    Args:
        iterable (Iterable[T]): any iterable
        args_length (int): length of the iterable
        show_progress (bool, optional): If true, will return a tqdm iterator.
            Defaults to False.

    Raises:
        ImportError: if tqdm is not installed and show_progress is True
    """
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError as err:
            raise ImportError(
                "You must install tqdm to use show_progress=True."
                "You can install tqdm with `pip install tqdm`."
            ) from err
        return list(tqdm(iterable, total=args_length))
    return list(iterable)


def _load_loader(
    loader_args: Dict[str, Any], loader_callable: Callable[..., BaseLoader]
) -> List[Document]:
    """Function used to instantiate the loader and load the data.

    Args:
        loader_args (Dict[str, Any]): list of loader_args used
            to initialize the loader. The keys of the dict are the name of
            the arguments and the values are the list of values for each.
        loader_callable (Callable[..., BaseLoader]): callable
            used to initialize the loader (class or init function )

    Returns:
        List[Document]: list of loaded documents
    """
    loader = loader_callable(**loader_args)
    return loader.load()


class BatchLoader(BaseLoader):
    """
    Meta loader use to load multiple data from a loader either in parallel
    or sequentially.

    Example:

    Using a class that is instantiate with __init__
    ```
    file_path_list : List[str] = [...]
    loader_files = BatchLoader(
        TextLoader,
        {
            "file_path": file_path_list
        },
        method="process",
        max_workers=4,
    )
    ```
    Using a static method to initialize the loader:
    ```
    youtube_video_urls : List[str] = [...]
    loader_youtube = BatchLoader(
        YoutubeLoader.from_url,
        {
            "url": youtube_video_urls
        },
    )
    ```

    Then you can load your data:
    ```
    documents = loader_files.load()
    ```
    """

    def __init__(
        self,
        loader_initializer_callable: Callable[..., BaseLoader],
        init_loader_args: Dict[str, List[Any]],
        *,
        method: str = "sequential",
        max_workers: int = 1,
        show_progress: bool = False,
    ):
        """Initialize the loader with a loader_callable (either a class or a
        function used to initialize the loader) and a list of loader_args.

        Possible methods are:
            - "sequential": load data sequentially
            - "thread": load data using threads
            - "process": load data using processes
            - "async": load data using asyncio

        Args:
            loader_initializer_callable (Callable[..., BaseLoader]): callable
                used to initialize the loader (class or init function )
            init_loader_args (Dict[str, List[Any]]): list of loader_args used
                to initialize the loader. The keys of the dict are the name of
                the arguments and the values are the list of values for each.
            method (str, optional): concurrent method to use.
                Defaults to "sequential".
            max_workers (int, optional): number of workers if 'thread' or
                'process' methods are used. Defaults to 1.
            show_progress: If true, will show a progress bar as the files are
                loaded. This forces an iteration through all matching files to
                count them prior to loading them.

        Raises:
            ValueError: if the length of the lists in loader_args are not equal
        """
        loader_args = list(zip(*init_loader_args.values()))
        self.loaders_args = [
            dict(zip(init_loader_args.keys(), loader_arg)) for loader_arg in loader_args
        ]
        self.loader_callable = loader_initializer_callable
        self.show_progress = show_progress
        self._partial_load = partial(_load_loader, loader_callable=self.loader_callable)
        self.method = method
        self.max_workers = max_workers

        if len(set(len(arg) for arg in init_loader_args.values())) != 1:
            raise ValueError("All loader_args lists must have the same length")

    def load(self) -> List[Document]:
        """Load data from the loader using the specified method. The list of
        documents is flattened.

        Possible methods are:
            - "sequential": load data sequentially
            - "thread": load data using threads
            - "process": load data using processes
            - "async": load data using asyncio

        Args:
            method (str, optional): concurrent method to use.
                Defaults to "sequential".
            max_workers (int, optional): number of workers if 'thread' or
                'process' methods are used. Defaults to 1.

        Raises:
            ValueError: if the method is not valid

        Returns:
            List[Document]: list of loaded documents
        """

        if self.method == "thread":
            return self._load_thread(self.max_workers)
        elif self.method == "process":
            return self._load_process(self.max_workers)
        elif self.method == "sequential":
            return self._load_sequential()
        elif self.method == "async":
            return self._async_load()
        else:
            raise ValueError(f"Invalid method {self.method}")

    async def _async_load(self) -> List[Document]:
        """Load data from the loader using asyncio.

        Returns:
            List[Document]: list of loaded documents flattened.
        """

        event_loop = asyncio.get_event_loop()
        tasks = [
            event_loop.run_in_executor(
                None,
                self._partial_load,
                loader_args,
            )
            for loader_args in self.loaders_args
        ]
        if self.show_progress:
            try:
                from tqdm.asyncio import tqdm
            except ImportError as err:
                raise ImportError(
                    "You must install tqdm to use show_progress=True."
                    "You can install tqdm with `pip install tqdm`."
                ) from err

            results = await tqdm.gather(*tasks)
        else:
            results = await asyncio.gather(*tasks)
        return [doc for docs in results for doc in docs]

    def _load_thread(self, max_workers: int) -> List[Document]:
        """Load data from the loader using threads executor.

        Args:
            max_workers (int): number of workers

        Returns:
            List[Document]: list of loaded documents flattened.
        """

        with ThreadPoolExecutor(max_workers) as executor:
            docs_list = _make_iterator_list(
                iterable=executor.map(
                    self._partial_load,
                    self.loaders_args,
                ),
                args_length=len(self.loaders_args),
                show_progress=self.show_progress,
            )

        return [doc for docs in docs_list for doc in docs]

    def _load_process(self, max_workers: int) -> List[Document]:
        """Load data from the loader using process pool executor.

        Args:
            max_workers (int): number of workers

        Returns:
            List[Document]: list of loaded documents flattened.
        """

        with multiprocessing.Pool(max_workers) as pool:
            docs_list = _make_iterator_list(
                iterable=pool.imap(
                    self._partial_load,
                    self.loaders_args,
                ),
                args_length=len(self.loaders_args),
                show_progress=self.show_progress,
            )

        return [doc for docs in docs_list for doc in docs]

    def _load_sequential(self) -> List[Document]:
        """Load data from the loader sequentially.

        Returns:
            List[Document]: list of loaded documents flattened.
        """
        docs_list = _make_iterator_list(
            iterable=map(self._partial_load, self.loaders_args),
            args_length=len(self.loaders_args),
            show_progress=self.show_progress,
        )

        return [doc for docs in docs_list for doc in docs]
