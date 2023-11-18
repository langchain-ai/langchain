from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.document import Document
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    from langchain.document_loaders import ApifyDatasetLoader


class ApifyWrapper(BaseModel):
    """Wrapper around Apify.
    To use, you should have the ``apify-client`` python package installed,
    and the environment variable ``APIFY_API_TOKEN`` set with your API key, or pass
    `apify_api_token` as a named parameter to the constructor.
    """

    apify_client: Any
    apify_client_async: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate environment.
        Validate that an Apify API token is set and the apify-client
        Python package exists in the current environment.
        """
        apify_api_token = get_from_dict_or_env(
            values, "apify_api_token", "APIFY_API_TOKEN"
        )

        try:
            from apify_client import ApifyClient, ApifyClientAsync

            values["apify_client"] = ApifyClient(apify_api_token)
            values["apify_client_async"] = ApifyClientAsync(apify_api_token)
        except ImportError:
            raise ImportError(
                "Could not import apify-client Python package. "
                "Please install it with `pip install apify-client`."
            )

        return values

    def call_actor(
        self,
        actor_id: str,
        run_input: Dict,
        dataset_mapping_function: Callable[[Dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
        """Run an Actor on the Apify platform and wait for results to be ready.
        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.
        """
        from langchain.document_loaders import ApifyDatasetLoader

        actor_call = self.apify_client.actor(actor_id).call(
            run_input=run_input,
            build=build,
            memory_mbytes=memory_mbytes,
            timeout_secs=timeout_secs,
        )

        return ApifyDatasetLoader(
            dataset_id=actor_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )

    async def acall_actor(
        self,
        actor_id: str,
        run_input: Dict,
        dataset_mapping_function: Callable[[Dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
        """Run an Actor on the Apify platform and wait for results to be ready.
        Args:
            actor_id (str): The ID or name of the Actor on the Apify platform.
            run_input (Dict): The input object of the Actor that you're trying to run.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to
                an instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                Actor run's default dataset.
        """
        from langchain.document_loaders import ApifyDatasetLoader

        actor_call = await self.apify_client_async.actor(actor_id).call(
            run_input=run_input,
            build=build,
            memory_mbytes=memory_mbytes,
            timeout_secs=timeout_secs,
        )

        return ApifyDatasetLoader(
            dataset_id=actor_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )

    def call_actor_task(
        self,
        task_id: str,
        task_input: Dict,
        dataset_mapping_function: Callable[[Dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
        """Run a saved Actor task on Apify and wait for results to be ready.
        Args:
            task_id (str): The ID or name of the task on the Apify platform.
            task_input (Dict): The input object of the task that you're trying to run.
                Overrides the task's saved input.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.
        """
        from langchain.document_loaders import ApifyDatasetLoader

        task_call = self.apify_client.task(task_id).call(
            task_input=task_input,
            build=build,
            memory_mbytes=memory_mbytes,
            timeout_secs=timeout_secs,
        )

        return ApifyDatasetLoader(
            dataset_id=task_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )

    async def acall_actor_task(
        self,
        task_id: str,
        task_input: Dict,
        dataset_mapping_function: Callable[[Dict], Document],
        *,
        build: Optional[str] = None,
        memory_mbytes: Optional[int] = None,
        timeout_secs: Optional[int] = None,
    ) -> "ApifyDatasetLoader":
        """Run a saved Actor task on Apify and wait for results to be ready.
        Args:
            task_id (str): The ID or name of the task on the Apify platform.
            task_input (Dict): The input object of the task that you're trying to run.
                Overrides the task's saved input.
            dataset_mapping_function (Callable): A function that takes a single
                dictionary (an Apify dataset item) and converts it to an
                instance of the Document class.
            build (str, optional): Optionally specifies the actor build to run.
                It can be either a build tag or build number.
            memory_mbytes (int, optional): Optional memory limit for the run,
                in megabytes.
            timeout_secs (int, optional): Optional timeout for the run, in seconds.
        Returns:
            ApifyDatasetLoader: A loader that will fetch the records from the
                task run's default dataset.
        """
        from langchain.document_loaders import ApifyDatasetLoader

        task_call = await self.apify_client_async.task(task_id).call(
            task_input=task_input,
            build=build,
            memory_mbytes=memory_mbytes,
            timeout_secs=timeout_secs,
        )

        return ApifyDatasetLoader(
            dataset_id=task_call["defaultDatasetId"],
            dataset_mapping_function=dataset_mapping_function,
        )
