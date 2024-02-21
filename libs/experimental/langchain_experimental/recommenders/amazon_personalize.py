from typing import Any, List, Mapping, Optional, Sequence


class AmazonPersonalize:
    """Amazon Personalize Runtime wrapper for executing real-time operations:
    https://docs.aws.amazon.com/personalize/latest/dg/API_Operations_Amazon_Personalize_Runtime.html

    Args:
        campaign_arn: str, Optional: The Amazon Resource Name (ARN) of the campaign
                                    to use for getting recommendations.
        recommender_arn: str, Optional: The Amazon Resource Name (ARN) of the
                                    recommender to use to get recommendations
        client: Optional:  boto3 client
        credentials_profile_name: str, Optional :AWS profile name
        region_name: str, Optional:  AWS region, e.g., us-west-2

    Example:
        .. code-block:: python

        personalize_client = AmazonPersonalize (
            campaignArn='<my-campaign-arn>' )
    """

    def __init__(
        self,
        campaign_arn: Optional[str] = None,
        recommender_arn: Optional[str] = None,
        client: Optional[Any] = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        self.campaign_arn = campaign_arn
        self.recommender_arn = recommender_arn

        if campaign_arn and recommender_arn:
            raise ValueError(
                "Cannot initialize AmazonPersonalize with both "
                "campaign_arn and recommender_arn."
            )

        if not campaign_arn and not recommender_arn:
            raise ValueError(
                "Cannot initialize AmazonPersonalize. Provide one of "
                "campaign_arn or recommender_arn"
            )

        try:
            if client is not None:
                self.client = client
            else:
                import boto3
                import botocore.config

                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    # use default credentials
                    session = boto3.Session()

                client_params = {}
                if region_name:
                    client_params["region_name"] = region_name

                service = "personalize-runtime"
                session_config = botocore.config.Config(user_agent_extra="langchain")
                client_params["config"] = session_config
                self.client = session.client(service, **client_params)

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )

    def get_recommendations(
        self,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        filter_arn: Optional[str] = None,
        filter_values: Optional[Mapping[str, str]] = None,
        num_results: Optional[int] = 10,
        context: Optional[Mapping[str, str]] = None,
        promotions: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata_columns: Optional[Mapping[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Get recommendations from Amazon Personalize:
        https://docs.aws.amazon.com/personalize/latest/dg/API_RS_GetRecommendations.html

        Args:
            user_id: str, Optional: The user identifier
                                    for which to retrieve recommendations
            item_id: str, Optional: The item identifier
                                    for which to retrieve recommendations
            filter_arn: str, Optional:  The ARN of the filter
                                    to apply to the returned recommendations
            filter_values: Mapping, Optional: The values
                                    to use when filtering recommendations.
            num_results: int, Optional: Default=10: The number of results to return
            context: Mapping, Optional: The contextual metadata
                                    to use when getting recommendations
            promotions: Sequence, Optional: The promotions
                                    to apply to the recommendation request.
            metadata_columns: Mapping, Optional: The metadata Columns to be returned
                                    as part of the response.

        Returns:
            response: Mapping[str, Any]: Returns an itemList and recommendationId.

        Example:
            .. code-block:: python

        personalize_client = AmazonPersonalize(campaignArn='<my-campaign-arn>' )\n
        response = personalize_client.get_recommendations(user_id="1")

        """
        if not user_id and not item_id:
            raise ValueError("One of user_id or item_id is required")

        if filter_arn:
            kwargs["filterArn"] = filter_arn
        if filter_values:
            kwargs["filterValues"] = filter_values
        if user_id:
            kwargs["userId"] = user_id
        if num_results:
            kwargs["numResults"] = num_results
        if context:
            kwargs["context"] = context
        if promotions:
            kwargs["promotions"] = promotions
        if item_id:
            kwargs["itemId"] = item_id
        if metadata_columns:
            kwargs["metadataColumns"] = metadata_columns
        if self.campaign_arn:
            kwargs["campaignArn"] = self.campaign_arn
        if self.recommender_arn:
            kwargs["recommenderArn"] = self.recommender_arn

        return self.client.get_recommendations(**kwargs)

    def get_personalized_ranking(
        self,
        user_id: str,
        input_list: List[str],
        filter_arn: Optional[str] = None,
        filter_values: Optional[Mapping[str, str]] = None,
        context: Optional[Mapping[str, str]] = None,
        metadata_columns: Optional[Mapping[str, Sequence[str]]] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Re-ranks a list of recommended items for the given user.
        https://docs.aws.amazon.com/personalize/latest/dg/API_RS_GetPersonalizedRanking.html

        Args:
            user_id: str, Required: The user identifier
                                    for which to retrieve recommendations
            input_list: List[str], Required: A list of items (by itemId) to rank
            filter_arn: str, Optional:  The ARN of the filter to apply
            filter_values: Mapping, Optional: The values to use
                                                when filtering recommendations.
            context: Mapping, Optional: The contextual metadata
                                            to use when getting recommendations
            metadata_columns: Mapping, Optional: The metadata Columns to be returned
                                    as part of the response.

        Returns:
            response: Mapping[str, Any]: Returns personalizedRanking
                                                and recommendationId.

        Example:
            .. code-block:: python

        personalize_client = AmazonPersonalize(campaignArn='<my-campaign-arn>' )\n
        response = personalize_client.get_personalized_ranking(user_id="1",
                                                        input_list=["123,"256"])

        """

        if filter_arn:
            kwargs["filterArn"] = filter_arn
        if filter_values:
            kwargs["filterValues"] = filter_values
        if user_id:
            kwargs["userId"] = user_id
        if input_list:
            kwargs["inputList"] = input_list
        if context:
            kwargs["context"] = context
        if metadata_columns:
            kwargs["metadataColumns"] = metadata_columns
        kwargs["campaignArn"] = self.campaign_arn

        return self.client.get_personalized_ranking(kwargs)
