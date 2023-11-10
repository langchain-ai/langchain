"""Metrics reporter that sends token usage to Amazon CloudWatch service."""

import datetime
import logging
from typing import Any, Dict, List, NamedTuple

from . import TokenUsageReport, TokenUsageReporter


logger = logging.getLogger(__name__)


class CloudWatchMetrics(NamedTuple):
    """A metrics value with units supported by Amazon CloudWatch."""

    value: int | float | None
    unit: str

    @property
    def is_empty(self) -> bool:
        """Returns True if this metrics does not contain a value."""
        return self.value is None

    @property
    def float_value(self) -> float:
        """Returns this value as a float number."""
        if self.value is not None:
            return float(self.value)
        else:
            raise ValueError("Value is None.")


class CloudWatchTokenUsageReporter(TokenUsageReporter):
    """A token usage reporter implementation that sends the metrics data to Amazon CloudWatch.

    Example usage:

        from langchain.chains import LLMChain

        reporter = CloudWatchTokenUsageReporter("openai_token_usage", {"project": "test_project"})
        handler = OpenAITokenUsageCallbackHandler(reporter)

        llm = AnyLLM(..., callbacks=[handler])
        prompt = ...
        chain = LLMChain(llm=llm, prompt=prompt)
        chain.run(...)
    """

    namespace: str
    dimensions: Dict[str, str]

    def __init__(
        self,
        namespace: str,
        dimensions: Dict[str, str] | None = None,
        boto3_session: Any | None = None,
    ) -> None:
        """A token usage reporter implementation that sends the metrics data to Amazon CloudWatch.

        Args:
            namespace (str): The Amazon CloudWatch namespace of the metrics.
            dimensions (Dict[str, str] | None, optional): Additional CloudWatch dimensions of the
                metrics. Defaults to None. The model name, if present in the callback, will be
                added to these dimensions.
            boto3_session (boto3.Session | None, optional): Optional pre-configured boto3 session.
                If left to the default None, the default boto3 session will be used.
        """
        try:
            import boto3
            import botocore
        except ImportError as err:
            raise ImportError(
                "boto3 package not found, please install with `pip install boto3`"
            ) from err

        self.dimensions = dimensions or {}
        self.namespace = namespace
        boto3_session = boto3_session or boto3.Session()
        self.cloudwatch = boto3_session.client("cloudwatch")
        self._botocore = botocore

    @staticmethod
    def _cw_dimensions(dimensions: Dict[str, str]) -> List[Dict[str, str]]:
        return [{"Name": name, "Value": value} for name, value in dimensions.items()]

    def _get_metrics_data(
        self,
        raw_metrics: Dict[str, CloudWatchMetrics],
        timestamp: datetime.datetime,
        extra_dimensions: Dict[str, str] | None = None,
    ) -> List[Dict[str, Any]]:
        dimensions = self.dimensions
        if extra_dimensions is not None:
            dimensions = {**dimensions, **extra_dimensions}
        cw_dimensions = CloudWatchTokenUsageReporter._cw_dimensions(dimensions)
        return [
            {
                "MetricName": metric_name,
                "Dimensions": cw_dimensions,
                "Timestamp": timestamp.astimezone(datetime.timezone.utc),
                "Value": metric_data.float_value,
                "Unit": metric_data.unit,
            }
            for metric_name, metric_data in raw_metrics.items()
            if not metric_data.is_empty
        ]

    def send_report(self, report: TokenUsageReport) -> None:
        """Reports token usage statistics of a single LLM run to Amazon CloudWatch.

        Args:
            report (TokenUsageReport): The report to be sent.
        """
        raw_metrics = {
            "completion_tokens": CloudWatchMetrics(report.completion_tokens, "Count"),
            "prompt_tokens": CloudWatchMetrics(report.prompt_tokens, "Count"),
            "total_tokens": CloudWatchMetrics(report.total_tokens, "Count"),
            "total_cost": CloudWatchMetrics(report.total_cost, "Count"),
            "first_token_time": CloudWatchMetrics(report.first_token_time, "Seconds"),
            "completion_time": CloudWatchMetrics(report.completion_time, "Seconds"),
        }
        extra_dimensions = {}
        if report.model_name is not None and len(report.model_name) > 0:
            extra_dimensions["model_name"] = report.model_name
        if report.caller_id is not None:
            extra_dimensions["caller_id"] = report.caller_id

        # send metrics
        metrics = self._get_metrics_data(
            raw_metrics=raw_metrics,
            timestamp=report.timestamp,
            extra_dimensions=extra_dimensions,
        )
        try:
            self.cloudwatch.put_metric_data(Namespace=self.namespace, MetricData=metrics)
        except self._botocore.exceptions.ClientError as error:
            logger.warning("Couldn't put metrics data in namespace %s", self.namespace)
            logger.warning(str(error))
        except AttributeError:
            logger.warning("CloudWatch client is not available.")
