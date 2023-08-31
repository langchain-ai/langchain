"""Evaluators for parsing strings."""
from abc import abstractmethod
from operator import eq
from typing import Any, Callable, Dict, Optional, Union, cast

from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown


class JsonValidityEvaluator(StringEvaluator):
    """Evaluates whether the prediction is valid JSON.

    This evaluator checks if the prediction is a valid JSON string. It does not
        require any input or reference.

    Attributes:
        requires_input (bool): Whether this evaluator requires an input
            string. Always False.
        requires_reference (bool): Whether this evaluator requires a
            reference string. Always False.
        evaluation_name (str): The name of the evaluation metric.
            Always "json".

    Examples:
        >>> evaluator = JsonValidityEvaluator()
        >>> prediction = '{"name": "John", "age": 30, "city": "New York"}'
        >>> evaluator.evaluate(prediction)
        {'score': 1}

        >>> prediction = '{"name": "John", "age": 30, "city": "New York",}'
        >>> evaluator.evaluate(prediction)
        {'score': 0, 'reasoning': 'Expecting property name enclosed in double quotes'}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "json_validity"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator. Defaults to None.
            reference (str, optional): Not used in this evaluator. Defaults to None.

        Returns:
            dict: A dictionary containing the evaluation score. The score is 1 if
            the prediction is valid JSON, and 0 otherwise.
                If the prediction is not valid JSON, the dictionary also contains
                a "reasoning" field with the error message.

        """
        try:
            parse_json_markdown(prediction)
            return {"score": 1}
        except Exception as e:
            return {"score": 0, "reasoning": str(e)}


class _JsonComparisonEvaluator(StringEvaluator):
    """Evaluated by comparing the predicted structured object to the
    reference structured object.

    It does not require an input string.

    Attributes:
        requires_input (bool): Whether this evaluator requires an
            input string. Always False.
        requires_reference (bool): Whether this evaluator requires
            a reference string. Always True.
        evaluation_name (str): The name of the evaluation metric.s

    Examples:
        >>> evaluator = JsonEqualityEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

        >>> evaluator = JsonEqualityEvaluator(operator=lambda x, y: x['a'] == y['a'])
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "json_equality"

    def _parse_json(
        self, string: str
    ) -> Union[dict, list, None, float, bool, int, str]:
        return parse_json_markdown(string)

    @abstractmethod
    def _compare_objects(self, prediction: Any, reference: Any) -> dict:
        """Compare the prediction and reference objects.

        Args:
            prediction (Any): The prediction object.
            reference (Any): The reference object.

        Returns:
            dict: A dictionary containing the evaluation score.
        """

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator.
            reference (str): The reference string to compare against.

        Returns:
            dict: A dictionary containing the evaluation score.
        """
        parsed = self._parse_json(prediction)
        label = self._parse_json(cast(str, reference))
        return self._compare_objects(parsed, label)


class JsonRecallEvaluator(_JsonComparisonEvaluator):
    """Evaluates the recall of JSON field extraction.

    Recall is calculated as (True Positives) / (True Positives + False Negatives).

    Attributes:
        operator (Callable): A custom function to compare field values.

    Examples:
        >>> evaluator = JsonRecallEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1, "b": 2}')
        {'score': 0.5}
    """

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = operator or eq

    @property
    def evaluation_name(self) -> str:
        return "json_recall"

    def _compare_objects(self, prediction: Any, reference: Any) -> dict:
        true_positives = 0
        total_actual = len(reference.keys())

        for k, v in reference.items():
            if k in prediction and self.operator(v, prediction[k]):
                true_positives += 1

        return {"score": true_positives / total_actual if total_actual > 0 else 0}


class JsonEqualityEvaluator(_JsonComparisonEvaluator):
    """Evaluates whether the prediction is equal to the reference after
        parsing both as JSON.

    This evaluator checks if the prediction, after parsing as JSON, is equal
        to the reference,
    which is also parsed as JSON. It does not require an input string.

    Attributes:
        requires_input (bool): Whether this evaluator requires an
            input string. Always False.
        requires_reference (bool): Whether this evaluator requires
            a reference string. Always True.
        evaluation_name (str): The name of the evaluation metric.
            Always "parsed_equality".

    Examples:
        >>> evaluator = JsonEqualityEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

        >>> evaluator = JsonEqualityEvaluator(operator=lambda x, y: x['a'] == y['a'])
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

    """

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = operator or eq

    @property
    def evaluation_name(self) -> str:
        return "json_equality"

    def _compare_objects(self, prediction: Any, reference: Any) -> dict:
        if isinstance(reference, list):
            if not isinstance(prediction, list):
                return {"score": 0}
            prediction = sorted(prediction, key=lambda x: str(x))
            reference = sorted(reference, key=lambda x: str(x))
        return {"score": self.operator(prediction, reference)}


class JsonAccuracyEvaluator(_JsonComparisonEvaluator):
    """Evaluates the accuracy of JSON field extraction.

    Accuracy is calculated as (True Positives + True Negatives)
        / (Total Predicted + Total Actual).

    Attributes:
        operator (Callable): A custom function to compare field values.

    Examples:
        >>> evaluator = JsonAccuracyEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1, "b": 2}', reference='{"a": 1, "b": 2}')
        {'score': 1.0}
    """

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = operator or eq

    @property
    def evaluation_name(self) -> str:
        return "json_accuracy"

    def _compare_objects(self, prediction: Any, reference: Any) -> dict:
        true_positives = 0
        total = len(set(prediction).union(reference))

        for k, v in reference.items():
            if k in prediction and self.operator(v, prediction[k]):
                true_positives += 1

        return {"score": true_positives / total if total > 0 else 0}


class JsonPrecisionEvaluator(_JsonComparisonEvaluator):
    """Evaluates the precision of JSON field extraction.

    Precision is calculated as (True Positives) / (True Positives + False Positives).

    Attributes:
        operator (Callable): A custom function to compare field values.

    Examples:
        >>> evaluator = JsonPrecisionEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1, "b": 2}', reference='{"a": 1, "b": 2}')
        {'score': 1.0}
        >>> evaluator.evaluate_strings('{"a": 1, "b": 3}', reference='{"a": 1, "b": 2}')
        {'score': 0.5}
    """

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = operator or eq

    @property
    def evaluation_name(self) -> str:
        return "json_precision"

    def _compare_objects(self, prediction: Any, reference: Any) -> dict:
        true_positives = 0
        total_predicted = len(prediction.keys())

        for k, v in prediction.items():
            if k in reference and self.operator(v, reference[k]):
                true_positives += 1

        return {"score": true_positives / total_predicted if total_predicted > 0 else 0}


class JsonIoUEvaluator(_JsonComparisonEvaluator):
    """Evaluates the Intersection over Union (IoU) of JSON field extraction.

    Attributes:
        operator (Callable): A custom function to compare field values.

    Examples:
        >>> evaluator = JsonIoUEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1, "b": 2}', reference='{"a": 1, "b": 2}')
        {'score': 1.0}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1, "b": 2}')
        {'score': 0.5}

    """

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = operator or eq

    @property
    def evaluation_name(self) -> str:
        return "json_iou"

    def _compare_objects(self, prediction: Any, reference: Any) -> dict:
        intersection = 0
        union = len(set(prediction).union(reference))

        for k, v in reference.items():
            if k in prediction and self.operator(prediction[k], v):
                intersection += 1

        return {"score": intersection / union if union > 0 else 0}


class JsonF1Evaluator(_JsonComparisonEvaluator):
    """Evaluates the F1 score of JSON field extraction.

    F1 is the harmonic mean of precision and recall.

    Attributes:
        operator (Callable): A custom function to compare field values.

    Examples:
        >>> evaluator = JsonF1Evaluator()
        >>> evaluator.evaluate_strings('{"a": 1, "b": 2}', reference='{"a": 1, "b": 2}')
        {'score': 1.0}
    """

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operator = operator or eq

    @property
    def evaluation_name(self) -> str:
        return "json_f1"

    def _compare_objects(self, prediction: Any, reference: Any) -> Dict[str, Any]:
        true_positives = 0
        total_actual = len(reference.keys())
        total_predicted = len(prediction.keys())

        for k, v in reference.items():
            if k in prediction and self.operator(v, prediction[k]):
                true_positives += 1

        if true_positives == 0:
            return {"score": 0}

        precision = true_positives / total_predicted
        recall = true_positives / total_actual

        f1 = 2 * (precision * recall) / (precision + recall)
        return {"score": f1}
