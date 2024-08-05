# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterator, List, Union

from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations.thought import SerializableThought, Thought
from graph_of_thoughts.parser import Parser
from graph_of_thoughts.prompter import Prompter

# from libs.core.langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

class OperationType(Enum):
    """
    Enum to represent different operation types that can be used as unique identifiers.
    """

    score: int = 0
    validate_and_improve: int = 1
    generate: int = 2
    improve: int = 3
    aggregate: int = 4
    keep_best_n: int = 5
    keep_valid: int = 6
    ground_truth_evaluator: int = 7
    selector: int = 8

    def get_label(self) -> str:
        match self:
            case OperationType.score:
                return "Score"
            case OperationType.validate_and_improve:
                return "Val. & Improve"
            case OperationType.generate:
                return "Generate"
            case OperationType.improve:
                return "Improve"
            case OperationType.aggregate:
                return "Aggregate"
            case OperationType.keep_best_n:
                return "Keep Best N"
            case OperationType.keep_valid:
                return "Keep Valid"
            case OperationType.ground_truth_evaluator:
                return "Ground Truth"
            case OperationType.selector:
                return "Selector"

    def get_verb(self, status: OperationStatus) -> str:
        if status == OperationStatus.PENDING:
            return self.get_infinitive_verb()
        elif status == OperationStatus.EXECUTING:
            return self.get_present_continuous_verb()
        elif status == OperationStatus.EXECUTED:
            return self.get_past_verb()

    def get_present_continuous_verb(self) -> str:
        match self:
            case OperationType.score:
                return "Scoring"
            case OperationType.validate_and_improve:
                return "Validating & Improving"
            case OperationType.generate:
                return "Generating"
            case OperationType.improve:
                return "Improving"
            case OperationType.aggregate:
                return "Aggregating"
            case OperationType.keep_best_n:
                return "Keeping Best"
            case OperationType.keep_valid:
                return "Keeping Valid"
            case OperationType.ground_truth_evaluator:
                return "Evaluating"
            case OperationType.selector:
                return "Selecting"

    def get_past_verb(self) -> str:
        match self:
            case OperationType.score:
                return "Scored"
            case OperationType.validate_and_improve:
                return "Val. & Improved"
            case OperationType.generate:
                return "Generated"
            case OperationType.improve:
                return "Improved"
            case OperationType.aggregate:
                return "Aggregated"
            case OperationType.keep_best_n:
                return "Kept Best"
            case OperationType.keep_valid:
                return "Kept Valid"
            case OperationType.ground_truth_evaluator:
                return "Evaluated"
            case OperationType.selector:
                return "Selected"

    def get_infinitive_verb(self) -> str:
        match self:
            case OperationType.score:
                return "To score"
            case OperationType.validate_and_improve:
                return "To validate and improve"
            case OperationType.generate:
                return "To generate"
            case OperationType.improve:
                return "To improve"
            case OperationType.aggregate:
                return "To aggregate"
            case OperationType.keep_best_n:
                return "To keep best"
            case OperationType.keep_valid:
                return "To keep valid"
            case OperationType.ground_truth_evaluator:
                return "To evaluate"
            case OperationType.selector:
                return "To select"

    def get_css_color(self) -> str:
        match self:
            case OperationType.score:
                return "#41D3BD"
            case OperationType.validate_and_improve:
                return "#791E94"
            case OperationType.generate:
                return "#DE6449"
            case OperationType.improve:
                return "#407899"
            case OperationType.aggregate:
                return "#FFD275 "
            case OperationType.keep_best_n:
                return "#BBBE64"
            case OperationType.keep_valid:
                return "#8E5572"
            case OperationType.ground_truth_evaluator:
                return "#BCAA99"
            case OperationType.selector:
                return "#7FB685"


class OperationStatus(Enum):
    """
    Enum to represent the status of an operation.
    """

    PENDING: int = 0
    EXECUTING: int = 1
    EXECUTED: int = 2

    def get_css_color(self) -> str:
        match self:
            case OperationStatus.PENDING:
                return "orange"
            case OperationStatus.EXECUTING:
                return "turquoise"
            case OperationStatus.EXECUTED:
                return "SpringGreen "


@dataclass
class OperationSummary:
    status: OperationStatus
    type: OperationType
    thoughts: List[SerializableThought]
    n_thoughts: int

    def fmt_op(
        self,
    ) -> str:
        return f"{self.type.get_past_verb()}"

    def fmt_thought(self) -> str:
        return f"{self.type.name}"


class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        Initializes a new Operation instance with a unique id, and empty predecessors and successors.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False
        self.status: OperationStatus = OperationStatus.PENDING

    def can_be_executed(self) -> bool:
        """
        Checks if the operation can be executed based on its predecessors.

        :return: True if all predecessors have been executed, False otherwise.
        :rtype: bool
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        """
        Iterates over all predecessors and aggregates their thoughts.

        :return: A list of all thoughts from the predecessors.
        :rtype: List[Thought]
        """
        previous_thoughts: List[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]

        return previous_thoughts

    def add_predecessor(self, operation: Operation) -> None:
        """
        Add a preceding operation and update the relationships.

        :param operation: The operation to be set as a predecessor.
        :type operation: Operation
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: Operation) -> None:
        """
        Add a succeeding operation and update the relationships.

        :param operation: The operation to be set as a successor.
        :type operation: Operation
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Execute the operation, assuring that all predecessors have been executed.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info(
            "Executing operation %d of type %s", self.id, self.operation_type
        )
        self.status = OperationStatus.EXECUTING
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug("Operation %d executed", self.id)

        if len(self.get_thoughts()) != self.get_n_thoughts():
            self.logger.warning(
                "Score operation %d has thought count mismatch (list %d, config %d)",
                self.id,
                len(self.get_thoughts()),
                self.get_n_thoughts(),
            )

        self.executed = True
        self.status = OperationStatus.EXECUTED

    @abstractmethod
    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Abstract method for the actual execution of the operation.
        This should be implemented in derived classes.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        """
        Abstract method to retrieve the thoughts associated with the operation.
        This should be implemented in derived classes.

        :return: List of associated thoughts.
        :rtype: List[Thought]
        """
        pass

    def get_summary(self) -> OperationSummary:
        """
        Abstract method to retrieve a summary of the operation.
        This should be implemented in derived classes.

        :return: A summary of the operation.
        :rtype: OperationSummary
        """
        return OperationSummary(
            status=self.status,
            type=self.operation_type,
            thoughts=[t.serialize() for t in self.get_thoughts()],
            n_thoughts=self.get_n_thoughts(),
        )

    @abstractmethod
    def get_n_thoughts(self) -> int:
        """
        Abstract method to retrieve the number of children (thoughts) of the
        operation.

        :returns: An integer with the number
        :type: int
        """


class Score(Operation):
    """
    Operation to score thoughts.
    """

    operation_type: OperationType = OperationType.score

    def __init__(
        self,
        num_samples: int = 1,
        combined_scoring: bool = False,
        scoring_function: Union[
            Callable[[Union[List[Dict], Dict]], Union[List[float], float]], None
        ] = None,
    ) -> None:
        """
        Initializes a new Score operation.

        :param num_samples: Number of samples to use for scoring. Defaults to 1.
        :type num_samples: int
        :param combined_scoring: Whether to score all thoughts together or individually. Defaults to False.
        :type combined_scoring: bool
        :param scoring_function: A function to score thoughts (if not using LM). Defaults to None.
        :type scoring_function: Takes a list of thought states or a single thought state and
                                returns a list of scores or a single score.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.thoughts: List[Thought] = []
        self.scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = scoring_function

    def get_n_thoughts(self) -> int:
        if self.combined_scoring:
            return 1
        return len(self.get_previous_thoughts()) or 1

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of scored thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the scoring operation by scoring the thoughts from the predecessors.
        If combined scoring is used, the thoughts are scored together, otherwise individually.
        If a scoring function is provided, it is used, otherwise the LM is prompted.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "Score operation needs at least one predecessor"

        if self.combined_scoring:
            previous_thoughts_states = [thought.state for thought in previous_thoughts]
            if self.scoring_function is not None:
                self.logger.debug(
                    "Using scoring function %s to score states", self.scoring_function
                )
                scores = self.scoring_function(previous_thoughts_states)
            else:
                #Langchain chain to score the thoughts
                prompt, var_dic = prompter.score_prompt(previous_thoughts_states)
                var_dic_list = [var_dic for i in range(self.num_samples)]
                self.logger.debug("Prompt for LM: %s", prompt)
                # Build the chain of operations
                chain = prompt | lm.load_llm() | StrOutputParser()
                # Execute the chain of operations as many times as the number of samples in parallel
                responses = chain.batch(var_dic_list)
                self.logger.debug("Responses from LM: %s", responses)
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
            for thought, score in zip(previous_thoughts, scores):
                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                self.thoughts.append(new_thought)
        else:
            for thought in previous_thoughts:
                new_thought = Thought.from_thought(thought)
                if self.scoring_function is not None:
                    self.logger.debug(
                        "Using scoring function %s to score state",
                        self.scoring_function,
                    )
                    score = self.scoring_function(thought.state)
                else:
                    #Langchain chain to score the thoughts
                    prompt, var_dic = prompter.score_prompt([thought.state])
                    var_dic_list = [var_dic for i in range(self.num_samples)]
                    self.logger.debug("Prompt for LM: %s", prompt)
                    # Build the chain of operations
                    chain = prompt | lm.load_llm() | StrOutputParser()
                    # Execute the chain of operations as many times as the number of samples in parallel
                    responses = chain.batch(var_dic_list)
                    self.logger.debug("Responses from LM: %s", responses)
                    score = parser.parse_score_answer([thought.state], responses)[0]

                new_thought.score = score
                self.thoughts.append(new_thought)

        self.logger.info(
            "Score operation %d scored %d thoughts",
            self.id,
            len(self.thoughts),
        )


class ValidateAndImprove(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(
        self,
        num_samples: int = 1,
        improve: bool = True,
        num_tries: int = 3,
        validate_function: Callable[[Dict], bool] = None,
    ) -> None:
        """
        Initializes a new ValidateAndImprove operation.

        :param num_samples: Number of samples to use for validation. Defaults to 1.
        :type num_samples: int
        :param improve: Whether to improve the thought if it is not valid. Defaults to True.
        :type improve: bool
        :param num_tries: Number of tries to improve the thought before giving up. Defaults to 3.
        :type num_tries: int
        :param validate_function: A function to validate thoughts (if not using LM). Defaults to None.
        :type validate_function: Takes a thought state and returns a boolean.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.improve: bool = improve
        self.num_tries: int = num_tries
        self.validate_function: Callable[[Dict], bool] = validate_function
        self.thoughts: List[List[Thought]] = []

    def get_n_thoughts(self) -> int:
        return len(self.get_previous_thoughts()) or 1

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the list of final thoughts, after validation and improvement.

        :return: List of final validated and improved thoughts.
        :rtype: List[Thought]
        """
        return [thought_list[-1] for thought_list in self.thoughts]

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the ValidateAndImprove operation by validating and improving the predecessors' thoughts.
        If a validation function is provided, it is used, otherwise the LM is prompted.
        If improvement is enabled, the LM is prompted to improve the thought, if it is not valid.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "ValidateAndImprove operation needs at least one predecessor"

        for thought in previous_thoughts:
            thought_list = []
            current_thought = Thought.from_thought(thought)
            current_try = 0
            while True:
                if self.validate_function is not None:
                    self.logger.debug(
                        "Using validate function %s to score states",
                        self.validate_function,
                    )
                    valid = self.validate_function(current_thought.state)
                else:
                    #Langchain chain to validate the thoughts
                    prompt, var_dic = prompter.validation_prompt(**current_thought.state)
                    var_dic_list = [var_dic for i in range(self.num_samples)]
                    self.logger.debug("Prompt for LM: %s", prompt)
                    # Build the chain of operations
                    chain = prompt | lm.load_llm() | StrOutputParser()
                    # Execute the chain of operations in parallel as many times as the number of samples
                    responses = chain.batch(var_dic_list)
                    self.logger.debug("Responses from LM: %s", responses)

                    valid = parser.parse_validation_answer(
                        current_thought.state, responses
                    )
                current_thought.valid = valid
                thought_list.append(current_thought)
                if (
                    not self.improve
                    or current_thought.valid
                    or current_try >= self.num_tries
                ):
                    break
                #Langchain chain to improve the thoughts
                improve_prompt, var_dic = prompter.improve_prompt(**current_thought.state)
                var_dic_list = [var_dic]    # The LLM is called only once to improve the thought
                self.logger.debug("Prompt for LM: %s", improve_prompt)
                # Build the chain of operations
                chain = improve_prompt | lm.load_llm() | StrOutputParser()
                # Execute the chain of operations once
                responses = chain.batch(var_dic_list)
                self.logger.debug("Responses from LM: %s", responses)
                state_update = parser.parse_improve_answer(
                    current_thought.state, responses
                )
                current_thought = Thought({**current_thought.state, **state_update})
                current_try += 1
            self.thoughts.append(thought_list)

        self.logger.info(
            "Validate and improve operation %d created %d valid thoughts from %d previous thoughts",
            self.id,
            len(
                [
                    thought_list[-1]
                    for thought_list in self.thoughts
                    if thought_list[-1].valid
                ]
            ),
            len(previous_thoughts),
        )


class Generate(Operation):
    """
    Operation to generate thoughts.
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
        self, num_branches_prompt: int = 1, num_branches_response: int = 1
    ) -> None:
        """
        Initializes a new Generate operation.

        :param num_branches_prompt: Number of responses that each prompt should generate (passed to prompter). Defaults to 1.
        :type num_branches_prompt: int
        :param num_branches_response: Number of responses the LM should generate for each prompt. Defaults to 1.
        :type num_branches_response: int
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of generated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def get_n_thoughts(self) -> int:
        return self.num_branches_response

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Generate operation by generating thoughts from the predecessors.
        The thoughts are generated by prompting the LM with the predecessors' thought states.
        If there are no predecessors, the kwargs are used as a base state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # no predecessors, use kwargs as base state
            previous_thoughts = [Thought(state=kwargs)]

        for thought in previous_thoughts:
            base_state = thought.state
            # Generate the prompt and the content of the variables that will be used in the prompt
            prompt, var_dic = prompter.generate_prompt(self.num_branches_prompt, **base_state)
                    # List of dictionaries with the variables that will be used in the prompt. The number of dictionaries is equal to the number of branches of the prompt.
            var_dic_list = [var_dic for i in range(self.num_branches_response)]
            self.logger.debug("Prompt for LM: %s", prompt)
            # Build the chain of operations
            chain = prompt | lm.load_llm() | StrOutputParser()
            # Execute the chain of operations as many times as the number of branches of the prompt in parallel
            responses = chain.batch(var_dic_list)
            self.logger.debug("Responses from LM: %s", responses)
            # Parse the responses to generate new thoughts
            for new_state in parser.parse_generate_answer(base_state, responses):
                new_state = {**base_state, **new_state}
                self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    "New thought %d created with state %s",
                    self.thoughts[-1].id,
                    self.thoughts[-1].state,
                )
        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                "Generate operation %d created more thoughts than expected",
                self.id,
            )
        self.logger.info(
            "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
        )


class Improve(Operation):
    """
    Operation to improve thoughts.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        """
        Initializes a new Improve operation.
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation after improvement.

        :return: List of improved thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def get_n_thoughts(self) -> int:
        return len(self.get_thoughts())

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Improve operation by improving the predecessors' thoughts.
        The thoughts are improved by prompting the LM with the predecessors' thought states.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert len(self.predecessors) > 0, "Needs at least one predecessor"

        for thought in previous_thoughts:
            #Langchain chain to improve the thoughts
            improve_prompt, var_dic  = prompter.improve_prompt(**thought.state)
            var_dic_list = [var_dic]    # The LLM is called only once to improve the thought
            self.logger.debug("Prompt for LM: %s", improve_prompt)
            # Build the chain of operations
            chain = improve_prompt | lm.load_llm() | StrOutputParser()
            # Execute the chain of operations once
            responses = chain.batch(var_dic_list)
            self.logger.debug("Responses from LM: %s", responses)
            state_update = parser.parse_improve_answer(thought.state, responses)
            self.thoughts.append(Thought({**thought.state, **state_update}))

        self.logger.info(
            "Improve operation %d improved %d thoughts", self.id, len(self.thoughts)
        )


class Aggregate(Operation):
    """
    Operation to aggregate thoughts.
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, num_responses: int = 1) -> None:
        """
        Initializes a new Aggregate operation.

        :param num_responses: Number of responses to use for aggregation. Defaults to 1.
        :type num_responses: int
        """
        super().__init__()
        self.thoughts: List[Thought] = []
        self.num_responses: int = num_responses

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation after aggregation.

        :return: List of aggregated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def get_n_thoughts(self) -> int:
        return 1

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Aggregate operation by aggregating the predecessors' thoughts.
        The thoughts are aggregated by prompting the LM with the predecessors' thought states.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "Aggregate operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            return

        # applied in order of score
        base_state: Dict = {}
        for thought in sorted(previous_thoughts, key=lambda thought: thought.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]
        #Langchain chain to aggregate the thoughts
        prompt, var_dic = prompter.aggregation_prompt(previous_thought_states)
        var_dic_list = [var_dic for i in range(self.num_responses)]
        self.logger.debug("Prompt for LM: %s", prompt)

        # Build the chain of operations
        chain = prompt | lm.load_llm() | StrOutputParser()
        # Execute the chain of operations in parallel as many times as the number of samples
        responses = chain.batch(var_dic_list)
        self.logger.debug("Responses from LM: %s", responses)

        parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

        if isinstance(parsed, dict):
            parsed = [parsed]
        for new_state in parsed:
            self.thoughts.append(Thought({**base_state, **new_state}))


class KeepBestN(Operation):
    """
    Operation to keep the best N thoughts from predecessors based on their score.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        """
        Initializes a new KeepBestN operation.

        :param n: Maximum number of thoughts to keep.
        :type n: int
        :param higher_is_better: Whether higher scores are better. Defaults to True.
        :type higher_is_better: bool
        :raises AssertionError: If `n` is not greater than zero.
        """
        super().__init__()
        self.n: int = n
        assert self.n > 0, "KeepBestN operation must keep at least one thought"
        self.higher_is_better: bool = higher_is_better
        self.thoughts: List[Thought] = []

    def get_n_thoughts(self) -> int:
        return self.n

    def get_best_n(self) -> List[Thought]:
        """
        Returns the best N thoughts from the predecessors based on their score.

        :return: List of best N thoughts.
        :rtype: List[Thought]
        :raises AssertionError: If not all predecessors have been executed.
        :raises AssertionError: If not all thoughts have been scored.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()
        assert all(
            previous_thought.scored for previous_thought in previous_thoughts
        ), "Not all thoughts have been scored"

        try:
            return sorted(
                previous_thoughts,
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]
        except:
            self.logger.error("Error in KeepBestN operation")
            self.logger.error(
                "Previous operation: %s", [op.id for op in self.predecessors]
            )
            self.logger.error("Previous thoughts: %s", previous_thoughts)
            self.logger.error(
                "Scores: %s", [thought.score for thought in previous_thoughts]
            )
            return sorted(
                [i for i in previous_thoughts if isinstance(i.score, float)],
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts kept by the operation.

        :return: List of kept thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the KeepBestN operation by keeping the best N thoughts from the predecessors according to their score.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        :raises AssertionError: If not all predecessors have been executed.
        :raises AssertionError: If not all thoughts have been scored.
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepBestN operation must have at least one predecessor"

        self.thoughts = [Thought.from_thought(thought) for thought in self.get_best_n()]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepBestN operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class KeepValid(Operation):
    """
    Operation to keep valid thoughts from predecessors.
    """

    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        """
        Initializes a new KeepValid operation.
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts kept by the operation.

        :return: List of kept thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def get_n_thoughts(self) -> int:
        return len(self.get_thoughts())

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the KeepValid operation by keeping the valid thoughts from the predecessors.
        Keeps unvalidated thoughts as well.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepValid operation must have at least one predecessor"

        self.thoughts: List[Thought] = [
            Thought.from_thought(thought)
            for thought in self.get_previous_thoughts()
            if not thought.validated or thought.valid
        ]

        if any(not thought.validated for thought in self.thoughts):
            self.logger.warning(
                "KeepValid operation %d has unvalidated thoughts", self.id
            )

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepValid operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class GroundTruth(Operation):
    """
    Operation to evaluate if thoughts correctly solve the problem, using a ground truth evaluator
    """

    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(self, ground_truth_evaluator: Callable[[Dict], bool]) -> None:
        """
        Initializes a new GroundTruth operation.

        :param ground_truth_evaluator: A function to evaluate if a thought solves the problem.
        :type ground_truth_evaluator: A function that takes a thought state and returns a boolean.
        """
        super().__init__()
        self.ground_truth_evaluator: Callable[[Dict], bool] = ground_truth_evaluator
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of evaluated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def get_n_thoughts(self) -> int:
        return len(self.get_previous_thoughts())

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the GroundTruth operation by evaluating the predecessors' thoughts using the ground truth evaluator function.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessor.
        """
        assert (
            len(self.predecessors) >= 1
        ), "GroundTruth operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            try:
                new_thought.solved = self.ground_truth_evaluator(new_thought.state)
            except:
                new_thought.solved = False
            self.thoughts.append(new_thought)

        self.logger.info(
            "GroundTruth operation %d evaluated %d thoughts and %d solved the problem",
            self.id,
            len(self.thoughts),
            len([thought for thought in self.thoughts if thought.solved]),
        )


class Selector(Operation):
    """
    Operation to select thoughts from predecessors.
    Useful for separating thoughts to perform different, subsequent operations on them.
    """

    operation_type: OperationType = OperationType.selector

    def __init__(self, selector: Callable[[List[Thought]], List[Thought]]) -> None:
        """
        Initializes a new Selector operation.

        :param selector: A function to select thoughts from the predecessors' thoughts.
        :type selector: A function that takes a list of thoughts and returns a list of thoughts.
        """
        super().__init__()
        self.selector: Callable[[List[Thought]], List[Thought]] = selector
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts selected by the operation.

        :return: List of selected thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def get_n_thoughts(self) -> int:
        return len(self.get_thoughts())

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Selector operation by selecting thoughts from the predecessors using the selector function.
        If the Selector has no predecessors, the selector function is called with a thought containing the kwargs as state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(kwargs)]

        self.thoughts = [
            Thought.from_thought(thought)
            for thought in self.selector(previous_thoughts)
        ]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s selected", thought.id, thought.state
            )

        self.logger.info(
            "Selector operation %d selected %d thoughts", self.id, len(self.thoughts)
        )
