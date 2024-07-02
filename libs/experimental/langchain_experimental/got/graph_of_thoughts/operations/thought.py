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
from typing import Dict, Iterator, Optional

from pydantic import BaseModel


class SerializableThought(BaseModel):
    id: int
    state: Dict
    score: float
    valid: bool
    solved: bool


class Thought:
    """
    Represents an LLM thought with its state, constructed by the parser, and various flags.
    """

    _ids: Iterator[int] = itertools.count(0)

    def __init__(self, state: Optional[Dict] = None) -> None:
        """
        Initializes a new Thought instance with a state and various default flags.

        :param state: The state of the thought. Defaults to None.
        :type state: Optional[Dict]
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Thought._ids)
        self.state: Dict = state
        self._score: float = 0.0
        self._valid: bool = False
        self._solved: bool = False
        self.scored: bool = False
        self.validated: bool = False
        self.compared_to_ground_truth: bool = False

    @staticmethod
    def from_thought(thought: Thought) -> Thought:
        """
        Creates a new thought from an existing one.

        :param thought: An instance of a Thought to clone.
        :return: A new Thought instance with properties copied from the input thought.
        """
        new_thought = Thought(thought.state)
        new_thought.score = thought.score
        new_thought.valid = thought.valid
        new_thought.solved = thought.solved
        new_thought.scored = thought.scored
        new_thought.validated = thought.validated
        new_thought.compared_to_ground_truth = thought.compared_to_ground_truth
        return new_thought

    @property
    def valid(self) -> bool:
        """
        Returns the validity of the thought.

        :return: The validity of the thought.
        :rtype: bool
        """
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        """
        Sets the validity of the thought and the validated flag.

        :param valid: The validity of the thought.
        :type valid: bool
        """
        self.validated = True
        self._valid = valid

    @property
    def score(self) -> float:
        """
        Returns the score of the thought.

        :return: The score of the thought.
        :rtype: float
        """
        return self._score

    @score.setter
    def score(self, new_score: float) -> None:
        """
        Sets the score of the thought and the scored flag.

        :param new_score: The score of the thought.
        :type new_score: float
        """
        self.scored = True
        self._score = new_score

    @property
    def solved(self) -> bool:
        """
        Returns the solved flag of the thought.

        :return: The solved flag of the thought.
        :rtype: bool
        """
        return self._solved

    @solved.setter
    def solved(self, solved: bool) -> None:
        """
        Sets the solved flag of the thought and the compared_to_ground_truth flag.

        :param solved: Whether the thought contains a solution to the problem.
        :type solved: bool
        """
        self.compared_to_ground_truth = True
        self._solved = solved

    def serialize(self) -> SerializableThought:
        return SerializableThought(
            id=self.id,
            state=self.state,
            score=self.score,
            solved=self.solved,
            valid=self.valid,
        )
