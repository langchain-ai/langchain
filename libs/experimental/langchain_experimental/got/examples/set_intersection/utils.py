# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# The source code is adapted from the sorting source code written by
# Nils Blach.
#
# main author: Robert Gerstenberger

from typing import Dict, List, Set


def string_to_list(string: str) -> List[int]:
    """
    Helper function to convert a list encoded inside a string into a Python
    list object of integer elements.

    :param string: Input string containing a list.
    :type string: str
    :return: List of integer elements.
    :rtype: List[int]
    :raise AssertionError: If input string does not contain a list.
    """

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return [int(num) for num in string[1:-1].split(",")]


def string_to_set(string: str) -> Set[int]:
    """
    Helper function to convert a list encoded inside a string into a Python
    set object of integer elements.

    :param string: Input string containing a list.
    :type string: str
    :return: Set of integer elements.
    :rtype: Set[int]
    :raise AssertionError: If input string does not contain a list.
    """

    assert string[0] == "[" and string[-1] == "]", "String is not a list."
    return {int(num) for num in string[1:-1].split(",")}


def test_set_intersection(state: Dict) -> bool:
    """
    Function to test whether the final solution matches ground truth.

    :param state: Thought state that represents the final solution.
    :type state: Dict
    :return: Returns whether the solution matches the ground truth.
    :rtype: bool
    """

    # convert string to list
    try:
        correct_list = string_to_list(state["result"])
        sorted_list = sorted(string_to_list(state["current"]))
        return sorted_list == correct_list
    except:
        return False


def num_errors(state: Dict) -> float:
    """
    Function to locally count the number of errors that serves as a score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """

    try:
        set1 = string_to_set(state["set1"])
        set2 = string_to_set(state["set2"])
        if "subset" in state and state["subset"] != "" and state["subset"] is not None:
            set2 = string_to_set(state["subset"])
        common = sorted(list(set1 & set2))
        llm_solution = sorted(string_to_list(state["current"]))
        num_errors = 0
        common_idx = 0
        llm_idx = 0
        while common_idx < len(common) and llm_idx < len(llm_solution):
            if common[common_idx] == llm_solution[llm_idx]:
                common_idx += 1
                llm_idx += 1
            elif common[common_idx] < llm_solution[llm_idx]:
                common_idx += 1
                num_errors += 1
            elif common[common_idx] > llm_solution[llm_idx]:
                llm_idx += 1
                num_errors += 1
        num_errors += len(common) - common_idx + len(llm_solution) - llm_idx
        return num_errors
    except:
        return 1000
