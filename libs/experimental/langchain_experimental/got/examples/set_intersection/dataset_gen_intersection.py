# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Robert Gerstenberger

import csv
import numpy as np


def scramble(array: np.ndarray, rng: np.random.Generator) -> None:
    """
    Helper function to change the order of the elements in an array randomly.

    :param array: Array to be scrambled.
    :type: numpy.ndarray
    :param rng: Random number generator.
    :type rng: numpy.random.Generator
    """

    size = array.shape[0]

    index_array = rng.integers(0, size, size)

    for i in range(size):
        temp = array[i]
        array[i] = array[index_array[i]]
        array[index_array[i]] = temp


if __name__ == "__main__":
    """
    Input(u)  : Set size.
    Input(v)  : Range of the integer numbers in the sets: 0..v (exclusive)
    Input(w)  : Seed for the random number generator.
    Input(x)  : Number of samples to be generated.
    Input(y)  : Filename for the output CSV file.
    Output(z) : Input sets and intersected set written a file in the CSV format.
                File contains the sample ID, input set 1, input set 2,
                intersection set.
    """

    set_size = 32  # size of the generated sets
    int_value_ubound = 64  # (exclusive) upper limit of generated numbers
    seed = 42  # seed of the random number generator
    num_sample = 100  # number of samples
    filename = "set_intersection_032.csv"  # output filename

    assert 2 * set_size <= int_value_ubound

    rng = np.random.default_rng(seed)

    intersection_sizes = rng.integers(set_size // 4, 3 * set_size // 4, num_sample)

    np.set_printoptions(
        linewidth=np.inf
    )  # no wrapping in the array fields in the output file

    with open(filename, "w") as f:
        fieldnames = ["ID", "SET1", "SET2", "INTERSECTION"]
        writer = csv.DictWriter(f, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_sample):
            intersection_size = intersection_sizes[i]

            full_set = np.arange(0, int_value_ubound, dtype=np.int16)

            scramble(full_set, rng)

            intersection = full_set[:intersection_size].copy()

            sorted_intersection = np.sort(intersection)

            set1 = full_set[:set_size].copy()
            set2 = np.concatenate(
                [intersection, full_set[set_size : 2 * set_size - intersection_size]]
            )

            scramble(set1, rng)
            scramble(set2, rng)

            writer.writerow(
                {
                    "ID": i,
                    "SET1": set1.tolist(),
                    "SET2": set2.tolist(),
                    "INTERSECTION": sorted_intersection.tolist(),
                }
            )
