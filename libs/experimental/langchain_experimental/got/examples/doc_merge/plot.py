# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import json
import os
import matplotlib.pyplot as plt


def get_complete_results(base_directory):
    results_complete = {}
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            results_complete[folder_name] = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".json"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        results_complete[folder_name].append(
                            {"key": int(file_name.split(".")[0]), "data": data}
                        )
        for key in results_complete.keys():
            results_complete[key] = sorted(
                results_complete[key], key=lambda x: x["key"]
            )
    return results_complete


def get_final_scores(results_complete):
    scores = {}
    for method in results_complete.keys():
        scores[method] = []
        for result in results_complete[method]:
            score = 0
            solved = False
            cost = 1
            prompt_tokens = 0
            completion_tokens = 0
            for op in reversed(result["data"]):
                if "cost" in op:
                    cost = op["cost"]
                    prompt_tokens = op["prompt_tokens"]
                    completion_tokens = op["completion_tokens"]
                if "operation" in op and op["operation"] == "score":
                    try:
                        score = max(op["scores"])
                        break
                    except:
                        continue
            scores[method].append(
                [result["key"], score, solved, prompt_tokens, completion_tokens, cost]
            )
        scores[method] = sorted(scores[method], key=lambda x: x[0])
    return scores


def get_plotting_data(base_directory):
    results_complete = get_complete_results(base_directory)
    scores = get_final_scores(results_complete)
    results_plotting = {
        method: {
            "scores": [x[1] for x in scores[method]],
            "solved": sum([1 for x in scores[method] if x[2]]),
            "costs": [x[5] for x in scores[method]],
        }
        for method in scores.keys()
    }
    return results_plotting


def plot_results(
    results,
    methods_order=["io", "cot", "tot", "got", "got2"],
    model="GPT-3.5",
    num_ndas=4,
    y_lower=0,
    y_upper=10,
    cost_upper=1.8,
    display_solved=True,
    annotation_offset=1,
    display_left_ylabel=False,
    display_right_ylabel=False,
):
    methods_order = [method for method in methods_order if method in results]
    scores_ordered = [
        [score for score in results[method]["scores"]] for method in methods_order
    ]
    total_costs = [sum(results[method]["costs"]) for method in methods_order]

    # Create figure and axis
    fig, ax = plt.subplots(dpi=150, figsize=(3.75, 5))

    # Create boxplots
    positions = range(1, len(methods_order) + 1)
    ax.boxplot(scores_ordered, positions=positions)

    fig_fontsize = 12

    # Set the ticks and labels
    methods_labels = ["IO", "CoT", "ToT", "GoT", "GoT2"]
    ax.set_xticks(range(1, len(methods_order) + 1))
    ax.set_xticks(range(1, len(methods_order) + 1))
    ax.set_xticklabels(methods_labels)
    # ax.set_xlabel("Approach")

    ax.set_ylim(y_lower, 12 if display_solved else 9.75)
    plt.yticks(fontsize=fig_fontsize)

    if display_left_ylabel:
        ax.set_ylabel(
            f"Score (out of 10); the higher the better", fontsize=fig_fontsize
        )

    # ax.set_title(f"Document Merging")

    ax2 = ax.twinx()
    ax2.bar(
        positions,
        total_costs,
        alpha=0.5,
        color="blue",
        label="Total Cost ($); the lower the better",
    )
    ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)
    ax2.set_ylim(0, cost_upper)
    number_of_ticks = len(ax.get_yticks())
    tick_interval = cost_upper / (number_of_ticks)
    ax2_ticks = [tick_interval * i for i in range(number_of_ticks)]

    # Set custom tick positions for ax2
    ax2.set_yticks(ax2_ticks)

    if display_right_ylabel:
        ax2.set_ylabel(
            "Total Cost ($); the lower the better",
            color="#1919ff",
            fontsize=fig_fontsize,
        )

    if display_solved:
        annotation_height = y_upper + annotation_offset
        count = 1
        for method in methods_order:
            if method not in results:
                continue
            solved = results[method]["solved"]
            ax.text(
                count, annotation_height, f"Solved: {solved}", ha="center", va="bottom"
            )
            count += 1

    model = model.replace(".", "").replace("-", "").lower()
    fig.savefig(f"doc_merge_{model}_{num_ndas}.pdf", bbox_inches="tight")


plot_results(
    get_plotting_data("results/"),
    num_ndas=4,
    display_solved=False,
    model="GPT-3.5",
    y_upper=10,
    display_left_ylabel=True,
    cost_upper=15,
)
