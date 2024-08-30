from typing import Dict, List, Optional, Tuple

from langchain.chains.llm import LLMChain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel

from langchain_experimental.video_captioning.models import VideoModel
from langchain_experimental.video_captioning.prompts import (
    JOIN_SIMILAR_VIDEO_MODELS_PROMPT,
    REMOVE_VIDEO_MODEL_DESCRIPTION_PROMPT,
)


class CaptionProcessor:
    def __init__(
        self,
        llm: BaseLanguageModel,
        verbose: bool = True,
        similarity_threshold: int = 80,
        use_unclustered_models: bool = False,
    ) -> None:
        self.llm = llm
        self.verbose = verbose

        # Set the percentage value for how similar two video model image
        # descriptions should be in order for us to cluster them into a group
        self._SIMILARITY_THRESHOLD = similarity_threshold
        # Set to True if you want to include video models which were not clustered.
        # Will likely result in closed-caption artifacts
        self._USE_NON_CLUSTERED_VIDEO_MODELS = use_unclustered_models

    def process(
        self,
        video_models: List[VideoModel],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[VideoModel]:
        # Remove any consecutive duplicates
        video_models = self._remove_consecutive_duplicates(video_models)

        # Holds the video models after clustering has been applied
        video_models_post_clustering = []
        # In this case, index represents a divider between clusters
        index = 0
        for start, end in self._get_model_clusters(video_models):
            start_vm, end_vm = video_models[start], video_models[end]

            if self._USE_NON_CLUSTERED_VIDEO_MODELS:
                # Append all the non-clustered models in between model clusters
                # staged for OpenAI combination
                video_models_post_clustering += video_models[index:start]
            index = end + 1

            # Send to llm for description combination
            models_to_combine = video_models[start:index]
            combined_description = self._join_similar_video_models(
                models_to_combine, run_manager
            )

            # Strip any prefixes that are redundant in the context of closed-captions
            stripped_description = self._remove_video_model_description_prefix(
                combined_description, run_manager
            )

            # Create a new video model which is the combination of all the models in
            # the cluster
            combined_and_stripped_model = VideoModel(
                start_vm.start_time, end_vm.end_time, stripped_description
            )

            video_models_post_clustering.append(combined_and_stripped_model)

        if self._USE_NON_CLUSTERED_VIDEO_MODELS:
            # Append any non-clustered models present after every clustered model
            video_models_post_clustering += video_models[index:]

        return video_models_post_clustering

    def _remove_consecutive_duplicates(
        self,
        video_models: List[VideoModel],
    ) -> List[VideoModel]:
        buffer: List[VideoModel] = []

        for video_model in video_models:
            # Join this model and the previous model if they have the same image
            # description
            if (
                len(buffer) > 0
                and buffer[-1].image_description == video_model.image_description
            ):
                buffer[-1].end_time = video_model.end_time

            else:
                buffer.append(video_model)

        return buffer

    def _remove_video_model_description_prefix(
        self, description: str, run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> str:
        conversation = LLMChain(
            llm=self.llm,
            prompt=REMOVE_VIDEO_MODEL_DESCRIPTION_PROMPT,
            verbose=True,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        # Get response from OpenAI using LLMChain
        response = conversation({"description": description})

        # Take out the Result: part of the response
        return response["text"].replace("Result:", "").strip()

    def _join_similar_video_models(
        self,
        video_models: List[VideoModel],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> str:
        descriptions = ""
        count = 1
        for video_model in video_models:
            descriptions += (
                f"Description {count}: " + video_model.image_description + ", "
            )
            count += 1

        # Strip trailing ", "
        descriptions = descriptions[:-2]

        conversation = LLMChain(
            llm=self.llm,
            prompt=JOIN_SIMILAR_VIDEO_MODELS_PROMPT,
            verbose=True,
            callbacks=run_manager.get_child() if run_manager else None,
        )
        # Get response from OpenAI using LLMChain
        response = conversation({"descriptions": descriptions})

        # Take out the Result: part of the response
        return response["text"].replace("Result:", "").strip()

    def _get_model_clusters(
        self, video_models: List[VideoModel]
    ) -> List[Tuple[int, int]]:
        # Word bank which maps lowercase words (case-insensitive) with trailing s's
        # removed (singular/plural-insensitive) to video model indexes in video_models
        word_bank: Dict[str, List[int]] = {}

        # Function which formats words to be inserted into word bank, as specified
        # above
        def format_word(w: str) -> str:
            return w.lower().rstrip("s")

        # Keeps track of the current video model index
        index = 0
        for vm in video_models:
            for word in vm.image_description.split():
                formatted_word = format_word(word)
                word_bank[formatted_word] = (
                    word_bank[formatted_word] if formatted_word in word_bank else []
                ) + [index]
            index += 1

        # Keeps track of the current video model index
        index = 0
        # Maps video model index to list of other video model indexes that have a
        # similarity score above the threshold
        sims: Dict[int, List[int]] = {}
        for vm in video_models:
            # Maps other video model index to number of words it shares in common
            # with this video model
            matches: Dict[int, int] = {}
            for word in vm.image_description.split():
                formatted_word = format_word(word)
                for match in word_bank[formatted_word]:
                    if match != index:
                        matches[match] = matches[match] + 1 if match in matches else 1
            if matches:
                # Get the highest number of words another video model shares with
                # this video model
                max_words_in_common = max(matches.values())

                # Get all video model indexes that share the maximum number of words
                # with this video model
                vms_with_max_words = [
                    key
                    for key, value in matches.items()
                    if value == max_words_in_common
                ]

                # Maps other video model index to its similarity score with this
                # video model
                sim_scores: Dict[int, float] = {}

                # Compute similarity score for all video models that share the
                # highest number of word occurrences with this video model
                for vm_index in vms_with_max_words:
                    sim_scores[vm_index] = video_models[vm_index].similarity_score(vm)

                # Get the highest similarity score another video model shares with
                # this video model
                max_score = max(sim_scores.values())

                # Get a list of all video models that have the maximum similarity
                # score to this video model
                vms_with_max_score = [
                    key for key, value in sim_scores.items() if value == max_score
                ]

                # Finally, transfer all video models with a high enough similarity
                # to this video model into the sims dictionary
                if max_score >= self._SIMILARITY_THRESHOLD:
                    sims[index] = []
                    for vm_index in vms_with_max_score:
                        sims[index].append(vm_index)

                index += 1

        # Maps video model index to boolean, indicates if we have already checked
        # this video model's similarity array so that we don't have infinite recursion
        already_accessed: Dict[int, bool] = {}

        # Recursively search video_model[vm_index]'s similarity matches to find the
        # earliest and latest video model in the cluster (start and end)
        def _find_start_and_end(vm_index: int) -> Tuple[int, int]:
            close_matches = sims[vm_index]
            first_vm, last_vm = min(close_matches), max(close_matches)
            first_vm, last_vm = min(vm_index, first_vm), max(vm_index, last_vm)

            if not already_accessed.get(vm_index, None):
                already_accessed[vm_index] = True
                for close_match in close_matches:
                    if close_match in sims:
                        if vm_index in sims[close_match]:
                            s, e = _find_start_and_end(close_match)
                            first_vm = min(s, first_vm)
                            last_vm = max(e, last_vm)

            return first_vm, last_vm

        # Add the video model cluster results into a set
        clusters = set()
        for vm_index in sims:
            clusters.add(_find_start_and_end(vm_index))

        # Filter the set to include only non-subset intervals
        filtered_clusters = set()
        for interval in clusters:
            start, end = interval[0], interval[1]
            is_subset = any(
                start >= other_start and end <= other_end
                for other_start, other_end in clusters
                if interval != (other_start, other_end)
            )
            if not is_subset:
                filtered_clusters.add(interval)

        # Sort these clusters into a list, sorted using the first element of the
        # tuple (index of video model in the cluster with earliest start time)
        sorted_clusters = sorted(filtered_clusters, key=lambda x: x[0])

        # Merge any overlapping clusters into one big cluster
        def _merge_overlapping_clusters(
            array: List[Tuple[int, int]],
        ) -> List[Tuple[int, int]]:
            if len(array) <= 1:
                return array

            def _merge(
                curr: Tuple[int, int], rest: List[Tuple[int, int]]
            ) -> List[Tuple[int, int]]:
                if curr[1] >= rest[0][0]:
                    return [(curr[0], rest[0][1])] + rest[1:]
                return [curr] + rest

            return _merge(array[0], _merge_overlapping_clusters(array[1:]))

        merged_clusters = _merge_overlapping_clusters(sorted_clusters)

        return merged_clusters
