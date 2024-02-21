import datetime
import torch
import os

from loguru import logger
from pathlib import Path
from safetensors.torch import save_file, load_file, _find_shared_tensors, _is_complete
from typing import List, Dict
from collections import defaultdict


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])]
        )
        if not complete_names:
            if len(shared) == 1:
                # Force contiguous
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model since you could be storing much more memory than needed. Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an issue."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mecanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove


def convert_file(pt_file: Path, sf_file: Path, discard_names: List[str]):
    """
    Convert a pytorch file to a safetensors file
    This will remove duplicate tensors from the file.

    Unfortunately, this might not respect *transformers* convention.
    Forcing us to check for potentially different keys during load when looking
    for specific tensors (making tensor sharing explicit).
    """
    loaded = torch.load(pt_file, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    to_removes = _remove_duplicate_names(loaded, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del loaded[to_remove]
    # Force tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_file)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_file, metadata=metadata)
    reloaded = load_file(sf_file)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def convert_files(pt_files: List[Path], sf_files: List[Path], discard_names: List[str]):
    assert len(pt_files) == len(sf_files)

    N = len(pt_files)
    # We do this instead of using tqdm because we want to parse the logs with the launcher

    for i, (pt_file, sf_file) in enumerate(zip(pt_files, sf_files)):
        # Skip blacklisted files
        if (
            "arguments" in pt_file.name
            or "args" in pt_file.name
            or "training" in pt_file.name
        ):
            continue

        start = datetime.datetime.now()
        convert_file(pt_file, sf_file, discard_names)
        elapsed = datetime.datetime.now() - start
        logger.info(f"Convert: [{i + 1}/{N}] -- Took: {elapsed}")
