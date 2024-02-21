import time
import os

from datetime import timedelta
from loguru import logger
from pathlib import Path
from typing import Optional, List

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub.utils import (
    LocalEntryNotFoundError,
    EntryNotFoundError,
    RevisionNotFoundError,  # Import here to ease try/except in other part of the lib
)

WEIGHTS_CACHE_OVERRIDE = os.getenv("WEIGHTS_CACHE_OVERRIDE", None)


def weight_hub_files(
    model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"
) -> List[str]:
    """Get the weights filenames on the hub"""
    api = HfApi()
    info = api.model_info(model_id, revision=revision)
    filenames = [
        s.rfilename
        for s in info.siblings
        if s.rfilename.endswith(extension)
        and len(s.rfilename.split("/")) == 1
        and "arguments" not in s.rfilename
        and "args" not in s.rfilename
        and "training" not in s.rfilename
    ]

    if not filenames:
        raise EntryNotFoundError(
            f"No {extension} weights found for model {model_id} and revision {revision}.",
            None,
        )

    return filenames


def try_to_load_from_cache(
    model_id: str, revision: Optional[str], filename: str
) -> Optional[Path]:
    """Try to load a file from the Hugging Face cache"""
    if revision is None:
        revision = "main"

    object_id = model_id.replace("/", "--")
    repo_cache = Path(HUGGINGFACE_HUB_CACHE) / f"models--{object_id}"

    if not repo_cache.is_dir():
        # No cache for this model
        return None

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"

    # Resolve refs (for instance to convert main to the associated commit sha)
    if refs_dir.is_dir():
        revision_file = refs_dir / revision
        if revision_file.exists():
            with revision_file.open() as f:
                revision = f.read()

    # Check if revision folder exists
    if not snapshots_dir.exists():
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = snapshots_dir / revision / filename
    return cached_file if cached_file.is_file() else None


def weight_files(
    model_id: str, revision: Optional[str] = None, extension: str = ".safetensors"
) -> List[Path]:
    """Get the local files"""
    # Local model
    if Path(model_id).exists() and Path(model_id).is_dir():
        local_files = list(Path(model_id).glob(f"*{extension}"))
        if not local_files:
            raise FileNotFoundError(
                f"No local weights found in {model_id} with extension {extension}"
            )
        return local_files

    try:
        filenames = weight_hub_files(model_id, revision, extension)
    except EntryNotFoundError as e:
        if extension != ".safetensors":
            raise e
        # Try to see if there are pytorch weights
        pt_filenames = weight_hub_files(model_id, revision, extension=".bin")
        # Change pytorch extension to safetensors extension
        # It is possible that we have safetensors weights locally even though they are not on the
        # hub if we converted weights locally without pushing them
        filenames = [
            f"{Path(f).stem.lstrip('pytorch_')}.safetensors" for f in pt_filenames
        ]

    if WEIGHTS_CACHE_OVERRIDE is not None:
        files = []
        for filename in filenames:
            p = Path(WEIGHTS_CACHE_OVERRIDE) / filename
            if not p.exists():
                raise FileNotFoundError(
                    f"File {p} not found in {WEIGHTS_CACHE_OVERRIDE}."
                )
            files.append(p)
        return files

    files = []
    for filename in filenames:
        cache_file = try_to_load_from_cache(
            model_id, revision=revision, filename=filename
        )
        if cache_file is None:
            raise LocalEntryNotFoundError(
                f"File {filename} of model {model_id} not found in "
                f"{os.getenv('HUGGINGFACE_HUB_CACHE', 'the local cache')}. "
                f"Please run `text-generation-server download-weights {model_id}` first."
            )
        files.append(cache_file)

    return files


def download_weights(
    filenames: List[str], model_id: str, revision: Optional[str] = None
) -> List[Path]:
    """Download the safetensors files from the hub"""

    def download_file(filename, tries=5, backoff: int = 5):
        local_file = try_to_load_from_cache(model_id, revision, filename)
        if local_file is not None:
            logger.info(f"File {filename} already present in cache.")
            return Path(local_file)

        for i in range(tries):
            try:
                logger.info(f"Download file: {filename}")
                start_time = time.time()
                local_file = hf_hub_download(
                    filename=filename,
                    repo_id=model_id,
                    revision=revision,
                    local_files_only=False,
                )
                logger.info(
                    f"Downloaded {local_file} in {timedelta(seconds=int(time.time() - start_time))}."
                )
                return Path(local_file)
            except Exception as e:
                if i + 1 == tries:
                    raise e
                logger.error(e)
                logger.info(f"Retrying in {backoff} seconds")
                time.sleep(backoff)
                logger.info(f"Retry {i + 1}/{tries - 1}")

    # We do this instead of using tqdm because we want to parse the logs with the launcher
    start_time = time.time()
    files = []
    for i, filename in enumerate(filenames):
        file = download_file(filename)

        elapsed = timedelta(seconds=int(time.time() - start_time))
        remaining = len(filenames) - (i + 1)
        eta = (elapsed / (i + 1)) * remaining if remaining > 0 else 0

        logger.info(f"Download: [{i + 1}/{len(filenames)}] -- ETA: {eta}")
        files.append(file)

    return files
