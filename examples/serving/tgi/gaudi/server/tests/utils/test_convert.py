from text_generation_server.utils.hub import (
    download_weights,
    weight_hub_files,
    weight_files,
)

from text_generation_server.utils.convert import convert_files


def test_convert_files():
    model_id = "bigscience/bloom-560m"
    pt_filenames = weight_hub_files(model_id, extension=".bin")
    local_pt_files = download_weights(pt_filenames, model_id)
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors" for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])

    found_st_files = weight_files(model_id)

    assert all([p in found_st_files for p in local_st_files])
