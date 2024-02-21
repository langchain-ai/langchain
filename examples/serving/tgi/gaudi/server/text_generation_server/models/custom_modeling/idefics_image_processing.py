# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Idefics."""

from typing import Callable, Dict, List, Optional, Union, Iterable
import numpy as np

from PIL import Image

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    resize,
    to_channel_dimension_format,
    rescale,
    normalize,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from io import BytesIO
import base64
import requests
from transformers import TensorType, is_torch_available


IDEFICS_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]
IDEFICS_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]


def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.
    Args:
        image_size (`int`, *optional*, defaults to `224`):
            Resize to image size
        image_num_channels (`int`, *optional*, defaults to `3`):
            Number of image channels.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 224,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        image_num_channels: Optional[int] = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size
        self.image_num_channels = image_num_channels
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(
        self,
        images: ImageInput,
        image_num_channels: Optional[int] = 3,
        image_size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        transform: Callable = None,
        **kwargs,
    ) -> TensorType.PYTORCH:
        """
        Preprocess a batch of images.
        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Resize to image size
            image_num_channels (`int`, *optional*, defaults to `self.image_num_channels`):
                Number of image channels.
            image_mean (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`):
                Mean to use if normalizing the image. This is a float or list of floats the length of the number of
                channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
                be overridden by the `image_mean` parameter in the `preprocess` method.
            image_std (`float` or `List[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`):
                Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
                number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
                method. Can be overridden by the `image_std` parameter in the `preprocess` method.
            transform (`Callable`, *optional*, defaults to `None`):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
                assumed - and then a preset of inference-specific transforms will be applied to the images
        Returns:
            a PyTorch tensor of the processed images
        """
        image_size = image_size if image_size is not None else self.image_size
        image_num_channels = (
            image_num_channels
            if image_num_channels is not None
            else self.image_num_channels
        )
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = (image_size, image_size)

        if len(images) == 0:
            return []

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # For training a user needs to pass their own set of transforms as a Callable.
        # For reference this is what was used in the original IDEFICS training:
        # transform = transforms.Compose([
        #     convert_to_rgb,
        #     transforms.RandomResizedCrop((size, size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=image_mean, std=image_std),
        # ])
        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in `transform` torch must be installed")
            import torch

            images = [transform(x) for x in images]
            return torch.stack(images)

        # for inference we do the exact transforms that were used to train IDEFICS
        images = [convert_to_rgb(x) for x in images]
        # further transforms expect numpy arrays
        images = [to_numpy_array(x) for x in images]
        images = [resize(x, size, resample=PILImageResampling.BICUBIC) for x in images]
        images = [self.rescale(image=image, scale=1 / 255) for image in images]
        images = [self.normalize(x, mean=image_mean, std=image_std) for x in images]
        images = [
            to_channel_dimension_format(x, ChannelDimension.FIRST) for x in images
        ]
        # TODO: this converts to torch tensors - switch to convert_to_tensors once it becomes available
        images = BatchFeature(
            data={"pixel_values": images}, tensor_type=TensorType.PYTORCH
        )["pixel_values"]

        return images

    def fetch_images(self, image_url_or_urls: Union[str, List[str]]):
        """
        Convert a single or a list of urls into the corresponding `PIL.Image` objects.
        If a single url is passed, the return value will be a single object. If a list is passed a list of objects is
        returned.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                " Safari/537.36"
            )
        }
        if isinstance(image_url_or_urls, list):
            return [self.fetch_images(x) for x in image_url_or_urls]
        elif isinstance(image_url_or_urls, str):
            image = image_url_or_urls

            if image.startswith("http://") or image.startswith("https://"):
                response = requests.get(image_url_or_urls, stream=True, headers=headers, timeout=(1, 5))
                response.raise_for_status()
                content = response.content
            elif image.startswith("data:"):
                # https://stackoverflow.com/questions/17090571/is-there-a-way-to-set-background-image-as-a-base64-encoded-image
                # data:image/png;base64,xxx
                image = image.split(",")[-1]
                content = base64.b64decode(image)
            else:
                raise ValueError(f"Unrecognized image {image}")

            try:
                image = Image.open(BytesIO(content))
                # image.verify()
            except Exception:
                raise ValueError(f"Could not load image from url {image_url_or_urls}")    
            return image
        else:
            raise ValueError(
                f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}"
            )

    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The rescaled image.
        """
        # return rescale(image, scale=scale, data_format=data_format, input_data_format=input_data_format, **kwargs)
        # requires 4.32
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Image mean to use for normalization.
            std (`float` or `Iterable[float]`):
                Image standard deviation to use for normalization.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The normalized image.
        """
        # TODO 4.32
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)


import transformers

transformers.IdeficsImageProcessor = IdeficsImageProcessor
