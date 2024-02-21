from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

extra_compile_args = ["-std=c++17"]
if not torch.version.hip:
    extra_compile_args.append("-arch=compute_80")

setup(
    name="custom_kernels",
    ext_modules=[
        CUDAExtension(
            name="custom_kernels.fused_bloom_attention_cuda",
            sources=["custom_kernels/fused_bloom_attention_cuda.cu"],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="custom_kernels.fused_attention_cuda",
            sources=["custom_kernels/fused_attention_cuda.cu"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
