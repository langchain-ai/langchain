from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="exllama_kernels",
    ext_modules=[
        CUDAExtension(
            name="exllama_kernels",
            sources=[
                "exllama_kernels/exllama_ext.cpp",
                "exllama_kernels/cuda_buffers.cu",
                "exllama_kernels/cuda_func/column_remap.cu",
                "exllama_kernels/cuda_func/q4_matmul.cu",
                "exllama_kernels/cuda_func/q4_matrix.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
